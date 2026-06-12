# =============================
# auth.py
# =============================
"""Simple login gate for the app, using only the Python standard library.

Credentials live in .streamlit/secrets.toml as PBKDF2-hashed passwords —
never plaintext. Fails CLOSED: if no users are configured, the app shows
setup instructions and refuses to load.

Generate a password hash:
    python auth.py mypassword

Then put the printed line into .streamlit/secrets.toml:
    [auth.users]
    admin = "<salt$hash printed by the command>"

NEVER commit secrets.toml to git. On Streamlit Community Cloud, paste the
same TOML into the app's Settings -> Secrets instead of using a file.
"""

import hashlib
import hmac
import secrets as pysecrets
import time

PBKDF2_ITERATIONS = 200_000


def hash_password(password, salt=None):
    """Return 'salt$hash' using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = pysecrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored):
    """Constant-time check of a password against a 'salt$hash' string."""
    try:
        salt, _ = stored.split("$", 1)
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)


def _get_users():
    import streamlit as st
    try:
        return dict(st.secrets["auth"]["users"])
    except (KeyError, FileNotFoundError):
        return {}


def check_credentials(username, password, users=None):
    if users is None:
        users = _get_users()
    stored = users.get(username)
    if stored is None:
        # Burn the same time as a real check so usernames can't be probed
        hash_password(password, "0" * 32)
        return False
    return verify_password(password, stored)


def require_login():
    """Call once at the top of app.py (after set_page_config). Renders the
    login page and halts the script until authenticated; afterwards returns
    the username."""
    import streamlit as st

    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    users = _get_users()

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.title("🔒 AI Stock Trend Predictor")

        if not users:
            st.error("No users configured — the app is locked by default.")
            st.markdown(
                "**Setup (one time):**\n"
                "1. Generate a password hash:\n"
                "```bash\npython auth.py yourpassword\n```\n"
                "2. Create `.streamlit/secrets.toml` next to app.py and paste:\n"
                "```toml\n[auth.users]\nadmin = \"<the salt$hash it printed>\"\n```\n"
                "3. Restart the app. Add one line per user for more accounts.\n\n"
                "⚠️ Never commit `secrets.toml` to git. On Streamlit Cloud, "
                "paste the TOML into **Settings → Secrets** instead."
            )
            st.stop()

        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary",
                                              use_container_width=True)

        if submitted:
            attempts = st.session_state.get("auth_attempts", 0)
            if attempts >= 3:
                time.sleep(min(attempts, 8))  # slow down brute force
            if check_credentials(username.strip(), password, users):
                st.session_state["auth_user"] = username.strip()
                st.session_state["auth_attempts"] = 0
                st.rerun()
            else:
                st.session_state["auth_attempts"] = attempts + 1
                st.error("Invalid username or password.")

        st.caption("Access is restricted. Contact the app owner for an account.")

    st.stop()


def logout_button():
    """Sidebar logout control; call inside `with st.sidebar:`."""
    import streamlit as st
    st.caption(f"Logged in as **{st.session_state.get('auth_user', '?')}**")
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state.pop("auth_user", None)
        st.rerun()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python auth.py <password>")
        sys.exit(1)
    print("\nAdd this to .streamlit/secrets.toml :\n")
    print("[auth.users]")
    print(f'admin = "{hash_password(sys.argv[1])}"')
    print("\n(change 'admin' to any username; one line per user)")
