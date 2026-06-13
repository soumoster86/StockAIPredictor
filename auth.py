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

    # ---- Hero header ----
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
            <div style="font-size:2.3rem; font-weight:800; letter-spacing:-0.02em;">
                📈 AI Stock Trend Predictor
            </div>
            <div style="font-size:1.05rem; color: rgba(230,233,239,0.6);
                        margin-top:0.25rem;">
                Machine-learning powered stock analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ---- First-run setup gate (fail closed) ----
    if not users:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
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

    # ---- Two-column hero: preview | login ----
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### What's inside")
        features = [
            ("📊", "Interactive Charts", "Candlesticks, moving averages, RSI"),
            ("🤖", "AI Signals", "Ensemble of neural net, XGBoost & forests"),
            ("📈", "Multi-Day Predictions", "1 / 3 / 5 / 10 / 20-day outlooks"),
            ("📉", "Honest Backtesting", "Walk-forward validation vs. buy & hold"),
            ("🎯", "Trade Planning", "Auto support/resistance, stops & sizing"),
            ("📝", "Signal Journal", "Every call scored against real prices"),
        ]
        for icon, title, desc in features:
            st.markdown(
                f"""
                <div style="display:flex; align-items:flex-start; gap:0.7rem;
                            padding:0.55rem 0.7rem; margin-bottom:0.45rem;
                            background:rgba(54,179,126,0.06);
                            border-left:3px solid #36b37e; border-radius:6px;">
                    <div style="font-size:1.4rem; line-height:1;">{icon}</div>
                    <div>
                        <div style="font-weight:600;">{title}</div>
                        <div style="font-size:0.85rem; color:rgba(230,233,239,0.6);">
                            {desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### 🔒 Secure Login")
        with st.form("login"):
            username = st.text_input("Username", placeholder="your username")
            password = st.text_input("Password", type="password",
                                      placeholder="your password")
            submitted = st.form_submit_button("Access Dashboard ➜", type="primary",
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

        st.caption("Access is restricted. Email **soumoster@gmail.com** "
                   "to request credentials.")

    # ---- Fixed footer ----
    st.markdown(
        """
        <div style="
            position: fixed; left: 0; bottom: 0; width: 100%;
            text-align: center; padding: 0.6rem 1rem;
            background: rgba(14, 17, 23, 0.92);
            color: rgba(250, 250, 250, 0.6);
            font-size: 0.8rem; z-index: 999;
            border-top: 1px solid rgba(255,255,255,0.06);">
            ⚠️ Educational purposes only — not financial advice &nbsp;·&nbsp;
            © 2026 Soumoster Analytics
        </div>
        """,
        unsafe_allow_html=True,
    )

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
