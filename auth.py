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
    import base64
    from pathlib import Path

    import streamlit as st

    # Theme CSS on login too (before authenticated shell).
    try:
        from ui.theme import footer_bar, inject_global_css
        inject_global_css()
    except Exception:
        footer_bar = None

    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    users = _get_users()

    _icon_b64 = ""
    for _name in ("icon_256.png", "icon_128.png", "icon.png"):
        _p = Path(__file__).parent / "src" / _name
        if _p.exists():
            _icon_b64 = base64.b64encode(_p.read_bytes()).decode()
            break

    # Title uses Streamlit heading + safe top padding so it never sits under
    # the fixed Deploy toolbar (that was clipping the top of "AI Stock…").
    _, mid, _ = st.columns([1, 2.4, 1])
    with mid:
        if _icon_b64:
            st.markdown(
                f'<div style="text-align:center;padding:0.85rem 0 0.35rem 0;">'
                f'<img src="data:image/png;base64,{_icon_b64}" alt="logo" '
                f'width="64" height="64" style="border-radius:16px;" /></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:center;padding:0.85rem 0 0.35rem 0;"
                "font-size:2.4rem;'>📈</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<h2 style='text-align:center;margin:0.25rem 0 0.2rem 0;"
            "line-height:1.3;font-weight:800;'>AI Stock Predictor</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;margin:0 0 0.4rem 0;"
            "color:rgba(232,236,241,0.65);font-size:0.95rem;line-height:1.45;'>"
            "Machine-learning signals · honest backtests · trade planning</p>",
            unsafe_allow_html=True,
        )
    st.divider()

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

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("**What you get**")
        features = [
            ("📊", "Interactive Charts", "Candlesticks, MAs, RSI, S/R levels"),
            ("🤖", "AI Signals", "Regularized ensemble · multi-horizon outlook"),
            ("📉", "Honest Backtests", "Walk-forward · costs · random benchmark"),
            ("🎯", "Trade Planning", "Stops, targets, position sizing"),
            ("📝", "Signal Journal", "Forward-test every call you log"),
            ("🌐", "Global Model", "Pooled training when artifacts are present"),
        ]
        for icon, title, desc in features:
            st.markdown(
                f"""
<div style="display:flex;gap:0.7rem;align-items:flex-start;padding:0.65rem 0.75rem;
            margin-bottom:0.4rem;border-radius:12px;background:#161d27;
            border:1px solid rgba(255,255,255,0.08);border-left:3px solid #36b37e;">
  <div style="font-size:1.25rem;line-height:1;">{icon}</div>
  <div>
    <div style="font-weight:650;color:#e8ecf1;font-size:0.95rem;">{title}</div>
    <div style="font-size:0.82rem;color:rgba(232,236,241,0.6);margin-top:0.1rem;">{desc}</div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("**Secure login**")
        with st.form("login"):
            username = st.text_input("Username", placeholder="your username")
            password = st.text_input(
                "Password", type="password", placeholder="your password",
            )
            submitted = st.form_submit_button(
                "Enter dashboard", type="primary", use_container_width=True,
            )

        if submitted:
            attempts = st.session_state.get("auth_attempts", 0)
            if attempts >= 3:
                time.sleep(min(attempts, 8))
            if check_credentials(username.strip(), password, users):
                st.session_state["auth_user"] = username.strip()
                st.session_state["auth_attempts"] = 0
                st.rerun()
            else:
                st.session_state["auth_attempts"] = attempts + 1
                st.error("Invalid username or password.")

        st.caption(
            "Restricted access · request credentials at **soumoster@gmail.com**"
        )

    if footer_bar is not None:
        footer_bar(
            "Educational purposes only — not financial advice · © 2026 Soumoster Analytics"
        )
    else:
        st.caption("Educational purposes only — not financial advice.")

    st.stop()


def logout_button():
    """Sidebar logout control; call inside `with st.sidebar:`."""
    import streamlit as st

    user = st.session_state.get("auth_user", "?")
    st.caption(f"Signed in as **{user}**")
    if st.button("Log out", use_container_width=True):
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
