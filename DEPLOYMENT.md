# Deploying to Streamlit Community Cloud

## 1. Prepare the GitHub repository

Files to commit:
```
app.py  ui/  model.py  data.py  journal.py  report.py  alerts.py  auth.py
train_global.py  stocks.csv  stocks_universe.csv  requirements.txt
requirements-dev.txt  .gitignore  .gitattributes  DEPLOYMENT.md
.github/workflows/ci.yml  scripts/check_models.py
global_models/   (optional pre-trained artifacts; prefer Git LFS)
```

Files that must NEVER be committed (already in `.gitignore`):
- `.streamlit/secrets.toml` — your passwords live here
- `journal.csv`, `journals/` — personal per-user forward-test data
- `alerts/` — alert de-dupe state
- `__pycache__/`, `.venv/`

```bash
git init
git add .
git commit -m "AI Stock Trend Predictor"
# create a PRIVATE repo on github.com, then:
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
A private repo is recommended since the app is access-restricted anyway.

## 2. Generate your login credentials

```bash
python auth.py yourpassword
```
Copy the printed `[auth.users]` block — you will paste it into the cloud
secrets panel in step 4. Do NOT create a secrets.toml in the repo.

## 3. Create the app

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **Create app** → "Yup, I have an app"
3. Repository: `<you>/<repo>` · Branch: `main` · Main file path: `app.py`
4. Pick a custom subdomain (e.g. `yourname-stock-app`)

## 4. Advanced settings (BEFORE clicking Deploy)

- **Python version: 3.12** (recommended; matches common torch CPU wheels)
- **Secrets**: paste the `[auth.users]` block from step 2

(Both can also be changed later under App → Settings.)

## 5. Deploy and verify

Click Deploy. The first build takes several minutes (torch). Then check:
- [ ] Login page appears; wrong password rejected; right one works
- [ ] A stock loads end-to-end (prediction, backtest, charts)
- [ ] Log a journal entry, confirm it appears

## Known limitations on Community Cloud

- **Journal storage**:
  - Default is **local CSV** under `journals/` — wiped on Cloud restart.
  - For permanence, enable **Supabase** (see below). Download CSV from the
    Journal tab as a backup either way.

### Nightly rankings (autopilot)

Workflow: `.github/workflows/nightly-rankings.yml`

- **Schedule:** weekdays 18:00 UTC (after NSE close)
- **Manual:** GitHub → Actions → *Nightly rankings* → *Run workflow*
- Writes/commits `rankings/rankings_latest.csv` + `rankings_meta.json`
- Uses `requirements-rankings.txt` (lighter than full UI deps)
- Commit message includes `[skip ci]`; CI also ignores `rankings/**` paths

Streamlit Cloud redeploys when `main` updates, so the Screener picks up fresh
precomputed results after each successful nightly run.

If the job fails often (Yahoo rate limits), increase `pause` in the workflow
dispatch inputs (e.g. `0.8`) or lower batch size.

### Optional: Supabase journal (survives redeploys)

1. Create a free project at https://supabase.com
2. SQL Editor → run `scripts/supabase_journal.sql`
3. Project Settings → API → copy **Project URL** and **service_role** key
4. Streamlit Cloud → App settings → Secrets:

```toml
[journal]
backend = "supabase"
supabase_url = "https://YOUR_PROJECT.supabase.co"
supabase_key = "YOUR_SERVICE_ROLE_KEY"
table = "signal_journal"
```

5. Redeploy. The Journal tab should show **Supabase (cloud-persistent)**.

### Optional: Alerts (Telegram / email)

When the Screener has scored results, the app can notify you of new top BUY
screens. De-dupes per rankings snapshot under `alerts/state.json` (local only;
not committed).

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather), get the token,
   then message your bot and fetch `chat_id` from
   `https://api.telegram.org/bot<token>/getUpdates`
2. Streamlit Cloud → App settings → Secrets:

```toml
[alerts]
enabled = true
min_buy_score = 60
min_probability = 0.55
top_n = 10
require_edge = true
max_risk = 8.0
telegram_bot_token = "123456:ABC..."
telegram_chat_id = "987654321"
# optional email
# smtp_host = "smtp.gmail.com"
# smtp_port = 587
# smtp_user = "you@gmail.com"
# smtp_password = "app-password"
# email_to = "you@gmail.com"
# email_from = "you@gmail.com"
```

3. In the app → **Screener** → **Alerts** → use **Dry run** first, then send.
   Set `enabled = true` for production; the UI can still force a send when
   channels are configured.

### One-click PDF / CSV report

On **Prediction**, download a CSV or PDF pack (signal, risk, plan, metrics) for
the selected stock. PDF needs `reportlab` (already in `requirements.txt`).

- **Resource limits (~1 GB)**: avoid opening many stocks × model types in
  one session; the app caps its model cache, but heavy use can still hit
  "over its resource limits" → reboot the app from the cloud dashboard.
- **yfinance rate limits**: cloud IPs are shared, so Yahoo sometimes
  refuses requests. "Could not fetch data" usually fixes itself within
  the hour (data is cached for 60 min once fetched). The default
  default watchlist is the full `stocks_universe.csv`; the Screener still
  hard-caps at 80 symbols per run so Yahoo/Cloud limits stay usable.
- **App sleeps** after ~12h of no traffic; first visitor wakes it (~1 min).
- **Login is per-session**: a hard refresh requires logging in again.

## Updating the app

Just `git push` — Community Cloud redeploys automatically.
