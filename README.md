# AI Stock Trend Predictor

An access-controlled Streamlit web app for AI-assisted stock analysis. It is
designed primarily for NSE stocks, but it can analyze any ticker supported by
Yahoo Finance.

The app predicts the probability of a meaningful future up move, turns that
probability into a long-only trading signal, validates the signal with
backtests and walk-forward testing, and helps the user plan position size,
stop loss, target, and risk.

> Important: This project is for education and decision support only. It is not
> investment advice, financial advice, or a recommendation service. In India,
> providing paid stock recommendations may require SEBI registration. Past
> performance does not guarantee future returns.

---

## Table of Contents

- [Features](#features)
- [What the App Predicts](#what-the-app-predicts)
- [Signal vs Scanner Screen Call](#signal-vs-scanner-screen-call)
- [How the Model Works](#how-the-model-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Local Setup](#local-setup)
- [Authentication Setup](#authentication-setup)
- [Running the App](#running-the-app)
- [Deployment](#deployment)
- [Testing](#testing)
- [Custom Stock Universe](#custom-stock-universe)
- [Known Limitations](#known-limitations)
- [Security Notes](#security-notes)
- [License](#license)

---

## Features

### Stock Analysis

- Select a stock from `stocks.csv` or enter a custom Yahoo Finance ticker.
- View latest close, daily move, and a 30-day sparkline.
- Analyze NSE tickers such as `RELIANCE.NS`, `INFY.NS`, `NETWEB.NS`, and any
  other Yahoo Finance-compatible symbol.
- A data-quality gate cleans bad ticks, non-positive prices, and unadjusted
  splits before any feature is computed, so corrupt input can't distort the model.
- Watchlist data is fetched in one batched request (not one call per stock),
  which is faster and far more resistant to Yahoo rate limits.

### AI Model Options

The app supports multiple model types:

- Ensemble: Neural Network + Random Forest + XGBoost, when XGBoost is installed.
- Neural Network: tabular PyTorch MLP.
- LSTM: sequence model using the latest 20 trading days.
- GRU: sequence model using the latest 20 trading days.
- Global model: a single model pre-trained on the pooled history of every stock
  in the watchlist (run `train_global.py` offline). When its artifacts are
  present the app prefers it automatically, falling back to a per-stock model
  otherwise. Toggle it from the sidebar.

All model types use the same feature pipeline, chronological train-validation-
test split, threshold tuning, backtesting, and explainability flow.

### Signal Generation

The main stock header shows one full-analysis signal:

- `BUY`: model probability is above the tuned entry threshold.
- `HOLD`: model probability is between the tuned entry and exit thresholds.
- `SELL`: model probability is below the tuned exit threshold.

This is a long-only signal. `SELL` means exit to cash or avoid a fresh long
entry. It does not mean short sell.

### Scanner

The Scanner tab screens the entire watchlist quickly. It uses a faster model
and default thresholds so that the app can rank many stocks at once.

The scanner output is labelled as a `Screen Call`, not the final trading
signal. Open the stock for the full model signal before making any decision.

### Risk and Trade Planning

- Auto-detected support and resistance from recent swing highs/lows.
- ATR-based stop loss and target.
- Reward-to-risk ratio.
- Position sizing calculator based on capital and risk per trade.
- Risk score from volatility, ATR, and one-year drawdown.

### Validation and Explainability

- Out-of-sample accuracy compared with a baseline.
- Backtest vs buy-and-hold.
- Sharpe ratio, max drawdown, exposure, win rate, and trade count.
- Walk-forward validation across multiple market periods.
- Probability calibration metrics: Brier score and expected calibration error.
- Optional isotonic calibration.
- Feature attribution showing what pushed the prediction up or down.
- Random-signal benchmark: ranks the strategy's Sharpe against hundreds of
  random strategies of the same trade frequency, to show whether the edge is
  real or just luck.

### Journal

- Log the app's signal and trade plan.
- Later resolve logged BUY plans against real price action.
- Tracks target hit, stop hit, expired plans, win rate, and average return.

---

## What the App Predicts

The model predicts whether the stock will make a meaningful up move over a
future horizon.

For the 1-day signal, the target is:

```text
Target_1 = 1 if next-day forward return > noise threshold
Target_1 = 0 otherwise
```

The noise threshold is defined in `data.py` and scales by horizon:

```text
threshold = NOISE_THRESHOLD * sqrt(horizon)
```

This means the probability shown by the app is not simply "will the price close
green by any tiny amount?" It is closer to:

```text
What is the probability of a meaningful up move?
```

---

## Signal vs Scanner Screen Call

This distinction is important.

### Full Signal

The signal at the top of the app is the full analysis signal for the selected
stock. It uses:

- The selected model type from the sidebar.
- The full feature set.
- Per-stock validation-based threshold tuning.
- Optional probability calibration.
- The latest available data for the selected ticker.

This is the primary signal to trust inside the app.

### Scanner Screen Call

The Scanner tab is a fast watchlist ranking tool. It uses:

- A lightweight tree model.
- Default thresholds.
- No per-stock threshold tuning.
- A speed-first workflow designed to scan many symbols.

Because of this, a stock can appear as `BUY` in the Scanner but show `SELL` in
the full analysis header. That is expected behavior, not a bug.

When they disagree:

```text
Use the full Signal for the selected stock.
Use the Scanner only to find names worth opening.
```

The app also displays a warning when the selected stock's Scanner call disagrees
with the full analysis signal.

---

## How the Model Works

### 1. Fetch Market Data

Daily OHLCV data is downloaded from Yahoo Finance through `yfinance`.

For Indian stocks, the app also attempts to fetch NIFTY 50 data through:

```text
^NSEI
```

If NIFTY data is unavailable, market-context features fall back to neutral
values so the app can continue running.

### 2. Engineer Features

The model uses 15 scale-free features:

Trend:

- `Close_MA20`
- `MA20_MA50`
- `MACD_hist`

Momentum:

- `Return`
- `Mom5`
- `Mom10`
- `Mom20`
- `RSI`

Volatility:

- `Vol20`
- `ATR_pct`

Volume:

- `Vol_ratio`

Market context:

- `Nifty_Ret`
- `Nifty_Mom20`
- `Rel_Str5`
- `Rel_Str20`

### 3. Create Targets

The app creates targets for multiple horizons:

```text
1, 3, 5, 10, and 20 trading days
```

The final rows that do not have future data are kept for live prediction but
excluded from training.

### 4. Split Chronologically

The main model uses a chronological split:

```text
64% train
16% validation
20% test
```

The scaler is fit only on the training slice to avoid look-ahead leakage.

### 5. Train Model

The selected model estimates the probability that the target is 1.

For the ensemble:

- Neural Network predicts probability.
- Random Forest predicts probability.
- XGBoost predicts probability, if installed.
- Final probability is the average of member probabilities.

### 6. Tune Thresholds

The app does not blindly use 50% as the trading cutoff.

Instead, it tunes two thresholds on the validation slice:

- Entry threshold: above this probability, signal `BUY`.
- Exit threshold: below this probability, signal `SELL`.

The objective is after-cost Sharpe ratio.

### 7. Evaluate on Test Data

The untouched test slice is used for:

- Accuracy.
- Baseline accuracy.
- Precision.
- Recall.
- Backtest.
- Calibration curve.
- Brier score.
- Expected calibration error.

### 8. Refit for Live Signal

After evaluation, the app refits a live model on all rows with known targets so
the displayed prediction uses the latest available history.

### Global Model (optional, pre-trained)

Instead of training one model per stock from about 1,200 rows, you can train a
single global model on the pooled history of the entire watchlist
(50,000+ rows) by running `python train_global.py`. This saves one model per
horizon into `global_models/`, which you commit to the repo. When present, the
app prefers the global model automatically and falls back to per-stock training
when it is missing or a stock has too little history. Thresholds and all
reported metrics stay stock-specific, so the comparison remains fair. Retrain
monthly as more real market history accumulates — never on the model's own
predictions.

---

## Project Structure

```text
.
├── app.py              # Streamlit UI and app workflow
├── auth.py             # Login gate and PBKDF2 password hashing
├── data.py             # Data download, feature engineering, targets
├── model.py            # Models, training, signals, backtests, trade planning
├── journal.py          # Signal journal and forward-test scoring
├── train_global.py     # Offline trainer for the pooled global model
├── stocks.csv          # Default watchlist
├── requirements.txt    # Python dependencies (pinned)
├── DEPLOYMENT.md       # Streamlit Community Cloud deployment guide
├── LICENSE             # Unlicense / public domain dedication
├── .streamlit/config.toml      # Theme (safe to commit; no secrets)
├── .github/workflows/ci.yml    # Runs the test suite on every push
├── tests/              # Unit + static integrity tests (conftest, test_*.py)
└── global_models/      # Created by train_global.py, then committed
                        #   global_h1..h20.joblib + global_meta.json
```

---

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- yfinance
- Plotly
- PyTorch
- scikit-learn
- XGBoost

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` uses the CPU-only PyTorch wheel for Streamlit Community
  Cloud.
- The pinned CPU wheel works best with Python 3.12 or lower.
- If the CPU-only torch pin fails locally on macOS/Linux, replace the torch
  lines with:

```text
torch>=2.0
```

---

## Authentication Setup

The app fails closed. If no user is configured, it will not load the stock
analysis UI.

Generate a password hash:

```bash
python auth.py yourpassword
```

Create this file:

```text
.streamlit/secrets.toml
```

Paste the generated block:

```toml
[auth.users]
admin = "salt$hash"
```

You can add multiple users:

```toml
[auth.users]
admin = "salt$hash"
user2 = "salt$hash"
```

Never commit `.streamlit/secrets.toml`.

---

## Running the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

Typical workflow:

1. Log in.
2. Select a stock from the sidebar.
3. Choose a model type.
4. Review the full Signal at the top.
5. Check the Probability tab, Trade Plan, Backtest, Walk-Forward, and Charts.
6. Use Scanner only as a watchlist ranking tool.
7. Log signals in the Journal if you want forward-test tracking.

---

## Deployment

The repository includes `DEPLOYMENT.md` for Streamlit Community Cloud.

Summary:

1. Push the repository to GitHub.
2. Create a Streamlit Community Cloud app.
3. Set the main file path to:

```text
app.py
```

4. Set Python version to 3.12 if available.
5. Paste your `[auth.users]` block into Streamlit Secrets.
6. Deploy.

Important cloud notes:

- The first build can take several minutes because of PyTorch.
- `journal.csv` is local storage and may be lost on cloud restarts.
- Yahoo Finance can rate-limit shared cloud IPs.
- Cached data and models are refreshed periodically.

---

## Testing

Run the test suite with:

```bash
pytest
```

Or run a focused test file:

```bash
pytest tests/test_model.py
```

The tests cover:

- Model utility math.
- Signal threshold behavior.
- Feature engineering behavior.
- Journal resolution.
- Auth hashing and validation.
- Static app integrity checks.

If `pytest` is missing:

```bash
pip install pytest
```

---

## Custom Stock Universe

The default stock list is stored in:

```text
stocks.csv
```

Expected columns:

```csv
Name,Symbol
Reliance Industries,RELIANCE.NS
Infosys,INFY.NS
```

You can also upload a CSV from the sidebar at runtime. Uploaded lists replace
the default list for that session.

---

## Known Limitations

- Daily single-stock direction is difficult to predict. Accuracy near 50% can
  be normal in honest out-of-sample testing.
- The app is a decision-support tool, not an automated trading system.
- Scanner calls are not full signals.
- Backtests assume simplified execution and approximate transaction costs.
- Long-horizon labels overlap, so long-horizon accuracy can look optimistic.
- yfinance data can be delayed, revised, missing, or temporarily rate-limited.
- Streamlit Community Cloud storage is ephemeral, so journal data should be
  downloaded regularly.
- Model performance can change across regimes.

---

## Security Notes

- Passwords are stored as PBKDF2-HMAC-SHA256 hashes.
- Password checks use constant-time comparison.
- The app slows repeated failed login attempts.
- The app refuses to load if no users are configured.
- Do not commit secrets, journal files, or private credentials.

Recommended `.gitignore` entries:

```gitignore
.streamlit/secrets.toml
journal.csv
__pycache__/
.venv/
```

---

## License

This project is released under the Unlicense. See `LICENSE` for details.

Market data is fetched through Yahoo Finance via `yfinance`; use is subject to
Yahoo Finance terms and data availability.

---

## Disclaimer

This software is provided "as is" without warranty. Use it at your own risk.
Always do your own research before making any financial decision.
