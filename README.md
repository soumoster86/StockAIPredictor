# 📈 AI Stock Trend Predictor

An access-controlled, end-to-end web application for AI-assisted stock analysis — built for NSE stocks, working with any Yahoo Finance ticker.

It is **not a crystal ball**. Daily stock direction is close to noise, and this project is honest about that. Instead, it imposes the *process* professional traders use and retail traders skip: probabilistic signals instead of certainty, systematic stops and position sizing, regime-aware validation, calibrated confidence, and a forward-testing journal that scores every signal against what the market actually did.

> ⚠️ **Educational and decision-support tool only — not investment advice.** In India, providing paid stock recommendations requires SEBI registration. Past performance does not predict future results.

---

## ✨ Features

| Area | What it does |
|---|---|
| 🔮 **Multi-horizon prediction** | Probability of an up-move over 1 / 3 / 5 / 10 / 20 days, each from its own model, with per-horizon out-of-sample accuracy vs. baseline |
| 🤖 **Selectable model engine** | Soft-voting **Ensemble (Neural Net + XGBoost + Random Forest)**, plus standalone NN, **LSTM**, and **GRU** sequence models — all flowing through identical validation machinery |
| 🗳️ **Model agreement** | Individual model votes shown side by side; disagreement flagged as lower conviction |
| 📊 **Honest backtesting** | Long-only (no overnight shorts — matching Indian cash-equity rules), realistic transaction costs charged per position change, Sharpe ratio, max drawdown, exposure, and a buy-and-hold benchmark on every chart |
| 🧪 **Walk-forward validation** | Expanding-window retraining across market regimes — fold-by-fold results that expose "lucky period" performance |
| 📏 **Probability calibration** | Reliability curve, Brier score vs. baseline, ECE, and an optional isotonic-regression fix so that "70%" actually means 70% |
| 💡 **Explainable signals** | Occlusion-based attribution: every BUY/SELL lists the factors pushing it ("MACD bullish crossover +4.2%", "Volatility elevated −2.1%") |
| 🎯 **Trade planning** | Auto-detected support & resistance (swing highs/lows, clustered), ATR-based stop loss and target with reward:risk, overlaid on the chart |
| 🧮 **Position sizing** | The professional formula — shares = (capital × risk%) ÷ (entry − stop) — so a stopped-out trade costs exactly the risk you chose |
| ⚖️ **Risk score** | 1–10 blend of annualized volatility, ATR, and worst 1-year drawdown, with position-sizing guidance |
| 📝 **Signal journal** | One-click logging of each signal + trade plan; later scored against real price action (target hit / stop hit / expired) with a running scorecard — the app's forward test |
| 📋 **Configurable universe** | Stock list lives in `stocks.csv` (editable) or upload your own watchlist CSV at runtime |
| 🔒 **Login gate** | Fails closed by default; PBKDF2-hashed multi-user credentials in Streamlit secrets; brute-force slowdown |
| 📖 **Built for non-traders too** | Every metric has a plain-language tooltip and there's a full glossary — Sharpe, drawdown, R:R all explained |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) (tabs, forms, widgets), [Plotly](https://plotly.com/python/) (candlesticks, gauges, equity & reliability curves) |
| Data | [yfinance](https://github.com/ranaroussi/yfinance) (market data), pandas / NumPy (feature engineering), CSV (stock universe, journal) |
| ML | [PyTorch](https://pytorch.org) (MLP, LSTM, GRU), [scikit-learn](https://scikit-learn.org) (Random Forest, scaling, isotonic calibration), [XGBoost](https://xgboost.readthedocs.io) |
| Security | Python standard library auth (PBKDF2-HMAC-SHA256, constant-time comparison), Streamlit secrets |
| Hosting | Streamlit Community Cloud (CPU-only PyTorch wheels), GitHub auto-deploy |

**Architecture** — five focused modules behind a unified predictor interface (`fit` / `predict_all` / `predict_last`), so every model type shares the same training splits, threshold tuning, backtest, and explainability code:

```
app.py        Streamlit UI: tabs, charts, tooltips, calculators
model.py      Models, ensemble, validation, backtesting, trade planning
data.py       Data fetching and feature engineering (11 scale-free features)
journal.py    Signal logging and forward-test resolution
auth.py       Login gate and password hashing (also a CLI hash generator)
stocks.csv    Editable stock universe for the dropdown
```

---

## 🚀 Quickstart (local)

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt

# One-time: create your login credentials
python auth.py yourpassword
mkdir -p .streamlit
# paste the printed [auth.users] block into .streamlit/secrets.toml

streamlit run app.py
```

> 💡 On macOS, if the pinned `torch==2.5.1+cpu` fails to install, remove the two torch lines in `requirements.txt` and use `torch>=2.0` instead (the CPU pin exists for cloud deployment).

**Never commit `.streamlit/secrets.toml`** — it's already in `.gitignore`.

---

## ☁️ Deployment

Full step-by-step guide in [`DEPLOYMENT.md`](DEPLOYMENT.md) — covers the GitHub setup, Streamlit Community Cloud configuration (Python 3.12 + secrets), the CPU-only PyTorch trick that keeps the build inside the cloud's resource limits, and the platform's known limitations (ephemeral journal storage, yfinance rate limits, app sleep).

---

## 🔬 How a prediction is made

1. **Fetch** five years of daily OHLCV data for the selected ticker.
2. **Engineer 11 scale-free features** across four groups — trend (price vs. MA20, MA20 vs. MA50, normalized MACD histogram), momentum (1/5/10/20-day returns, RSI), volatility (rolling σ, ATR%), volume (vs. 20-day average).
3. **Label** each day per horizon: did the forward return beat a noise threshold that scales with √horizon? Look-ahead rows stay `NaN` — never silently mislabeled.
4. **Split chronologically** (64% train / 16% validation / 20% test); fit the scaler on train only — no leakage.
5. **Train** the selected model(s); class imbalance corrected; seeded for reproducibility.
6. **Tune** entry/exit signal thresholds on the validation slice by maximizing after-cost Sharpe; optionally calibrate probabilities with isotonic regression.
7. **Report** everything against the untouched test slice — accuracy vs. majority-class baseline, cost-aware backtest vs. buy-and-hold, calibration quality.
8. **Plan the trade** — support/resistance from swing-point clustering, ATR-anchored stop and target, position size from the user's capital and risk tolerance.
9. **Log and verify** — journal the signal; the app scores it later against real price action.

---

## 🧭 Honest limitations

- Daily single-stock direction is extremely hard to predict; a ~50% walk-forward win rate is the *expected* honest outcome, and the app is designed to tell you so rather than hide it.
- Backtests assume trades at the close and approximate Indian costs at ~0.2% per round trip; live slippage varies.
- The journal stores to a local CSV — ephemeral on cloud hosts; download it regularly.
- Long-horizon accuracy figures are optimistic due to overlapping label windows (flagged in-app).

## 🗺️ Roadmap

Watchlist scanner across the full stock universe · NIFTY market-context features · random-signal benchmark for the walk-forward results · persistent journal backend (Google Sheets / Supabase) · per-user journal filtering.

---

## 📄 License & attribution

Released for educational use. Market data via Yahoo Finance through the `yfinance` library — subject to Yahoo's terms of service.
