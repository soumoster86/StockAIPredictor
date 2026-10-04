# =============================
# model.py
# =============================
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from data import FEATURES, HORIZONS
from execution import (
    DEFAULT_NOTIONAL,
    RISK_FREE_RATE,
    daily_rate,
    equity_stats,
    locked_sessions,
    market_frame,
    side_costs,
    simulate_bracket,
    valid_rows,
)

logger = logging.getLogger(__name__)

SEED = 42
TRANSACTION_COST = 0.001
TRADING_DAYS = 252
DEFAULT_THRESHOLDS = (0.55, 0.45)
ENTRY_GRID = np.round(np.arange(0.50, 0.71, 0.05), 2)
EXIT_GRID = np.round(np.arange(0.30, 0.51, 0.05), 2)
SEQ_WINDOW = 20  # lookback days for LSTM/GRU
PLAN_MAX_HOLD_DAYS = 20  # plan expiry — matches journal.MAX_HOLD_DAYS

MODEL_TYPES = ["Ensemble (NN + XGBoost + RF)", "Neural Network", "LSTM", "GRU"]


# =====================================================================
# Networks
# =====================================================================

class StockModel(nn.Module):
    """Tabular MLP with light dropout. Outputs raw logits; sigmoid at inference."""

    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class SequenceNet(nn.Module):
    """LSTM/GRU over a window of daily feature vectors; the final hidden
    state feeds a linear head. Outputs raw logits."""

    def __init__(self, input_size, rnn_type="lstm", hidden=32, dropout=0.15):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):           # x: (batch, time, features)
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1, :]))


# =====================================================================
# Torch training helpers
# =====================================================================

def _pos_weight(y_t):
    pos_frac = max(float(y_t.mean()), 1e-6)
    return torch.tensor([(1.0 - pos_frac) / pos_frac])


def _train_torch(model, X_t, y_t, epochs, lr=1e-3, weight_decay=1e-4,
                 val_frac=0.15, patience=12):
    """Full-batch Adam with L2 + early stopping on a chronological val slice.

    The last `val_frac` of the training rows are held out for early stopping
    only (still inside the caller's train_end — no test leakage). Falls back
    to fixed epochs when the set is too small to split."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_t))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = int(X_t.shape[0])
    n_val = int(n * val_frac) if n >= 60 else 0
    if n_val >= 15:
        X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
        X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    else:
        X_tr, y_tr, X_val, y_val = X_t, y_t, None, None

    best_state, best_val, stall = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        if X_val is None:
            continue
        model.eval()
        with torch.no_grad():
            vloss = float(criterion(model(X_val), y_val).item())
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _sigmoid_probs(model, X_t):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).numpy().flatten()


def _make_windows(Xs, row_indices, window):
    """Stack one (window, n_features) slice ending at each row index."""
    return np.stack([Xs[i - window + 1:i + 1] for i in row_indices]).astype(np.float32)


# =====================================================================
# Unified predictor interface
# Every predictor exposes:
#   .window        int, lookback rows needed per prediction (1 for tabular)
#   .fit(Xs, y, train_end)   train on rows [0, train_end)
#   .predict_all(Xs)         prob per row; NaN for the first window-1 rows
#   .predict_last(Xs)        prob for the final row
# =====================================================================

class TabularNNPredictor:
    window = 1
    name = "Neural Net"

    def fit(self, Xs, y, train_end):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        X_t = torch.tensor(Xs[:train_end], dtype=torch.float32)
        y_t = torch.tensor(y[:train_end], dtype=torch.float32).view(-1, 1)
        self.model = _train_torch(StockModel(Xs.shape[1]), X_t, y_t, epochs=100)
        return self

    def predict_all(self, Xs):
        return _sigmoid_probs(self.model, torch.tensor(Xs, dtype=torch.float32))

    def predict_last(self, Xs):
        return float(self.predict_all(Xs[-1:])[0])


class TreePredictor:
    """Wraps a sklearn-style classifier (RandomForest or XGBoost)."""
    window = 1

    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    def fit(self, Xs, y, train_end):
        self.estimator.fit(Xs[:train_end], y[:train_end])
        return self

    def predict_all(self, Xs):
        return self.estimator.predict_proba(Xs)[:, 1]

    def predict_last(self, Xs):
        return float(self.predict_all(Xs[-1:])[0])


class EnsemblePredictor:
    """Soft-voting ensemble: averages the probability-up of all members."""
    window = 1
    name = "Ensemble"

    def __init__(self, members):
        self.members = members

    def fit(self, Xs, y, train_end):
        for m in self.members:
            m.fit(Xs, y, train_end)
        return self

    def predict_all(self, Xs):
        return np.mean([m.predict_all(Xs) for m in self.members], axis=0)

    def predict_last(self, Xs):
        return float(np.mean([m.predict_last(Xs) for m in self.members]))

    def member_probs_last(self, Xs):
        """Per-model probabilities for the latest row — shows agreement."""
        return {m.name: m.predict_last(Xs) for m in self.members}


class SequencePredictor:
    """LSTM/GRU over SEQ_WINDOW-day feature sequences."""

    def __init__(self, rnn_type):
        self.rnn_type = rnn_type
        self.window = SEQ_WINDOW
        self.name = rnn_type.upper()

    def fit(self, Xs, y, train_end):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        rows = np.arange(self.window - 1, train_end)
        X_t = torch.tensor(_make_windows(Xs, rows, self.window))
        y_t = torch.tensor(y[rows], dtype=torch.float32).view(-1, 1)
        self.model = _train_torch(
            SequenceNet(Xs.shape[1], self.rnn_type), X_t, y_t, epochs=60
        )
        return self

    def predict_all(self, Xs):
        n = len(Xs)
        probs = np.full(n, np.nan)
        rows = np.arange(self.window - 1, n)
        X_t = torch.tensor(_make_windows(Xs, rows, self.window))
        probs[rows] = _sigmoid_probs(self.model, X_t)
        return probs

    def predict_last(self, Xs):
        X_t = torch.tensor(Xs[-self.window:][None, :, :].astype(np.float32))
        return float(_sigmoid_probs(self.model, X_t)[0])


class CalibratedPredictor:
    """Wraps any predictor and remaps its probabilities through an isotonic
    regression fitted on validation data — so '70%' means what it says.
    Everything else (window, member votes) delegates to the base predictor."""

    def __init__(self, base, iso):
        self.base = base
        self.iso = iso
        self.window = getattr(base, 'window', 1)
        self.name = f"{base.name} (calibrated)"

    def _map(self, p):
        p = np.asarray(p, dtype=float)
        out = np.full_like(p, np.nan)
        m = np.isfinite(p)
        out[m] = self.iso.predict(p[m])
        return out

    def fit(self, *args, **kwargs):
        return self

    def predict_all(self, Xs):
        return self._map(self.base.predict_all(Xs))

    def predict_last(self, Xs):
        return float(self.iso.predict([self.base.predict_last(Xs)])[0])

    def __getattr__(self, item):  # delegate e.g. member_probs_last
        return getattr(self.base, item)


def calibration_metrics(probs, y, n_bins=8):
    """Reliability data: do stated probabilities match observed frequencies?
    Returns Brier score (lower = better; squared error of the probabilities),
    the Brier of always predicting the base rate (the score to beat),
    expected calibration error (avg gap between stated and actual, weighted
    by bin size), and the per-bin curve for plotting."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(probs)
    probs, y = probs[m], y[m]

    brier = float(np.mean((probs - y) ** 2))
    base_rate = float(y.mean())
    brier_baseline = float(np.mean((base_rate - y) ** 2))

    df = pd.DataFrame({'p': probs, 'y': y})
    try:
        df['bin'] = pd.qcut(df['p'], n_bins, duplicates='drop')
    except ValueError:
        df['bin'] = 0
    curve = (df.groupby('bin', observed=True)
               .agg(predicted=('p', 'mean'), actual=('y', 'mean'), count=('y', 'size'))
               .reset_index(drop=True))
    ece = float(np.sum(curve['count'] / len(df) * np.abs(curve['predicted'] - curve['actual'])))

    return {'brier': brier, 'brier_baseline': brier_baseline, 'ece': ece,
            'curve': curve.to_dict('records'), 'base_rate': base_rate}


def _tree_ensemble_members():
    """Stronger regularized trees for the soft-voting ensemble."""
    members = [
        TabularNNPredictor(),
        TreePredictor(RandomForestClassifier(
            n_estimators=400, max_depth=5, min_samples_leaf=25,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1,
        ), "Random Forest"),
    ]
    if HAS_XGB:
        members.append(TreePredictor(XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.5, reg_alpha=0.1, min_child_weight=5,
            eval_metric="logloss", random_state=SEED,
        ), "XGBoost"))
    else:
        # sklearn histogram GBDT is a solid XGB stand-in when xgboost
        # is unavailable (macOS / slim envs).
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            members.append(TreePredictor(HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.06, max_iter=200,
                l2_regularization=1.0, min_samples_leaf=25,
                random_state=SEED,
            ), "HistGBDT"))
        except ImportError:
            pass
    return members


def make_predictor(model_type):
    if model_type.startswith("Ensemble"):
        return EnsemblePredictor(_tree_ensemble_members())
    if model_type == "LSTM":
        return SequencePredictor("lstm")
    if model_type == "GRU":
        return SequencePredictor("gru")
    return TabularNNPredictor()


# =====================================================================
# Pure strategy / analytics logic (unchanged math, no torch)
# =====================================================================

def build_positions(probs, entry, exit_, can_buy=None, can_sell=None):
    """Long/flat positions with entry/exit hysteresis.

    With `can_buy` / `can_sell` masks (circuit locks), the position follows
    the signal's intent only when the order could actually fill: a blocked
    buy keeps you flat and a blocked sell keeps you long until the next
    session where the order goes through."""
    raw = np.where(probs > entry, 1.0, np.where(probs < exit_, 0.0, np.nan))
    intent = pd.Series(raw).ffill().fillna(0.0).to_numpy()
    if can_buy is None and can_sell is None:
        return intent

    n = len(intent)
    can_buy = np.ones(n, bool) if can_buy is None else np.asarray(can_buy, bool)
    can_sell = np.ones(n, bool) if can_sell is None else np.asarray(can_sell, bool)
    positions = np.empty(n)
    cur = 0.0
    for i in range(n):
        want = intent[i]
        if (want > cur and can_buy[i]) or (want < cur and can_sell[i]):
            cur = want
        positions[i] = cur
    return positions


def performance_stats(positions, returns, cost=TRANSACTION_COST, gap=None,
                      sell_cost=None, rf=0.0):
    """Strategy statistics for a long/flat position series.

    Legacy mode (`gap=None`): `returns[t]` is the close-to-close return
    earned by `positions[t]`, and `cost` is charged per unit of change.

    Next-open mode (`gap` given): the order decided at t's close fills at
    t+1's open, so the overnight `gap[t]` is earned by the position held
    yesterday and the session's `returns[t]` (open → close) by the new one.
    `cost` is the buy cost and `sell_cost` the sell cost; either may be a
    per-row array (liquidity-dependent slippage).

    `rf` is the annual risk-free rate: days in cash earn it, and Sharpe is
    measured on returns in excess of it."""
    positions = np.asarray(positions, dtype=float)
    returns = np.asarray(returns, dtype=float)
    buy_c = np.asarray(cost, dtype=float)
    sell_c = buy_c if sell_cost is None else np.asarray(sell_cost, dtype=float)

    delta = np.diff(positions, prepend=0.0)
    costs = np.where(delta > 0, delta * buy_c, -delta * sell_c)

    if gap is None:
        gross = returns * positions
        held = positions == 1
    else:
        prev = np.concatenate([[0.0], positions[:-1]])
        gross = (1.0 + prev * np.asarray(gap, dtype=float)) * (1.0 + positions * returns) - 1.0
        held = (positions == 1) | (prev == 1)

    strategy_returns = gross - costs + daily_rate(rf) * (1.0 - positions)
    eq = equity_stats(strategy_returns, rf)

    in_market = positions == 1
    win_rate = float((gross[held] > 0).mean()) if held.any() else float('nan')

    return {
        **eq,
        'exposure': float(in_market.mean()),
        'win_rate': win_rate,
        'n_trades': int((delta > 0).sum()),
        'total_costs': float(costs.sum()),
    }


def strategy_stats(probs, entry, exit_, market, cost=TRANSACTION_COST):
    """Simulate the threshold strategy on `market`.

    `market` is either a market frame from execution.market_frame (next-open
    fills, real costs, circuit locks) or a plain array of close-to-close
    returns (legacy, flat `cost`)."""
    if isinstance(market, pd.DataFrame):
        positions = build_positions(probs, entry, exit_,
                                    market['can_buy'].to_numpy(),
                                    market['can_sell'].to_numpy())
        return market_stats(positions, market)
    return performance_stats(build_positions(probs, entry, exit_), market, cost)


def market_stats(positions, market):
    """performance_stats for a fixed position series on a market frame."""
    return performance_stats(positions, market['intraday'].to_numpy(),
                             cost=market['buy_cost'].to_numpy(),
                             gap=market['gap'].to_numpy(),
                             sell_cost=market['sell_cost'].to_numpy(),
                             rf=RISK_FREE_RATE)


def tune_thresholds(probs, returns, cost=TRANSACTION_COST):
    """Grid-search entry/exit thresholds for the best after-cost Sharpe.
    `returns` may be a market frame or legacy close-to-close returns."""
    best = DEFAULT_THRESHOLDS
    best_score = -np.inf
    for entry in ENTRY_GRID:
        for exit_ in EXIT_GRID:
            if exit_ >= entry:
                continue
            stats = strategy_stats(probs, entry, exit_, returns, cost)
            score = stats['sharpe'] if np.isfinite(stats['sharpe']) else stats['total_return']
            if score > best_score:
                best_score = score
                best = (float(entry), float(exit_))
    return best


def rating_from_prob(prob):
    if prob > 0.80:
        return "Strong Buy"
    if prob >= 0.65:
        return "Buy"
    if prob >= 0.45:
        return "Neutral"
    return "Sell"


def compute_risk_score(data):
    vol_ann = float(data['Vol20'].iloc[-1]) * np.sqrt(TRADING_DAYS)
    atr_pct = float(data['ATR_pct'].iloc[-1])
    close_1y = data['Close'].tail(TRADING_DAYS)
    max_dd_1y = float((close_1y / close_1y.cummax() - 1.0).min())

    c_vol = min(vol_ann / 0.60, 1.0)
    c_atr = min(atr_pct / 0.05, 1.0)
    c_dd = min(abs(max_dd_1y) / 0.50, 1.0)

    score = float(np.clip(round(1.0 + 9.0 * (c_vol + c_atr + c_dd) / 3.0, 1), 1.0, 10.0))
    level = "Low" if score <= 3 else "Medium" if score <= 7 else "High"

    return {'score': score, 'level': level, 'volatility_annualized': vol_ann,
            'atr_pct': atr_pct, 'max_drawdown_1y': max_dd_1y}


def _classification_metrics(probs, y_true):
    y_pred = (probs > 0.5).astype(float)
    accuracy = float((y_pred == y_true).mean())
    majority = float(max(y_true.mean(), 1 - y_true.mean()))
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    precision = tp / max(float((y_pred == 1).sum()), 1.0)
    recall = tp / max(float((y_true == 1).sum()), 1.0)
    return {'accuracy': accuracy, 'baseline_accuracy': majority,
            'precision': precision, 'recall': recall}


def _masked(data, target_col):
    sub = data[data[target_col].notna()]
    return sub[FEATURES].values, sub[target_col].values.astype(float), sub.index


def _tune_on_slice(all_probs, mkt, start, end):
    """Tune thresholds on rows [start:end) with a usable next session."""
    probs = all_probs[start:end]
    sub = mkt.iloc[start:end]
    mask = valid_rows(sub) & np.isfinite(probs)
    return tune_thresholds(probs[mask], sub[mask])


# =====================================================================
# Main entry points
# =====================================================================

def train_model(data, model_type="Neural Network", calibrate=False, cost_profile="NSE"):
    """1-day model of the chosen type. Chronological 64/16/20 split;
    scaler fit on train only; thresholds tuned on validation (next-open
    fills, `cost_profile` costs); metrics from the untouched test slice.
    With calibrate=True, an isotonic regression fitted on the validation
    slice remaps probabilities so they match observed frequencies."""
    X, y, dates = _masked(data, 'Target_1')
    n = len(X)
    if n < 300:
        raise ValueError("Need at least 300 rows of feature data to train.")

    mkt = market_frame(data, dates, profile=cost_profile)

    test_n = int(n * 0.20)
    val_n = int(n * 0.16)
    train_end = n - test_n - val_n
    val_end = n - test_n

    scaler = StandardScaler().fit(X[:train_end])
    Xs = scaler.transform(X)

    predictor = make_predictor(model_type).fit(Xs, y, train_end)
    all_probs = predictor.predict_all(Xs)

    iso = None
    if calibrate:
        raw_val = all_probs[train_end:val_end]
        v_m = np.isfinite(raw_val)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(raw_val[v_m], y[train_end:val_end][v_m])
        predictor = CalibratedPredictor(predictor, iso)
        all_probs = predictor._map(all_probs)

    thresholds = _tune_on_slice(all_probs, mkt, train_end, val_end)

    test_probs = all_probs[val_end:]
    metrics = _classification_metrics(test_probs, y[val_end:])
    metrics['entry_threshold'], metrics['exit_threshold'] = thresholds
    metrics['calibration'] = calibration_metrics(test_probs, y[val_end:])
    metrics['calibrated'] = bool(calibrate)

    # The evaluation model above intentionally stops at the training slice so
    # validation/test metrics remain honest. For the live signal, refit on all
    # rows with known targets so today's prediction uses the latest history.
    #
    # NOTE: this is a deliberate ~2x training cost — the ensemble is trained
    # once for honest metrics and once more here for the deployed signal. It is
    # acceptable because get_trained() in ui/services.py caches the whole call
    # (per symbol/model), so the double fit happens at most once per hour.
    live_scaler = StandardScaler().fit(X)
    live_Xs = live_scaler.transform(X)
    live_predictor = make_predictor(model_type).fit(live_Xs, y, n)
    if iso is not None:
        live_predictor = CalibratedPredictor(live_predictor, iso)

    return live_predictor, live_scaler, metrics, test_probs, thresholds, dates[val_end:]


def predict(predictor, scaler, data, thresholds=DEFAULT_THRESHOLDS):
    """Signal for the latest close. Sequence models internally use the
    last SEQ_WINDOW rows; tabular models use the last row."""
    entry, exit_ = thresholds
    Xs = scaler.transform(data[FEATURES].values)
    prob = predictor.predict_last(Xs)

    if prob > entry:
        return "BUY", prob
    elif prob < exit_:
        return "SELL", prob
    else:
        return "HOLD", prob


def explain_prediction(predictor, scaler, data):
    """Occlusion attribution, model-agnostic: set one feature to its
    training mean (0 in scaled space) — across the full lookback window
    for sequence models — and measure the probability shift."""
    Xs = scaler.transform(data[FEATURES].values).astype(np.float32)
    base_prob = predictor.predict_last(Xs)
    w = getattr(predictor, 'window', 1)

    contributions = []
    for j, feat in enumerate(FEATURES):
        X_masked = Xs.copy()
        X_masked[-w:, j] = 0.0
        masked_prob = predictor.predict_last(X_masked)
        contributions.append({
            'feature': feat,
            'value': float(data[feat].values[-1]),
            'contribution': base_prob - masked_prob,
        })

    contributions.sort(key=lambda d: abs(d['contribution']), reverse=True)
    return base_prob, contributions


def multi_horizon_forecast(data, model_type="Neural Network"):
    """One model of the chosen type per horizon."""
    Xs_latest_src = data[FEATURES].values
    rows = []

    for h in HORIZONS:
        X, y, _ = _masked(data, f'Target_{h}')
        n = len(X)
        if n < 300:
            continue

        split = int(n * 0.8)
        scaler = StandardScaler().fit(X[:split])
        Xs = scaler.transform(X)

        predictor = make_predictor(model_type).fit(Xs, y, split)
        all_probs = predictor.predict_all(Xs)
        test_probs = all_probs[split:]
        t_mask = np.isfinite(test_probs)
        cm = _classification_metrics(test_probs[t_mask], y[split:][t_mask])

        # Report metrics from the held-out split, but use a refit model for
        # the latest probability so the live forecast is not trained on stale
        # history.
        live_scaler = StandardScaler().fit(X)
        live_Xs = live_scaler.transform(X)
        live_predictor = make_predictor(model_type).fit(live_Xs, y, n)
        prob = live_predictor.predict_last(live_scaler.transform(Xs_latest_src))

        rows.append({
            'Horizon': f"{h} Day" if h == 1 else f"{h} Days",
            'Probability Up': prob,
            'Rating': rating_from_prob(prob),
            'Test Accuracy': cm['accuracy'],
            'Baseline': cm['baseline_accuracy'],
        })

    return pd.DataFrame(rows)


def _test_market(test_probs, prices, test_index, cost_profile, notional):
    """Market frame for the test window, trimmed to rows with both a
    probability and a usable next session."""
    probs = np.asarray(test_probs, dtype=float)
    mkt = market_frame(prices, test_index, profile=cost_profile, notional=notional)
    valid = valid_rows(mkt) & np.isfinite(probs)
    return probs[valid], mkt[valid]


def backtest(test_probs, prices, test_index, thresholds=DEFAULT_THRESHOLDS,
             cost_profile="NSE", notional=DEFAULT_NOTIONAL):
    """Backtest pre-computed probabilities on the held-out period.

    `prices` is the OHLCV frame (a Close Series still works, close-to-close).
    Orders decided at each close fill at the next open, pay the profile's
    statutory costs plus liquidity-based slippage, and are blocked on
    circuit-locked sessions. Buy & Hold is simulated with the same rules."""
    probs, mkt = _test_market(test_probs, prices, test_index, cost_profile, notional)
    idx = mkt.index

    positions = build_positions(probs, *thresholds,
                                mkt['can_buy'].to_numpy(), mkt['can_sell'].to_numpy())
    intent = build_positions(probs, *thresholds)
    stats = market_stats(positions, mkt)

    always_in = build_positions(np.ones(len(mkt)), 0.5, 0.4,
                                mkt['can_buy'].to_numpy(), mkt['can_sell'].to_numpy())
    bh = market_stats(always_in, mkt)

    equity = pd.Series(stats.pop('equity'), index=idx, name='Strategy')
    buy_hold = pd.Series(bh['equity'], index=idx, name='Buy & Hold')
    stats['buy_hold_return'] = float(buy_hold.iloc[-1] - 1.0)
    stats['round_trip_cost'] = float((mkt['buy_cost'] + mkt['sell_cost']).median())
    stats['blocked_days'] = int((positions != intent).sum())
    stats['cost_profile'] = cost_profile

    return stats, equity, buy_hold


def random_signal_benchmark(test_probs, prices, test_index,
                            thresholds=DEFAULT_THRESHOLDS, n_random=300, seed=SEED,
                            cost_profile="NSE", notional=DEFAULT_NOTIONAL):
    """Is the strategy's performance better than luck?

    Generates `n_random` random long-only strategies that hold for the SAME
    number of days as the real strategy (matched exposure), then reports
    where the real strategy's Sharpe and total return fall within those
    random distributions, as percentiles.

    Matching exposure is the crux: a random strategy that trades more or
    less than the model would be an unfair yardstick. We keep the count of
    in-market days identical and only randomize WHICH days — so the only
    thing being tested is signal quality, not activity level.

    Returns a dict, or None if the strategy never takes a position."""
    probs, mkt = _test_market(test_probs, prices, test_index, cost_profile, notional)
    n = len(mkt)
    if n < 20:
        return None

    real_pos = build_positions(probs, *thresholds,
                               mkt['can_buy'].to_numpy(), mkt['can_sell'].to_numpy())
    n_in_market = int((real_pos == 1).sum())
    if n_in_market == 0 or n_in_market == n:
        return None  # nothing (or everything) held -> no meaningful comparison

    real = market_stats(real_pos, mkt)
    real_sharpe = real["sharpe"]
    real_return = real["total_return"]

    rng = np.random.default_rng(seed)
    rand_sharpes = np.empty(n_random)
    rand_returns = np.empty(n_random)
    for i in range(n_random):
        pos = np.zeros(n)
        pos[rng.choice(n, size=n_in_market, replace=False)] = 1.0
        s = market_stats(pos, mkt)
        rand_sharpes[i] = s["sharpe"] if np.isfinite(s["sharpe"]) else 0.0
        rand_returns[i] = s["total_return"]

    rs = real_sharpe if np.isfinite(real_sharpe) else 0.0
    return {
        "real_sharpe": float(rs),
        "real_return": float(real_return),
        "sharpe_percentile": float((rand_sharpes < rs).mean() * 100),
        "return_percentile": float((rand_returns < real_return).mean() * 100),
        "rand_sharpes": rand_sharpes,
        "rand_returns": rand_returns,
        "n_random": n_random,
        "exposure_days": n_in_market,
        "total_days": n,
    }


def walk_forward(data, model_type="Neural Network", n_splits=4, min_train=300,
                 calibrate=False, cost_profile="NSE"):
    """Expanding-window walk-forward validation of the chosen model type."""
    X, y, dates = _masked(data, 'Target_1')
    n = len(X)
    fold_size = (n - min_train) // n_splits
    if fold_size < 40:
        raise ValueError("Not enough history for walk-forward validation.")

    mkt = market_frame(data, dates, profile=cost_profile)

    rows = []
    for i in range(n_splits):
        train_total = min_train + i * fold_size
        test_end = train_total + fold_size if i < n_splits - 1 else n

        val_n = max(int(train_total * 0.16), 40)
        fit_end = train_total - val_n

        scaler = StandardScaler().fit(X[:fit_end])
        Xs = scaler.transform(X)

        predictor = make_predictor(model_type).fit(Xs, y, fit_end)
        all_probs = predictor.predict_all(Xs)

        if calibrate:
            raw_val = all_probs[fit_end:train_total]
            v_m0 = np.isfinite(raw_val)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            iso.fit(raw_val[v_m0], y[fit_end:train_total][v_m0])
            predictor = CalibratedPredictor(predictor, iso)
            all_probs = predictor._map(all_probs)

        entry, exit_ = _tune_on_slice(all_probs, mkt, fit_end, train_total)

        test_probs = all_probs[train_total:test_end]
        test_mkt = mkt.iloc[train_total:test_end]
        t_mask = valid_rows(test_mkt) & np.isfinite(test_probs)
        test_mkt = test_mkt[t_mask]

        stats = strategy_stats(test_probs[t_mask], entry, exit_, test_mkt)
        accuracy = float(
            ((test_probs[t_mask] > 0.5) == y[train_total:test_end][t_mask]).mean()
        )
        buy_hold = strategy_stats(np.ones(len(test_mkt)), 0.5, 0.4, test_mkt)['total_return']

        rows.append({
            'Fold': i + 1,
            'Test Start': dates[train_total].date(),
            'Test End': dates[test_end - 1].date(),
            'Accuracy': accuracy,
            'Win Rate': stats['win_rate'],
            'Strategy Return': stats['total_return'],
            'Buy & Hold': buy_hold,
            'Sharpe': stats['sharpe'],
            'Max Drawdown': stats['max_drawdown'],
            'Exposure': stats['exposure'],
            'Trades': stats['n_trades'],
            'Entry Thr': entry,
            'Exit Thr': exit_,
        })

    return pd.DataFrame(rows)


# =====================================================================
# Trade planning (pure, no torch)
# =====================================================================

def find_support_resistance(data, lookback=252, swing_window=10, cluster_pct=0.015):
    """Detect swing highs/lows over the past `lookback` days, merge levels
    that sit within `cluster_pct` of each other, and return the nearest
    support below and resistance above the current price.

    A swing high is a day whose High is the highest within ±swing_window
    days (and symmetrically for swing lows)."""
    sub = data.tail(lookback)
    price = float(data['Close'].iloc[-1])
    w = 2 * swing_window + 1

    swing_highs = sub['High'][
        sub['High'] == sub['High'].rolling(w, center=True).max()
    ].dropna().values
    swing_lows = sub['Low'][
        sub['Low'] == sub['Low'].rolling(w, center=True).min()
    ].dropna().values

    def cluster(levels):
        merged = []
        for lv in sorted(float(x) for x in levels):
            if merged and (lv - merged[-1][-1]) / price < cluster_pct:
                merged[-1].append(lv)
            else:
                merged.append([lv])
        return [float(np.mean(g)) for g in merged]

    support_levels = cluster(swing_lows)
    resistance_levels = cluster(swing_highs)

    return {
        'support': max((lv for lv in support_levels if lv < price), default=None),
        'resistance': min((lv for lv in resistance_levels if lv > price), default=None),
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'price': price,
    }


def compute_trade_plan(data, support=None, resistance=None,
                       atr_stop_mult=1.5, atr_target_mult=3.0, min_rr=1.5,
                       min_stop_atr_mult=0.5):
    """ATR-based entry/stop/target for a long trade at the current price.

    Stop: 1.5x ATR below entry — tightened to just below support when a
    support level sits inside that band (structure beats formula). Support
    is ignored when it would make the stop tighter than `min_stop_atr_mult`
    × ATR (noise-tight floors create absurd reward:risk ratios).
    Target: nearest resistance if it offers at least `min_rr` reward:risk,
    otherwise 3x ATR above entry."""
    entry = float(data['Close'].iloc[-1])
    atr = float(data['ATR_pct'].iloc[-1]) * entry
    min_risk = max(min_stop_atr_mult * atr, 1e-9)

    stop = entry - atr_stop_mult * atr
    stop_basis = f"{atr_stop_mult:.1f}× ATR below entry"
    if support is not None and stop < support < entry:
        struct_stop = support * 0.995
        if entry - struct_stop >= min_risk:
            stop = struct_stop
            stop_basis = "just below the nearest support"
        # else: keep ATR stop — support is noise-close to entry

    # Absolute floor so risk distance never collapses below min_stop_atr_mult×ATR
    if entry - stop < min_risk:
        stop = entry - min_risk
        stop_basis = f"{min_stop_atr_mult:.1f}× ATR below entry (floor)"

    risk_per_share = entry - stop

    target = entry + atr_target_mult * atr
    target_basis = f"{atr_target_mult:.1f}× ATR above entry"
    if resistance is not None and resistance > entry and risk_per_share > 0:
        rr_at_resistance = (resistance - entry) / risk_per_share
        if rr_at_resistance >= min_rr:
            target = resistance
            target_basis = "the nearest resistance"

    return {
        'entry': entry,
        'stop': float(stop),
        'target': float(target),
        'risk_per_share': float(risk_per_share),
        'reward_risk': float((target - entry) / risk_per_share) if risk_per_share > 0 else float('nan'),
        'atr': float(atr),
        'stop_basis': stop_basis,
        'target_basis': target_basis,
    }


def position_size(capital, risk_pct, entry, stop):
    """How many shares to buy so that hitting the stop loses exactly
    `risk_pct` of capital. The professional formula:
        shares = (capital × risk%) / (entry − stop)
    Capped so the position never costs more than the available capital."""
    risk_amount = capital * risk_pct / 100.0
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0 or capital <= 0:
        return None

    shares = int(risk_amount // risk_per_share)
    max_affordable = int(capital // entry)
    capped = shares > max_affordable
    shares = min(shares, max_affordable)

    position_value = shares * entry
    return {
        'shares': shares,
        'risk_amount': float(risk_amount),
        'actual_risk': float(shares * risk_per_share),
        'position_value': float(position_value),
        'pct_of_capital': float(position_value / capital),
        'capped_by_capital': capped,
    }


def _trade_daily_returns(close, res):
    """Daily returns of one filled bracket trade, from fill day to exit day.

    Fill day: fill (+buy cost) → close. Middle days: close → close. Exit day:
    prior close → exit price (−sell cost). They compound to the trade's net
    return, so equity curves and drawdowns are marked to market daily."""
    dates = close.loc[res["fill_date"]:res["exit_date"]].index
    c = close.loc[dates].to_numpy(float)
    fill_basis = res["fill_price"] * (1.0 + res["buy_cost"])
    exit_net = res["exit_price"] * (1.0 - res["sell_cost"])
    if len(dates) == 1:
        return pd.Series([exit_net / fill_basis - 1.0], index=dates)
    r = np.empty(len(dates))
    r[0] = c[0] / fill_basis - 1.0
    r[1:-1] = c[1:-1] / c[:-2] - 1.0
    r[-1] = exit_net / c[-2] - 1.0
    return pd.Series(r, index=dates)


def plan_backtest(test_probs, data, test_index, entry_threshold,
                  cost_profile="NSE", notional=DEFAULT_NOTIONAL,
                  max_days=PLAN_MAX_HOLD_DAYS):
    """Backtest the trade plan the app actually recommends.

    Whenever the model's probability crosses `entry_threshold` while flat,
    build the Plan tab's trade from data known that day (support/resistance,
    ATR stop, target) and simulate it with execution.simulate_bracket: fill
    at the next open, gap-aware stop/target exits, circuit locks, costs,
    expiry after `max_days`. One trade at a time; the next signal is taken
    from the exit day's close. Cash earns the risk-free rate between trades.

    Returns (stats, trades DataFrame, daily equity Series)."""
    probs = pd.Series(np.asarray(test_probs, dtype=float), index=test_index)
    close = data['Close'].astype(float)
    locks = locked_sessions(data)
    costs = side_costs(data, cost_profile, notional)

    window = close.loc[test_index[0]:].index[1:]   # first tradeable session onward
    daily = pd.Series(daily_rate(RISK_FREE_RATE), index=window)
    in_trade = pd.Series(False, index=window)

    trades, no_fill = [], 0
    i = 0
    while i < len(test_index):
        dt = test_index[i]
        p = probs.iloc[i]
        if not (np.isfinite(p) and p > entry_threshold):
            i += 1
            continue
        hist = data.loc[:dt]
        sr = find_support_resistance(hist)
        plan = compute_trade_plan(hist, sr['support'], sr['resistance'])
        res = simulate_bracket(data, dt, plan['stop'], plan['target'], max_days=max_days,
                               profile=cost_profile, notional=notional,
                               locks=locks, costs=costs)
        if res["status"] == "NO FILL":
            no_fill += 1
            i += 1
            continue
        if res["fill_date"] is None:   # signal on the last bar: nothing to trade yet
            break
        exit_date = res["exit_date"] if res["exit_date"] is not None else close.index[-1]
        if res["status"] == "OPEN":     # still running: mark to the last close, no sell yet
            res = {**res, "exit_date": exit_date, "sell_cost": 0.0}
        tr = _trade_daily_returns(close, res).reindex(window).dropna()
        daily.loc[tr.index] = tr
        in_trade.loc[tr.index] = True
        trades.append({"signal_date": dt, "probability": float(p), "stop": plan['stop'],
                       "target": plan['target'], "reward_risk": plan['reward_risk'],
                       **{k: res[k] for k in ("fill_date", "fill_price", "exit_date",
                                              "exit_price", "status", "days",
                                              "gross_return", "outcome_return")}})
        if res["status"] == "OPEN":
            break
        # Next signal: the exit day's close (fills the session after). The
        # exit is always after the signal day, so this always advances.
        i = max(int(test_index.searchsorted(exit_date)), i + 1)

    trades_df = pd.DataFrame(trades)
    eq = equity_stats(daily.to_numpy(), RISK_FREE_RATE)
    equity = pd.Series(eq.pop("equity"), index=window, name="Trade plan")
    closed = trades_df[trades_df["status"] != "OPEN"] if len(trades_df) else trades_df
    rets = closed["outcome_return"] if len(closed) else pd.Series(dtype=float)
    gains, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    stats = {
        **eq,
        "n_trades": int(len(closed)),
        "n_open": int(len(trades_df) - len(closed)),
        "n_no_fill": int(no_fill),
        "win_rate": float((rets > 0).mean()) if len(rets) else float("nan"),
        "target_rate": float((closed["status"] == "TARGET HIT").mean()) if len(closed) else float("nan"),
        "stop_rate": float((closed["status"] == "STOP HIT").mean()) if len(closed) else float("nan"),
        "avg_return": float(rets.mean()) if len(rets) else float("nan"),
        "avg_days": float(closed["days"].mean()) if len(closed) else float("nan"),
        "profit_factor": float(gains / losses) if losses > 0 else float("nan"),
        "exposure": float(in_trade.mean()) if len(in_trade) else 0.0,
    }
    return stats, trades_df, equity


# =====================================================================
# Watchlist scanner (fast screen across many stocks)
# =====================================================================

def make_fast_predictor():
    """Lightweight tree model for scanning many stocks quickly. XGBoost
    when available (faster, usually a touch better), else Random Forest.
    Sequence models and the full ensemble are deliberately excluded —
    a 20-stock scan must stay inside cloud memory/time budgets."""
    if HAS_XGB:
        return TreePredictor(XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.07,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.5, min_child_weight=5,
            eval_metric="logloss", random_state=SEED, n_jobs=2,
        ), "XGBoost")
    return TreePredictor(RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=20,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=SEED, n_jobs=2,
    ), "Random Forest")


def _scan_result(prob, cm, model_name, thresholds=DEFAULT_THRESHOLDS, source="per-stock"):
    entry, exit_ = thresholds
    signal = "BUY" if prob > entry else "SELL" if prob < exit_ else "HOLD"
    return {
        'probability': float(prob),
        'signal': signal,
        'rating': rating_from_prob(prob),
        'accuracy': cm['accuracy'],
        'baseline': cm['baseline_accuracy'],
        'model': model_name,
        'source': source,
    }


def buy_score(probability, accuracy=None, baseline=None, risk=None,
              reward_risk=None, to_support=None):
    """Composite 0–100 score for ranking *long* candidates in the screener.

    Higher is better. Weights favor model probability, then out-of-sample
    edge, then structure (R:R, calmer risk, proximity to support). This is a
    ranking heuristic for the watchlist — not a guarantee or recommendation.
    """
    p = float(probability) if probability is not None and np.isfinite(probability) else 0.5
    p = float(np.clip(p, 0.0, 1.0))

    edge = 0.0
    if accuracy is not None and baseline is not None:
        try:
            edge = max(0.0, float(accuracy) - float(baseline))
        except (TypeError, ValueError):
            edge = 0.0

    risk_term = 0.5
    if risk is not None and np.isfinite(risk):
        # 1 (calm) → 1.0, 10 (wild) → 0.0
        risk_term = 1.0 - float(np.clip((float(risk) - 1.0) / 9.0, 0.0, 1.0))

    rr_term = 0.0
    if reward_risk is not None and np.isfinite(reward_risk):
        # R:R 1.0 → 0, 3.0+ → 1
        rr_term = float(np.clip((float(reward_risk) - 1.0) / 2.0, 0.0, 1.0))

    support_term = 0.4  # neutral when unknown
    if to_support is not None and np.isfinite(to_support):
        ts = float(to_support)
        if ts < 0:
            support_term = 0.0  # already through support — riskier entry
        elif ts <= 0.08:
            support_term = 1.0 - ts / 0.08  # closer to floor = better
        else:
            support_term = max(0.0, 1.0 - (ts - 0.08) / 0.20)  # far above floor

    score = (
        50.0 * p
        + 20.0 * min(edge / 0.08, 1.0)
        + 12.0 * risk_term
        + 12.0 * rr_term
        + 6.0 * support_term
    )
    return float(np.clip(score, 0.0, 100.0))


def rank_buy_candidates(scan_df, min_prob=0.55, max_risk=8.0,
                        require_edge=True, top_n=10):
    """Filter and rank a scan DataFrame into best long candidates.

    Expects columns produced by `run_scan` (Screen, Probability Up, …).
    Returns a copy sorted by Buy Score descending (empty if none qualify).
    """
    if scan_df is None or len(scan_df) == 0:
        return pd.DataFrame()

    df = scan_df.copy()
    if "Buy Score" not in df.columns:
        scores = []
        for _, row in df.iterrows():
            scores.append(buy_score(
                row.get("Probability Up"),
                row.get("Test Acc"),
                row.get("Baseline"),
                row.get("Risk"),
                row.get("Reward Risk"),
                row.get("To Support"),
            ))
        df["Buy Score"] = scores

    buys = df[df["Screen"] == "BUY"].copy() if "Screen" in df.columns else df.copy()
    if buys.empty:
        return buys

    if "Probability Up" in buys.columns:
        buys = buys[buys["Probability Up"] >= float(min_prob)]
    if max_risk is not None and "Risk" in buys.columns:
        buys = buys[buys["Risk"] <= float(max_risk)]
    if require_edge and "Test Acc" in buys.columns and "Baseline" in buys.columns:
        buys = buys[buys["Test Acc"] >= buys["Baseline"]]

    if buys.empty:
        return buys

    buys = buys.sort_values("Buy Score", ascending=False).reset_index(drop=True)
    buys.insert(0, "Rank", range(1, len(buys) + 1))
    if top_n is not None:
        buys = buys.head(int(top_n))
    return buys


def quick_scan(data, thresholds=DEFAULT_THRESHOLDS):
    """One-stock quick screen: train a fast tree model (80/20 chronological
    split, scaler fit on train only), report the latest probability-up,
    signal at default thresholds, and honest out-of-sample accuracy vs
    baseline. Returns None when there's too little history.

    This is a SCREEN, not the full analysis: default thresholds, no
    ensemble, no threshold tuning — open the stock for the real thing."""
    X, y, _ = _masked(data, 'Target_1')
    n = len(X)
    if n < 300:
        return None

    split = int(n * 0.8)
    scaler = StandardScaler().fit(X[:split])
    Xs = scaler.transform(X)

    predictor = make_fast_predictor().fit(Xs, y, split)
    test_probs = predictor.predict_all(Xs)[split:]
    cm = _classification_metrics(test_probs, y[split:])

    # Keep the out-of-sample scanner metrics from the 80/20 split, then refit
    # for the displayed live probability using all labeled history.
    live_scaler = StandardScaler().fit(X)
    live_Xs = live_scaler.transform(X)
    live_predictor = make_fast_predictor().fit(live_Xs, y, n)
    prob = live_predictor.predict_last(live_scaler.transform(data[FEATURES].values))
    return _scan_result(prob, cm, predictor.name, thresholds, source="per-stock")


def quick_scan_global(data, bundle=None, thresholds=DEFAULT_THRESHOLDS):
    """Screen one stock with a pre-trained global model — no per-stock fit.

    Freezes the global weights, scores this stock's last-20% chronology for
    indicative accuracy, and reports the latest probability. Prefer this in
    the watchlist scanner when global artifacts are present (orders of
    magnitude faster than training a tree per name)."""
    if bundle is None:
        bundle = load_global_model(1)
    if bundle is None:
        return None

    predictor, scaler = bundle["predictor"], bundle["scaler"]
    X, y, _ = _masked(data, 'Target_1')
    if len(X) < 80:
        return None

    Xs = scaler.transform(X)
    split = max(int(len(X) * 0.8), 1)
    test_probs = predictor.predict_all(Xs)[split:]
    t_ok = np.isfinite(test_probs)
    if t_ok.sum() < 10:
        return None
    cm = _classification_metrics(test_probs[t_ok], y[split:][t_ok])
    prob = predictor.predict_last(scaler.transform(data[FEATURES].values))
    return _scan_result(prob, cm, "Global", thresholds, source="global")


# =====================================================================
# Global model: one model trained on the pooled history of many stocks.
# (Phase 1 — trained offline by train_global.py, loaded here.)
# =====================================================================

# Override with env GLOBAL_MODEL_DIR for external/LFS-synced model stores
# (e.g. /data/models or a mounted volume on cloud).
GLOBAL_MODEL_DIR = os.environ.get("GLOBAL_MODEL_DIR", "global_models")
GLOBAL_META_FILE = "global_meta.json"


def _day_ids_from_index(index):
    """Calendar day keys as int (days since Unix epoch).

    Unit-agnostic: pandas may store indexes as datetime64[ns] or [us];
    using .asi8 + Timestamp would mis-decode us as ns and collapse dates
    to 1970. Day-level ints avoid that class of bug entirely."""
    ts = pd.to_datetime(index)
    # .normalize() -> midnight; .days is timezone-safe for tz-naive series
    return (ts.normalize() - pd.Timestamp("1970-01-01")).days.to_numpy(dtype=np.int64)


def pool_training_data(per_stock_frames, target_col):
    """Stack labeled rows from every stock. Returns X, y, day_ids, n_stocks.

    day_ids (int days since epoch) enable time-based train/test splits so
    the global model is not evaluated on randomly shuffled future rows."""
    X_parts, y_parts, d_parts, used = [], [], [], 0
    for _sym, d in per_stock_frames.items():
        if d is None or d.empty or target_col not in d.columns:
            continue
        sub = d[d[target_col].notna()]
        if len(sub) < 100:
            continue
        X_parts.append(sub[FEATURES].values)
        y_parts.append(sub[target_col].values.astype(float))
        d_parts.append(_day_ids_from_index(sub.index))
        used += 1
    if not X_parts:
        return None, None, None, 0
    return (np.vstack(X_parts), np.concatenate(y_parts),
            np.concatenate(d_parts), used)


def train_global_predictor(per_stock_frames, target_col, model_type="Ensemble",
                           train_frac=0.8, embargo_days=None):
    """Train one global model with a **time-ordered** pool split.

    All labeled rows across stocks are sorted by calendar day. The earliest
    `train_frac` of unique trading days form the training set; later days
    form the holdout. An embargo of `embargo_days` (default: horizon inferred
    from target_col, else 5) drops borderline days so overlapping multi-day
    labels cannot leak across the cut.

    The final returned predictor is refit on **all** labeled rows so the
    deployed artifact uses the full history; reported metrics still come from
    the held-out time slice of the evaluation fit only.
    """
    X, y, day_ids, n_stocks = pool_training_data(per_stock_frames, target_col)
    if X is None or len(X) < 500:
        raise ValueError(f"Not enough pooled data for {target_col} "
                         f"({0 if X is None else len(X)} rows).")

    # Infer embargo from Target_h name when possible
    if embargo_days is None:
        embargo_days = 5
        if isinstance(target_col, str) and target_col.startswith("Target_"):
            try:
                embargo_days = max(int(target_col.split("_", 1)[1]), 1)
            except ValueError:
                pass

    order = np.argsort(day_ids, kind="mergesort")
    X, y, day_ids = X[order], y[order], day_ids[order]

    unique_days = np.unique(day_ids)
    cut_i = max(int(len(unique_days) * train_frac), 1)
    cut_i = min(cut_i, len(unique_days) - 1)
    cut_day = int(unique_days[cut_i - 1])

    train_mask = day_ids <= cut_day
    test_mask = day_ids > cut_day + int(embargo_days)

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    cut_date = str(pd.Timestamp("1970-01-01") + pd.Timedelta(days=cut_day))[:10]
    if n_train < 300 or n_test < 50:
        raise ValueError(
            f"Time split too thin for {target_col}: "
            f"train={n_train}, test={n_test} (need ≥300/50). "
            f"Cut day={cut_date}, embargo={embargo_days}d."
        )

    # Evaluation fit: train on past only, score the future holdout
    scaler_eval = StandardScaler().fit(X[train_mask])
    Xs_eval = scaler_eval.transform(X)
    # Contiguous [train | rest] layout for predictor.fit(train_end=...)
    eval_order = np.concatenate([np.where(train_mask)[0], np.where(~train_mask)[0]])
    Xs_eval_ord = Xs_eval[eval_order]
    y_eval_ord = y[eval_order]
    predictor_eval = make_predictor(model_type)
    predictor_eval.fit(Xs_eval_ord, y_eval_ord, n_train)

    test_probs = predictor_eval.predict_all(Xs_eval)[test_mask]
    t_ok = np.isfinite(test_probs)
    metrics = _classification_metrics(test_probs[t_ok], y[test_mask][t_ok])
    metrics["n_stocks"] = int(n_stocks)
    metrics["n_rows"] = int(len(X))
    metrics["n_train"] = n_train
    metrics["n_test"] = n_test
    metrics["split"] = "time"
    metrics["cut_date"] = cut_date
    metrics["embargo_days"] = int(embargo_days)

    # Deploy fit: full history, same feature space
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    predictor = make_predictor(model_type)
    predictor.fit(Xs, y, len(X))

    return predictor, scaler, metrics


def save_global_model(predictor, scaler, horizon, directory=GLOBAL_MODEL_DIR):
    import os

    import joblib
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"global_h{horizon}.joblib")
    joblib.dump({"predictor": predictor, "scaler": scaler,
                 "features": list(FEATURES), "horizon": horizon}, path)
    return path


def load_global_model(horizon, directory=GLOBAL_MODEL_DIR):
    import os

    import joblib
    path = os.path.join(directory, f"global_h{horizon}.joblib")
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        # A corrupted/unreadable artifact should be distinguishable from a
        # missing one — otherwise it silently falls back to the per-stock path.
        logger.warning("Failed to load global model %s", path, exc_info=True)
        return None
    if list(bundle.get("features", [])) != list(FEATURES):
        logger.warning(
            "Global model %s has a mismatched feature set; ignoring it.", path
        )
        return None
    return bundle


def global_model_available(directory=GLOBAL_MODEL_DIR):
    return load_global_model(1, directory) is not None


# =====================================================================
# Phase 2 — using the global model for a single stock.
#
# The global predictor + scaler are pre-trained (Phase 1). Here we apply
# them to ONE stock: score its rows, then tune entry/exit thresholds on
# THIS stock's own validation slice and report metrics on its untouched
# test slice. So the global path returns the exact same tuple as
# train_model() and flows through backtest/predict/explain unchanged —
# only the model's *weights* come from the pooled training, not a fresh
# per-stock fit. Thresholds and evaluation stay stock-specific.
# =====================================================================

def predict_with_global(data, bundle, calibrate=False, cost_profile="NSE"):
    """Drop-in replacement for train_model() that uses a loaded global
    model instead of training a per-stock one. `bundle` is the dict from
    load_global_model(1). Returns the same 6-tuple as train_model()."""
    predictor = bundle["predictor"]
    scaler = bundle["scaler"]

    X, y, dates = _masked(data, 'Target_1')
    n = len(X)
    if n < 120:
        raise ValueError("Need at least 120 rows to evaluate the global model.")

    mkt = market_frame(data, dates, profile=cost_profile)

    # Same chronological split as train_model, for honest thresholds/metrics.
    test_n = int(n * 0.20)
    val_n = int(n * 0.16)
    val_end = n - test_n

    Xs = scaler.transform(X)               # the global model's own scaler
    all_probs = predictor.predict_all(Xs)

    eval_predictor = predictor
    if calibrate:
        raw_val = all_probs[n - test_n - val_n:val_end]
        v_m = np.isfinite(raw_val)
        if v_m.sum() > 10:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            iso.fit(raw_val[v_m], y[n - test_n - val_n:val_end][v_m])
            eval_predictor = CalibratedPredictor(predictor, iso)
            all_probs = eval_predictor._map(all_probs)

    thresholds = _tune_on_slice(all_probs, mkt, n - test_n - val_n, val_end)

    test_probs = all_probs[val_end:]
    metrics = _classification_metrics(test_probs, y[val_end:])
    metrics['entry_threshold'], metrics['exit_threshold'] = thresholds
    metrics['calibration'] = calibration_metrics(test_probs, y[val_end:])
    metrics['calibrated'] = bool(calibrate)
    metrics['source'] = 'global'
    # Thresholds/backtest use this stock's chronology, but the global weights
    # were trained on a pooled universe that usually includes this name —
    # so accuracy here is "stock-specific thresholds", not pure OOS.
    metrics['oos_note'] = (
        "Global weights are pooled (may include this stock). "
        "Thresholds and the test slice below are stock-specific; "
        "treat accuracy as indicative, not fully out-of-sample. "
        "Prefer Walk-Forward for a stricter check."
    )

    # Live signal: the global predictor already encodes all its training;
    # no per-stock refit needed. Return it directly with its own scaler.
    return eval_predictor, scaler, metrics, test_probs, thresholds, dates[val_end:]


def multi_horizon_global(data, directory=GLOBAL_MODEL_DIR):
    """Multi-horizon forecast using the saved global models, one per
    horizon. Falls back to None for any horizon whose global model is
    missing, so the caller can fill gaps with the per-stock path."""
    rows = {}
    Xsrc = data[FEATURES].values
    for h in HORIZONS:
        bundle = load_global_model(h, directory)
        if bundle is None:
            rows[h] = None
            continue
        X, y, _ = _masked(data, f'Target_{h}')
        if len(X) < 80:
            rows[h] = None
            continue
        scaler, predictor = bundle["scaler"], bundle["predictor"]
        split = int(len(X) * 0.8)
        test_probs = predictor.predict_all(scaler.transform(X))[split:]
        cm = _classification_metrics(test_probs, y[split:])
        prob = predictor.predict_last(scaler.transform(Xsrc))
        rows[h] = {
            'Horizon': f"{h} Day" if h == 1 else f"{h} Days",
            'Probability Up': float(prob),
            'Rating': rating_from_prob(prob),
            'Test Accuracy': cm['accuracy'],
            'Baseline': cm['baseline_accuracy'],
        }
    return rows
