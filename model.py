# =============================
# model.py
# =============================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from data import FEATURES, HORIZONS

SEED = 42
TRANSACTION_COST = 0.001
TRADING_DAYS = 252
DEFAULT_THRESHOLDS = (0.55, 0.45)
ENTRY_GRID = np.round(np.arange(0.50, 0.71, 0.05), 2)
EXIT_GRID = np.round(np.arange(0.30, 0.51, 0.05), 2)
SEQ_WINDOW = 20  # lookback days for LSTM/GRU

MODEL_TYPES = ["Ensemble (NN + XGBoost + RF)", "Neural Network", "LSTM", "GRU"]


# =====================================================================
# Networks
# =====================================================================

class StockModel(nn.Module):
    """Tabular MLP. Outputs raw logits; sigmoid applied at inference."""

    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class SequenceNet(nn.Module):
    """LSTM/GRU over a window of daily feature vectors; the final hidden
    state feeds a linear head. Outputs raw logits."""

    def __init__(self, input_size, rnn_type="lstm", hidden=32):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):           # x: (batch, time, features)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


# =====================================================================
# Torch training helpers
# =====================================================================

def _pos_weight(y_t):
    pos_frac = max(float(y_t.mean()), 1e-6)
    return torch.tensor([(1.0 - pos_frac) / pos_frac])


def _train_torch(model, X_t, y_t, epochs, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_t))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
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


def make_predictor(model_type):
    if model_type.startswith("Ensemble"):
        members = [
            TabularNNPredictor(),
            TreePredictor(RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=20,
                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
            ), "Random Forest"),
        ]
        if HAS_XGB:
            members.append(TreePredictor(XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=SEED,
            ), "XGBoost"))
        return EnsemblePredictor(members)
    if model_type == "LSTM":
        return SequencePredictor("lstm")
    if model_type == "GRU":
        return SequencePredictor("gru")
    return TabularNNPredictor()


# =====================================================================
# Pure strategy / analytics logic (unchanged math, no torch)
# =====================================================================

def build_positions(probs, entry, exit_):
    raw = np.where(probs > entry, 1.0, np.where(probs < exit_, 0.0, np.nan))
    return pd.Series(raw).ffill().fillna(0.0).to_numpy()


def performance_stats(positions, returns, cost=TRANSACTION_COST):
    positions = np.asarray(positions, dtype=float)
    returns = np.asarray(returns, dtype=float)

    changes = np.abs(np.diff(positions, prepend=0.0))
    strategy_returns = returns * positions - cost * changes
    equity = np.cumprod(1.0 + strategy_returns)

    std = strategy_returns.std()
    sharpe = (float(strategy_returns.mean() / std * np.sqrt(TRADING_DAYS))
              if std > 0 else float('nan'))

    running_max = np.maximum.accumulate(equity)
    max_drawdown = float((equity / running_max - 1.0).min())

    in_market = positions == 1
    win_rate = float((returns[in_market] > 0).mean()) if in_market.any() else float('nan')

    return {
        'total_return': float(equity[-1] - 1.0),
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'exposure': float(in_market.mean()),
        'win_rate': win_rate,
        'n_trades': int((np.diff(positions, prepend=0.0) > 0).sum()),
        'equity': equity,
    }


def tune_thresholds(probs, returns, cost=TRANSACTION_COST):
    best = DEFAULT_THRESHOLDS
    best_score = -np.inf
    for entry in ENTRY_GRID:
        for exit_ in EXIT_GRID:
            if exit_ >= entry:
                continue
            stats = performance_stats(build_positions(probs, entry, exit_), returns, cost)
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


# =====================================================================
# Main entry points
# =====================================================================

def train_model(data, model_type="Neural Network"):
    """1-day model of the chosen type. Chronological 64/16/20 split;
    scaler fit on train only; thresholds tuned on validation; metrics
    from the untouched test slice. Returns probabilities (not tensors)
    so every model type flows through the same backtest."""
    X, y, dates = _masked(data, 'Target_1')
    n = len(X)
    if n < 300:
        raise ValueError("Need at least 300 rows of feature data to train.")

    next_ret = data['Close'].pct_change().shift(-1).loc[dates].values

    test_n = int(n * 0.20)
    val_n = int(n * 0.16)
    train_end = n - test_n - val_n
    val_end = n - test_n

    scaler = StandardScaler().fit(X[:train_end])
    Xs = scaler.transform(X)

    predictor = make_predictor(model_type).fit(Xs, y, train_end)
    all_probs = predictor.predict_all(Xs)

    val_probs = all_probs[train_end:val_end]
    val_rets = next_ret[train_end:val_end]
    mask = np.isfinite(val_rets) & np.isfinite(val_probs)
    thresholds = tune_thresholds(val_probs[mask], val_rets[mask])

    test_probs = all_probs[val_end:]
    metrics = _classification_metrics(test_probs, y[val_end:])
    metrics['entry_threshold'], metrics['exit_threshold'] = thresholds

    return predictor, scaler, metrics, test_probs, thresholds, dates[val_end:]


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

        prob = predictor.predict_last(scaler.transform(Xs_latest_src))

        rows.append({
            'Horizon': f"{h} Day" if h == 1 else f"{h} Days",
            'Probability Up': prob,
            'Rating': rating_from_prob(prob),
            'Test Accuracy': cm['accuracy'],
            'Baseline': cm['baseline_accuracy'],
        })

    return pd.DataFrame(rows)


def backtest(test_probs, prices, test_index, thresholds=DEFAULT_THRESHOLDS):
    """Backtest pre-computed probabilities on the held-out period."""
    probs = np.asarray(test_probs, dtype=float)

    next_returns = prices.pct_change().shift(-1)
    rets = next_returns.loc[test_index].to_numpy()
    valid = np.isfinite(rets) & np.isfinite(probs)
    probs, rets, idx = probs[valid], rets[valid], test_index[valid]

    positions = build_positions(probs, *thresholds)
    stats = performance_stats(positions, rets)

    equity = pd.Series(stats.pop('equity'), index=idx, name='Strategy')
    buy_hold = pd.Series(np.cumprod(1.0 + rets), index=idx, name='Buy & Hold')
    stats['buy_hold_return'] = float(buy_hold.iloc[-1] - 1.0)

    return stats, equity, buy_hold


def walk_forward(data, model_type="Neural Network", n_splits=4, min_train=300):
    """Expanding-window walk-forward validation of the chosen model type."""
    X, y, dates = _masked(data, 'Target_1')
    n = len(X)
    fold_size = (n - min_train) // n_splits
    if fold_size < 40:
        raise ValueError("Not enough history for walk-forward validation.")

    next_ret = data['Close'].pct_change().shift(-1).loc[dates].values

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

        val_probs = all_probs[fit_end:train_total]
        val_rets = next_ret[fit_end:train_total]
        v_mask = np.isfinite(val_rets) & np.isfinite(val_probs)
        entry, exit_ = tune_thresholds(val_probs[v_mask], val_rets[v_mask])

        test_probs = all_probs[train_total:test_end]
        test_rets = next_ret[train_total:test_end]
        t_mask = np.isfinite(test_rets) & np.isfinite(test_probs)

        stats = performance_stats(
            build_positions(test_probs[t_mask], entry, exit_), test_rets[t_mask]
        )
        accuracy = float(
            ((test_probs[t_mask] > 0.5) == y[train_total:test_end][t_mask]).mean()
        )
        buy_hold = float(np.prod(1.0 + test_rets[t_mask]) - 1.0)

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
