"""Sequence models for fleet-level emissions dynamics (LSTM + baselines).

Scope
-----
The per-flight models in ``predictive_models.py`` are cross-sectional. This
module works on the *aggregated* series -- daily fleet CO2 -- where the question
changes from "what does this flight emit" to "where is the fleet's emissions
trajectory going", which is a forecasting problem with seasonality (strong
day-of-week effect in air traffic) and regime shifts (fuel price, capacity
changes, geopolitical re-routing).

Discipline
----------
A deep sequence model on a short series is a good way to produce an impressive
plot and a useless forecast. Three guards are built in:

1. **Baselines first.** ``seasonal_naive`` (y[t-7]) and ``drift`` are computed
   on the same folds. An LSTM that does not beat seasonal-naive on a
   weekly-seasonal series has learned nothing; the comparison is not optional
   and is printed alongside every result.
2. **Scaling fitted on train only.** The scaler is fitted inside each fold, on
   the training window. Fitting it on the whole series before splitting leaks
   the test distribution's location and scale -- one of the most common and
   least visible errors in time-series deep learning.
3. **Walk-forward evaluation**, never a random split (see ``validation.py``).

torch is optional. Without it, :func:`fit_lstm` raises a clear message and the
baselines still run, so the pipeline never breaks on a missing GPU stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_daily", "make_windows", "seasonal_naive_forecast",
    "drift_forecast", "fit_lstm", "walk_forward_forecast", "TORCH_AVAILABLE",
]

try:  # pragma: no cover
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def aggregate_daily(
    df: pd.DataFrame,
    time_col: str = "actual_departure_day",
    value_cols: Sequence[str] = ("co2_kg", "distance_km"),
    by: str | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Collapse the flight table to a regular time series.

    ``asfreq`` after the groupby is deliberate: a missing operational day must
    appear as NaN, not vanish. A silently shortened index turns a 7-day
    seasonality into a 6-day one and every lag feature after it is wrong.
    """
    work = df.copy(deep=False)
    work[time_col] = pd.to_datetime(work[time_col]).dt.normalize()
    keys = [time_col] + ([by] if by else [])
    agg = {c: "sum" for c in value_cols if c in work.columns}
    agg["n_flights"] = (time_col, "size") if False else None  # placeholder
    out = work.groupby(keys, observed=True).agg(
        **{c: (c, "sum") for c in value_cols if c in work.columns},
        n_flights=(time_col, "size"),
    ).reset_index()

    if by is None:
        out = out.set_index(time_col).asfreq(freq)
    return out


def make_windows(
    series: np.ndarray, lookback: int, horizon: int, exog: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised windows ``(X, y)`` with a stride-trick view.

    ``sliding_window_view`` returns a *view*, not a copy: memory stays O(n)
    instead of O(n . lookback). On a 5-year daily series with lookback 28 that
    is the difference between 14 KB and 400 KB -- irrelevant here, decisive the
    moment the same code runs on minute-level data, which is exactly the pattern
    that shows up in market microstructure work.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    s = np.asarray(series, dtype="float32").ravel()
    n = s.size
    total = lookback + horizon
    if n < total:
        raise ValueError(f"series too short: {n} points for lookback+horizon={total}")

    windows = sliding_window_view(s, total)          # (n - total + 1, total)
    X = windows[:, :lookback]
    y = windows[:, lookback:]

    if exog is not None:
        e = np.asarray(exog, dtype="float32")
        if e.ndim == 1:
            e = e[:, None]
        ew = sliding_window_view(e, lookback, axis=0)     # (n-lookback+1, k, L)
        ew = ew[: X.shape[0]].transpose(0, 2, 1)          # (m, L, k)
        X = np.concatenate([X[:, :, None], ew], axis=2)
    else:
        X = X[:, :, None]

    return np.ascontiguousarray(X), np.ascontiguousarray(y)


# ---------------------------------------------------------------------------
# Baselines -- must be beaten before an LSTM means anything
# ---------------------------------------------------------------------------
def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int = 7
                            ) -> np.ndarray:
    """y_hat[t+h] = y[t+h-season]. The bar for any weekly-seasonal series."""
    h = np.asarray(history, dtype="float64").ravel()
    if h.size < season:
        return np.repeat(h[-1], horizon)
    return np.array([h[-season + (i % season)] for i in range(horizon)])


def drift_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Linear extrapolation through the first and last observation."""
    h = np.asarray(history, dtype="float64").ravel()
    if h.size < 2:
        return np.repeat(h[-1] if h.size else np.nan, horizon)
    slope = (h[-1] - h[0]) / (h.size - 1)
    return h[-1] + slope * np.arange(1, horizon + 1)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:  # pragma: no cover - exercised only where torch exists

    class EmissionsLSTM(nn.Module):
        """Stacked LSTM -> linear head producing the whole horizon at once.

        Direct multi-horizon output rather than recursive one-step rollout:
        recursion compounds its own error and, more importantly, makes the
        training loss inconsistent with the evaluation metric.
        """

        def __init__(self, n_features: int = 1, hidden_size: int = 64,
                     num_layers: int = 2, horizon: int = 7, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, horizon),
            )

        def forward(self, x):                       # (B, L, F) -> (B, H)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


@dataclass
class LSTMResult:
    model: object
    scaler_mean: float
    scaler_std: float
    train_loss: list[float]
    val_loss: list[float]
    epochs_run: int


def fit_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2,
    epochs: int = 60, batch_size: int = 64, learning_rate: float = 1e-3,
    patience: int = 10, seed: int = 42, device: str | None = None,
) -> LSTMResult:
    """Train the LSTM with train-only standardisation and early stopping.

    The scaler statistics are computed on ``X_train`` alone and returned with
    the model, so inference applies exactly the transform the model was trained
    under -- and so the leak is impossible by construction rather than by
    convention.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not installed. `pip install torch` for the LSTM path; the "
            "baselines and the gradient-boosting models run without it."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    mu = float(X_train[..., 0].mean())
    sd = float(X_train[..., 0].std()) or 1.0

    def _prep(X, y):
        Xs = X.copy().astype("float32")
        Xs[..., 0] = (Xs[..., 0] - mu) / sd
        ys = ((y - mu) / sd).astype("float32")
        return (torch.from_numpy(Xs).to(device), torch.from_numpy(ys).to(device))

    Xtr, ytr = _prep(X_train, y_train)
    has_val = X_val is not None and y_val is not None and len(X_val) > 0
    if has_val:
        Xva, yva = _prep(X_val, y_val)

    model = EmissionsLSTM(n_features=X_train.shape[2], hidden_size=hidden_size,
                          num_layers=num_layers, horizon=y_train.shape[1],
                          dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(patience // 3, 2))
    loss_fn = nn.HuberLoss(delta=1.0)   # robust to the occasional operational outlier

    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    tr_hist, va_hist = [], []
    best, best_state, bad = np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
        tr_hist.append(total / len(ds))

        if has_val:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(Xva), yva).item()
            va_hist.append(v)
            sched.step(v)
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    logger.info("early stop at epoch %d (best val %.5f)", epoch, best)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return LSTMResult(model=model, scaler_mean=mu, scaler_std=sd,
                      train_loss=tr_hist, val_loss=va_hist,
                      epochs_run=len(tr_hist))


def predict_lstm(result: LSTMResult, X: np.ndarray) -> np.ndarray:
    """Inverse-transform back to kg CO2."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch not installed")
    Xs = X.copy().astype("float32")
    Xs[..., 0] = (Xs[..., 0] - result.scaler_mean) / result.scaler_std
    result.model.eval()
    with torch.no_grad():
        out = result.model(torch.from_numpy(Xs)).cpu().numpy()
    return out * result.scaler_std + result.scaler_mean


def walk_forward_forecast(
    series: pd.Series,
    lookback: int = 28,
    horizon: int = 7,
    n_folds: int = 4,
    season: int = 7,
    use_lstm: bool = True,
    **lstm_kwargs,
) -> pd.DataFrame:
    """Compare LSTM against seasonal-naive and drift on rolling origins.

    Returns one row per (fold, method) with MAE, RMSE and MAPE. The LSTM column
    is populated only if torch is available; otherwise the baselines still give
    a complete, publishable comparison.
    """
    y = series.dropna().to_numpy(dtype="float64")
    n = y.size
    rows = []

    for k in range(n_folds):
        # Rolling origin: each fold's test block sits immediately after its
        # training window, and no future data touches the fit.
        end_train = n - (n_folds - k) * horizon
        if end_train < lookback + horizon:
            continue
        hist, actual = y[:end_train], y[end_train:end_train + horizon]

        preds = {
            "seasonal_naive": seasonal_naive_forecast(hist, horizon, season),
            "drift": drift_forecast(hist, horizon),
        }

        if use_lstm and TORCH_AVAILABLE:
            try:
                Xtr, ytr = make_windows(hist, lookback, horizon)
                cut = max(int(0.85 * len(Xtr)), 1)
                res = fit_lstm(Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:],
                               **lstm_kwargs)
                last = hist[-lookback:].reshape(1, lookback, 1)
                preds["lstm"] = predict_lstm(res, last).ravel()
            except Exception as exc:  # pragma: no cover
                logger.warning("fold %d: LSTM failed (%s)", k, exc)

        for name, p in preds.items():
            err = actual - p[: actual.size]
            nz = np.abs(actual) > 1e-9
            rows.append({
                "fold": k, "method": name, "n": actual.size,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mape": float(np.mean(np.abs(err[nz] / actual[nz])) * 100)
                if nz.any() else np.nan,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        logger.info("\n%s", out.groupby("method")[["mae", "rmse", "mape"]].mean())
    return out
