"""Supervised models for per-flight emissions, with a leakage-safe backtest.

Positioning
-----------
The ICAO CEM is a *physical* model: two parameters per segment, fitted on
certification data, and it will never be beaten on a mission it was fitted for.
The learned models here do something the CEM structurally cannot -- condition on
the operational context that the CEM ignores by construction: airport pair,
taxi congestion, wind component, day of week, aircraft age, operator procedure.

So the framing is deliberately residual-first:

    co2_observed = CEM(distance, type)  +  f(operational context)  +  eps
                   \\_______________/       \\__________________/
                    physics, exact          what ML is actually for

Learning the residual instead of the level is not a stylistic choice. It
(a) keeps the physics exact and auditable, (b) removes the dominant variance so
the learner spends capacity on the part that is genuinely uncertain, and
(c) makes the result interpretable: a feature importance on the residual answers
"what makes this flight burn more than the book value", which is the question a
fleet planner asks. Fitting the level instead gives a model whose top feature is
"distance", at 95 % importance, and which tells nobody anything.

Both modes are supported via ``target_mode``; ``residual`` is the default.

Dependencies
------------
XGBoost and LightGBM are optional. Absent, the registry silently falls back to
scikit-learn's ``HistGradientBoostingRegressor`` -- same algorithm family,
comparable accuracy, no install friction. Nothing in the API changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .validation import PurgedWalkForwardSplit, check_no_leakage

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSpec", "BacktestResult", "available_models", "build_model",
    "make_preprocessor", "backtest", "fit_final", "feature_importance",
    "permutation_importance_df",
]

# --- optional heavy dependencies -------------------------------------------
try:  # pragma: no cover
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

try:  # pragma: no cover
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False


@dataclass
class ModelSpec:
    name: str
    factory: Callable[..., Any]
    params: dict = field(default_factory=dict)
    needs_dense_numeric: bool = True


def available_models(random_state: int = 42, n_jobs: int = -1) -> dict[str, ModelSpec]:
    """Registry of usable regressors, resolved against what is installed."""
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import Ridge

    reg: dict[str, ModelSpec] = {
        "ridge": ModelSpec(
            "ridge", Ridge,
            {"alpha": 1.0, "random_state": random_state},
        ),
        "random_forest": ModelSpec(
            "random_forest", RandomForestRegressor,
            {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 20,
             "n_jobs": n_jobs, "random_state": random_state},
        ),
        "hist_gradient_boosting": ModelSpec(
            "hist_gradient_boosting", HistGradientBoostingRegressor,
            {"max_iter": 400, "learning_rate": 0.06, "max_depth": 8,
             "min_samples_leaf": 40, "l2_regularization": 1.0,
             "early_stopping": True, "validation_fraction": 0.1,
             "random_state": random_state},
        ),
    }
    if _HAS_XGB:
        reg["xgboost"] = ModelSpec(
            "xgboost", XGBRegressor,
            {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 8,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "reg_lambda": 1.0, "min_child_weight": 10,
             "tree_method": "hist", "n_jobs": n_jobs,
             "random_state": random_state},
        )
    if _HAS_LGBM:
        reg["lightgbm"] = ModelSpec(
            "lightgbm", LGBMRegressor,
            {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 40,
             "n_jobs": n_jobs, "random_state": random_state, "verbose": -1},
        )
    return reg


def build_model(name: str, random_state: int = 42, n_jobs: int = -1, **overrides):
    """Instantiate a regressor by name, with graceful fallback."""
    registry = available_models(random_state, n_jobs)
    if name not in registry:
        fallback = "hist_gradient_boosting"
        logger.warning("%s unavailable (not installed); falling back to %s",
                       name, fallback)
        name = fallback
    spec = registry[name]
    return spec.factory(**{**spec.params, **overrides})


def make_preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str],
    scale_numeric: bool = False,
):
    """ColumnTransformer: median-impute numerics, one-hot low-cardinality cats.

    ``handle_unknown="infrequent_if_exist"`` matters in a walk-forward setting:
    a test fold will contain aircraft types and airports absent from the
    training window, and a splitter that raises on unseen categories turns a
    valid backtest into a crash on fold 3.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                       StandardScaler)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), list(numeric)),
            ("cat", Pipeline([
                # pandas `category` dtype must be materialised as plain strings:
                # SimpleImputer and OneHotEncoder both try a float cast on a
                # Categorical block and fail on the first ICAO code.
                ("as_str", FunctionTransformer(
                    lambda X: X.astype("object").fillna("__missing__").astype(str),
                    feature_names_out="one-to-one")),
                ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=50, sparse_output=False)),
            ]), list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass
class BacktestResult:
    model_name: str
    target_mode: str
    folds: pd.DataFrame
    predictions: pd.DataFrame
    fit_seconds: float
    leakage_checks: list[dict]

    @property
    def summary(self) -> pd.Series:
        """Mean +/- std across folds. The std is the number that matters.

        A model whose fold-to-fold R2 ranges over 0.3 has not been validated,
        whatever its mean. Reporting the mean alone is how backtests lie.
        """
        m = self.folds[["mae", "rmse", "r2", "mape"]]
        out = {}
        for c in m.columns:
            out[f"{c}_mean"] = m[c].mean()
            out[f"{c}_std"] = m[c].std(ddof=1)
        out["n_folds"] = len(self.folds)
        out["fit_seconds"] = self.fit_seconds
        return pd.Series(out, name=self.model_name)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[keep], y_pred[keep]
    nz = np.abs(yt) > 1e-9
    return {
        "n": int(keep.sum()),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "mape": float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100)
        if nz.any() else np.nan,
    }


def backtest(
    df: pd.DataFrame,
    model_name: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    target: str = "co2_kg",
    baseline: str | None = "cem_co2_kg",
    target_mode: str = "residual",
    time_col: str = "actual_departure_day",
    splitter=None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> BacktestResult:
    """Walk-forward backtest with purged folds.

    Parameters
    ----------
    target_mode
        ``"residual"``: learn ``target - baseline``, predict ``baseline + f(x)``.
        ``"level"``: learn ``target`` directly.
    baseline
        Column holding the physical CEM prediction. Required in residual mode.

    Metrics are always computed on the **level**, whatever the training target,
    so the two modes are directly comparable and so the reported MAE is in kg of
    CO2 rather than in kg of residual.
    """
    from sklearn.pipeline import Pipeline

    if target_mode == "residual" and (baseline is None or baseline not in df.columns):
        raise ValueError("residual mode needs a `baseline` column (CEM prediction)")

    splitter = splitter or PurgedWalkForwardSplit(n_splits=5, test_size=7, embargo=1)
    features = list(numeric) + list(categorical)

    work = df.dropna(subset=[target] + ([baseline] if baseline else []))
    work = work.sort_values(time_col).reset_index(drop=True)

    y_level = work[target].to_numpy(dtype="float64")
    base = (work[baseline].to_numpy(dtype="float64") if baseline
            else np.zeros_like(y_level))
    y_fit = y_level - base if target_mode == "residual" else y_level
    groups = work[time_col]

    fold_rows, pred_frames, checks = [], [], []
    t0 = time.perf_counter()

    for k, (tr, te) in enumerate(splitter.split(work, y_fit, groups=groups)):
        checks.append(check_no_leakage(tr, te, groups,
                                       getattr(splitter, "embargo", 0)))
        pipe = Pipeline([
            ("prep", make_preprocessor(numeric, categorical,
                                       scale_numeric=(model_name == "ridge"))),
            ("model", build_model(model_name, random_state, n_jobs)),
        ])
        pipe.fit(work.iloc[tr][features], y_fit[tr])
        pred_fit = pipe.predict(work.iloc[te][features])
        pred_level = base[te] + pred_fit if target_mode == "residual" else pred_fit

        m = _metrics(y_level[te], pred_level)
        m |= {"fold": k, "n_train": len(tr), "n_test": len(te),
              "test_start": str(groups.iloc[te].min().date()),
              "test_end": str(groups.iloc[te].max().date())}
        fold_rows.append(m)

        pred_frames.append(pd.DataFrame({
            "fold": k, "index": work.index[te],
            "y_true": y_level[te], "y_pred": pred_level, "baseline": base[te],
        }))
        logger.info("fold %d: MAE %.1f kg | RMSE %.1f | R2 %.4f | n=%d",
                    k, m["mae"], m["rmse"], m["r2"], m["n"])

    elapsed = time.perf_counter() - t0

    # Baseline-only reference: the CEM's own out-of-sample error, so the
    # learned model is scored against physics rather than against zero.
    if baseline:
        base_metrics = _metrics(y_level, base)
        logger.info("CEM baseline (all rows): MAE %.1f kg | R2 %.4f",
                    base_metrics["mae"], base_metrics["r2"])

    return BacktestResult(
        model_name=model_name, target_mode=target_mode,
        folds=pd.DataFrame(fold_rows),
        predictions=pd.concat(pred_frames, ignore_index=True) if pred_frames
        else pd.DataFrame(),
        fit_seconds=elapsed, leakage_checks=checks,
    )


def fit_final(
    df: pd.DataFrame,
    model_name: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    target: str = "co2_kg",
    baseline: str | None = "cem_co2_kg",
    target_mode: str = "residual",
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Refit on the full sample once the backtest has settled the choice."""
    from sklearn.pipeline import Pipeline

    features = list(numeric) + list(categorical)
    work = df.dropna(subset=[target] + ([baseline] if baseline else []))
    y = work[target].to_numpy(dtype="float64")
    if target_mode == "residual":
        y = y - work[baseline].to_numpy(dtype="float64")

    pipe = Pipeline([
        ("prep", make_preprocessor(numeric, categorical,
                                   scale_numeric=(model_name == "ridge"))),
        ("model", build_model(model_name, random_state, n_jobs)),
    ])
    pipe.fit(work[features], y)
    return pipe


def feature_importance(pipeline, top: int = 25) -> pd.DataFrame:
    """Native importances, mapped back to post-encoding feature names.

    Impurity-based importances are biased toward high-cardinality features; use
    :func:`permutation_importance_df` for anything that will be quoted.
    """
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    else:
        raise AttributeError(f"{type(model).__name__} exposes no importances")
    return (pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top).reset_index(drop=True))


def permutation_importance_df(
    pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5,
    random_state: int = 42, n_jobs: int = -1, sample: int | None = 50_000,
) -> pd.DataFrame:
    """Permutation importance with a std, on a subsample.

    Model-agnostic and unbiased with respect to cardinality, at the cost of
    ``n_repeats x p`` extra prediction passes -- hence the subsample, which cuts
    the cost by ~25x on this dataset with no material change in ranking.
    """
    from sklearn.inspection import permutation_importance

    if sample is not None and len(X) > sample:
        idx = np.random.default_rng(random_state).choice(len(X), sample, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]

    res = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                                 random_state=random_state, n_jobs=n_jobs)
    return (pd.DataFrame({"feature": X.columns,
                          "importance_mean": res.importances_mean,
                          "importance_std": res.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))
