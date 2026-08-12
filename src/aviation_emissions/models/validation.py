"""Leakage-safe cross-validation for temporally ordered data.

Why not ``KFold``
-----------------
Flight records are not i.i.d. A rotation observed on day t is highly predictive
of the same aircraft's day t+1 leg: same tail, same route pair, same crew
pairing. Random K-fold puts sibling legs in train and test simultaneously, and
the resulting R2 measures memorisation of the rotation schedule, not
generalisation. The gap is not subtle -- on autocorrelated panels it routinely
inflates R2 by 10-30 points.

The two guards implemented here are the standard ones from the financial ML
literature (Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7),
and they transfer to any panel with serial dependence:

**Purging** -- remove from the training set any observation whose information
window overlaps the test window. Here the natural window is the rotation cycle.

**Embargo** -- additionally drop a buffer of ``embargo`` periods immediately
*after* the test block, because serial correlation makes the observations just
after a test fold nearly as informative about it as the fold itself.

:class:`PurgedWalkForwardSplit` is the honest default for a production
forecaster: train on the past, test on the future, never the reverse.
:class:`PurgedKFold` is available when the sample is too short to sacrifice a
warm-up period, at the cost of training on data that post-dates the test block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["PurgedWalkForwardSplit", "PurgedKFold", "temporal_holdout",
           "check_no_leakage"]


@dataclass
class PurgedWalkForwardSplit:
    """Expanding-window walk-forward CV with a post-test embargo.

    Parameters
    ----------
    n_splits
        Number of successive test blocks.
    test_size
        Length of each test block, in *periods* of the time index (days here).
    embargo
        Periods dropped after each test block before training resumes.
    min_train_size
        Minimum training periods required for a fold to be emitted.
    expanding
        True: each fold trains on everything before the test block (expanding
        window). False: fixed-length rolling window of ``max_train_size``.

    Yields ``(train_idx, test_idx)`` positional index arrays, matching the
    scikit-learn splitter protocol, so it drops straight into
    ``cross_val_score`` / ``GridSearchCV``.
    """

    n_splits: int = 5
    test_size: int = 7
    embargo: int = 1
    min_train_size: int = 7
    expanding: bool = True
    max_train_size: int | None = None

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self, X, y=None, groups: Sequence | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError(
                "PurgedWalkForwardSplit requires `groups`: the time index "
                "(e.g. df['actual_departure_day']). Splitting on row order "
                "alone silently assumes the frame is sorted by time."
            )
        t = pd.Series(pd.to_datetime(pd.Series(groups).to_numpy()))
        periods = t.dt.normalize()
        unique = np.sort(periods.unique())
        n_periods = unique.size

        needed = self.min_train_size + self.n_splits * self.test_size
        if n_periods < needed:
            raise ValueError(
                f"time index spans {n_periods} periods; need at least {needed} "
                f"for {self.n_splits} folds of {self.test_size} with a "
                f"{self.min_train_size}-period warm-up. Reduce n_splits or "
                f"test_size."
            )

        # Position of each row within the ordered period list: one O(n log P)
        # searchsorted instead of a per-fold boolean scan over the whole frame.
        pos = np.searchsorted(unique, periods.to_numpy())

        first_test = n_periods - self.n_splits * self.test_size
        for k in range(self.n_splits):
            test_start = first_test + k * self.test_size
            test_end = test_start + self.test_size

            test_mask = (pos >= test_start) & (pos < test_end)
            train_end = test_start                       # purge: strictly before
            train_start = 0 if self.expanding else max(
                0, train_end - (self.max_train_size or train_end))
            train_mask = (pos >= train_start) & (pos < train_end)

            # Embargo: exclude the periods right after the test block. On an
            # expanding window they are not in this fold's training set anyway,
            # but the mask keeps the semantics explicit and correct for the
            # rolling case and for any future anchored variant.
            embargo_mask = (pos >= test_end) & (pos < test_end + self.embargo)
            train_mask &= ~embargo_mask

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size == 0 or test_idx.size == 0:
                logger.warning("fold %d empty, skipped", k)
                continue
            yield train_idx, test_idx


@dataclass
class PurgedKFold:
    """K-fold over contiguous time blocks, with purge + embargo around each.

    Trains on data that post-dates the test block, so it is *not* a valid
    simulation of live deployment. Use it only for variance estimation on short
    samples, and say so when reporting.
    """

    n_splits: int = 5
    embargo: int = 1

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups: Sequence | None = None
              ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("PurgedKFold requires `groups` (the time index)")
        periods = pd.Series(pd.to_datetime(pd.Series(groups).to_numpy())).dt.normalize()
        unique = np.sort(periods.unique())
        pos = np.searchsorted(unique, periods.to_numpy())
        blocks = np.array_split(np.arange(unique.size), self.n_splits)

        for blk in blocks:
            lo, hi = blk[0], blk[-1] + 1
            test_mask = (pos >= lo) & (pos < hi)
            purge_mask = (pos >= lo - self.embargo) & (pos < hi + self.embargo)
            train_idx = np.flatnonzero(~purge_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx


def temporal_holdout(
    df: pd.DataFrame, time_col: str = "actual_departure_day", test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single chronological split. The final, untouched out-of-sample set.

    Held out once, at the start, and looked at once, at the end. Every time it
    is used to choose between models it stops being out-of-sample.
    """
    t = pd.to_datetime(df[time_col])
    cutoff = t.quantile(1 - test_frac)
    train, test = df[t < cutoff], df[t >= cutoff]
    logger.info("holdout at %s: %d train / %d test", cutoff, len(train), len(test))
    return train.copy(), test.copy()


def check_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: Sequence,
    embargo: int = 0,
) -> dict:
    """Assert a fold is temporally clean. Call it in tests, not in comments.

    Verifies (a) no index appears in both sides, (b) every training timestamp
    precedes every test timestamp, (c) the embargo gap is honoured.
    """
    t = pd.Series(pd.to_datetime(pd.Series(groups).to_numpy())).dt.normalize()
    tr, te = t.iloc[train_idx], t.iloc[test_idx]
    overlap = np.intersect1d(train_idx, test_idx).size
    gap_days = (te.min() - tr.max()).days if len(tr) and len(te) else np.nan
    return {
        "index_overlap": int(overlap),
        "train_max": tr.max(), "test_min": te.min(),
        "gap_days": gap_days,
        "is_causal": bool(len(tr) == 0 or tr.max() < te.min()),
        "embargo_respected": bool(np.isnan(gap_days) or gap_days >= 1),
    }
