"""Group statistics for the pre-registered hypotheses (n = 9 subjects, non-parametric)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe(d: np.ndarray) -> dict:
    """Median, IQR and n of a vector of per-subject values."""
    d = np.asarray(d, float)
    q1, q3 = np.percentile(d, [25, 75])
    return {"median": float(np.median(d)), "iqr_lo": float(q1), "iqr_hi": float(q3), "n": int(len(d))}


def wilcoxon_greater(x: np.ndarray, y: np.ndarray | float = 0.0) -> dict:
    """One-sided Wilcoxon signed-rank test of x - y > 0 (exact for small n, no ties)."""
    d = np.asarray(x, float) - (np.asarray(y, float) if np.ndim(y) else y)
    r = stats.wilcoxon(d, alternative="greater")
    return {"W": float(r.statistic), "p": float(r.pvalue), **{f"delta_{k}": v for k, v in describe(d).items()}}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rank correlation (two-sided p)."""
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p_two_sided": float(r.pvalue), "n": int(len(x))}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def pick(df: pd.DataFrame, **kw) -> pd.Series:
    """Per-subject auc_mean for one condition, indexed by subject."""
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.set_index("subject")["auc_mean"].sort_index()
