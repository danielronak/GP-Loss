"""Small, dependency-light statistics used by the analyses (numpy only)."""
import itertools
import math

import numpy as np


def exact_permutation_test(a, b, alternative="greater"):
    """Exact permutation test on the difference in means, mean(a) - mean(b).

    Enumerates every relabelling of the pooled values (C(n_a+n_b, n_a) splits;
    252 for 5 vs 5). With 5 vs 5 the smallest attainable one-sided p is 1/252.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, na = len(pooled), len(a)
    observed = a.mean() - b.mean()
    total = pooled.sum()
    count = n_splits = 0
    for idx in itertools.combinations(range(n), na):
        sa = pooled[list(idx)].sum()
        diff = sa / na - (total - sa) / (n - na)
        if alternative == "greater":
            hit = diff >= observed - 1e-12
        elif alternative == "less":
            hit = diff <= observed + 1e-12
        else:
            hit = abs(diff) >= abs(observed) - 1e-12
        count += hit
        n_splits += 1
    return observed, count / n_splits


def hierarchical_bootstrap_diff(groups_a, groups_b, n_boot=10000, seed=0, ci=0.95):
    """CI for mean(A) - mean(B) where each group is a list of runs and each run
    is an array of per-seed scores. Resamples runs, then seeds within runs."""
    rng = np.random.default_rng(seed)

    def draw(groups):
        runs = [np.asarray(groups[i], float) for i in rng.integers(0, len(groups), len(groups))]
        return np.mean([r[rng.integers(0, len(r), len(r))].mean() for r in runs])

    diffs = np.array([draw(groups_a) - draw(groups_b) for _ in range(n_boot)])
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def paired_t_test(a, b):
    """Two-sided paired t-test; returns (mean_diff, t, p). Uses scipy if present."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n < 2 or np.allclose(d.std(ddof=1), 0):
        return float(d.mean()), float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1))
    except ImportError:
        p = float("nan")
    return float(d.mean()), float(t), p
