import math

import numpy as np
import pytest

from src import stats
from src.anti_convergence import (adaptive_mutation_rate, calculate_novelty_bonus, get_tree_structure_hash,
                                  trivial_set, tree_distance)
from src.config import load_yaml
from src.gp_core import build_pset
from src.phase0 import analyze, run_order

CFG = load_yaml("configs/phase0_ceiling_test.yaml")


def _rows(scores_by_loss, seeds):
    rows = []
    for loss_id, per_seed in scores_by_loss.items():
        for s, acc in zip(seeds, per_seed):
            rows.append({"loss_id": loss_id, "eval_seed": s, "test_acc": acc})
    return rows


def _fake(base_means, exp_means, ce=88.0, mse=86.5, jitter=0.1, seed=0):
    rng = np.random.default_rng(seed)
    seeds = CFG["final_eval"]["seeds"]
    d = {"ref_CrossEntropy": ce + jitter * rng.standard_normal(5),
         "ref_MSE": mse + jitter * rng.standard_normal(5)}
    for s, m in zip(CFG["search_seeds"], base_means):
        d[f"baseline_s{s}"] = m + jitter * rng.standard_normal(5)
    for s, m in zip(CFG["search_seeds"], exp_means):
        d[f"tier1_s{s}"] = m + jitter * rng.standard_normal(5)
    return _rows(d, seeds)


def test_permutation_test_exact_values():
    delta, p = stats.exact_permutation_test([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
    assert delta == 5 and p == pytest.approx(1 / 252)
    _, p_same = stats.exact_permutation_test([1, 2, 3], [1, 2, 3])
    assert p_same > 0.4


def test_clear_lift_passes():
    out = analyze(CFG, _fake([86.5, 86.4, 86.6, 86.5, 86.3], [87.8, 87.9, 87.7, 88.0, 87.9]))
    assert out["decision"] == "PASS"
    assert out["p_permutation_one_sided"] < 0.05
    assert out["fraction_of_gap_closed"] > 0.5


def test_no_lift_fails():
    out = analyze(CFG, _fake([86.5, 86.4, 86.6, 86.5, 86.3], [86.4, 86.6, 86.5, 86.3, 86.5]))
    assert out["decision"] == "FAIL"


def test_significant_but_tiny_lift_fails():
    """Beyond noise but closes < 25% of a 1.5pp gap -> not a meaningful lift."""
    out = analyze(CFG, _fake([86.5] * 5, [86.8] * 5, jitter=0.01))
    assert out["p_permutation_one_sided"] < 0.05
    assert out["decision"] == "FAIL"


def test_incomplete_and_provisional():
    rows = _fake([86.5] * 5, [87.9] * 5)
    only_one = [r for r in rows if not r["loss_id"].endswith(("_s1", "_s2", "_s3", "_s4"))]
    assert analyze(CFG, only_one)["decision"] == "INCOMPLETE"
    three = [r for r in rows if not r["loss_id"].endswith(("_s3", "_s4"))]
    out = analyze(CFG, three)
    assert out["decision"].startswith("PROVISIONAL")
    assert out["min_attainable_p"] == pytest.approx(1 / math.comb(6, 3))


def test_run_order_interleaves_arms():
    order = run_order(CFG)
    assert order[:4] == [("baseline", 0), ("tier1", 0), ("baseline", 1), ("tier1", 1)]


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    a = [list(88 + 0.2 * rng.standard_normal(5)) for _ in range(5)]
    b = [list(86 + 0.2 * rng.standard_normal(5)) for _ in range(5)]
    lo, hi = stats.hierarchical_bootstrap_diff(a, b, n_boot=2000)
    assert lo < 2.0 < hi


def test_anti_convergence_port():
    pset = build_pset("baseline")
    assert trivial_set(pset) == {f"{op}(error)" for op in
                                 ["square", "abs", "neg", "sqrt", "log", "sin", "cos", "tanh", "sigmoid", "exp"]}
    assert "square(p_target)" in trivial_set(build_pset("tier1"))
    assert tree_distance("square(error)", "square(error)") == 0.0
    assert get_tree_structure_hash("square(sub(error, 0.7243))") == "square(sub(error, C))"
    archive = ["square(sub(error, 0.1))", "square(sub(error, 0.5))"]
    assert calculate_novelty_bonus("square(sub(error, 0.9))", archive) == 0.5
    assert calculate_novelty_bonus("abs(error)", archive) == 2.0
    assert [adaptive_mutation_rate(d) for d in (0.1, 0.3, 0.9)] == [0.5, 0.4, 0.3]
