import pytest
import torch
import torch.nn.functional as F

from src.gp_core import (EvolvedLossFunction, OPERATOR_SETS, build_pset, is_crossentropy_like,
                         make_toolbox, tree_from_string, uses_terminal)


def _batch(b=16, c=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, c, generator=g) * 3, torch.randint(0, c, (b,), generator=g)


def test_pset_arities():
    assert build_pset("baseline").arguments == ["error"]
    assert build_pset("tier1").arguments == ["error", "p_target"]
    assert build_pset("tier1_logits").arguments == ["error", "p_target", "logits"]
    with pytest.raises(ValueError):
        build_pset("nope")


def test_neg_log_p_target_is_exactly_crossentropy():
    """The single most important Phase 0 correctness check (Kickoff §4.2)."""
    pset = build_pset("tier1")
    loss = EvolvedLossFunction(tree_from_string("neg(log(p_target))", pset), pset)
    logits, y = _batch()
    torch.testing.assert_close(loss(logits, y), F.cross_entropy(logits, y), rtol=1e-5, atol=1e-6)
    # gradients match too
    l1 = logits.clone().requires_grad_(True)
    l2 = logits.clone().requires_grad_(True)
    loss(l1, y).backward()
    F.cross_entropy(l2, y).backward()
    torch.testing.assert_close(l1.grad, l2.grad, rtol=1e-4, atol=1e-6)
    assert loss.fallback_count == 0


def test_square_error_is_mse_sum_over_classes():
    pset = build_pset("baseline")
    loss = EvolvedLossFunction(tree_from_string("square(error)", pset), pset)
    logits, y = _batch()
    err = F.one_hot(y, 10).float() - logits.softmax(1)
    torch.testing.assert_close(loss(logits, y), err.pow(2).sum(1).mean())


def test_mixed_tree_broadcasts_p_target_across_classes():
    pset = build_pset("tier1")
    loss = EvolvedLossFunction(tree_from_string("add(square(error), neg(log(p_target)))", pset), pset)
    logits, y = _batch()
    err = F.one_hot(y, 10).float() - logits.softmax(1)
    expected = (err.pow(2).sum(1) + 10 * F.cross_entropy(logits, y, reduction="none")).mean()
    torch.testing.assert_close(loss(logits, y), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("expr", [
    "tanh(sqrt(0.43))", "max(error, 1.2)", "min(0.5, error)", "square(0.72)",
    "add(max(error, error), sqrt(0.438))", "mul(error, tanh(0.5))",
])
def test_constant_subtrees_no_longer_crash(expr):
    """Phase 4b bug: torch unary ops / maximum raise on Python floats."""
    pset = build_pset("baseline")
    loss = EvolvedLossFunction(tree_from_string(expr, pset), pset)
    logits, y = _batch()
    out = loss(logits.requires_grad_(True), y)
    assert torch.isfinite(out)
    assert loss.fallback_count == 0


def test_constant_only_tree_is_finite_but_gradient_free():
    pset = build_pset("baseline")
    loss = EvolvedLossFunction(tree_from_string("sqrt(0.64)", pset), pset)
    logits, y = _batch()
    logits.requires_grad_(True)
    loss(logits, y).backward()
    assert logits.grad.abs().max() == 0


def test_fallback_is_recorded():
    pset = build_pset("baseline")
    loss = EvolvedLossFunction(tree_from_string("square(error)", pset), pset)
    loss.func = lambda e: e * float("nan")
    logits, y = _batch()
    out = loss(logits, y)
    assert torch.isfinite(out) and loss.fallback_count == 1


@pytest.mark.parametrize("operator_set", OPERATOR_SETS)
def test_random_trees_are_finite_on_extreme_inputs(operator_set):
    """Every operator is domain-safe: random trees never produce NaN/Inf."""
    import random
    random.seed(0)
    pset = build_pset(operator_set)
    tb = make_toolbox(pset)
    logits = torch.tensor([[50.0, -50.0, 0.0, 1e-8, 20.0, -20.0, 3.0, -3.0, 0.5, 100.0]] * 4)
    y = torch.tensor([0, 1, 3, 9])
    n_fallback = 0
    for _ in range(300):
        loss = EvolvedLossFunction(tb.individual(), pset)
        lg = logits.clone().requires_grad_(True)
        out = loss(lg, y)
        assert torch.isfinite(out)
        n_fallback += loss.fallback_count
    # tier1_logits can overflow via logits; everything else should almost never fall back
    if operator_set != "tier1_logits":
        assert n_fallback <= 3


def test_tier2_ops_bounded():
    from src.gp_core import safe_expm1, safe_log1p, safe_pow, safe_recip
    x = torch.tensor([-1e6, -1.0, -1e-9, 0.0, 1e-9, 1.0, 1e6])
    for f in (safe_recip, safe_log1p, safe_expm1):
        assert torch.isfinite(f(x)).all()
    assert torch.isfinite(safe_pow(x, torch.tensor(10.0))).all()


def test_helpers():
    pset = build_pset("tier1")
    t = tree_from_string("neg(log(p_target))", pset)
    assert is_crossentropy_like(t) and uses_terminal(t, "p_target")
    assert not uses_terminal(t, "error")
    assert not is_crossentropy_like(tree_from_string("square(error)", pset))


def test_max_depth_enforced_by_toolbox():
    import random
    random.seed(1)
    pset = build_pset("baseline")
    tb = make_toolbox(pset, max_depth=5)
    pop = tb.population(n=30)
    for _ in range(200):
        a, b = tb.clone(random.choice(pop)), tb.clone(random.choice(pop))
        a, b = tb.mate(a, b)
        (a,) = tb.mutate(a)
        assert a.height <= 5 and b.height <= 5
