import torch
import torch.nn as nn

from src.architectures import SimpleCNN, TinyMLP
from src.data import make_synthetic, train_val_split
from src.gp_core import EvolvedLossFunction, build_pset, tree_from_string
from src.smoke_test import KillLog, single_batch_check
from src.utils import train_and_evaluate

PSET = build_pset("tier1")


def _loss(expr, pset=PSET):
    return EvolvedLossFunction(tree_from_string(expr, pset), pset)


def _check(loss):
    torch.manual_seed(0)
    model = TinyMLP()
    data = make_synthetic(64)
    return single_batch_check(model, loss, data.x, data.y)


def test_smoke_passes_real_losses():
    assert _check(_loss("square(error)")) is None
    assert _check(_loss("neg(log(p_target))")) is None
    assert _check(nn.CrossEntropyLoss()) is None


def test_smoke_kills_constant_tree():
    assert _check(_loss("sqrt(0.64)")) == "zero_grad"


def test_smoke_kills_nan_tree():
    loss = _loss("square(error)")
    loss.func = lambda e, p: e * float("nan")
    assert _check(loss) == "invalid_output"


def test_smoke_kills_constant_but_connected_loss():
    class Const(nn.Module):
        def forward(self, logits, y):
            return logits.sum() * 0 + 1.0 + 1e-3 * logits.softmax(1).sum()  # softmax sums to 1
    assert _check(Const()) in ("zero_grad", "constant_loss")


def test_smoke_does_not_touch_global_rng():
    data = make_synthetic(8)
    torch.manual_seed(123)
    model = TinyMLP()
    before = torch.get_rng_state()
    single_batch_check(model, _loss("square(error)"), data.x, data.y)
    assert torch.equal(before, torch.get_rng_state())


def test_train_and_evaluate_learns_and_is_deterministic():
    data = make_synthetic(1200, seed=1)
    train, val = train_val_split(data, 300)
    r1 = train_and_evaluate(TinyMLP, nn.CrossEntropyLoss(), train, val, epochs=3, seed=7)
    r2 = train_and_evaluate(TinyMLP, nn.CrossEntropyLoss(), train, val, epochs=3, seed=7)
    assert not r1.killed and r1.accuracy > 50
    assert r1.accuracy == r2.accuracy


def test_train_and_evaluate_smoke_kill_returns_zero():
    data = make_synthetic(400)
    train, val = train_val_split(data, 100)
    r = train_and_evaluate(TinyMLP, _loss("sqrt(0.64)"), train, val, epochs=2)
    assert r.killed and r.kill_reason == "zero_grad" and r.accuracy == 0.0


def test_early_kill_for_wrong_sign_loss():
    """log(p_target) maximizes the loss of the correct class -> below chance."""
    data = make_synthetic(1200, seed=2)
    train, val = train_val_split(data, 300)
    r = train_and_evaluate(TinyMLP, _loss("log(p_target)"), train, val, epochs=3)
    assert r.killed and r.kill_reason == "early_kill_epoch1"


def test_simple_cnn_shape():
    assert SimpleCNN()(torch.randn(3, 1, 28, 28)).shape == (3, 10)


def test_kill_log():
    k = KillLog()
    k.record(1, "zero_grad")
    k.record(1, "zero_grad")
    k.record(2, "nonfinite_loss")
    assert k.counts(1) == {"zero_grad": 2}
    assert k.total() == {"zero_grad": 2, "nonfinite_loss": 1}
