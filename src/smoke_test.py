"""Smoke-test / early-kill filter (Spec §15). Always on.

Check 1 (single batch, ~free) runs inside train_and_evaluate before any
optimizer step. Check 2 (one-epoch early kill) is applied there after epoch 1.
"""
from collections import Counter

import torch

KILL_REASONS = (
    "invalid_output",     # loss wrapper hit its NaN/exception fallback
    "nonfinite_loss",
    "nonfinite_grad",
    "zero_grad",          # degenerate tree ignoring its input
    "constant_loss",      # same loss for different logits
    "early_kill_epoch1",  # still near chance after one epoch
    "unstable_training",  # most steps produced non-finite gradients
)


def single_batch_check(model, loss_module, x, y):
    """Return a kill reason (str) or None if the loss looks trainable.

    Uses the real (freshly initialized) model's logits on one batch; no
    parameters are updated. The perturbation uses a private generator so the
    global torch RNG (and thus the training run) is unaffected.
    """
    before = getattr(loss_module, "fallback_count", 0)
    with torch.no_grad():
        base = model(x).float()
    logits = base.clone().requires_grad_(True)
    loss = loss_module(logits, y)
    if getattr(loss_module, "fallback_count", 0) > before:
        return "invalid_output"
    if not torch.isfinite(loss):
        return "nonfinite_loss"
    loss.backward()
    g = logits.grad
    if g is None or not torch.isfinite(g).all():
        return "nonfinite_grad"
    if g.abs().max().item() < 1e-12:
        return "zero_grad"
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(base.shape, generator=gen).to(base.device)
    with torch.no_grad():
        loss2 = loss_module(base + 2.0 * noise, y)
    if getattr(loss_module, "fallback_count", 0) > before:
        return "invalid_output"
    l1, l2 = loss.item(), loss2.item()
    if abs(l1 - l2) <= 1e-7 * (1.0 + abs(l1)):
        return "constant_loss"
    return None


class KillLog:
    """Per-generation counts of which check killed how many individuals."""

    def __init__(self):
        self.by_generation = {}

    def record(self, generation, reason):
        self.by_generation.setdefault(generation, Counter())[reason] += 1

    def counts(self, generation):
        return dict(self.by_generation.get(generation, {}))

    def total(self):
        c = Counter()
        for v in self.by_generation.values():
            c.update(v)
        return dict(c)
