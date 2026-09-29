"""Seeding and the generic train/eval loop (smoke test folded in)."""
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from .data import iterate_minibatches
from .smoke_test import single_batch_check


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EvalResult:
    accuracy: float                  # % on the evaluation set (0.0 if killed)
    killed: bool = False
    kill_reason: Optional[str] = None
    epoch1_accuracy: Optional[float] = None
    epochs_run: int = 0
    fallback_batches: int = 0
    skipped_steps: int = 0
    seconds: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@torch.no_grad()
def evaluate_accuracy(model, data, batch_size=1000):
    model.eval()
    correct = 0
    for x, y in iterate_minibatches(data, batch_size, shuffle=False):
        correct += (model(x).argmax(1) == y).sum().item()
    return 100.0 * correct / max(len(data), 1)


def train_and_evaluate(model_fn, loss_module, train, eval_data, epochs, device=None,
                       lr=1e-3, batch_size=128, seed=0, early_kill=True,
                       early_kill_data=None, early_kill_chance_multiple=1.5,
                       max_skip_fraction=0.5):
    """Train model_fn() with loss_module and return an EvalResult.

    The single-batch smoke test always runs before the first optimizer step.
    `seed` fixes model init and batch order; it only touches the torch RNG,
    never python `random` (which drives the GP search).
    """
    t0 = time.time()
    device = device or train.x.device
    torch.manual_seed(seed)
    model = model_fn().to(device)
    gen = torch.Generator().manual_seed(seed)
    fb0 = getattr(loss_module, "fallback_count", 0)

    def result(acc, reason=None, **kw):
        return EvalResult(
            accuracy=acc, killed=reason is not None, kill_reason=reason,
            fallback_batches=getattr(loss_module, "fallback_count", 0) - fb0,
            seconds=time.time() - t0, **kw)

    model.train()
    xb, yb = train.x[:batch_size], train.y[:batch_size]
    reason = single_batch_check(model, loss_module, xb, yb)
    if reason:
        return result(0.0, reason)
    fb0 = getattr(loss_module, "fallback_count", 0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(len(train) / batch_size)
    skipped, epoch1_acc = 0, None
    for epoch in range(epochs):
        model.train()
        skipped_epoch = 0
        for x, y in iterate_minibatches(train, batch_size, generator=gen):
            opt.zero_grad(set_to_none=True)
            loss = loss_module(model(x), y)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            if not torch.isfinite(gnorm):
                skipped_epoch += 1
                continue
            opt.step()
        skipped += skipped_epoch
        if skipped_epoch > max_skip_fraction * steps_per_epoch:
            return result(0.0, "unstable_training", epochs_run=epoch + 1, skipped_steps=skipped)
        if epoch == 0 and early_kill:
            ek = early_kill_data if early_kill_data is not None else eval_data
            epoch1_acc = evaluate_accuracy(model, ek)
            chance = 100.0 / train.num_classes
            if epoch1_acc < early_kill_chance_multiple * chance:
                return result(0.0, "early_kill_epoch1", epoch1_accuracy=epoch1_acc,
                              epochs_run=1, skipped_steps=skipped)

    acc = evaluate_accuracy(model, eval_data)
    return result(acc, None, epoch1_accuracy=epoch1_acc, epochs_run=epochs, skipped_steps=skipped)
