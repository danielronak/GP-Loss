"""GP primitives, operator sets and the differentiable loss wrapper.

Ported from the Phase 4b notebook (Optimized_loss_GP.ipynb). Deviations from
that code are deliberate bug fixes and are marked "FIX:" below.

Operator sets (Spec §6.2, §14.2):
  baseline : error-only terminal + the Phase 4 operator set
  tier1    : baseline + p_target terminal (softmax prob of the correct class)
  tier1_logits : tier1 + raw logits terminal
  tier2    : tier1 + pow / recip / log1p / expm1 (bounded)
"""
import operator
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from deap import base, creator, gp, tools

OPERATOR_SETS = ("baseline", "tier1", "tier1_logits", "tier2")

# ---------------------------------------------------------------------------
# DEAP creator classes. Created at import so that dill checkpoints can be
# loaded in a fresh runtime (dill.load needs creator.Individual to exist).
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# ---------------------------------------------------------------------------
# Constant handling.
# FIX: torch.maximum/minimum and every unary torch op raise TypeError on a
# plain Python float, so in Phase 4b any tree with a constant-only subtree
# (e.g. tanh(sqrt(0.43)), max(error, 1.2)) hit the exception fallback and
# scored chance accuracy. These helpers let constants flow through as tensors.
# ---------------------------------------------------------------------------
def _t(x):
    return x if torch.is_tensor(x) else torch.tensor(float(x))


def _pair(a, b):
    a, b = _t(a), _t(b)
    if a.device != b.device:
        if a.dim() == 0:
            a = a.to(b.device)
        else:
            b = b.to(a.device)
    return a, b


# --- binary (baseline) ---
def p_add(a, b):
    return torch.add(*_pair(a, b))


def p_mul(a, b):
    return torch.mul(*_pair(a, b))


def p_sub(a, b):
    return torch.sub(*_pair(a, b))


def p_max(a, b):
    return torch.maximum(*_pair(a, b))


def p_min(a, b):
    return torch.minimum(*_pair(a, b))


# --- unary (baseline) ---
def p_abs(x):
    return torch.abs(_t(x))


def p_square(x):
    return torch.square(_t(x))


def safe_sqrt(x):
    return torch.sqrt(torch.clamp(torch.abs(_t(x)), min=1e-7))


def safe_log(x):
    return torch.log(torch.clamp(torch.abs(_t(x)), min=1e-7))


def p_neg(x):
    return torch.neg(_t(x))


def p_sin(x):
    return torch.sin(_t(x))


def p_cos(x):
    return torch.cos(_t(x))


def p_tanh(x):
    return torch.tanh(_t(x))


def p_sigmoid(x):
    return torch.sigmoid(_t(x))


def p_softplus(x):
    return F.softplus(_t(x))


def p_elu(x):
    return F.elu(_t(x))


# --- Tier 2 (Spec §14.2): every op is domain-clamped (Spec §14.5) ---
def safe_pow(a, b):
    """|a|^b with base in [1e-4, 1e2] and exponent in [0, 3]."""
    a, b = _pair(a, b)
    return torch.pow(torch.clamp(torch.abs(a), 1e-4, 1e2), torch.clamp(b, 0.0, 3.0))


def safe_recip(x):
    """1/x with |x| clamped to >= 1e-3, sign preserved (0 treated as +)."""
    x = _t(x)
    return torch.where(x >= 0, 1.0 / torch.clamp(x, min=1e-3), 1.0 / torch.clamp(x, max=-1e-3))


def safe_log1p(x):
    return torch.log1p(torch.clamp(_t(x), min=-1.0 + 1e-6))


def safe_expm1(x):
    return torch.expm1(torch.clamp(_t(x), max=5.0))


def _rand_const():
    # Module-level function (not a lambda) so the ephemeral is picklable.
    return random.uniform(0, 2)


# (function, arity, name). Names match Phase 4b so tree strings and the
# trivial-tree list stay comparable with the prior project.
BASELINE_PRIMITIVES = [
    (p_add, 2, "add"),
    (p_mul, 2, "mul"),
    (p_sub, 2, "sub"),
    (p_max, 2, "max"),
    (p_min, 2, "min"),
    (p_abs, 1, "abs"),
    (p_square, 1, "square"),
    (safe_sqrt, 1, "sqrt"),
    (safe_log, 1, "log"),
    (p_neg, 1, "neg"),
    (p_sin, 1, "sin"),
    (p_cos, 1, "cos"),
    (p_tanh, 1, "tanh"),
    (p_sigmoid, 1, "sigmoid"),
    (p_softplus, 1, "softplus"),
    (p_elu, 1, "elu"),
]

TIER2_PRIMITIVES = [
    (safe_pow, 2, "pow"),
    (safe_recip, 1, "recip"),
    (safe_log1p, 1, "log1p"),
    (safe_expm1, 1, "expm1"),
]

_ARGUMENTS = {
    "baseline": ["error"],
    "tier1": ["error", "p_target"],
    "tier1_logits": ["error", "p_target", "logits"],
    "tier2": ["error", "p_target"],
}


def build_pset(operator_set="baseline"):
    """Build the DEAP primitive set for one of OPERATOR_SETS."""
    if operator_set not in OPERATOR_SETS:
        raise ValueError(f"unknown operator_set {operator_set!r}; choose from {OPERATOR_SETS}")
    args = _ARGUMENTS[operator_set]
    pset = gp.PrimitiveSet("MAIN", len(args))
    prims = BASELINE_PRIMITIVES + (TIER2_PRIMITIVES if operator_set == "tier2" else [])
    for fn, arity, name in prims:
        pset.addPrimitive(fn, arity, name=name)
    pset.addEphemeralConstant("const", _rand_const)
    pset.renameArguments(**{f"ARG{i}": a for i, a in enumerate(args)})
    return pset


# Register the ephemeral class in deap.gp at import (needed before dill.load).
build_pset("baseline")


def make_toolbox(pset, max_depth=5, init_min_depth=1, init_max_depth=3):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=init_min_depth, max_=init_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    return toolbox


def tree_from_string(expr, pset):
    """Parse a tree string (e.g. from a results CSV) back into an Individual."""
    return creator.Individual(gp.PrimitiveTree.from_string(expr, pset))


# ---------------------------------------------------------------------------
# Loss wrapper
# ---------------------------------------------------------------------------
class EvolvedLossFunction(nn.Module):
    """Compiles a GP tree into a differentiable classification loss.

    Terminals: error = one_hot - softmax (B, C); p_target = softmax prob of
    the correct class, shape (B, 1) so it broadcasts against error;
    logits = raw logits (B, C).

    Reduction (unchanged from Phase 4b): 2-D output -> sum over classes, mean
    over batch; 1-D -> mean. A pure p_target tree yields (B, 1), so
    neg(log(p_target)) is exactly CrossEntropy. In mixed trees any p_target
    term is broadcast across the C classes and so enters the sum C times.

    NaN/Inf or an exception falls back to 1000*mean(probs), a constant with
    zero gradient. `fallback_count` records how often that happened so the
    smoke test can kill such individuals.
    """

    PENALTY = 1000.0

    def __init__(self, tree, pset):
        super().__init__()
        self.tree = tree
        self.arg_names = list(pset.arguments)
        self.func = gp.compile(expr=tree, pset=pset)
        self.fallback_count = 0

    def forward(self, logits, targets):
        logits = logits.float()
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, probs.size(1)).to(probs.dtype)
        inputs = {"error": one_hot - probs}
        if "p_target" in self.arg_names:
            inputs["p_target"] = probs.gather(1, targets.view(-1, 1))
        if "logits" in self.arg_names:
            inputs["logits"] = logits
        try:
            loss_val = self.func(*[inputs[a] for a in self.arg_names])
            if not torch.is_tensor(loss_val):
                loss_val = torch.tensor(loss_val, device=logits.device, dtype=torch.float32)
            loss_val = loss_val.to(logits.device)
            if not loss_val.requires_grad:
                loss_val = loss_val + 0.0 * probs.sum()  # gradient-connectivity guard
            if loss_val.dim() > 1:
                loss_val = torch.mean(torch.sum(loss_val, dim=1))
            elif loss_val.dim() == 1:
                loss_val = torch.mean(loss_val)
            if not torch.isfinite(loss_val):
                self.fallback_count += 1
                return self.PENALTY * torch.mean(probs)
            return loss_val
        except Exception:
            self.fallback_count += 1
            return self.PENALTY * torch.mean(probs)


def uses_terminal(tree, name):
    # renameArguments updates .value (the name used in the tree string); .name stays "ARGn"
    return any(isinstance(node, gp.Terminal) and node.value == name for node in tree)


def is_crossentropy_like(tree):
    """True if the tree contains log(p_target), the CrossEntropy ingredient."""
    return "log(p_target)" in str(tree)
