"""The five Phase 4b anti-convergence mechanisms (Spec §6.4). Always on.

Ported unchanged from Optimized_loss_GP.ipynb, except that thresholds and
schedules are parameters (defaults equal the Phase 4b constants).

  1. fitness sharing           apply_fitness_sharing
  2. novelty bonus             calculate_novelty_bonus
  3. trivial-tree penalty      apply_trivial_penalty
  4. adaptive mutation rate    adaptive_mutation_rate
  5. diversity-aware selection tournaments run on shared fitness (evolve.py),
                               driven by calculate_population_diversity
"""
import re

import numpy as np

# Single-op trees penalized in Phase 4b (the op list is kept verbatim).
LEGACY_TRIVIAL_OPS = ["square", "abs", "neg", "sqrt", "log", "sin", "cos", "tanh", "sigmoid", "exp"]

_CONST_RE = re.compile(r"\d+\.\d+")


def trivial_set(pset):
    """Single-op trees on each input terminal, e.g. square(error), square(p_target).

    For the baseline set this is exactly the Phase 4b list.
    """
    return {f"{op}({arg})" for op in LEGACY_TRIVIAL_OPS for arg in pset.arguments}


def tree_distance(tree1, tree2):
    """Phase 4b structural distance: normalized character-level comparison.

    NOTE: this is not a true edit distance (a one-char prefix shift changes
    every aligned position). It is kept as-is for the in-loop mechanisms so
    they match Phase 4b; H2 analysis should also use a real tree edit distance.
    """
    s1, s2 = str(tree1), str(tree2)
    if s1 == s2:
        return 0.0
    len_diff = abs(len(s1) - len(s2))
    char_diff = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return (len_diff + char_diff) / max(len(s1), len(s2))


def get_tree_structure_hash(tree):
    """Structural signature ignoring constant values."""
    return _CONST_RE.sub("C", str(tree))


def apply_fitness_sharing(population, sigma=0.3):
    for ind in population:
        niche_count = sum(1 for other in population if tree_distance(ind, other) < sigma)
        ind.shared_fitness = ind.fitness.values[0] / max(niche_count, 1)


def calculate_novelty_bonus(tree, archive, schedule=(2.0, 1.0, 0.5, 0.0)):
    """Bonus by how often this structure appeared before: schedule[min(count, 3)]."""
    tree_hash = get_tree_structure_hash(tree)
    count = sum(1 for t in archive if get_tree_structure_hash(t) == tree_hash)
    return schedule[min(count, len(schedule) - 1)]


def apply_trivial_penalty(tree, trivial, penalty=0.95):
    return penalty if str(tree) in trivial else 1.0


def adaptive_mutation_rate(diversity, low_threshold=0.2, mid_threshold=0.4,
                           high_rate=0.5, mid_rate=0.4, base_rate=0.3):
    if diversity < low_threshold:
        return high_rate
    if diversity < mid_threshold:
        return mid_rate
    return base_rate


def calculate_population_diversity(population):
    if len(population) < 2:
        return 1.0
    pairs = [tree_distance(a, b) for i, a in enumerate(population) for b in population[i + 1:]]
    return float(np.mean(pairs)) if pairs else 0.0
