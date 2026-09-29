"""EvolutionConfig (Spec §6.9) plus YAML loading."""
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import yaml


@dataclass
class EvolutionConfig:
    seed: int = 42                      # GP search seed (python `random`)
    operator_set: str = "baseline"      # see gp_core.OPERATOR_SETS

    # GP (Spec §6.9 defaults)
    population_size: int = 20
    generations: int = 20
    tree_max_depth: int = 5
    init_min_depth: int = 1
    init_max_depth: int = 3
    tournament_size: int = 3
    elitism: int = 3
    crossover_rate: float = 0.5

    # Anti-convergence (Phase 4b constants)
    mutation_rate_base: float = 0.3
    mutation_rate_mid: float = 0.4
    mutation_rate_high: float = 0.5
    diversity_mid_threshold: float = 0.4
    diversity_low_threshold: float = 0.2
    fitness_sharing_sigma: float = 0.3
    novelty_bonus_schedule: Tuple[float, ...] = (2.0, 1.0, 0.5, 0.0)
    trivial_tree_penalty: float = 0.95

    # Fitness evaluation
    epochs_per_individual: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    # Seed for model init + batch order. Identical for every individual in a
    # run (common random numbers), so fitness differences come from the loss.
    eval_seed: Optional[int] = None

    # Smoke test / early kill (Spec §15). The single-batch check is always on.
    early_kill: bool = True
    early_kill_chance_multiple: float = 1.5   # kill if epoch-1 acc < 1.5 x chance
    early_kill_val_size: int = 2000

    def __post_init__(self):
        self.novelty_bonus_schedule = tuple(self.novelty_bonus_schedule)
        if self.eval_seed is None:
            self.eval_seed = self.seed

    def to_dict(self):
        d = asdict(self)
        d["novelty_bonus_schedule"] = list(self.novelty_bonus_schedule)
        return d

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown EvolutionConfig keys: {sorted(unknown)}")
        return cls(**d)

    def replace(self, **kw):
        d = self.to_dict()
        d.update(kw)
        if "seed" in kw and "eval_seed" not in kw:
            d["eval_seed"] = None
        return EvolutionConfig.from_dict(d)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)
