# ACLE — Architecture-Conditioned Loss Evolution

Tree-based GP (DEAP) that evolves classification losses, with the question
"does the best loss depend on the architecture?". Background and reasoning:
`ARCHITECTURE_CONDITIONED_LOSS_EVOLUTION_SPEC (1).md`; build brief:
`ACLE_CODING_AGENT_KICKOFF.md`; how the work is split between Claude Code
and Colab: `workflow_strategy.md`; the prior project: `evolutionary_loss_research_paper.md`
(use the notebooks for exact numbers).

## Status

Phase 0 (operator-expansion ceiling test, the hard gate) is built and ready to run:
`notebooks/ACLE_Phase0.ipynb`. The architecture zoo and transfer matrix are
not built yet, on purpose (they wait for the Phase 0 result).

## Layout

```
src/
  gp_core.py          operator sets (baseline / tier1 / tier2), EvolvedLossFunction
  anti_convergence.py the five Phase 4b mechanisms (ported)
  smoke_test.py       single-batch validity check + kill log (always on)
  utils.py            seeding, train_and_evaluate (smoke test folded in)
  data.py             GPU-resident tensor datasets (Fashion-MNIST / MNIST)
  evolve.py           evolve_loss_for_architecture(): checkpoint/resume every generation
  phase0.py           Phase 0 protocol: runs, final eval, pre-registered PASS/FAIL
  stats.py            permutation test, hierarchical bootstrap, paired t-test
  architectures.py    SimpleCNN (Phase 0 model)
configs/phase0_ceiling_test.yaml
notebooks/ACLE_Phase0.ipynb   thin Colab launcher
tests/                        CPU unit tests
```

## Standalone use (any model)

```python
from src.config import EvolutionConfig
from src.data import tensors_from_loader
from src.evolve import evolve_loss_for_architecture

train = tensors_from_loader(my_train_loader, num_classes=10, device="cuda")
val = tensors_from_loader(my_val_loader, num_classes=10, device="cuda")
result = evolve_loss_for_architecture(
    model_fn=lambda: MyNet(num_classes=10), train_data=train, val_data=val,
    config=EvolutionConfig(operator_set="tier1", population_size=20, generations=20),
    checkpoint_dir="/content/drive/MyDrive/ACLE/my_run")
print(result.best_tree, result.best_fitness)
```

Calling it again with the same `checkpoint_dir` resumes.

## Tests

```
pip install -r requirements.txt
python -m pytest -q
```
