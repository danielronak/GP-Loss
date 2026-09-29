# ACLE Workflow Strategy: Claude Code + Colab Pro

## The Core Principle

**Claude Code writes code. Colab burns GPU. GitHub connects them.**

```
┌──────────────┐      git push       ┌──────────┐      git pull       ┌──────────────┐
│  Local Machine│  ───────────────►  │  GitHub   │  ───────────────►  │  Colab Pro    │
│  (Claude Code) │                    │  (Repo)   │                    │  (GPU Runtime) │
│               │                    │           │  ◄───────────────  │               │
│  Write code   │      results       │           │   checkpoints +    │  Run evolution │
│  Debug logic  │  ◄─────────────── │           │   results via      │  Train models  │
│  Analyze data │   (download/Drive) │           │   Google Drive     │  Phase 0 gate  │
└──────────────┘                    └──────────┘                    └──────────────┘
```

---

## 0. Claude's review notes: suggested changes (2026-09-29)

These notes come from reading the spec, the kickoff brief and all four prior notebooks.

> **Decisions (2026-09-29):** Fashion-MNIST for Phase 0 and the zoo. A smaller Phase 0 is fine only if it can genuinely answer the question, so Phase 0 is now **5 cheap evolution runs per arm** (see §0.3). Phase 0 runs on the **free Colab tier**. Implemented in `src/phase0.py` + `notebooks/ACLE_Phase0.ipynb`.

### 0.1 How we actually work (differs from the diagram above)

- **Claude Code runs in a cloud container, not on your local machine.** The loop is still the same: Claude writes `src/` + tests, runs CPU unit tests in the container (PyTorch CPU and DEAP install there), then pushes. You pull in Colab.
- **Branch:** Claude pushes to `claude/stoic-knuth-jenuuo`, not `main`. Colab must clone or pull *that branch* (see the updated setup cell in §2). Merge to `main` whenever you like.
- **Private repo?** If the repo is private, Colab needs a GitHub token to clone. Store a fine-grained PAT in Colab's *Secrets* panel (e.g. `GH_TOKEN`) and clone with `https://$GH_TOKEN@github.com/...`. Never paste the token into a notebook cell.
- **Getting results back to Claude:** Claude cannot see your Drive. Each run will write a small `results/*.json|csv` summary and print a compact text summary at the end. Either (a) paste that printed summary into chat, or (b) commit `results/` from Colab (needs the token above). Checkpoints stay on Drive.

### 0.2 Headroom problem: MNIST is probably too easy **[DECIDE]**

Your Phase 1 notebook got CE = 98.56% and the evolved loss = 98.43% after 3 epochs, a **0.13%** gap. At 8 epochs A/B will all sit at ~99%. With a ceiling that tight:
- Phase 0 cannot show a "ceiling lift", because there is no gap left to close.
- The transfer matrix cannot show diagonal dominance, because every cell is ~99% ± noise.

**Suggestion:** use **Fashion-MNIST** (CE ≈ 90–92%, same 28×28×1 shape, same cost) for Phase 0 and the zoo. A reduced-train-set MNIST is an alternative. The spec already allows Fashion-MNIST.

### 0.3 Compute realism **[DECIDE]**

pop 20 × gen 20 is up to 400 evaluations × 8 epochs = ~3,200 epochs per evolution. Full 60k-image MNIST through a `DataLoader` on a T4 runs about 8–15 s/epoch, which is **~7–13 h per evolution**, not the ~1–2 h estimated. Phase 0 with 5 seeds × 2 operator sets would mean 10 evolutions. Planned levers (all cheap to build):
1. **Preload the dataset as GPU tensors** and batch by slicing (no DataLoader/PIL). Often 5–10× faster at this scale.
2. **Fitness on a fixed subset** (e.g. 10k train / 5k val). Final numbers still use the full held-out test set.
3. **Evaluation cache keyed by tree string**, so elites and unchanged crossover children don't retrain. The old loop retrained them.
4. Smoke-test filter (already planned). It kills the ~2%-accuracy "random" individuals that filled the old logs.
5. **First Colab action: a 2-minute timing cell** (one evaluation per architecture). We size pop/gen/epochs from real numbers.

~~3 evolution seeds per operator set~~. **Revised, as built:** 3 runs per arm cannot answer the question. Search randomness (which tree a run happens to find) is the dominant noise, so the run is the unit of replication. With 3 vs 3 runs the smallest attainable permutation p-value is 1/20 = 0.05, so a lift can never be called significant. With **5 vs 5 it is 1/252**. Levers 1–4 make each run cheap enough to afford that. Protocol (`src/phase0.py`, `configs/phase0_ceiling_test.yaml`):
- SimpleCNN, Fashion-MNIST. Fitness = accuracy on a fixed 5k val subset after 5 epochs on a fixed 10k train subset; pop 20, 15 generations.
- The arms differ only in operator set (`baseline` vs `tier1`). Runs with the same seed share model inits and batch order.
- Each run's val-best tree is retrained on the full 50k train split for 10 epochs and scored on the untouched test set over 5 shared seeds. CE and MSE are reference rows on the same seeds.
- **PASS** iff the one-sided exact permutation p < 0.05 over runs **and** the lift is ≥ max(0.2 pp, 25% of the CE-vs-baseline gap). The rule is fixed in code before any data is seen.
- Run order interleaves the arms, so a partial budget still gives a balanced (provisional) answer.

Time is estimated by the notebook's benchmark cell; paste its output to Claude before the long run.

### 0.4 A design subtlety in Phase 0 worth knowing up front

With `p_target` exposed, `neg(log(p_target))` **is exactly CrossEntropy** and is reachable at tree depth 2. So the expanded search space literally contains CE. In practice Phase 0 answers "does GP find CE (or something at least as good)?" That is still a useful gate, but frame it that way. Two implementation consequences:
- **Shapes (as built):** `error` is `(B, C)`; `p_target` is `(B, 1)`. The reduction is a sum over the class dimension, then a mean over the batch. A pure `neg(log(p_target))` tree is therefore **exactly** CE (unit-tested, gradients included). In mixed trees such as `add(square(error), neg(log(p_target)))`, the `p_target` term broadcasts across classes and is counted C times.
- **Trivial penalty:** do *not* put `neg(log(p_target))` in `TRIVIAL_SET`. Instead, log whenever evolution rediscovers CE, since that is itself a Phase 0 finding.

### 0.5 Bugs in the Phase 4b reference code (`Optimized_loss_GP.ipynb`): fix while porting

The kickoff says "port verbatim". These should still be fixed, and each fix will be noted in code comments:
1. **Resume re-runs the saved generation.** The checkpoint is written *after evaluation but before breeding*. On resume, the loop re-processes that same population: Gen 22 stats in the notebook are identical to Gen 21. It also appends the population to the novelty archive a second time, which skews novelty bonuses. Fix: checkpoint the bred next generation, or breed first on resume.
2. **`range(start_gen, start_gen + n_generations)`** runs *n more* generations on resume, not up to *n total*. That is why the "20-generation" run ended at gen 21/31. Fix: loop to `config.generations`.
3. **RNG state is not checkpointed**, so resume isn't reproducible. The plan in §4 already fixes this.
4. **Fitness was measured on the same `test_loader` later used for the final comparison.** The new code keeps train / val (fitness) / test (reporting only) strictly separate.
5. **`dill.load` fails in a fresh runtime unless DEAP `creator` classes exist first** (verified). Importing `src.gp_core` will always create them, so a post-restart resume just works.
6. **Elitism ranks by *shared* fitness,** so the raw-best tree can drop out of the population. The notebook had to dig it out of the archive by hand. Add a hall-of-fame tracking best-by-raw-accuracy.
7. The kickoff's early-kill threshold "<15% on 10-class" should be expressed relative to chance (`k / num_classes`). Chance was 2% on 50-way Omniglot.
8. *(found while building)* **Constant subtrees crashed.** `torch.maximum`/`minimum` and every unary torch op raise `TypeError` on a Python float. Any tree like `tanh(sqrt(0.43))`, `max(error, 1.2)` or `square(0.72)` threw, and the wrapper silently turned that into the zero-gradient penalty, so the tree scored chance. The Phase 4b logs are full of these "Raw=2.00%" individuals. Fixed with constant-safe primitives.
9. *(found while building)* **Max-depth limit bypassed.** The loop called `gp.cxOnePoint`/`gp.mutUniform` directly, not the `staticLimit`-decorated `toolbox.mate`/`toolbox.mutate`, so trees could grow past depth 5. Fixed.
10. *(found while building)* The penalty fallback `1000 * mean(probs)` is a constant (softmax rows sum to 1), so it has zero gradient. NaN trees therefore trained on nothing but were still scored. The smoke test now kills them up front.

All fixes are listed in the docstring of `src/evolve.py` / `src/gp_core.py`.

### 0.6 H2 metric note

`tree_distance` is a character-by-character `zip` of the two expression strings, not a true edit distance. Small prefix changes shift everything. Keep it as-is *inside* the evolution loop so the mechanisms match Phase 4b. For the **H2 structural-divergence analysis**, also report a proper tree edit distance (Zhang–Shasha, e.g. the `zss` package) so the paper's key Tier-1 number rests on a defensible metric.

---

## 1. Compute Budget Planning (100 Compute Units)

### What 100 units buys you

| GPU Type | Approx. Hours | Cost per Hour (CU) | Best For |
|----------|--------------|---------------------|----------|
| T4       | ~50 hrs      | ~2 CU/hr            | Evolution runs (many short trainings) |
| L4       | ~33 hrs      | ~3 CU/hr            | Faster evolution, transfer matrix |
| A100     | ~12 hrs      | ~8 CU/hr            | Only if you need speed on one big run |

### Recommended allocation

| Phase | Task | Est. GPU Hours | Recommended GPU | Est. CU |
|-------|------|---------------|-----------------|---------|
| **Phase 0** | Ceiling test (baseline vs expanded, ≥5 seeds) | 3–5 hrs | T4 | ~6–10 |
| **Phase A** | Sanity-check 3 architectures with CrossEntropy | 0.5 hrs | T4 | ~1 |
| **Phase B** | Test `evolve_loss_for_architecture` (2 gens) | 0.5 hrs | T4 | ~1 |
| **Phase C** | Full evolution: Arch-A, B, C (pop=20, gen=20) | 4–8 hrs | T4 or L4 | ~8–24 |
| **Phase D** | Structural divergence analysis | ~0 (CPU) | — | 0 |
| **Phase E** | Transfer matrix (9 cells + 3 CE × 5 seeds) | 2–3 hrs | T4 or L4 | ~4–9 |
| **Phase G** | Confirmatory re-runs (10–20 seeds, targeted) | 3–5 hrs | T4 or L4 | ~6–15 |
| **Buffer** | Debugging, re-runs, Colab disconnects | 3–5 hrs | T4 | ~6–10 |
| | | **Total** | | **~32–70 CU** |

> [!TIP]
> You have enough budget — even with generous buffer. Use **T4 by default** (cheapest per CU). Only upgrade to L4 if a single evolution run is taking too long and you want to compress wall-clock time.

> [!CAUTION]
> **Do NOT leave a Colab GPU runtime idle.** Colab Pro still consumes compute units while connected, even if no code is running. Disconnect the runtime the moment a phase finishes.

---

## 2. Code Architecture: Write Locally, Run Remotely

### What Claude Code builds (local, no GPU needed)

All Python source files go in `src/`. These are **pure library code** — no Colab-specific logic, no `drive.mount()`, no `!pip install`. Claude Code can write, lint, and unit-test everything locally:

```
GP-Loss/
├── src/
│   ├── __init__.py
│   ├── gp_core.py              # Operator set, EvolvedLossFunction, tree utils
│   ├── anti_convergence.py     # All 5 anti-convergence mechanisms
│   ├── smoke_test.py           # Single-batch validity + one-epoch early-kill
│   ├── architectures.py        # Arch-A/B/C + registry
│   ├── evolve.py               # evolve_loss_for_architecture() + checkpointing
│   ├── transfer_matrix.py      # Cross-architecture matrix + stats
│   ├── config.py               # EvolutionConfig dataclass
│   └── utils.py                # Seeding, label remap, train_and_evaluate
├── configs/
│   ├── phase0_ceiling_test.yaml
│   └── phase1_mnist_zoo.yaml
├── notebooks/                  # Thin Colab runners (see below)
│   ├── ACLE_Phase0.ipynb
│   ├── ACLE_Phase1_Zoo.ipynb
│   └── ACLE_TransferMatrix.ipynb
├── tests/                      # Local unit tests (no GPU needed)
│   ├── test_gp_core.py
│   ├── test_smoke_test.py
│   └── test_architectures.py
├── requirements.txt
└── README.md
```

### What Colab notebooks do (thin launchers only)

Each notebook is **minimal** — just setup + a function call. All logic lives in `src/`.

```python
# === Cell 1: Setup (same in every notebook) ===
from google.colab import drive
drive.mount('/content/drive')

!pip install -q deap dill pyyaml

# Clone or pull latest code (branch Claude pushes to; see §0.1)
import os
BRANCH = 'claude/stoic-knuth-jenuuo'
if not os.path.exists('/content/GP-Loss'):
    # private repo: from google.colab import userdata; token = userdata.get('GH_TOKEN')
    # and use f'https://{token}@github.com/danielronak/GP-Loss.git'
    !git clone -b {BRANCH} https://github.com/danielronak/GP-Loss.git /content/GP-Loss
else:
    !cd /content/GP-Loss && git fetch origin {BRANCH} && git checkout {BRANCH} && git pull origin {BRANCH}

import sys
sys.path.insert(0, '/content/GP-Loss')

# === Cell 2: Run the actual experiment ===
from src.evolve import evolve_loss_for_architecture
from src.architectures import ARCHITECTURE_ZOO
from src.config import EvolutionConfig
# ... call the function, save results to Drive
```

> [!IMPORTANT]
> **Notebooks should never contain significant logic.** If you find yourself writing more than ~10 lines of actual algorithm code in a notebook cell, stop — that code belongs in `src/` where Claude Code can properly maintain it.

---

## 3. The Git Workflow

### The cycle

```
1. Claude Code writes/edits src/ files locally
2. You review and commit:  git add . && git commit -m "..." && git push
3. In Colab: !cd /content/GP-Loss && git pull origin claude/stoic-knuth-jenuuo
4. Run the experiment
5. Results save to Google Drive (checkpoints/) or to results/ in the repo
6. Download results locally if needed for analysis
7. Back to step 1
```

### Branch strategy (keep it simple)

- Claude pushes to `claude/stoic-knuth-jenuuo`; Colab pulls that branch. Merge to `main` at milestones (e.g. after Phase 0) if you want a clean `main`.
- Commit often, with descriptive messages. You'll want the history when writing the paper.
- Tag meaningful milestones: `git tag phase0-pass` or `git tag phase0-fail`.

---

## 4. Checkpointing Strategy (Survival Plan for Colab Disconnects)

Colab **will** disconnect. Plan for it.

### Where checkpoints go

```python
# In every notebook's setup:
CHECKPOINT_BASE = '/content/drive/MyDrive/ACLE_Checkpoints'

# Each run gets a unique directory:
# /content/drive/MyDrive/ACLE_Checkpoints/phase0_baseline_seed42/
# /content/drive/MyDrive/ACLE_Checkpoints/phase0_expanded_seed42/
# /content/drive/MyDrive/ACLE_Checkpoints/evolution_A_seed42/
# etc.
```

### What gets checkpointed (every generation)

```python
checkpoint = {
    'generation': gen,
    'population': population,       # dill-serialized DEAP individuals
    'archive': archive,             # for novelty calculation
    'best_fitness': best_fitness,
    'diversity_trajectory': [...],
    'fitness_trajectory': [...],
    'config': config,               # full config for reproducibility
    'rng_state': {                  # for exact reproducibility
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
}
```

### The `run_manifest.json` (written at run start)

```json
{
    "run_id": "phase0_expanded_seed42",
    "checkpoint_dir": "/content/drive/MyDrive/ACLE_Checkpoints/phase0_expanded_seed42/",
    "started_at": "2026-09-26T01:00:00",
    "config": { "...": "..." },
    "phase": "phase0",
    "operator_set": "expanded_tier1"
}
```

### Recovery after disconnect

```python
# evolve.py handles this automatically:
result = evolve_loss_for_architecture(
    model_fn=...,
    train_loader=...,
    val_loader=...,
    config=config,
    checkpoint_dir=checkpoint_dir,
    device=device,
    resume=True,  # <-- finds latest checkpoint and continues
)
```

> [!WARNING]
> **Always verify the checkpoint path at run start.** The spec documents a prior incident where hours were lost because the save path and load path didn't match. The code prints the absolute path and writes `run_manifest.json` to prevent this.

---

## 5. Phase-by-Phase Execution Plan

### Phase 0 (the hard gate) — ~3–5 GPU hours

**Before touching Colab:**
1. Claude Code builds: `gp_core.py`, `anti_convergence.py`, `smoke_test.py`, `utils.py`, `config.py`
2. Claude Code extends `EvolvedLossFunction` for `p_target`/`logits` terminals
3. Claude Code writes local unit tests (CPU-only, mock data):
   - Test that a hand-crafted `-log(p_target)` tree computes CrossEntropy ← **critical**
   - Test smoke-test filter catches NaN/zero-grad/constant loss
   - Test checkpointing round-trip with dill
4. Claude Code builds `evolve.py` with resume support
5. Claude Code creates `notebooks/ACLE_Phase0.ipynb` (thin launcher)
6. `git push`

**On Colab (T4):**
1. Pull repo, install deps
2. Run baseline evolution (5 seeds) — checkpoint to Drive
3. Run expanded-operator evolution (5 seeds) — checkpoint to Drive
4. Compare best-achievable fitness across seeds
5. Record PASS/FAIL

**After Colab:**
1. Download results
2. Claude Code analyzes results locally, records outcome
3. Decision: expanded operators go forward, or baseline only

### Phase 1 (Zoo + Transfer Matrix) — ~7–14 GPU hours

**Before Colab:**
1. Claude Code builds `architectures.py` (A/B/C)
2. Claude Code creates `notebooks/ACLE_Phase1_Zoo.ipynb`
3. `git push`

**On Colab — Session 1 (sanity check + evolutions):**
1. Sanity-check each architecture with CrossEntropy
2. Run full evolution for Arch-A (20 gen), checkpoint every gen
3. If time remains: start Arch-B

**On Colab — Session 2 (continue evolutions):**
1. Resume Arch-B if needed
2. Run Arch-C
3. Save evolved loss trees to Drive

**On Colab — Session 3 (transfer matrix):**
1. Build transfer matrix (all losses × all architectures × 5 seeds)
2. Save raw CSV to Drive

**After Colab:**
1. Download matrix CSV
2. Claude Code runs statistical analysis locally (paired t-tests, specificity scores — **no GPU needed**)
3. If promising: schedule confirmatory re-runs (10–20 seeds) on Colab

---

## 6. Saving Compute Units: Key Tactics

| Tactic | Why |
|--------|-----|
| **Disconnect runtime when not running code** | Idle runtimes still consume CU |
| **Use T4 unless wall-clock time is critical** | T4 is ~2 CU/hr vs L4's ~3 CU/hr |
| **Run analysis/stats locally** | Paired t-tests, tree distances, plotting = CPU work |
| **Batch seeds in one session** | Avoid setup overhead across sessions |
| **Smoke-test filter kills doomed individuals early** | Reclaims 20–40% of wasted evaluation compute |
| **Don't debug on GPU** | Write and test code locally with Claude Code first |
| **Checkpoint every generation** | Never redo work after a disconnect |
| **Use `resume=True` always** | Seamless recovery |

---

## 7. Results Flow

```
Colab (GPU)                          Local (Claude Code)
─────────────                        ───────────────────
Checkpoints → Google Drive      ──►  Download for archival
Raw CSVs    → Google Drive      ──►  Download, analyze, plot
Evolved trees → Google Drive    ──►  Download, inspect structure
                                     Statistical analysis (no GPU)
                                     Paper writing assistance
                                     Visualization / plotting
```

### What to save to the repo's `results/` directory (committed to Git):
- `transfer_matrix.csv` (summary)
- `transfer_matrix_raw.csv` (per-seed)
- `evolution_trajectories.csv`
- `structural_divergence.json`
- `phase0_result.json` (PASS/FAIL + evidence)

### What stays on Drive only (too large for Git):
- Per-generation checkpoint files (dill-serialized populations)
- Full archive histories

---

## 8. Recommended Session Structure

Each Colab session should follow this pattern:

```python
# Cell 1: Mount + Setup (30 seconds)
drive.mount(...)
!pip install -q deap dill pyyaml
!cd /content/GP-Loss && git pull origin claude/stoic-knuth-jenuuo
sys.path.insert(0, '/content/GP-Loss')

# Cell 2: Configure (set seed, phase, architecture)
config = EvolutionConfig(seed=42, ...)
checkpoint_dir = f'{CHECKPOINT_BASE}/evolution_A_seed42/'

# Cell 3: Run (this is the one that takes hours)
result = evolve_loss_for_architecture(...)

# Cell 4: Save results (run IMMEDIATELY after Cell 3)
save_results(result, ...)

# Cell 5: Disconnect runtime (don't waste CU)
# Runtime → Disconnect and delete runtime
```

> [!TIP]
> **Run multiple seeds sequentially in one session** rather than one seed per session. The setup overhead (mount, install, pull, data download) is the same either way, and you avoid forgetting to disconnect.

---

## 9. Summary: Who Does What

| Task | Where | Tool |
|------|-------|------|
| Write Python source code | Local | Claude Code |
| Write/edit unit tests | Local | Claude Code |
| Run unit tests (CPU) | Local | Claude Code / terminal |
| Statistical analysis | Local | Claude Code |
| Plotting / visualization | Local | Claude Code |
| Write paper sections | Local | Claude Code |
| Run GP evolution | Colab | T4/L4 GPU |
| Train neural networks | Colab | T4/L4 GPU |
| Build transfer matrix | Colab | T4/L4 GPU |
| Store checkpoints | Google Drive | Automatic |
| Version control | GitHub | git push/pull |
