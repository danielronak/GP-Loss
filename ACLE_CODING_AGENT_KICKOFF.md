# ACLE — Coding Agent Kickoff

**What this is:** A self-contained build brief for a coding agent starting implementation of the Architecture-Conditioned Loss Evolution (ACLE) project. This is the *action* document. The companion file `ARCHITECTURE_CONDITIONED_LOSS_EVOLUTION_SPEC.md` is the *reasoning* document — read it when you need the "why" behind any instruction here; every section reference below (e.g. "Spec §14") points into it.

**Your environment:** Python 3.10+, PyTorch 2.0+, DEAP 1.4+, running on a single GPU (Google Colab T4 assumed). Compute is scarce — the entire project budget is ~9–17 GPU-hours. Checkpoint aggressively; assume the runtime can disconnect at any time.

---

## 0. The one rule that overrides everything

**Build and run Phase 0 (Section 4 below) before building anything else in the research pipeline.** Phase 0 is a hard gate. Its result decides whether half the later work happens at all. Do not build the architecture zoo or the transfer matrix until Phase 0 has run and its pass/fail result is recorded. If you find yourself writing zoo/matrix code before Phase 0 has produced a number, stop.

---

## 1. Build order (checklist)

Work top to bottom. Do not skip ahead.

- [ ] **1.1** Set up repo structure (Section 2 below) and `requirements.txt`.
- [ ] **1.2** Port validated infrastructure from prior project (Section 3): operator set, `EvolvedLossFunction`, anti-convergence mechanisms, dill checkpointing, seeding/eval utilities. This is *reuse*, not new design — do not redesign it.
- [ ] **1.3** Implement the **smoke-test / early-kill filter** (Section 5) and fold it into the evaluation entry point so no evaluation can run without it. Always on from here forward.
- [ ] **1.4** Extend `EvolvedLossFunction` to accept the new terminals needed for Phase 0 (`p_target`, and optionally `logits`) — Section 4.2. Unit-test this before running anything.
- [ ] **1.5** **Run Phase 0 — the operator-expansion ceiling test (Section 4). HARD GATE.** Record PASS/FAIL across ≥5 seeds.
- [ ] **1.6** Implement `architectures.py` — Arch-A/B/C (Section 6) — and sanity-check each trains with plain CrossEntropy.
- [ ] **1.7** Implement `evolve_loss_for_architecture(...)` (Section 7), the core reusable API. Test end-to-end on one architecture for 2 generations.
- [ ] **1.8** Run full per-architecture evolutions (A, B, C) → produce `loss_A`, `loss_B`, `loss_C`.
- [ ] **1.9** Structural-divergence analysis (H2): pairwise tree distances across architectures vs run-to-run noise.
- [ ] **1.10** Build the transfer matrix (Section 8): every loss × every architecture × N seeds + CrossEntropy baseline row.
- [ ] **1.11** Statistics: paired t-tests for diagonal dominance (H1), specificity scores.
- [ ] **1.12** If a Tier-1/Tier-2 result appears (Spec §4, §9), re-run relevant cells at higher seed count (Spec §10) before claiming significance anywhere.

Operator set used from step 1.6 onward depends on the Phase 0 result: **expanded set if Phase 0 PASSED, baseline set if it FAILED.**

---

## 2. Repository structure

```
architecture-conditioned-loss-evolution/
├── src/
│   ├── gp_core.py              # operator set, EvolvedLossFunction, tree utils   [REUSE + Phase-0 extension]
│   ├── anti_convergence.py     # fitness sharing, novelty, trivial penalty, adaptive mutation, diversity  [REUSE]
│   ├── smoke_test.py           # single-batch validity + one-epoch early-kill  [NEW]
│   ├── architectures.py        # Arch-A/B/C + zoo registry  [NEW]
│   ├── evolve.py               # evolve_loss_for_architecture(), checkpointing  [NEW, from prior run_anti_convergence_evolution]
│   ├── transfer_matrix.py      # cross-architecture matrix + stats  [NEW]
│   └── utils.py                # seeding, label remap, generic train/eval loop  [REUSE]
├── configs/
│   ├── phase0_ceiling_test.yaml
│   └── phase1_mnist_zoo.yaml
├── notebooks/
│   ├── ACLE_Phase0_CeilingTest.ipynb
│   └── ACLE_Phase1_Zoo.ipynb
├── checkpoints/                # dill checkpoints, one subfolder per run; write run_manifest.json per run
├── results/                    # transfer_matrix.csv, evolution_trajectories.csv, structural_divergence.json
└── requirements.txt
```

---

## 3. Reused infrastructure (port verbatim, do not redesign)

All of the following was built and validated in the prior project. Reproduce it faithfully. Full code skeletons are in Spec §13; operator/loss/anti-convergence details in Spec §6.2–6.4.

**Operator set (baseline).** `add, mul, sub, max, min` (binary); `abs, square, sqrt(safe), log(safe), neg, sin, cos, tanh, sigmoid, softplus, elu` (unary); ephemeral constant `uniform(0,2)`. Input terminal renamed to `error`. **Never add `exp` or `div`** — confirmed NaN/Inf sources. Max tree depth 5.

**`EvolvedLossFunction(nn.Module).** Compiles a DEAP tree to a differentiable loss. Computes `error = one_hot(target) - softmax(logits)`, applies the tree, includes: NaN/Inf → penalty fallback (`1000.0 * probs.mean()`); gradient-connectivity guard (`loss + 0.0*probs.sum()`) for degenerate trees; reduction to scalar. (Section 4.2 extends its input signature for Phase 0.)

**Anti-convergence (always on, mandatory).** `tree_distance` (normalized string-edit), `get_tree_structure_hash` (regex-strip constants), `apply_fitness_sharing(sigma=0.3)`, `calculate_novelty_bonus` (schedule `[2.0,1.0,0.5,0.0]` for 0/1/2/3+ prior occurrences), `apply_trivial_penalty(0.95)` over a `TRIVIAL_SET` of single-op-on-error trees, `adaptive_mutation_rate` (0.5 if diversity<0.2, 0.4 if <0.4, else 0.3), `calculate_population_diversity` (mean pairwise distance).

**Checkpointing.** `dill` (NOT `pickle` — DEAP's ephemeral-constant lambdas break pickle). Checkpoint after **every** generation. Write a `run_manifest.json` at run start (run id → absolute checkpoint dir → config) and print the absolute checkpoint path at run start — a prior session lost hours to a wrong-path mismatch (Spec §7 pitfall 4).

**Seeding / eval utils.** `set_all_seeds(seed)` covering `random`, `numpy`, `torch`, `torch.cuda`. Generic `train_and_evaluate(model_fn, loss_module, train_loader, test_loader, epochs, device)` returning test accuracy — this is where the smoke-test filter is folded in (Section 5).

---

## 4. Phase 0 — Operator Expansion Ceiling Test (HARD GATE) — Spec §14

### 4.1 The question Phase 0 answers

Was the prior project's failure to beat CrossEntropy a *search-space expressiveness* problem (the operators literally cannot express a competitive loss) or a *search-quality* problem? RL and richer search can only fix the latter and only matter if the ceiling is movable. Phase 0 finds out whether it's movable, cheaply, before any further investment. If the ceiling doesn't move, the operator-expansion + RL direction is dead and you proceed with the baseline set.

### 4.2 Implementation: extend the loss wrapper's inputs first

The baseline wrapper only exposed `error`. Phase 0 needs the tree to access target-class probability directly (because `-log(p_target)` *is* CrossEntropy, the most likely missing ingredient). Extend `EvolvedLossFunction` and the primitive set to expose new terminals:

- `p_target` — softmax probability of the correct class, shape `(batch,)` or broadcast-compatible.
- `safe_log(p_target)` must be reachable (log already exists as an operator; the point is that its argument can now be the target probability, not only `error`).
- Optionally `logits` — raw pre-softmax scores, for margin-style losses.

This changes the compiled tree's input arity. Update `gp.PrimitiveSet("MAIN", n_inputs)` accordingly and `renameArguments` for each. **Unit-test that a hand-written tree reproducing `-log(p_target)` actually computes CrossEntropy** before running the search — this is the single most important correctness check in Phase 0.

### 4.3 Operators to add, in strict order (do NOT skip to Tier 3)

- **Tier 1 (add first, cheap, most likely to matter):** `p_target` access + `log(p_target)`; optionally `logits` terminal.
- **Tier 2 (add second, still cheap):** `pow` (small evolved/int exponent), reciprocal-with-clamp, `log1p`, bounded `expm1`.
- **Tier 3 (only if Tier 1+2 raised the ceiling AND you accept the cost):** gradient-norm / Jacobian terms, prediction entropy, batch-level divergence terms. **COST WARNING:** these are non-pointwise and can cut evaluations 3–5×, worsening fitness noise. Measure per-eval cost before a full run. Most likely you never need Tier 3.

Every new operator ships with NaN/Inf-safe domain clamping (Spec §14.5). The smoke-test filter (Section 5) is the backstop but operators must still be individually safe.

### 4.4 Protocol

1. One architecture (Arch-A/ShallowWide or the user's SimpleCNN), MNIST or Fashion-MNIST.
2. Two evolutions, identical in every way (pop, generations, seed, epochs/individual, fixed data split) except operator set: **baseline** vs **baseline + Tier 1**. Anti-convergence + smoke-test active in both.
3. Compare **best-achievable fitness (the ceiling)**, not the average — this is a top-of-distribution question.
4. If Tier 1 lifts the ceiling, optionally add Tier 2, then (cautiously) Tier 3.
5. **Validate any apparent lift across ≥5 seeds** before calling PASS (no single-seed claims — Spec §7 pitfall 5).

### 4.5 Pass / fail

- **PASS:** expanded-operator best accuracy is meaningfully above baseline best accuracy beyond seed noise (ideally closes a visible fraction of the prior 1.5–4% gap to CrossEntropy; shrinking-not-closing still passes). → Carry expanded operators into all later phases. Learned Method 4 becomes *potentially* justified (Section 5.4 / Spec §16.3).
- **FAIL:** no ceiling lift beyond noise. → Limiter is not expressiveness; RL cannot help. Proceed to the zoo study with the **baseline** operator set; frame the paper on architecture-specialization only (Spec §4 Tiers 1/2), not on beating CrossEntropy. Record this — it's a clean, citable justification for the scoping and for excluding RL.

Both outcomes are useful. Record the result in `results/` and in the run manifest.

---

## 5. Smoke-Test / Early-Kill Filter (always on) — Spec §15

The cheap, deterministic, non-RL core of "predictive operator pruning." Reclaims compute wasted on doomed individuals. On from Phase 0 onward, in every phase.

### 5.1 Checks (cheapest first), applied before an individual gets its full training budget

1. **Single-batch validity (near-zero cost):** run the loss on one batch; kill (fitness 0 / mark invalid) if loss is NaN/Inf, non-finite after `.backward()`, gradient w.r.t. logits is all-zero (degenerate tree ignoring input), or loss is constant across differing inputs.
2. **One-epoch early-kill (cheap, optional):** train 1 epoch; kill anything still ≤ near-random accuracy (e.g. <15% on 10-class MNIST). Won't recover; reallocate the budget.

### 5.2 Implementation

Fold into `train_and_evaluate` so it is impossible to evaluate without it. **Log kills per generation and which check caught them** — a kill-rate spike after adding an operator tier flags that operator as unstable, and the aggregate kill-rate is a reportable efficiency metric ("pre-filter eliminated X% of evaluations at Y% of full cost").

### 5.3 Do NOT build the learned RL version yet

Keep it heuristic. Upgrade to a learned viability predictor only if all three hold (Spec §16.3): Phase 0 passed; exotic operators produce frequent lethal-but-valid offspring the heuristic misses (per the kill logs); and you've reclaimed enough compute to afford training the predictor. Absent all three, leave it heuristic.

---

## 6. Architecture zoo — Spec §5.1, §6.6

Three architectures spanning clean, interpretable axes; all cheap at MNIST scale. **Contract each must satisfy (also the contract for any user-supplied model):** a zero-argument factory returning a freshly-initialized `nn.Module` that takes the project's fixed input shape and returns raw logits `(batch, num_classes)`. Nothing else constrained.

- **Arch-A `ShallowWideCNN`** — 2 conv layers, wide channels (64→128), global avg pool, 1 FC. (Short gradient path, high per-layer capacity.)
- **Arch-B `DeepNarrowCNN`** — 6 conv layers, narrow channels (16→32→64), small kernels, **no normalization**. (Long gradient path, vanishing-gradient-prone, low per-layer capacity.)
- **Arch-C `PureMLP`** — flatten, 3 FC hidden layers, no convolution. (No spatial inductive bias — maximally different "nature.")

Register in `ARCHITECTURE_ZOO = {name: factory}`. Sanity-check each reaches sane CrossEntropy baseline accuracy (~97–99% A/B, lower for C) before proceeding. Optional Phase-2 additions (Arch-D = B+BatchNorm to isolate normalization; CIFAR-scale variants) are future work.

---

## 7. Core API — `evolve_loss_for_architecture(...)` — Spec §6.7, §13

The main reusable deliverable. Everything else calls this. It is a generalization of the prior project's validated `run_anti_convergence_evolution()` — the only substantive change is that the fitness-evaluation model is a parameter (`model_fn`) instead of hardcoded.

```python
def evolve_loss_for_architecture(
    model_fn,          # zero-arg factory -> fresh nn.Module (the fitness architecture)
    train_loader,
    val_loader,        # used for fitness; keep a SEPARATE held-out test set for final numbers
    config,            # EvolutionConfig dataclass (Spec §6.9)
    checkpoint_dir,
    device,
    resume=True,
) -> EvolvedLossResult:  # best_tree, best_fitness, final_population, trajectory
    ...
```

Requirements: anti-convergence mechanisms active; smoke-test filter active; checkpoint every generation with `dill`; resume from latest checkpoint if `resume`; write `run_manifest.json` and print absolute checkpoint path at start; collect per-generation fitness + diversity trajectory for later analysis. Defaults for `EvolutionConfig` in Spec §6.9 (pop 20, gen 20, 8 epochs/individual, tournament 3, elitism 3, crossover 0.5, adaptive mutation, sharing sigma 0.3).

**Standalone-usage requirement:** this function must work on its own — a user supplies their own `model_fn` and gets an evolved loss back — independent of the zoo/matrix study. That single-architecture mode is the MVP; the research study is just calling it three times and comparing.

---

## 8. Transfer matrix — Spec §5.3, §6.8

Tests H1 (does each architecture do best with its *own* evolved loss). For every (loss origin *i*, training architecture *j*) pair, train architecture *j* from scratch with `loss_i`, across N seeds, record mean±std test accuracy. Add a CrossEntropy baseline row (and optionally a "universal loss" control — one loss evolved against the *average* accuracy across all three architectures).

```
                Trained-on A   Trained-on B   Trained-on C
loss_A            M[A][A]        M[A][B]        M[A][C]
loss_B            M[B][A]        M[B][B]        M[B][C]
loss_C            M[C][A]        M[C][B]        M[C][C]
CrossEntropy      CE[A]          CE[B]          CE[C]
```

**Use a paired design:** same seeds / same data splits / same inits across all losses compared within a cell, so differences reflect the loss, not split luck (this directly fixes the prior variance problem). Save raw per-(loss,arch,seed) rows AND a summary matrix to `results/`. Compute per-architecture `specificity_score(i) = M[i][i] - mean(M[i][j], j≠i)`; test diagonal-vs-offdiagonal with **paired** t-tests across seeds.

Exploratory pass: N=5 seeds. Confirmatory pass (only for cells that look promising): N=10–20 (Spec §10 — reliably detecting a 1% effect at observed variance would formally need ~50–70 seeds; be honest, report confidence intervals, don't overclaim precision, scale seeds to the strength of the claim).

---

## 9. Non-negotiable rules (from prior-project scar tissue) — Spec §7

1. **Remap dataset labels to contiguous `0..num_classes-1`** whenever subsetting/few-shot, and assert `max(label) < num_classes` before training. (A label/class mismatch triggers an opaque CUDA `device-side assert`.)
2. **After any CUDA assert, restart the runtime.** The context is corrupted; debugging in-place wastes time.
3. **`dill`, never `pickle`,** for checkpoints.
4. **Print absolute checkpoint path + write `run_manifest.json` at every run start.** No guessing paths on recovery.
5. **No result is real at one seed.** Minimum 5 seeds before any claim, even informal. The prior project's single strongest "win" (+2.49%) reversed to −2.97% under 5-seed validation.
6. **Anti-convergence mechanisms are always on.** There is no baseline run without them; we already know that converges to trivial MSE in 1–2 generations.
7. **`exp` and `div` stay excluded** from the operator set (behind extra clamping only if a future extension insists, tested in isolation first).
8. **Fitness evaluation uses a fixed data split within a run; vary seeds only between runs.** Keep a separate held-out test set touched only for final reported numbers, never for evolution.

---

## 10. Definition of done (Phase 1)

- Phase 0 run, PASS/FAIL recorded across ≥5 seeds, operator set for later phases decided accordingly.
- `evolve_loss_for_architecture` works standalone on an arbitrary user `model_fn`.
- `loss_A`, `loss_B`, `loss_C` evolved and saved with trajectories.
- Structural-divergence numbers computed (H2).
- Full transfer matrix + CrossEntropy baseline saved to `results/`, exploratory (N=5) pass done.
- Paired-t-test diagonal-dominance analysis + specificity scores computed (H1).
- Any promising comparison re-run at higher seed count before being called significant.
- All outcomes are reportable (Spec §9 decision tree) — there is no "failed project" branch, only different findings.

---

**Start at Section 4 (Phase 0). Do not build the zoo or the matrix until Phase 0 has produced a recorded PASS/FAIL.**
