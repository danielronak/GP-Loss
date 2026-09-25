# Architecture-Conditioned Evolutionary Loss Function Discovery
## Project Specification (Working Name: ACLE — Architecture-Conditioned Loss Evolution)

**Status:** Not yet started. This document is the complete handoff spec.
**Audience:** This document is written to be usable by three different readers at once: (1) the researcher (you), (2) an academic reviewer/professor with no prior context, and (3) a coding agent that will implement it. Every section states not just *what* to do but *why*, so all three can verify the reasoning.

---

## Table of Contents

1. [TL;DR](#1-tldr)
2. [Background: What We Already Know](#2-background-what-we-already-know)
3. [The New Idea](#3-the-new-idea-architecture-conditioned-loss-evolution)
4. [Research Questions and Hypotheses](#4-research-questions-and-hypotheses)
5. [Experimental Design](#5-experimental-design)
6. [Technical Specification](#6-technical-specification-for-implementation)
7. [Known Pitfalls — Do Not Repeat These](#7-known-pitfalls--do-not-repeat-these)
8. [Step-by-Step Execution Plan](#8-step-by-step-execution-plan)
9. [How to Interpret Results](#9-how-to-interpret-results--decision-tree)
10. [Statistical Power: How Many Seeds Do We Actually Need?](#10-statistical-power-how-many-seeds-do-we-actually-need)
11. [Paper Framing](#11-paper-framing)
12. [Glossary](#12-glossary)
13. [Appendix: Reusable Code Skeletons](#13-appendix-reusable-code-skeletons)
14. [Phase 0: Operator Expansion — The Ceiling Test (Gate)](#14-phase-0-operator-expansion--the-ceiling-test-gate)
15. [The Smoke-Test / Early-Kill Filter (Always On)](#15-the-smoke-test--early-kill-filter-always-on)
16. [RL Methods: What We Considered and What We Decided](#16-rl-methods-what-we-considered-and-what-we-decided)

---

## 1. TL;DR

**The old question was:** "Can genetic programming evolve a loss function that beats CrossEntropy?" We tested this across 4 phases (MNIST, CIFAR-10, imbalanced data, few-shot CIFAR-10, Omniglot, and Omniglot with aggressive anti-convergence mechanisms). The answer was consistently **no** — evolution converges to MSE or simple MSE variants, and these lose to CrossEntropy by 1.5–4% in every scenario, even with 30 generations and 5 different diversity-preserving mechanisms.

**The new question is:** "Does the *best* loss function shape depend on the architecture it's paired with — and can we evolve a loss that is measurably *specialized* to one architecture over another?" This is a different and more tractable claim. We are not trying to beat a 70-year-old, information-theoretically-motivated loss function in absolute terms. We are testing whether loss-function preference is architecture-dependent at all — a question nobody in our literature review has directly tested.

**The method:** Evolve a separate loss function for each of several architectures (same GP system, same anti-convergence mechanisms we already built and validated). Then build a **transfer matrix**: train every architecture with every evolved loss (not just its own) and see whether each architecture performs best with *its own* evolved loss ("diagonal dominance"). This is testable, falls out naturally from infrastructure we already have, and is informative *even if the answer is no* (that itself would be a real finding: "loss function preference appears architecture-invariant").

**Why this escapes the old failure mode:** We are no longer trying to out-engineer CrossEntropy from scratch. We are asking a comparative, relative question (does loss A fit architecture A better than loss B does?) instead of an absolute one (is loss A better than CrossEntropy?). A 1% specialization effect, if statistically real, is a genuine, novel, publishable finding regardless of whether any evolved loss ever beats CrossEntropy outright.

**Immediate next action:** Do **Phase 0 first** (Section 14) — a cheap ceiling test that expands the operator set and checks whether a richer search space raises the achievable accuracy *at all* before any further work is committed. Only if Phase 0 passes do we proceed to build `evolve_loss_for_architecture(model_fn, data, config)` (Section 6.7), test it on one architecture end-to-end, extend to the 3-architecture "zoo," and build the transfer matrix (Section 5). Two capabilities are always-on from the start regardless of Phase 0: the **smoke-test / early-kill filter** (Section 15) which reclaims compute wasted on doomed individuals, and the five anti-convergence mechanisms (Section 6.4).

**On reinforcement learning:** We seriously evaluated adding RL to guide the search (see Section 16 for the full reasoning). Decision: **no heavy RL up front.** RL is a search-*efficiency* multiplier, not a search-*space* expander — it can only help you reach the ceiling faster, never raise it. Since our prior evidence suggests the limiter may be the operator set (the ceiling) rather than search intelligence, the correct first move is to expand operators (Phase 0) and measure whether the ceiling rises. The one RL idea worth adopting — a viability predictor for doomed offspring ("Method 4") — is introduced in its cheap, non-RL form as the smoke-test filter (Section 15) and only upgraded to a learned predictor *if* Phase 0 passes and exotic operators produce frequent-enough failures to be worth modeling. One RL idea (node-level credit assignment, "Method 2") is explicitly rejected because it is in direct tension with this project's core hypothesis (Section 16).

---

## 2. Background: What We Already Know

This section exists so a reader with zero prior context (e.g. a professor, or a coding agent with no memory of past sessions) understands why we are pivoting, not starting from scratch. 
IMPORTANT: Some of the formulas/values present in the research paper are innacurate, hence only use it to understand methodology and look at the ipynb files for accurate values. 

### 2.1 Prior work summary

We built a tree-based genetic programming (GP) system, using DEAP, that represents a loss function as an expression tree over a per-class error signal `error = one_hot(target) - softmax(logits)`. Trees are evolved (mutation, crossover, tournament selection, elitism) and each individual's fitness is the validation accuracy of a neural network trained with that individual's tree compiled into a PyTorch loss function.

We ran this across four progressively harder phases:

| Phase | Dataset | Setup | Best Evolved Loss | Evolved Acc | CrossEntropy Acc | Diff |
|---|---|---|---|---|---|---|
| 1 | MNIST | SimpleCNN, pop=5, gen=2 | `square(error)` (MSE) | 98.41% | 98.84% | −0.43% |
| 2 | CIFAR-10 (balanced) | ResNet-18, pop=20, gen=10 | `exp(error)` | 85.39% | 86.93% | −1.54% |
| 3a | CIFAR-10 (100:1 imbalanced) | Same loss as Phase 2 | `exp(error)` | 45.20% | 51.38% | −6.18% |
| 3b | CIFAR-10 (few-shot, 500/class, 5 seeds) | Same loss as Phase 2 | `exp(error)` | 61.88% ± 2.35% | 64.84% ± 4.12% | −2.97% (p=0.23, n.s.) |
| 4a | Omniglot (5-shot, 50-way) | Custom CNN, improved operators (added tanh/sigmoid/softplus/elu, removed exp/div), pop=20, gen=15 | `square(error)` (MSE) | 58.53% | 60.13% | −1.60% |
| 4b | Omniglot (5-shot, 50-way), **anti-convergence mechanisms**, gen=30, 5-fold validation | `square(sub(error, 0.724))` (biased MSE) | 56.35% ± 2.49% | 60.24% ± 4.87% | −3.89% (p=0.18, n.s.) |

**Evolved loss lost in 6/6 tested scenarios.** None of the differences favoring evolved losses were statistically significant. A full write-up of this work exists as a standalone research paper (see companion document `evolutionary_loss_research_paper.md`) — read that first if you want the full experimental narrative; this document only extracts what's operationally relevant going forward.

### 2.2 Why universal loss evolution struggles (and why this matters for the pivot)

Four converging explanations emerged, and all four remain true for the new project — they are not solved by pivoting, but the new research question is designed to be interesting *despite* them:

1. **CrossEntropy has real mathematical structure MSE lacks.** It's derived from maximum likelihood estimation, is a proper scoring rule, and has unbounded gradient when the prediction is very wrong. MSE saturates. This is a structural, not incidental, disadvantage for evolved MSE-like losses.
2. **The search space is biased toward simplicity.** Shorter trees train faster and are numerically safer (fewer chances of NaN/Inf), so evolution hill-climbs toward `square(error)` almost immediately (generation 1–2 in 5 of 6 experiments) and struggles to escape.
3. **Fitness evaluation is noisy, especially in few-shot settings.** In Phase 4b, CrossEntropy's own accuracy varied from 51.5% to 65.5% across 5 folds (±4.87% std) on the *identical* method — meaning single-seed "wins" (we saw one, +2.49% on one CIFAR-10 seed) are not trustworthy and evaporate under multi-seed validation. **This is the single most important lesson from the whole prior project.**
4. **Compute budget is tiny relative to what's needed to explore this space properly.** Our total evolution budget across all 4 phases was ~11 GPU-hours. Google's AutoML-Zero used ~15,000 GPU-hours for a related (broader) search. We are not going to out-compute our way to beating CrossEntropy in absolute terms with a laptop-scale budget.

None of these four problems go away in the new project. What changes is that we stop asking a question these four problems make nearly unwinnable ("beat CrossEntropy") and start asking a question they don't preclude ("is the optimum different for different architectures, even if none of the optima individually beat CrossEntropy").

### 2.3 What still works and will be reused directly

The following infrastructure was built, debugged, and validated in the prior project and should be reused, not rebuilt:

- **The primitive/operator set** (Section 6.2): `add, mul, sub, max, min, abs, square, sqrt, log, neg, sin, cos, tanh, sigmoid, softplus, elu` plus an ephemeral random constant in `[0, 2]`. `exp` and `div` were deliberately removed after causing instability (gradient explosion / division-by-near-zero NaNs).
- **The `EvolvedLossFunction` wrapper** (Section 6.3) that compiles a DEAP tree into a differentiable PyTorch `nn.Module`, with NaN/Inf guards and a gradient-connectivity trick (`loss + 0.0 * probs.sum()`) for degenerate trees that don't depend on the input.
- **The five anti-convergence mechanisms** (Section 6.4): fitness sharing, novelty bonus, trivial-tree penalty, adaptive mutation rate, and diversity-aware tournament selection. These measurably delayed (though did not prevent) convergence to MSE in Phase 4b (convergence pushed from generation 1 to generation 10; final population diversity 0.30 vs 0.12 without these mechanisms). **These are now a mandatory default, not an experimental add-on** — see Section 7.
- **Checkpointing to Google Drive after every generation** using `dill` (not `pickle`, which fails on DEAP's lambda-based ephemeral constants).
- **The K-fold / multi-seed validation pattern**: never trust a single-seed comparison; always validate the final head-to-head with ≥5 independent seeds and a paired t-test.

---

## 3. The New Idea: Architecture-Conditioned Loss Evolution

### 3.1 Core hypothesis, stated plainly

Different network architectures have different "natures": depth affects how gradients propagate, width affects capacity and overfitting tendency, the presence or absence of normalization changes activation statistics, and the choice between convolutional and fully-connected layers changes inductive bias. CrossEntropy is a single fixed function applied identically regardless of any of this. **Our hypothesis is that the loss landscape an architecture actually experiences during training is shaped by that architecture's own properties, and that a loss function evolved specifically against one architecture will fit that architecture measurably better than a loss function evolved against a different architecture — even if neither beats CrossEntropy outright.**

### 3.2 Why this reframing escapes the previous failure mode

The previous project's core problem was that it always reduced to a single absolute comparison: *evolved loss vs. CrossEntropy*, on one architecture, hoping to win outright. That comparison is extremely hard to win because of the four reasons in Section 2.2, and none of those reasons are fixed by trying harder within the same framing.

The new project instead reduces to a **relative, paired comparison**: *does loss evolved-for-A do better on architecture A than loss evolved-for-B does on architecture A?* This sidesteps the "CrossEntropy is really good" problem entirely, because CrossEntropy is not the yardstick for the primary claim — it only appears as a secondary baseline row. The primary claim is about **specialization**, which is a comparison between two things we generate ourselves under identical conditions, not a comparison against a 70-year-old, carefully-motivated fixed function.

This also turns our biggest recurring negative result — "evolution converges to MSE" — into a *testable variable* rather than a dead end. If every architecture converges to the *same* MSE variant regardless of its structure, that is itself informative (it would suggest loss-function preference is architecture-invariant — a real, reportable finding). If different architectures converge to *measurably different* tree structures, and those structures cross-transfer worse than they self-transfer, that is a positive, novel finding. **Every outcome of this experiment is informative and publishable in some form.** That property was missing from the old design, where a negative result (evolved loses to CE) was the only likely outcome and the paper had to be framed defensively around it.

### 3.3 What "architecture-conditioned" means operationally (start simple)

There are two ways to implement "the loss evolves for the network":

- **(a) Separate evolution runs per architecture** (simple, tractable, what we will do first): run the exact same GP system independently for each architecture in a small "zoo," using that architecture as the fitness-evaluation network. Compare the resulting trees and their cross-architecture transfer performance.
- **(b) A single population conditioned on architecture descriptors as GP terminals** (harder, more elegant, a natural extension once (a) works): give the tree access to terminals like `depth`, `width`, `has_batchnorm` so a *single* evolved expression could in principle produce different loss shapes for different architectures. This is more ambitious, more novel, but much riskier as a starting point (bigger search space, harder to interpret, unclear if DEAP's primitive-tree machinery handles conditional/architecture-aware terminals cleanly).

**Decision: start with (a).** It reuses 100% of the validated infrastructure in Section 2.3 and only requires new orchestration and evaluation code (Section 6), not new GP machinery. Note (b) explicitly in the paper's "Future Work" as the natural next step once (a) produces a positive or informative result.

---

## 4. Research Questions and Hypotheses

- **RQ1 (Specificity):** Does a loss function evolved for architecture *A* perform better on architecture *A* than a loss function evolved for a different architecture *B*, when both are evaluated on *A*?
  - **H1:** For each architecture *i* in the zoo, mean test accuracy using `loss_i` on architecture *i* is greater than the mean test accuracy using `loss_j` (j ≠ i) on architecture *i*, and this "diagonal dominance" is statistically significant (paired t-test across seeds, α = 0.05).

- **RQ2 (Structural divergence):** Do the loss functions evolved for different architectures actually differ in tree structure, or does evolution converge to the same solution (e.g., MSE) regardless of architecture?
  - **H2:** The mean pairwise structural distance (Section 6.4, `tree_distance`) between the best trees evolved for different architectures is greater than the mean structural distance between independent evolutionary runs on the *same* architecture (i.e., cross-architecture differences exceed run-to-run noise).

- **RQ3 (Absolute improvement, secondary):** Does any architecture-specialized loss beat CrossEntropy on its *own* architecture?
  - **H3 (bonus, not required for the paper to be valid):** For at least one architecture *i*, `loss_i` on architecture *i* exceeds CrossEntropy on architecture *i* by ≥1%, with p < 0.05 across ≥10 seeds.

- **RQ4 (Scaling, stated but not tested):** Does the specialization effect (if found) grow, shrink, or stay constant as architectures become larger/more heterogeneous?
  - **H4 (hypothesis for future work, explicitly out of scope for this compute budget):** We predict the effect grows with architectural heterogeneity, because more different "natures" should produce more different optimal loss shapes. This is stated as a testable hypothesis for follow-up work with more compute, not something we claim to demonstrate here.

### What counts as success — three tiers, so we never end up in the previous project's all-or-nothing bind again

1. **Tier 1 (minimum publishable result):** H2 holds — evolved trees are structurally different across architectures beyond run-to-run noise, whether or not this translates to accuracy differences. This alone is a valid, novel empirical finding about the GP search space.
2. **Tier 2 (core claim):** H1 holds — statistically significant diagonal dominance in the transfer matrix. This is the paper's central contribution: loss functions can be *specialized* by architecture.
3. **Tier 3 (headline result, not required):** H3 holds for at least one architecture — a specialized evolved loss beats CrossEntropy outright, with proper multi-seed validation. This would be the strongest possible outcome, but the paper stands on Tier 1/2 alone if this doesn't materialize.

---

## 5. Experimental Design

### 5.1 The architecture zoo (defining "nature" cheaply)

To keep Phase 1 tractable on a single Colab T4, use MNIST or Fashion-MNIST (28×28 grayscale, fast to train) and three architectures that differ along clean, interpretable axes while holding everything else (input size, output classes, training budget) constant:

| Name | Description | "Nature" it represents |
|---|---|---|
| **Arch-A: ShallowWide** | 2 conv layers (64→128 channels), large kernels, global average pool, 1 FC layer | Short gradient path, high per-layer capacity |
| **Arch-B: DeepNarrow** | 6 conv layers (16→32→64 channels), small kernels, no normalization | Long gradient path, prone to vanishing gradients, low per-layer capacity |
| **Arch-C: PureMLP** | Flatten input, 3 fully-connected hidden layers, no convolution | No spatial inductive bias — maximally different "nature" from A and B |

This gives 3 architectures spanning depth (A vs B), and representation type (A/B vs C), while all three are cheap enough to train in seconds per epoch at MNIST scale. If Phase 1 shows a signal, Phase 2 optionally adds:

| Name (Phase 2, optional) | Description |
|---|---|
| **Arch-D: DeepNarrow+BN** | Same as B but with BatchNorm after every conv — isolates the effect of normalization specifically |
| CIFAR-10 scale versions of A/B/C | Tests whether Phase 1 findings hold on harder data |

### 5.2 Per-architecture evolution protocol

For each architecture in the zoo, run one full evolution using the anti-convergence GP system from Section 6:

- Population 20, generations 20 (MNIST-scale individuals train fast — see Section 5.5 for time budget)
- Fitness = validation accuracy after training that architecture with the candidate loss tree for a fixed, short schedule (start at 8 epochs; increase if fitness signal looks too noisy — see Section 6.5)
- All five anti-convergence mechanisms active by default (not optional — see Section 7)
- Checkpoint after every generation (same pattern as Phase 4b)
- Output: `loss_A`, `loss_B`, `loss_C` — one best tree per architecture, plus the full final population and evolution history (fitness trajectory, diversity trajectory) for each, saved for the structural-divergence analysis (H2)

### 5.3 The cross-architecture transfer matrix (the key new experiment)

This is the experiment that actually tests H1. Construct a full matrix: every evolved loss, trained on every architecture, from scratch, averaged over multiple seeds.

Let the zoo be {A, B, C}. For every pair (loss origin *i*, training architecture *j*), train architecture *j* from scratch using `loss_i`, repeated across `N_SEEDS` seeds (start with `N_SEEDS = 5` for the exploratory pass, scale to 10+ for the confirmatory pass — see Section 10), and record mean ± std test accuracy.

```
                  Trained on Arch A   Trained on Arch B   Trained on Arch C
loss_A (from A)        M[A][A]             M[A][B]             M[A][C]
loss_B (from B)        M[B][A]             M[B][B]             M[B][C]
loss_C (from C)        M[C][A]             M[C][B]             M[C][C]
CrossEntropy (base)    CE[A]               CE[B]               CE[C]
loss_universal*        U[A]                U[B]                U[C]
```

`*loss_universal` is an optional control condition: one additional evolution run whose fitness is the *average* accuracy across all three architectures simultaneously (instead of one architecture alone). This tells us whether a "jack of all trades" loss exists and how it compares to both CrossEntropy and the specialized losses. Include this if time allows; it is not required for H1/H2.

**H1 is tested by checking whether the diagonal (`M[A][A]`, `M[B][B]`, `M[C][C]`) is significantly greater than the corresponding off-diagonal entries in the same row**, via paired t-tests across seeds (e.g., is `M[A][A]` > `M[A][B]` and `M[A][A]` > `M[A][C]`, paired by seed). Report a single summary "specificity score" per architecture:

```
specificity_score(i) = M[i][i] - mean(M[i][j] for j != i)
```

A positive, statistically significant specificity score across most/all architectures is the core positive result.

### 5.4 Statistical validation plan

- Exploratory pass: 5 seeds per matrix cell (as in Phase 4b) — enough to see if there's any signal worth pursuing.
- Confirmatory pass: if Tier 1 or Tier 2 success criteria look plausible after the exploratory pass, re-run the relevant matrix cells with 10–20 seeds (see Section 10 for why) before claiming significance in the paper.
- Always use a **paired** design: the same set of seeds (same data splits, same weight initializations where feasible) across all losses being compared in a given cell of the matrix, so that differences are due to the loss function, not noise from different random splits. This is a direct fix for the variance problem identified in Phase 4b.

### 5.5 Scaling plan: start small, hypothesize the rest

Given the ~11 GPU-hour budget of the entire prior project, Phase 1 of this project should target a similar or smaller budget:

- 3 evolution runs (pop=20, gen=20, ~8 epochs/individual, MNIST-scale) ≈ 1–2 GPU-hours each ≈ 3–6 GPU-hours total
- Transfer matrix exploratory pass: 3×3 = 9 cells + 3 CE baseline cells (+3 optional universal-loss cells) × 5 seeds × ~8 epochs each ≈ 1–2 GPU-hours total
- **Total Phase 1 budget: roughly 5–8 GPU-hours**, comparable to or less than the prior project's total spend.

If Phase 1 produces a Tier 1 or Tier 2 result, Phase 2 (CIFAR-10-scale zoo, more seeds, optional Arch-D) is proposed in the paper as future work with an explicit statement along these lines: *"Given the resource constraints of this study, we validate our hypothesis on a compact architecture zoo at MNIST scale. We predict — but do not test here — that the specialization effect will be more pronounced on larger, more heterogeneous architectures (ResNet variants, transformers, etc.), where differences in gradient flow and inductive bias are more extreme. We leave this scaling study to future work with access to greater compute."* This turns the compute limitation into an explicit, honest, and reasonable scoping statement rather than a hidden weakness.

---

## 6. Technical Specification (For Implementation)

This section is written so a coding agent can implement directly from it. Wherever code is shown, it is either a direct carry-over from the validated prior implementation (marked **[REUSE]**) or new orchestration logic for this project (marked **[NEW]**).

### 6.1 Repository / directory structure

```
architecture-conditioned-loss-evolution/
├── src/
│   ├── gp_core.py              # [REUSE] primitive set, EvolvedLossFunction, tree utilities
│   ├── anti_convergence.py     # [REUSE] fitness sharing, novelty, trivial penalty, adaptive mutation
│   ├── architectures.py        # [NEW] Arch-A/B/C model definitions, shared interface
│   ├── evolve.py                # [NEW] evolve_loss_for_architecture(), checkpointing
│   ├── transfer_matrix.py       # [NEW] cross-architecture evaluation matrix + stats
│   └── utils.py                  # [REUSE] seeding, label remapping, safe train/eval loops
├── configs/
│   └── phase1_mnist_zoo.yaml     # architecture list, hyperparameters (Section 6.9)
├── notebooks/
│   └── ACLE_Phase1_MNIST.ipynb
├── checkpoints/
│   └── acle_phase1/
│       ├── evolution_A/          # per-generation checkpoints, one folder per architecture
│       ├── evolution_B/
│       ├── evolution_C/
│       └── transfer_matrix/      # raw per-cell, per-seed results
├── results/
│   ├── transfer_matrix.csv
│   ├── evolution_trajectories.csv
│   └── structural_divergence.json
└── docs/
    └── ARCHITECTURE_CONDITIONED_LOSS_EVOLUTION_SPEC.md   # this file
```

### 6.2 Tree representation and operator set **[REUSE]**

Carry over exactly from Phase 4:

```python
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(torch.add, 2, name="add")
pset.addPrimitive(torch.mul, 2, name="mul")
pset.addPrimitive(torch.sub, 2, name="sub")
pset.addPrimitive(torch.maximum, 2, name="max")
pset.addPrimitive(torch.minimum, 2, name="min")
pset.addPrimitive(torch.abs, 1, name="abs")
pset.addPrimitive(torch.square, 1, name="square")
pset.addPrimitive(safe_sqrt, 1, name="sqrt")     # clamp(abs(x), min=1e-7) before sqrt
pset.addPrimitive(safe_log, 1, name="log")       # clamp(abs(x), min=1e-7) before log
pset.addPrimitive(torch.neg, 1, name="neg")
pset.addPrimitive(torch.sin, 1, name="sin")
pset.addPrimitive(torch.cos, 1, name="cos")
pset.addPrimitive(torch.tanh, 1, name="tanh")
pset.addPrimitive(torch.sigmoid, 1, name="sigmoid")
pset.addPrimitive(F.softplus, 1, name="softplus")
pset.addPrimitive(F.elu, 1, name="elu")
pset.addEphemeralConstant("const", lambda: random.uniform(0, 2))
pset.renameArguments(ARG0='error')
```

Do **not** re-add `exp` or `div` — both were confirmed sources of NaN/Inf instability in Phases 1–3. Max tree depth stays at 5 (bloat control, validated as reasonable across all prior phases).

**This is the *baseline* operator set only.** Phase 0 (Section 14) expands it, and the expansion is now the gating first experiment of the whole project — because there is a real possibility that the prior project's failure to beat CrossEntropy was a *search-space expressiveness* problem (the operators literally could not express a CrossEntropy-competitive loss) rather than a search-quality problem. See Section 14 for the exact operators to add, the order to add them in, and the pass/fail criterion. Do not begin the architecture-zoo study (Section 5) until Phase 0 has been run and its result recorded.

### 6.3 The `EvolvedLossFunction` wrapper **[REUSE]**

```python
class EvolvedLossFunction(nn.Module):
    def __init__(self, tree, pset):
        super().__init__()
        self.tree = tree
        self.func = gp.compile(expr=tree, pset=pset)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        error = one_hot - probs
        try:
            loss_val = self.func(error)
            if not torch.is_tensor(loss_val):
                loss_val = torch.tensor(loss_val, device=logits.device, dtype=torch.float32)
            if not loss_val.requires_grad:
                loss_val = loss_val + 0.0 * probs.sum()   # gradient-connectivity guard
            if loss_val.dim() > 1:
                loss_val = torch.mean(torch.sum(loss_val, dim=1))
            elif loss_val.dim() == 1:
                loss_val = torch.mean(loss_val)
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                return 1000.0 * torch.mean(probs)          # penalty fallback
            return loss_val
        except Exception:
            return 1000.0 * torch.mean(probs)
```

### 6.4 Anti-convergence mechanisms **[REUSE, now mandatory]**

All five mechanisms from Phase 4b, unchanged. These are no longer an "experimental extra" — they are the default configuration for every evolution run in this project, because Phase 4b demonstrated they meaningfully delay (from generation 1 to generation 10) convergence to trivial solutions.

```python
def tree_distance(tree1, tree2):
    """Structural distance via normalized string edit approximation."""
    s1, s2 = str(tree1), str(tree2)
    if s1 == s2:
        return 0.0
    len_diff = abs(len(s1) - len(s2))
    char_diff = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return (len_diff + char_diff) / max(len(s1), len(s2))

def get_tree_structure_hash(tree):
    """Structural signature ignoring specific constant values."""
    import re
    return re.sub(r'\d+\.\d+', 'C', str(tree))

def apply_fitness_sharing(population, sigma=0.3):
    for ind in population:
        niche_count = sum(1 for other in population if tree_distance(ind, other) < sigma)
        ind.shared_fitness = ind.fitness.values[0] / max(niche_count, 1)

def calculate_novelty_bonus(tree, archive):
    tree_hash = get_tree_structure_hash(tree)
    count = sum(1 for t in archive if get_tree_structure_hash(t) == tree_hash)
    return {0: 2.0, 1: 1.0, 2: 0.5}.get(count, 0.0)

def apply_trivial_penalty(tree, trivial_set):
    return 0.95 if str(tree) in trivial_set else 1.0

def adaptive_mutation_rate(diversity_score):
    if diversity_score < 0.2: return 0.5
    if diversity_score < 0.4: return 0.4
    return 0.3

def calculate_population_diversity(population):
    if len(population) < 2: return 1.0
    pairs = [tree_distance(a, b) for i, a in enumerate(population) for b in population[i+1:]]
    return float(np.mean(pairs)) if pairs else 0.0
```

`TRIVIAL_SET` should be defined per-project as the set of single-operation trees on `error` (e.g. `square(error)`, `abs(error)`, `sqrt(error)`, `tanh(error)`, etc.) — reuse the list from Phase 4b.

### 6.5 Fitness evaluation protocol — reliability improvements

Phase 4b's biggest weakness was fitness noise (CrossEntropy itself varied ±4.87% across folds). For this project, apply these mitigations, in order of priority:

0. **Smoke-test every individual before spending a full training budget on it** (Section 15). This is the cheap, always-on core of RL "Method 4" (predictive operator pruning), implemented as a heuristic rather than a learned model. Before committing 8 epochs to a candidate tree, run it on a single batch and kill it immediately if the loss is NaN/Inf, if its gradient is zero/undefined, or (optionally) if it is still at near-random accuracy after 1 epoch. Prior logs were full of individuals scoring ~10% (random), 0.84%, or "Loss exploded" — every one of those was a wasted full-length training run. This reclaims compute directly and matters *more* once Phase 0 adds exotic operators with new failure modes.

1. **Fix the data split per architecture zoo run**: use the *same* train/val split (same seed) across all individuals within one evolution run, so fitness differences reflect the loss function, not split luck. (Vary the seed only *between* independent runs used for statistical validation, never *within* one evolutionary search.)
2. **Increase epochs per individual if the fitness ranking looks unstable**: start at 8 epochs (MNIST-scale, cheap); if repeated runs of the same tree produce wildly different fitness, increase to 12–15 before assuming the tree itself is bad.
3. **Optional (more expensive) — average fitness over 2 short training runs per individual** with different weight initializations, if budget allows. This directly reduces the variance that undermined single-run fitness comparisons in Phase 3b/4b, at ~2x evaluation cost.
4. **Reserve a true held-out test set, separate from the validation set used for fitness**, and only touch it for the final transfer-matrix numbers reported in the paper — never for evolution itself. (This wasn't always cleanly separated in prior phases; enforce it explicitly here.)

### 6.6 Architecture zoo implementation

Each architecture must implement a common interface so the rest of the pipeline is architecture-agnostic:

```python
# architectures.py
class ShallowWideCNN(nn.Module):
    """Arch-A: 2 conv layers, wide channels, short gradient path."""
    def __init__(self, num_classes=10, in_channels=1):
        ...
    def forward(self, x):
        ...  # returns logits, shape (batch, num_classes)

class DeepNarrowCNN(nn.Module):
    """Arch-B: 6 conv layers, narrow channels, no normalization."""
    ...

class PureMLP(nn.Module):
    """Arch-C: fully-connected, no convolution."""
    ...

ARCHITECTURE_ZOO = {
    "A_ShallowWide": lambda: ShallowWideCNN(num_classes=10),
    "B_DeepNarrow": lambda: DeepNarrowCNN(num_classes=10),
    "C_PureMLP": lambda: PureMLP(num_classes=10),
}
```

**Contract for any architecture added to the zoo (this is also the contract the user's own custom `SimpleCNN`-style models must satisfy to plug into the system):** a zero-argument factory function that returns a freshly-initialized `nn.Module`, accepting the project's fixed input shape and returning raw logits of shape `(batch, num_classes)`. Nothing else about internal structure is constrained — this is what makes the framework general-purpose across "any network the user codes."

### 6.7 Core deliverable: the general evolution API **[NEW — this is the main thing to build first]**

This is the single most important function in the whole project. Everything else (the zoo, the transfer matrix) is just calling this multiple times and comparing outputs.

```python
def evolve_loss_for_architecture(
    model_fn: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: EvolutionConfig,
    checkpoint_dir: str,
    resume: bool = True,
) -> EvolvedLossResult:
    """
    Runs one full anti-convergence GP evolution using model_fn() as the
    fitness-evaluation architecture. Checkpoints every generation to
    checkpoint_dir (dill-serialized) so it can resume after a Colab
    disconnect. Returns the best tree, its fitness, the full final
    population (for structural-divergence analysis), and the per-generation
    fitness/diversity trajectory.
    """
```

`EvolutionConfig` should be a small dataclass/YAML-loadable struct holding: `population_size`, `generations`, `epochs_per_individual`, `tournament_size`, `elitism`, `crossover_rate`, `mutation_rate_bounds`, `fitness_sharing_sigma`, `novelty_bonus_schedule`, `trivial_penalty`, `seed`. Default values are given in Section 6.9.

This function is a direct generalization of the `run_anti_convergence_evolution()` function already written and validated in Phase 4b — the only real change is that the model class is now a parameter (`model_fn`) instead of hardcoded `OmniglotCNN`.

### 6.8 Transfer matrix runner **[NEW]**

```python
def build_transfer_matrix(
    architecture_zoo: Dict[str, Callable[[], nn.Module]],
    evolved_losses: Dict[str, EvolvedLossFunction],   # keyed by origin architecture name
    data_loaders: Dict[str, Tuple[DataLoader, DataLoader]],  # per-architecture-compatible loaders
    n_seeds: int = 5,
    epochs: int = 15,
    include_crossentropy_baseline: bool = True,
) -> TransferMatrixResult:
    """
    For every (loss_origin, training_architecture) pair, trains
    training_architecture from scratch with loss_origin's evolved loss,
    repeated across n_seeds, and records mean/std test accuracy.
    Also trains every architecture with plain CrossEntropy as a baseline
    row if include_crossentropy_baseline=True.
    Returns a structured result with the full matrix, paired seed-level
    data (for the paired t-tests in Section 5.3), and specificity scores.
    """
```

Output should be saved as both a raw CSV (one row per (loss_origin, arch, seed) triple, for reproducible statistics) and a summary CSV (mean ± std per cell, matching the display matrix in Section 5.3).

### 6.9 Default hyperparameters

```yaml
# configs/phase1_mnist_zoo.yaml
seed: 42
dataset: mnist          # or fashion_mnist
architecture_zoo: [A_ShallowWide, B_DeepNarrow, C_PureMLP]

evolution:
  population_size: 20
  generations: 20
  epochs_per_individual: 8
  tree_max_depth: 5
  tournament_size: 3
  elitism: 3
  crossover_rate: 0.5
  mutation_rate_base: 0.3
  mutation_rate_high_diversity_threshold: 0.4   # diversity < this -> mutation 0.4
  mutation_rate_low_diversity_threshold: 0.2    # diversity < this -> mutation 0.5
  fitness_sharing_sigma: 0.3
  novelty_bonus_schedule: [2.0, 1.0, 0.5, 0.0]  # for 0, 1, 2, 3+ prior occurrences
  trivial_tree_penalty: 0.95

transfer_matrix:
  n_seeds_exploratory: 5
  n_seeds_confirmatory: 15    # see Section 10
  epochs_final_training: 15
  include_universal_loss_control: true
```

### 6.10 What the "user codes a network and evolution starts" workflow looks like end to end

This is the concrete usage pattern that fulfills the original goal ("I code a network... and the algorithm starts the evolution with that model and finds a loss function"):

```python
# 1. User writes their own architecture, satisfying the contract in 6.6
def my_custom_net():
    return SimpleCNN(num_classes=10)

# 2. User calls the core API directly, no zoo/matrix required for basic use
result = evolve_loss_for_architecture(
    model_fn=my_custom_net,
    train_loader=my_train_loader,
    val_loader=my_val_loader,
    config=EvolutionConfig(population_size=20, generations=20, epochs_per_individual=8),
    checkpoint_dir="checkpoints/my_run/",
)

print(result.best_tree)          # e.g. square(sub(error, 0.42))
print(result.best_fitness)       # validation accuracy achieved
```

This single-architecture usage mode should work standalone, independent of the multi-architecture research study — it's the minimum viable product, and the transfer-matrix study (Section 5.3) is what turns repeated use of this same function into a paper.

---

## 7. Known Pitfalls — Do Not Repeat These

These are concrete bugs and false starts hit during the prior project. Listed here explicitly so they are not re-discovered at cost.

1. **CUDA `device-side assert triggered` from label/class-count mismatch.** Happens when a dataset's raw labels (e.g. Omniglot's ~1600 original class IDs) aren't remapped to a contiguous `0..num_classes-1` range matching the model's output layer. **Fix:** always build an explicit `old_label -> new_label` map when constructing any few-shot/subset dataset, and verify `max(labels) < num_classes` before training.
2. **A CUDA assert corrupts the GPU context for the rest of the session.** Re-running the failing cell after a fix will keep failing with the same or a different opaque CUDA error. **Fix:** `Runtime > Restart Runtime` is mandatory after any CUDA assert, not optional — don't waste time debugging around it in the same session.
3. **`pickle` cannot serialize DEAP's ephemeral-constant lambdas.** **Fix:** always use `dill`, not `pickle`, for checkpointing populations. (A `RuntimeWarning` about "Ephemeral const function cannot be pickled" at `pset` construction time is harmless as long as `dill` is used for the actual save/load — it comes from DEAP's own internal pickling check, not from our checkpoint code.)
4. **Silently saving checkpoints to the wrong Drive path.** In one prior session, checkpoints were saved to `/content/drive/MyDrive/Colab_Project_Checkpoints/` while recovery code looked in `/content/drive/MyDrive/EC_Project/checkpoints/...` — hours of confusion. **Fix:** always print the absolute checkpoint path at the top of every evolution run, and write a small `run_manifest.json` (run id → absolute checkpoint dir → config used) at the start of each run so recovery never requires guessing.
5. **Declaring a win from one seed.** The single strongest-looking positive result in the entire prior project (+2.49% evolved-vs-CrossEntropy on CIFAR-10 few-shot, seed 42) completely reversed under 5-seed validation (mean −2.97%). **Fix, now a hard rule for this project: no result is reported, even informally, until it has been checked across at least 5 seeds. A single-seed number is a pilot check, never a claim.**
6. **Evolution converges to a trivial single-operation tree within 1–2 generations if anti-convergence mechanisms are off.** This happened in 3 of the 4 phases that didn't use them. **Fix:** the five anti-convergence mechanisms (Section 6.4) are always on by default in this project; there is no "baseline" evolution run without them planned, because we already know what that produces.
7. **Unstable operators cause NaN cascades.** `exp` and `div` were both individually confirmed sources of exploding/undefined loss values. **Fix:** they remain excluded from the operator set (Section 6.2); if a future extension wants to re-add them, they must go behind extra-aggressive clamping and be tested in isolation first.
8. **Fitness noise from tiny few-shot fitness-evaluation sets.** Fitness computed on very small validation slices (e.g. 15 examples/class) swings heavily run-to-run. **Fix:** see Section 6.5's mitigations (fixed split per run, sufficient epochs, optional multi-run averaging).

---

## 8. Step-by-Step Execution Plan

| Phase | Task | Est. GPU time | Gate to proceed |
|---|---|---|---|
| **0** | **Operator-expansion ceiling test (Section 14).** Add richer operators, run plain/lightly-anti-convergent GP on ONE architecture, check whether the achievable accuracy ceiling rises toward CrossEntropy. Smoke-test filter (Section 15) active. | **~2–4 hrs** | **HARD GATE. If the ceiling does not rise, do NOT invest in richer operators or any RL; proceed to the zoo study with the baseline operator set and frame the paper around specialization only. If it rises, carry the expanded operator set into all later phases and Method 4-as-learned-predictor becomes justified.** |
| A | Build `architectures.py` (Arch-A/B/C) and confirm each trains normally with plain CrossEntropy on MNIST (sanity check — expect ~97–99% for A/B, somewhat lower for C) | <30 min | All three architectures train without error and reach sane baseline accuracy |
| B | Port `evolve_loss_for_architecture()` from the validated Phase 4b code, generalized to accept `model_fn` (Section 6.7) | — (implementation only) | Runs end-to-end on Arch-A for 2 generations without crashing, checkpoints correctly |
| C | Run full evolution (pop=20, gen=20) independently on Arch-A, Arch-B, Arch-C | ~3–6 hrs total | Each run produces a best tree, fitness trajectory, and final population saved to disk |
| D | Structural divergence check (H2): compute pairwise `tree_distance` between the three best trees, and between multiple independent runs on the same architecture (if budget allows a repeat run) as a noise baseline | <30 min | Have a concrete divergence number to report either way |
| E | Build the transfer matrix: all (loss origin × training architecture) pairs, 5 seeds each, plus CrossEntropy baseline row | ~1–2 hrs | Full matrix + baseline populated, saved as CSV |
| F | Statistical analysis: paired t-tests for diagonal dominance (H1), specificity scores per architecture | <30 min | Know whether Tier 1 / Tier 2 / Tier 3 (Section 4) was achieved |
| G | If Tier 1 or Tier 2 achieved and result looks promising: confirmatory re-run of the relevant matrix cells with 10–20 seeds (Section 10) before writing anything up as significant | ~2–4 hrs (targeted, not the full matrix) | Confirmed or disconfirmed at proper statistical power |
| H | Write up regardless of outcome (see Section 11) | — | — |

**Total estimated budget: ~9–17 GPU-hours** (Phase 0 gate + the zoo study), all achievable on Colab's free tier across a handful of sessions, using the same checkpoint-every-generation discipline validated in Phase 4b. Note that Phase 0 is a genuine gate: in the pessimistic case where the ceiling doesn't rise, you spend only the ~2–4 Phase-0 hours before knowing to drop the operator-expansion/RL direction, which is exactly the point of sequencing it first.

---

## 9. How to Interpret Results — Decision Tree

```
Did evolved trees differ structurally across architectures (H2)?
├── NO  → Report as a real (if less exciting) finding: "loss-function
│         preference under this GP search appears architecture-invariant."
│         Still worth writing up — nobody has shown this either way before.
│
└── YES → Is there significant diagonal dominance in the transfer matrix (H1)?
          ├── NO  → Trees differ structurally but don't transfer differently in
          │         practice. Report as: "structural specialization occurs but
          │         does not translate to measurable performance specialization
          │         at this scale/budget." Still a valid, informative negative
          │         result, and motivates the compute-scaling hypothesis (H4)
          │         as future work.
          │
          └── YES → Tier 2 achieved — this is the paper's core positive result.
                    Does any loss_i beat CrossEntropy on its own architecture (H3)?
                    ├── NO  → Still a strong paper: "loss functions specialize to
                    │         architecture even though none surpass the standard
                    │         baseline in absolute terms." Honest, novel, complete.
                    └── YES → Tier 3 — strongest possible outcome. Lead the paper
                              with this result, supported by the specialization
                              framing as the explanation for *why* it happened.
```

Every branch of this tree produces a legitimate, honestly-reportable paper. That is the design goal of this pivot.

---

## 10. Statistical Power: How Many Seeds Do We Actually Need?

You explicitly want to know that a ~1% effect, if found, is defensible. Be honest about this rather than hopeful: prior phases showed CrossEntropy's own accuracy varying by up to ±4.87% (std) across folds under few-shot conditions. Using a standard paired-difference sample-size approximation,

```
n ≈ (z_(α/2) + z_β)² · σ_d² / δ²
```

with a two-tailed α = 0.05 (`z ≈ 1.96`), 80% power (`z ≈ 0.84`), an observed paired-difference std of `σ_d ≈ 3%` (a reasonable middle estimate from our prior variance), and a target effect size `δ = 1%`:

```
n ≈ (1.96 + 0.84)² · 3² / 1² ≈ 7.84 · 9 ≈ 70
```

**This is an uncomfortable but important number: reliably detecting a genuine 1% mean effect at the variance levels we've actually observed would require on the order of 50–70 seeds per comparison** — far more than the 5 used in prior exploratory work. Two honest paths forward, both should be used together:

1. **Reduce variance by design**, which lowers the required *n* quadratically: use a paired design with identical data splits and initializations across compared conditions (already planned in Section 5.4), use larger validation sets than the 15-examples/class extremes seen in Phase 4, and consider averaging fitness/final accuracy over 2–3 short runs per condition. Halving `σ_d` (e.g. 3% → 1.5%) cuts the required `n` by 4×.
2. **Report effect sizes and confidence intervals honestly, not just p-values, and scale seed count to the claim being made**: 5 seeds is fine for an *exploratory* pass to decide whether to invest more (Section 5.4). Before writing "loss X significantly outperforms loss Y" anywhere in the paper, that specific comparison needs the higher seed count (10–20 as a practical, budget-aware compromise; note in the paper that this is powered to detect effects of roughly 1.5–2%, not 1%, given the compute budget, and report the actual confidence interval rather than overclaiming precision).

This section should be cited directly in the paper's Methods/Limitations to preempt exactly the criticism a careful reviewer would raise.

---

## 11. Paper Framing

**Central question:** *"Can tree-based genetic programming evolve novel loss functions?"* Our own prior work already suggests a nuanced answer: not ones that beat CrossEntropy outright under realistic compute budgets — but that may be the wrong question. This project proposes and tests a better one: *"Are the loss functions genetic programming discovers specific to the architecture they were evolved against?"*

**Candidate titles:**
- *"Does One Loss Fit All Architectures? Evolving Architecture-Conditioned Loss Functions via Genetic Programming"*
- *"Architecture-Specific Loss Function Discovery: Testing Whether Genetic Programming Specializes Losses to Network Structure"*
- *"Beyond Beating CrossEntropy: Evidence for Architecture-Dependent Loss Function Preference"*

**Why this is a stronger paper than the original plan, independent of outcome:** the original framing had exactly one path to a positive result (evolved beats CrossEntropy) and that path was empirically shown, across 6 separate experiments, to be extremely narrow. This framing has three tiers of legitimate positive/informative outcomes (Section 4), a built-in, honest treatment of statistical power (Section 10), and an explicit, reasonable compute-scaling hypothesis (H4) that turns "we only tested small architectures" from a weakness into a stated, falsifiable prediction for future work.

**Target venue:** GECCO remains the best fit (genetic programming + a clearly-scoped empirical AutoML question). IJCAI/AAAI AutoML tracks or workshop venues (NeurIPS/ICML AutoML workshops) as backups, consistent with the prior project's venue analysis.

**Related work to add beyond the prior paper's references:** work on architecture-aware training (e.g., learning-rate/optimizer choices that are known to interact with depth and normalization) should be cited as motivation for why architecture-conditioned *training components in general* are known to matter — even though, to our knowledge, this has not been tested specifically for the loss function itself. (Verify this literature-review claim before submission — search for existing "architecture-aware loss function" or "loss function transfer across architectures" work specifically, since this is the paper's central novelty claim and must be checked against the most current literature, not assumed from training-data-era knowledge.)

---

## 12. Glossary

- **Tree-based GP:** Genetic programming where candidate solutions are expression trees (here, mathematical formulas for a loss function) rather than fixed-length strings.
- **Fitness sharing:** Anti-convergence mechanism that divides an individual's fitness by the number of similar individuals in the population, discouraging the population from clustering around one solution.
- **Novelty bonus:** A small fitness boost given to structurally unique trees, independent of their accuracy, to reward exploration.
- **Trivial tree:** A loss function that is just one operation applied directly to the error signal (e.g. `square(error)`) — these are down-weighted because they dominate the search space too easily.
- **Diagonal dominance:** In the transfer matrix (Section 5.3), the property that each architecture performs best with the loss evolved specifically for it (the matrix's diagonal entries beat that row's off-diagonal entries).
- **Specificity score:** `M[i][i] - mean(M[i][j] for j != i)` — a single number per architecture summarizing how much better its own evolved loss does compared to losses evolved for other architectures.
- **Paired t-test:** A statistical test comparing two conditions measured on the *same* set of seeds/splits, which controls for run-to-run variance shared by both conditions (much more powerful than comparing independent, unpaired sets of runs).
- **Structural distance (`tree_distance`):** A normalized measure of how different two expression trees are, used to quantify diversity within a population and divergence between architectures' evolved losses.

---

## 13. Appendix: Reusable Code Skeletons

The following are condensed reference implementations carried over and adapted from the validated Phase 4b code. Treat these as a starting skeleton to adapt, not final production code — a coding agent implementing this project should still add proper error handling, logging, and tests.

```python
# --- utils.py: seeding and safe training loop [REUSE, generalized] ---

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_evaluate(model_fn, loss_module, train_loader, test_loader,
                        epochs, device, lr=1e-3):
    """Generic train+eval loop, architecture-agnostic. Returns test accuracy."""
    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_module(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                return 0.0   # evolution should treat this as a failed individual
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
```

```python
# --- evolve.py: generalized evolution loop skeleton [NEW, built from Phase 4b] ---

def evolve_loss_for_architecture(model_fn, train_loader, val_loader, config,
                                  checkpoint_dir, device, resume=True):
    os.makedirs(checkpoint_dir, exist_ok=True)
    set_all_seeds(config.seed)

    population = None
    start_gen = 0
    archive = []

    if resume:
        ckpt = find_latest_checkpoint(checkpoint_dir)   # reuse Phase 4b pattern
        if ckpt:
            population, start_gen, archive = ckpt['population'], ckpt['generation'], ckpt['archive']

    if population is None:
        population = toolbox.population(n=config.population_size)

    for gen in range(start_gen, config.generations):
        for i, ind in enumerate(population):
            if not ind.fitness.valid:
                raw = train_and_evaluate(
                    model_fn,
                    EvolvedLossFunction(ind, pset),
                    train_loader, val_loader,
                    epochs=config.epochs_per_individual,
                    device=device,
                )
                novelty = calculate_novelty_bonus(ind, archive)
                penalty = apply_trivial_penalty(ind, TRIVIAL_SET)
                ind.fitness.values = ((raw + novelty) * penalty,)
                ind.raw_fitness = raw

        apply_fitness_sharing(population, sigma=config.fitness_sharing_sigma)
        population.sort(key=lambda x: x.shared_fitness, reverse=True)
        diversity = calculate_population_diversity(population)
        archive.extend(population)

        save_checkpoint(checkpoint_dir, population, gen + 1,
                         metadata={'diversity': diversity, 'archive': archive})

        if gen < config.generations - 1:
            mut_rate = adaptive_mutation_rate(diversity)
            population = make_next_generation(population, config, mut_rate)

    population.sort(key=lambda x: x.raw_fitness, reverse=True)
    return EvolvedLossResult(
        best_tree=population[0],
        best_fitness=population[0].raw_fitness,
        final_population=population,
        # trajectory data collected across the loop above
    )
```

```python
# --- transfer_matrix.py: skeleton [NEW] ---

def build_transfer_matrix(architecture_zoo, evolved_losses, data_loaders,
                           n_seeds, epochs, device, include_crossentropy_baseline=True):
    rows = []
    for loss_name, tree in evolved_losses.items():
        for arch_name, model_fn in architecture_zoo.items():
            train_loader, test_loader = data_loaders[arch_name]
            for seed in range(n_seeds):
                set_all_seeds(seed)
                acc = train_and_evaluate(
                    model_fn, EvolvedLossFunction(tree, pset),
                    train_loader, test_loader, epochs, device,
                )
                rows.append({'loss_origin': loss_name, 'trained_on': arch_name,
                             'seed': seed, 'accuracy': acc})

    if include_crossentropy_baseline:
        for arch_name, model_fn in architecture_zoo.items():
            train_loader, test_loader = data_loaders[arch_name]
            for seed in range(n_seeds):
                set_all_seeds(seed)
                acc = train_and_evaluate(
                    model_fn, nn.CrossEntropyLoss(),
                    train_loader, test_loader, epochs, device,
                )
                rows.append({'loss_origin': 'CrossEntropy', 'trained_on': arch_name,
                             'seed': seed, 'accuracy': acc})

    return pd.DataFrame(rows)   # feed into paired t-tests / specificity scores
```

---

---

## 14. Phase 0: Operator Expansion — The Ceiling Test (Gate)

### 14.1 Why this exists and why it comes first

The prior project's central negative result (evolution never beats CrossEntropy) was obtained with an operator set of basic algebra plus bounded activations. That set may be **too impoverished to express a CrossEntropy-competitive loss at all**. CrossEntropy is `-log(p_target)`; the project's error-based formulation (`error = one_hot - probs`) does not even give clean access to `log(p)` on the target class. So the prior evidence proves the search fails *within a limited expressive space* — it does **not** prove the target is absent from a richer space.

This distinction controls the entire RL question (Section 16). **RL can only help you reach the achievable ceiling faster; it cannot raise the ceiling.** Only the operator set determines the ceiling. Therefore, before spending anything on richer search (RL or otherwise), we must first find out whether a richer operator set raises the ceiling *at all*. If it doesn't, no amount of search intelligence will help and the whole operator-expansion + RL direction is dead — cheaply. If it does, the expansion is validated and RL Method 4 becomes justified as a downstream efficiency gain.

**This is a hard gate.** Do not proceed to richer-operator work or any RL until Phase 0 has run and its result is recorded.

### 14.2 What to add, in strict priority order

Add operators in the following order, cheapest-and-most-likely-to-matter first. **Do not jump straight to Jacobians** — they are the most expensive and the least likely to be the missing ingredient.

**Tier 1 — direct probability access (add first; cheap; most likely to matter).** The prior formulation only exposed `error = one_hot - probs`. Give the tree access to the raw target-class probability and log-probability, so it can express CrossEntropy-like shapes:
- A terminal or unary op exposing `p_target` (the softmax probability of the correct class).
- `safe_log` applied to `p_target` directly (not just to error). This is the single most important addition: `-log(p_target)` *is* CrossEntropy, so if a competitive loss exists in reach, this is the operator most likely to unlock it.
- Consider exposing `logits` (pre-softmax) as an additional input signal, since some effective losses operate on margins rather than probabilities.

**Tier 2 — richer pointwise nonlinearities (add second; still cheap).** `pow` with a small integer/evolved exponent, reciprocal-with-clamp, `log1p`, `expm1`-with-clamp (a safer bounded cousin of the `exp` that previously caused instability). These broaden the expressible shapes without leaving pointwise-cheap territory.

**Tier 3 — structural / higher-order operators (add ONLY if Tier 1+2 raised the ceiling and you want to push further; expensive).** Gradient-norm or Jacobian-based terms (e.g. penalizing `‖∂loss/∂logits‖`), entropy of the prediction distribution, pairwise/batch-level terms (something like a divergence between predicted and target distributions across the batch). **Cost warning:** these are not pointwise. A Jacobian/gradient-norm term computed per batch, per individual, per generation can add a large constant factor to every evaluation and could cut your number of evaluations by 3–5×, worsening the fitness-noise bottleneck that everything else in this spec is fighting. Only enter Tier 3 with eyes open, and measure the per-evaluation cost before committing a full run.

### 14.3 Protocol

1. Take the existing anti-convergence GP (Section 6.4), on ONE architecture (use Arch-A/ShallowWide or the user's own SimpleCNN), on MNIST or Fashion-MNIST.
2. Run **two** short evolutions: one with the **baseline** operator set (Section 6.2), one with **baseline + Tier 1** operators. Same population size, generations, seed, epochs-per-individual, and the same fixed data split (Section 6.5) for both, so the only difference is the operator set.
3. The smoke-test filter (Section 15) is active in both runs.
4. Compare the **best achievable fitness** (the ceiling), not the average. We are asking "can the richer space express a better loss," which is a question about the top of the distribution.
5. If Tier 1 raises the ceiling meaningfully, optionally repeat adding Tier 2, then (cautiously, cost permitting) Tier 3.

### 14.4 Pass / fail criterion

- **PASS:** best-achievable accuracy with expanded operators is meaningfully higher than with the baseline set — ideally closing a visible fraction of the prior gap to CrossEntropy (recall prior gaps were 1.5–4%). Even shrinking the gap without closing it is a pass, because it demonstrates the ceiling is operator-limited and therefore movable. **Validate any apparent improvement across ≥5 seeds before calling it a pass** (Section 7, pitfall 5 — no single-seed claims, ever).
- **FAIL:** expanded operators produce no ceiling improvement beyond seed noise. Conclusion: the limiter is not expressiveness, richer search (RL) cannot help, and you proceed to the zoo study (Section 5) with the baseline operator set, framing the paper purely around architecture-specialization (Tiers 1/2 of Section 4), not around beating CrossEntropy.

Either outcome is useful and should be recorded in the paper. A FAIL result is a clean, citable justification for *why* the paper doesn't chase absolute improvement and why it doesn't use RL — turning a limitation into a defensible, evidence-based scoping decision.

### 14.5 Operator-expansion implementation notes

- Every new operator must ship with the same NaN/Inf-safe wrapping discipline as the existing set (clamp domains before `log`/`sqrt`/reciprocal; bound anything exp-like). The smoke-test filter (Section 15) is the safety net, but operators should still be individually safe.
- New terminals (`p_target`, `logits`) require extending the `EvolvedLossFunction.forward` signature so the compiled tree can receive them, not just `error`. This is a real (but small) change to the wrapper in Section 6.3 — the tree's input arity grows. Implement and unit-test this before running Phase 0.
- Keep max tree depth at 5 initially even with more operators; only raise it if Phase 0 clearly benefits from deeper expressions and the smoke-test filter is keeping evaluation cost under control.

---

## 15. The Smoke-Test / Early-Kill Filter (Always On)

### 15.1 What it is and where it comes from

This is the pragmatic, non-RL core of what the RL literature calls "predictive operator pruning" (the "Method 4" idea evaluated in Section 16). Instead of training a learned value function to predict whether an individual is viable, we use cheap deterministic checks to kill doomed individuals *before* spending a full training budget on them. It is **always on**, in every phase including Phase 0, because it strictly reclaims wasted compute and never changes which non-doomed loss wins.

### 15.2 The checks, cheapest first

Applied to each candidate individual before it receives its full `epochs_per_individual` training budget:

1. **Single-batch validity check (near-zero cost).** Run the compiled loss on exactly one batch. Kill the individual (assign fitness 0 / mark invalid) if: the loss is NaN or Inf; the loss is not finite after backward; the gradient w.r.t. logits is all-zero (degenerate tree that ignores its input — the `loss + 0.0*probs.sum()` guard keeps it differentiable but such a tree can't learn); or the loss is constant across different inputs.
2. **One-epoch early-kill (cheap).** Optionally, train for a single epoch and kill anything still at or below near-random accuracy (e.g. <15% on a 10-class MNIST task). A loss that hasn't moved off chance after a full epoch on easy data is not going to recover in the remaining budget; the compute is better spent elsewhere.

### 15.3 Why it matters more after Phase 0

With the baseline operator set, most doomed individuals were the familiar NaN/explosion cases. Once Phase 0 adds exotic operators (especially any Tier 3 Jacobian/higher-order terms), the space of ways an individual can be lethal-but-syntactically-valid grows — precisely the regime where a viability pre-filter pays off most. This is also the trigger condition for upgrading the filter from heuristic to learned (Section 16.3).

### 15.4 Implementation note

Fold the check into the fitness-evaluation entry point (the `train_and_evaluate` skeleton in Section 13), so it is impossible to run an evaluation without it. Log how many individuals per generation are killed by the filter and at which check — this number is itself a useful diagnostic (a spike in kills after adding an operator tier tells you that operator is unstable) and a reportable efficiency metric for the paper ("the pre-filter eliminated X% of evaluations at Y% of full cost").

---

## 16. RL Methods: What We Considered and What We Decided

We seriously evaluated reinforcement-learning approaches for guiding the GP search and maximizing fitness. This section records the reasoning so the decision is defensible to a reviewer and so a future implementer does not re-open a settled question without new evidence.

### 16.1 The four methods considered

1. **EMDP (evolution-as-MDP):** a learned neural policy takes the whole population state and outputs guided crossover/mutation steps, steering away from genetic dead-ends.
2. **Sub-program credit assignment (node-level RL):** Q-values are assigned to individual subtrees/nodes so reproduction protects effective blocks and mutates weak ones, instead of scoring only whole trees.
3. **Dynamic fitness-landscape shaping:** an RL agent detects stagnation and temporarily rewrites the fitness function to penalize the local optimum's traits, forcing dispersal.
4. **Predictive operator pruning:** an RL agent predicts whether a genetic operation will yield a viable child and filters out doomed ones before evaluation.

### 16.2 The decision, and the single principle behind it

**Adopt none of the heavy RL methods (1, 2, 3) up front. Adopt the cheap, non-RL core of Method 4 immediately (Section 15). Reconsider a learned Method 4 only if Phase 0 passes.**

The governing principle: **RL is a search-efficiency multiplier, not a search-space expander.** It can only help the population reach the achievable ceiling faster; it cannot raise the ceiling. The prior project's evidence suggests the ceiling itself may be the problem (operator expressiveness), which is a Phase-0 question, not an RL question. Spending on RL before knowing whether the ceiling is movable would be optimizing the speed of arrival at a destination we may not want.

Two further specific problems make the heavy methods poor fits for *this* project:

- **Noisy reward.** Every one of these methods trains on a reward derived from fitness, and our fitness has ±4.87%-scale noise (Section 10) against a target effect of ~1%. Training an RL policy (especially EMDP, Method 1) on a signal whose noise exceeds the effect largely teaches the policy to chase noise, and EMDP additionally needs many full evolution episodes just to train the policy — multiplying an already-tight compute budget.
- **Method 2 is in direct tension with this project's core hypothesis.** Node-level credit assignment assumes a subtree has stable, transferable value. But this project's entire thesis is that a loss's value is *architecture-specific and contextual*. If that hypothesis is true, per-node credit is not a stable learnable quantity — it differs per architecture and per surrounding context — and credit assignment breaks down exactly where we need it. The more right the architecture-conditioning hypothesis is, the *less* Method 2 can work. It is therefore rejected for this project and noted in Future Work only for the *architecture-fixed* regime, where its transferability assumption is at least satisfiable.

Method 3 (landscape shaping) is not adopted as RL because we already have a hardcoded equivalent (fitness sharing + novelty bonus + trivial penalty, Section 6.4) and, more importantly, the prior project showed that *escaping* the local optimum did not help — the escaped solutions lost by more. Making the escape smarter optimizes for something that already didn't translate to accuracy.

### 16.3 When (and only when) to upgrade Method 4 from heuristic to learned

The heuristic smoke-test filter (Section 15) captures most of Method 4's benefit for near-zero cost and complexity. Upgrade it to an actual learned viability predictor **only if all of the following hold:** (a) Phase 0 passed, so richer operators are in play and worth optimizing around; (b) the exotic operators produce lethal-but-valid offspring frequently enough that the heuristic filter's kill-rate logs show substantial wasted evaluation the simple checks miss; and (c) you have reclaimed enough compute from the heuristic filter to afford training the predictor. Absent all three, the heuristic filter stays as-is.

### 16.4 What goes in the paper about RL

State the decision honestly as a scoping choice grounded in evidence: the project deliberately does not employ RL-guided search because (i) the binding constraint is search-space expressiveness and fitness noise, not search intelligence, and RL addresses neither; (ii) node-level credit assignment is theoretically incompatible with the architecture-specialization hypothesis under test; and (iii) the one RL idea with a favorable cost/benefit profile (viability pre-filtering) is realized in a cheaper deterministic form. Position learned RL guidance — especially EMDP and a learned viability predictor at richer operator scales — as explicit Future Work contingent on greater compute and cleaner (lower-variance) fitness estimation. This mirrors the compute-scaling hypothesis (H4) framing: a stated, reasonable, falsifiable direction rather than a hidden gap.

---

**End of specification.**

This document is self-contained: Section 2 gives the full prior context, Sections 3–5 define the new experiment precisely enough to defend to a reviewer, Sections 6–8 give an implementer everything needed to start coding immediately, Sections 9–11 close the loop on how to interpret and write up whatever comes out (including the case where the answer is no), and Sections 14–16 add the operator-expansion gate, the always-on compute-reclaiming filter, and the reasoned RL decision that together determine what gets built and in what order. **The single most important sequencing rule in this document: Phase 0 (Section 14) is a hard gate that runs before everything else and decides whether the richer-operator/RL direction lives or dies.**
