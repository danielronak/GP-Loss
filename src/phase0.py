"""Phase 0: operator-expansion ceiling test (Spec §14, Kickoff §4). HARD GATE.

Design (pre-registered here, before any data is seen):

  * One architecture (SimpleCNN), Fashion-MNIST, one fixed train/val split.
  * Two arms that differ only in the operator set: `baseline` vs `tier1`
    (baseline + p_target terminal). Each arm gets R independent evolution
    runs (search seeds). Arm runs with the same seed share the eval seed
    (same inits and batch order).
  * Fitness = val accuracy on a fixed subset (cheap). The val-best tree of
    every run (its "ceiling") is then retrained from scratch on the full
    training split and scored on the untouched official test set, over S
    training seeds that are shared by every loss (paired design). CE and
    MSE (square(error)) are retrained on the same seeds as reference rows.
  * Statistic: run-level score = mean test accuracy over the S seeds.
    delta = mean(tier1 runs) - mean(baseline runs). Search randomness is the
    dominant noise source, so the run (not the training seed) is the unit
    of replication; an exact one-sided permutation test over runs.
  * PASS iff p < alpha AND delta >= max(min_effect_pp,
    min_gap_fraction * (CE - baseline)). "Beyond noise" and "closes a
    meaningful fraction of the gap to CrossEntropy" (Spec §14.4).
  * With R = 5 per arm the smallest attainable p is 1/252, so the test can
    reach p < 0.05. With R = 3 it cannot (min p = 1/20 = 0.05), which is
    why this protocol uses 5 cheap runs per arm rather than 3.
"""
import csv
import json
import math
import os
import time

import numpy as np
import torch

from . import stats
from .architectures import MODEL_REGISTRY
from .config import EvolutionConfig, load_yaml
from .data import load_torchvision, train_val_split
from .evolve import RESULT_FILE, evolve_loss_for_architecture
from .gp_core import EvolvedLossFunction, build_pset, is_crossentropy_like, tree_from_string
from .utils import train_and_evaluate

FINAL_CSV = "final_eval_raw.csv"
RUNS_CSV = "runs.csv"
RESULT_JSON = "phase0_result.json"
FINAL_FIELDS = ["loss_id", "arm", "search_seed", "tree", "eval_seed", "test_acc",
                "killed", "kill_reason", "seconds"]


def load_config(path_or_dict):
    cfg = load_yaml(path_or_dict) if isinstance(path_or_dict, str) else dict(path_or_dict)
    return cfg


def run_dir(output_dir, arm, seed):
    return os.path.join(output_dir, "runs", f"{arm}_s{seed}")


def run_order(cfg):
    """Interleave arms (b0, t0, b1, t1, ...) so a partial budget stays balanced."""
    return [(arm, s) for s in cfg["search_seeds"] for arm in cfg["arms"]]


def evolution_config(cfg, arm, seed):
    return EvolutionConfig.from_dict({**cfg["evolution"], "seed": seed,
                                      "operator_set": arm, "eval_seed": seed})


def prepare_data(cfg, data_root, device):
    d = cfg["data"]
    train_full, test = load_torchvision(cfg["dataset"], data_root, device)
    train, val = train_val_split(train_full, d["val_size"], d["split_seed"])
    return {
        "train": train, "val": val, "test": test,
        "fit_train": train.take(d["fitness_train_size"], seed=d["split_seed"]),
        "fit_val": val.take(d["fitness_val_size"], seed=d["split_seed"]),
    }


def _device(device):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("⚠️  No GPU found: Runtime > Change runtime type > T4 GPU.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Step 1: timing benchmark
# ---------------------------------------------------------------------------
def benchmark(cfg, data, device=None, assumed_fresh_fraction=0.7):
    """Time one fitness evaluation and one final-eval training; print a budget."""
    device = _device(device)
    model_fn = MODEL_REGISTRY[cfg["architecture"]]
    ev = evolution_config(cfg, "baseline", 0)
    pset = build_pset("baseline")
    mse = EvolvedLossFunction(tree_from_string("square(error)", pset), pset)
    train_and_evaluate(model_fn, mse, data["fit_train"].take(512), data["fit_val"].take(512),
                       epochs=1, device=device, early_kill=False)  # warm-up (cudnn, allocator)
    t = time.time()
    train_and_evaluate(model_fn, mse, data["fit_train"], data["fit_val"],
                       epochs=ev.epochs_per_individual, device=device, lr=ev.lr,
                       batch_size=ev.batch_size, seed=0, early_kill=False)
    t_fit = time.time() - t
    fe = cfg["final_eval"]
    t = time.time()
    train_and_evaluate(model_fn, torch.nn.CrossEntropyLoss(), data["train"], data["test"],
                       epochs=1, device=device, batch_size=ev.batch_size, seed=0, early_kill=False)
    t_final = (time.time() - t) * fe["epochs"]
    n_runs = len(cfg["arms"]) * len(cfg["search_seeds"])
    fit_evals = n_runs * ev.population_size * ev.generations * assumed_fresh_fraction
    n_final = (n_runs + len(fe["references"])) * len(fe["seeds"])
    est = {"seconds_per_fitness_eval": round(t_fit, 2),
           "seconds_per_final_training": round(t_final, 1),
           "est_evolution_hours": round(fit_evals * t_fit / 3600, 2),
           "est_final_eval_hours": round(n_final * t_final / 3600, 2)}
    est["est_total_hours"] = round(est["est_evolution_hours"] + est["est_final_eval_hours"], 2)
    print(f"⏱️  {est}")
    print(f"   ({n_runs} runs x {ev.population_size} pop x {ev.generations} gen, "
          f"assuming {assumed_fresh_fraction:.0%} of slots need fresh training)")
    return est


# ---------------------------------------------------------------------------
# Step 2: evolution runs (resumable; re-run after a disconnect)
# ---------------------------------------------------------------------------
def run_evolutions(cfg, data, output_dir, device=None, time_budget_minutes=None, verbose=False):
    device = _device(device)
    model_fn = MODEL_REGISTRY[cfg["architecture"]]
    deadline = time.time() + 60 * time_budget_minutes if time_budget_minutes else None
    for arm, seed in run_order(cfg):
        rdir = run_dir(output_dir, arm, seed)
        if os.path.exists(os.path.join(rdir, RESULT_FILE)):
            print(f"✔️  {arm}_s{seed} already complete")
            continue
        remaining = None
        if deadline:
            remaining = (deadline - time.time()) / 60
            if remaining <= 0:
                print("⏸️  Session time budget used up; re-run this cell to continue.")
                return False
        print(f"\n===== {arm}_s{seed} =====")
        res = evolve_loss_for_architecture(
            model_fn, data["fit_train"], data["fit_val"], evolution_config(cfg, arm, seed),
            checkpoint_dir=rdir, device=device, resume=True, run_id=f"phase0_{arm}_s{seed}",
            phase="phase0", time_budget_minutes=remaining, verbose=verbose)
        if not res.completed:
            return False
    write_runs_csv(cfg, output_dir)
    print("\n✅ All evolution runs complete.")
    return True


def load_runs(cfg, output_dir):
    runs = []
    for arm, seed in run_order(cfg):
        path = os.path.join(run_dir(output_dir, arm, seed), RESULT_FILE)
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            r["arm"], r["search_seed"] = arm, seed
            runs.append(r)
    return runs


def write_runs_csv(cfg, output_dir):
    rows = []
    for r in load_runs(cfg, output_dir):
        traj = r["trajectory"]
        rows.append({
            "arm": r["arm"], "search_seed": r["search_seed"], "best_tree": r["best_tree"],
            "best_val_acc": r["best_fitness"],
            "crossentropy_like": is_crossentropy_like(r["best_tree"] or ""),
            "n_unique_evaluated": r["n_unique_evaluated"],
            "kills_total": sum(r["kill_totals"].values()),
            "kill_totals": json.dumps(r["kill_totals"]),
            "final_diversity": traj[-1]["diversity"] if traj else None,
            "minutes": round(sum(t["seconds"] for t in traj) / 60, 1),
        })
    path = os.path.join(output_dir, RUNS_CSV)
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Step 3: final held-out evaluation (resumable)
# ---------------------------------------------------------------------------
def final_eval_jobs(cfg, output_dir):
    jobs = [("ref_" + name, "reference", "", spec) for name, spec in cfg["final_eval"]["references"].items()]
    for r in load_runs(cfg, output_dir):
        jobs.append((f"{r['arm']}_s{r['search_seed']}", r["arm"], r["search_seed"], r["best_tree"]))
    return jobs


def _loss_for(spec, arm):
    if spec == "ce":
        return torch.nn.CrossEntropyLoss()
    pset = build_pset("baseline" if arm == "reference" else arm)
    return EvolvedLossFunction(tree_from_string(spec, pset), pset)


def _read_final(output_dir):
    path = os.path.join(output_dir, FINAL_CSV)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_final_eval(cfg, data, output_dir, device=None, time_budget_minutes=None):
    device = _device(device)
    model_fn = MODEL_REGISTRY[cfg["architecture"]]
    fe, ev = cfg["final_eval"], cfg["evolution"]
    deadline = time.time() + 60 * time_budget_minutes if time_budget_minutes else None
    path = os.path.join(output_dir, FINAL_CSV)
    done = {(r["loss_id"], int(r["eval_seed"])) for r in _read_final(output_dir)}
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FINAL_FIELDS).writeheader()
    jobs = final_eval_jobs(cfg, output_dir)
    todo = [(j, s) for j in jobs for s in fe["seeds"] if (j[0], s) not in done]
    print(f"Final eval: {len(todo)} trainings to go ({len(jobs)} losses x {len(fe['seeds'])} seeds)")
    for (loss_id, arm, search_seed, spec), seed in todo:
        if deadline and time.time() > deadline:
            print("⏸️  Session time budget used up; re-run this cell to continue.")
            return False
        res = train_and_evaluate(model_fn, _loss_for(spec, arm), data["train"], data["test"],
                                 epochs=fe["epochs"], device=device, lr=ev.get("lr", 1e-3),
                                 batch_size=ev.get("batch_size", 128), seed=seed, early_kill=False)
        row = {"loss_id": loss_id, "arm": arm, "search_seed": search_seed, "tree": spec,
               "eval_seed": seed, "test_acc": round(res.accuracy, 4), "killed": res.killed,
               "kill_reason": res.kill_reason or "", "seconds": round(res.seconds, 1)}
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FINAL_FIELDS).writerow(row)
            f.flush()
            os.fsync(f.fileno())
        print(f"  {loss_id:14s} seed {seed}: {res.accuracy:.2f}%  ({res.seconds:.0f}s)")
    print("✅ Final evaluation complete.")
    return True


# ---------------------------------------------------------------------------
# Step 4: analysis and PASS/FAIL (CPU only; can run anywhere from the CSVs)
# ---------------------------------------------------------------------------
def analyze(cfg, final_rows, runs=None):
    dec = cfg["decision"]
    base_arm, exp_arm = cfg["arms"]
    by_loss = {}
    for r in final_rows:
        by_loss.setdefault(r["loss_id"], {})[int(r["eval_seed"])] = float(r["test_acc"])
    seeds = list(cfg["final_eval"]["seeds"])

    def scores(loss_id):
        d = by_loss.get(loss_id, {})
        return [d[s] for s in seeds if s in d]

    def arm_runs(arm):
        out = {}
        for s in cfg["search_seeds"]:
            sc = scores(f"{arm}_s{s}")
            if len(sc) == len(seeds):
                out[s] = sc
        return out

    runs_b, runs_e = arm_runs(base_arm), arm_runs(exp_arm)
    run_b = [float(np.mean(v)) for v in runs_b.values()]
    run_e = [float(np.mean(v)) for v in runs_e.values()]
    refs = {name: scores("ref_" + name) for name in cfg["final_eval"]["references"]}
    ref_means = {k: (float(np.mean(v)) if v else None) for k, v in refs.items()}

    out = {
        "protocol": {"dataset": cfg["dataset"], "architecture": cfg["architecture"],
                     "arms": cfg["arms"], "search_seeds": cfg["search_seeds"],
                     "final_eval_seeds": seeds, "final_eval_epochs": cfg["final_eval"]["epochs"],
                     "evolution": cfg["evolution"], "decision": dec},
        "n_complete_runs": {base_arm: len(run_b), exp_arm: len(run_e)},
        "run_scores": {base_arm: dict(zip(map(str, runs_b), run_b)),
                       exp_arm: dict(zip(map(str, runs_e), run_e))},
        "reference_means": ref_means,
        "reference_std": {k: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for k, v in refs.items()},
    }
    if runs:
        out["best_trees"] = {f"{r['arm']}_s{r['search_seed']}": r["best_tree"] for r in runs}
        out["crossentropy_like_best_trees"] = {
            arm: sum(1 for r in runs if r["arm"] == arm and is_crossentropy_like(r["best_tree"] or ""))
            for arm in cfg["arms"]}
    if len(run_b) < 2 or len(run_e) < 2:
        out["decision"] = "INCOMPLETE"
        return out

    delta, p = stats.exact_permutation_test(run_e, run_b, alternative="greater")
    ci = stats.hierarchical_bootstrap_diff(list(runs_e.values()), list(runs_b.values()))
    ce = ref_means.get("CrossEntropy")
    gap = (ce - float(np.mean(run_b))) if ce is not None else None
    required = max(dec["min_effect_pp"], dec["min_gap_fraction"] * max(gap or 0.0, 0.0))
    complete = len(run_b) == len(run_e) == len(cfg["search_seeds"])
    passed = p < dec["alpha"] and delta >= required
    out.update({
        "mean_run_score": {base_arm: float(np.mean(run_b)), exp_arm: float(np.mean(run_e))},
        "ceiling_max_run": {base_arm: max(run_b), exp_arm: max(run_e)},
        "delta_pp": float(delta), "delta_ci95": ci, "p_permutation_one_sided": float(p),
        "min_attainable_p": 1 / math.comb(len(run_b) + len(run_e), len(run_e)),
        "gap_ce_minus_baseline_pp": gap, "required_delta_pp": required,
        "fraction_of_gap_closed": (delta / gap) if gap and gap > 0 else None,
        "decision": ("PASS" if passed else "FAIL") if complete else
                    ("PROVISIONAL_PASS" if passed else "PROVISIONAL_FAIL"),
        "no_headroom": gap is not None and gap < dec["min_effect_pp"],
    })
    return out


def analyze_dir(cfg, output_dir, write=True):
    out = analyze(cfg, _read_final(output_dir), load_runs(cfg, output_dir))
    if write:
        with open(os.path.join(output_dir, RESULT_JSON), "w") as f:
            json.dump(out, f, indent=2)
    print_summary(out)
    return out


def print_summary(out):
    print("\n" + "=" * 70)
    print(f"PHASE 0 DECISION: {out['decision']}")
    print("=" * 70)
    for k in ("n_complete_runs", "mean_run_score", "ceiling_max_run", "reference_means",
              "delta_pp", "delta_ci95", "p_permutation_one_sided", "min_attainable_p",
              "gap_ce_minus_baseline_pp", "required_delta_pp", "fraction_of_gap_closed",
              "no_headroom", "crossentropy_like_best_trees"):
        if k in out:
            print(f"{k:28s} {out[k]}")
    print("\n--- copy everything between the lines back to Claude ---")
    print(json.dumps({k: v for k, v in out.items() if k != "protocol"}, default=str))
    print("--- end ---")
