"""evolve_loss_for_architecture: the core reusable API (Spec §6.7).

Generalizes run_anti_convergence_evolution() from Optimized_loss_GP.ipynb:
the fitness model is a parameter (model_fn) instead of a hardcoded class.
Selection, elitism, crossover/mutation and the anti-convergence mechanisms
follow Phase 4b. These Phase 4b bugs are fixed:

  * Resume re-ran the checkpointed generation: the checkpoint was written
    after evaluation but before breeding, so the same population was
    re-scored and appended to the novelty archive twice. The state is now
    saved *after* breeding, so resume starts with the next generation.
  * A resumed run did `n_generations` *more* generations; now `generations`
    is the total.
  * RNG state was not checkpointed. Python `random` (drives the GP) is now
    saved; fitness evaluation only uses a fixed per-run torch seed.
  * Crossover/mutation called gp.cxOnePoint/gp.mutUniform directly and so
    bypassed the max-depth-5 staticLimit decorators. They now go through the
    decorated toolbox.mate/toolbox.mutate.
  * The raw-best tree could be dropped (elitism ranks by shared fitness).
    An all-time hall of fame by raw accuracy is kept.
  * Killed individuals no longer receive a novelty bonus (Phase 4b gave a
    crashed tree 0% + 2% bonus).

Additions: smoke-test filter with per-generation kill log, an evaluation
cache keyed by tree string (elites/unchanged children are not retrained),
common random numbers (same init and batch order for every individual in a
run), an optional wall-clock budget, and an atomic checkpoint every generation
plus an eval-cache flush after every evaluation so a Colab disconnect loses at
most one training run.
"""
import datetime
import json
import os
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import dill

from . import anti_convergence as ac
from .config import EvolutionConfig
from .gp_core import EvolvedLossFunction, build_pset, is_crossentropy_like, make_toolbox
from .smoke_test import KillLog
from .utils import EvalResult, train_and_evaluate

STATE_FILE = "state.dill"
CACHE_FILE = "eval_cache.json"
MANIFEST_FILE = "run_manifest.json"
RESULT_FILE = "result.json"

# Config keys that may change on resume (e.g. to extend a finished run).
_RESUMABLE_CHANGES = {"generations"}


@dataclass
class EvolvedLossResult:
    best_tree: Optional[str]
    best_fitness: Optional[float]              # raw validation accuracy (%)
    hall_of_fame: list                         # [(tree_str, raw_acc)], best first
    final_population: list
    trajectory: list
    kill_log: dict
    completed: bool
    checkpoint_dir: str
    config: dict = field(default_factory=dict)

    def summary(self):
        return {k: getattr(self, k) for k in
                ("best_tree", "best_fitness", "completed", "checkpoint_dir")}


class _StopRun(Exception):
    pass


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _atomic_write(path, data, binary=False):
    tmp = path + ".tmp"
    with open(tmp, "wb" if binary else "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, path + ".prev")
    os.replace(tmp, path)


def _load_state(ckpt_dir):
    for name in (STATE_FILE, STATE_FILE + ".prev"):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return dill.load(f)
            except Exception as e:  # corrupted partial write on Drive
                print(f"⚠️  could not read {path}: {e}")
    return None


def _load_cache(ckpt_dir):
    for name in (CACHE_FILE, CACHE_FILE + ".prev"):
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return {k: EvalResult.from_dict(v) for k, v in json.load(f).items()}
            except Exception as e:
                print(f"⚠️  could not read {path}: {e}")
    return {}


def _check_config(saved, current):
    diffs = {k: (saved.get(k), current.get(k)) for k in set(saved) | set(current)
             if saved.get(k) != current.get(k) and k not in _RESUMABLE_CHANGES}
    if diffs:
        raise ValueError(
            "checkpoint was created with a different config; use a new checkpoint_dir "
            f"or resume=False. Differences (saved, current): {diffs}")


def evolve_loss_for_architecture(
    model_fn: Callable,
    train_data,
    val_data,
    config: EvolutionConfig,
    checkpoint_dir: str,
    device=None,
    resume: bool = True,
    run_id: Optional[str] = None,
    phase: Optional[str] = None,
    time_budget_minutes: Optional[float] = None,
    verbose: bool = False,
    log: Callable = print,
    _stop_after_evaluations: Optional[int] = None,
) -> EvolvedLossResult:
    """Run one anti-convergence GP evolution with model_fn() as the fitness model.

    train_data / val_data are data.TensorData (use data.tensors_from_loader to
    convert a DataLoader). Fitness = accuracy (%) on val_data. Keep a separate
    test set for reported numbers. Re-calling with the same checkpoint_dir
    resumes where the run stopped. If the time budget runs out, the result has
    completed=False; call again to continue.
    """
    ckpt_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    run_id = run_id or os.path.basename(ckpt_dir.rstrip("/"))
    deadline = time.time() + 60 * time_budget_minutes if time_budget_minutes else None
    log(f"📁 Checkpoint dir (absolute): {ckpt_dir}")

    pset = build_pset(config.operator_set)
    toolbox = make_toolbox(pset, config.tree_max_depth, config.init_min_depth, config.init_max_depth)
    trivial = ac.trivial_set(pset)
    ek_data = val_data.take(config.early_kill_val_size, seed=0) if config.early_kill else None

    state = _load_state(ckpt_dir) if resume else None
    manifest_path = os.path.join(ckpt_dir, MANIFEST_FILE)
    if state is not None:
        _check_config(state["config"], config.to_dict())
        random.setstate(state["py_random"])
        population = state["population"]
        log(f"▶️  Resuming {run_id} at generation {state['generation'] + 1}/{config.generations}")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest.setdefault("resumed_at", []).append(datetime.datetime.now().isoformat())
            manifest["config"] = config.to_dict()
            _atomic_write(manifest_path, json.dumps(manifest, indent=2))
    else:
        random.seed(config.seed)
        population = toolbox.population(n=config.population_size)
        state = {
            "generation": 0, "population": population, "last_evaluated": None,
            "archive": [], "hall_of_fame": {}, "trajectory": [], "kill_log": KillLog(),
            "config": config.to_dict(), "py_random": random.getstate(),
        }
        manifest = {
            "run_id": run_id, "checkpoint_dir": ckpt_dir, "phase": phase,
            "operator_set": config.operator_set, "config": config.to_dict(),
            "started_at": datetime.datetime.now().isoformat(),
            "git_commit": _git_commit(), "device": str(device or train_data.x.device),
        }
        _atomic_write(manifest_path, json.dumps(manifest, indent=2))
        _atomic_write(os.path.join(ckpt_dir, STATE_FILE), dill.dumps(state), binary=True)
        log(f"🆕 New population ({config.population_size} trees, operator_set={config.operator_set})")

    cache = _load_cache(ckpt_dir) if resume else {}
    archive, hof, kill_log = state["archive"], state["hall_of_fame"], state["kill_log"]
    trajectory = state["trajectory"]
    n_fresh = 0

    def evaluate(ind):
        nonlocal n_fresh
        if deadline and time.time() > deadline:
            raise _StopRun("time budget reached")
        if _stop_after_evaluations is not None and n_fresh >= _stop_after_evaluations:
            raise _StopRun("evaluation limit reached")
        py_state = random.getstate()  # evaluation must never perturb the GP RNG
        res = train_and_evaluate(
            model_fn, EvolvedLossFunction(ind, pset), train_data, val_data,
            epochs=config.epochs_per_individual, device=device, lr=config.lr,
            batch_size=config.batch_size, seed=config.eval_seed,
            early_kill=config.early_kill, early_kill_data=ek_data,
            early_kill_chance_multiple=config.early_kill_chance_multiple)
        random.setstate(py_state)
        n_fresh += 1
        return res

    completed = True
    try:
        while state["generation"] < config.generations:
            gen = state["generation"] + 1
            t_gen = time.time()
            fresh, hits = 0, 0
            for i, ind in enumerate(population):
                if ind.fitness.valid:  # elite carried over with its fitness
                    continue
                key = str(ind)
                if key in cache:
                    res, hits = cache[key], hits + 1
                else:
                    res = evaluate(ind)
                    cache[key] = res
                    fresh += 1
                    _atomic_write(os.path.join(ckpt_dir, CACHE_FILE),
                                  json.dumps({k: v.to_dict() for k, v in cache.items()}))
                    if res.killed:
                        kill_log.record(gen, res.kill_reason)
                ind.killed = res.killed
                ind.raw_fitness = res.accuracy
                if res.killed:
                    ind.fitness.values = (0.0,)
                else:
                    novelty = ac.calculate_novelty_bonus(ind, archive, config.novelty_bonus_schedule)
                    mult = ac.apply_trivial_penalty(ind, trivial, config.trivial_tree_penalty)
                    ind.fitness.values = ((res.accuracy + novelty) * mult,)
                if verbose:
                    tag = f"KILLED({res.kill_reason})" if res.killed else f"raw={res.accuracy:.2f}%"
                    log(f"  ind {i + 1:2d}: {tag:28s} fit={ind.fitness.values[0]:.2f} | {key[:70]}")

            ac.apply_fitness_sharing(population, sigma=config.fitness_sharing_sigma)
            population.sort(key=lambda x: x.shared_fitness, reverse=True)
            diversity = ac.calculate_population_diversity(population)
            archive.extend(toolbox.clone(ind) for ind in population)
            for ind in population:
                if not ind.killed and ind.raw_fitness > hof.get(str(ind), -1.0):
                    hof[str(ind)] = ind.raw_fitness

            raws = [ind.raw_fitness for ind in population]
            gen_best = max(population, key=lambda x: x.raw_fitness)
            hof_best = max(hof.items(), key=lambda kv: kv[1]) if hof else (None, None)
            mut_rate = ac.adaptive_mutation_rate(
                diversity, config.diversity_low_threshold, config.diversity_mid_threshold,
                config.mutation_rate_high, config.mutation_rate_mid, config.mutation_rate_base)
            entry = {
                "generation": gen,
                "best_raw": max(raws), "mean_raw": sum(raws) / len(raws),
                "best_shared": max(ind.shared_fitness for ind in population),
                "mean_shared": sum(ind.shared_fitness for ind in population) / len(population),
                "diversity": diversity,
                "unique_structures": len({ac.get_tree_structure_hash(i) for i in population}),
                "n_evaluated": fresh, "cache_hits": hits,
                "kills": kill_log.counts(gen), "n_killed": sum(1 for i in population if i.killed),
                "crossentropy_like": sum(1 for i in population if is_crossentropy_like(i)),
                "next_mutation_rate": mut_rate,
                "gen_best_tree": str(gen_best), "hof_best_tree": hof_best[0],
                "hof_best_raw": hof_best[1], "seconds": time.time() - t_gen,
            }
            trajectory.append(entry)
            kills = ", ".join(f"{k}={v}" for k, v in entry["kills"].items()) or "none"
            log(f"gen {gen:2d}/{config.generations} | best {entry['best_raw']:.2f}% "
                f"mean {entry['mean_raw']:.2f}% | div {diversity:.3f} | "
                f"new {fresh} cached {hits} | kills: {kills} | {entry['seconds']:.0f}s | "
                f"{str(gen_best)[:60]}")

            # Breed the next generation (also after the last one, so a finished
            # run can be extended with more generations deterministically).
            state["last_evaluated"] = [toolbox.clone(ind) for ind in population]
            population = _breed(population, config, mut_rate, toolbox)
            state.update(generation=gen, population=population, py_random=random.getstate())
            _atomic_write(os.path.join(ckpt_dir, STATE_FILE), dill.dumps(state), binary=True)
    except _StopRun as e:
        completed = False
        log(f"⏸️  Stopped early ({e}) during generation {state['generation'] + 1}; "
            "re-run the same call to resume.")

    ranked = sorted(hof.items(), key=lambda kv: kv[1], reverse=True)
    result = EvolvedLossResult(
        best_tree=ranked[0][0] if ranked else None,
        best_fitness=ranked[0][1] if ranked else None,
        hall_of_fame=ranked,
        final_population=state["last_evaluated"] or [],
        trajectory=trajectory,
        kill_log={str(g): dict(c) for g, c in kill_log.by_generation.items()},
        completed=completed and state["generation"] >= config.generations,
        checkpoint_dir=ckpt_dir,
        config=config.to_dict(),
    )
    if result.completed:
        payload = {
            "run_id": run_id, "phase": phase, "operator_set": config.operator_set,
            "best_tree": result.best_tree, "best_fitness": result.best_fitness,
            "hall_of_fame_top20": ranked[:20], "trajectory": trajectory,
            "kill_log": result.kill_log, "kill_totals": kill_log.total(),
            "n_unique_evaluated": len(cache), "config": config.to_dict(),
            "finished_at": datetime.datetime.now().isoformat(),
        }
        _atomic_write(os.path.join(ckpt_dir, RESULT_FILE), json.dumps(payload, indent=2))
        log(f"🏁 {run_id}: best val acc {result.best_fitness:.2f}% | {result.best_tree}")
    return result


def _breed(population, config, mut_rate, toolbox):
    """Phase 4b breeding: elitism by shared fitness + shared-fitness tournaments."""
    next_gen = [toolbox.clone(ind) for ind in population[:config.elitism]]
    k = min(config.tournament_size, len(population))
    while len(next_gen) < config.population_size:
        p1 = max(random.sample(population, k), key=lambda x: x.shared_fitness)
        p2 = max(random.sample(population, k), key=lambda x: x.shared_fitness)
        child = toolbox.clone(p1)
        if random.random() < config.crossover_rate:
            child, _ = toolbox.mate(child, toolbox.clone(p2))
        if random.random() < mut_rate:
            (child,) = toolbox.mutate(child)
        del child.fitness.values
        next_gen.append(child)
    return next_gen
