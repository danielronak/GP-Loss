"""End-to-end CPU tests of the evolution loop, checkpointing and resume."""
import json
import os
import subprocess
import sys

import dill
import pytest

from src.architectures import TinyMLP
from src.config import EvolutionConfig
from src.data import make_synthetic, train_val_split
from src.evolve import CACHE_FILE, MANIFEST_FILE, RESULT_FILE, STATE_FILE, evolve_loss_for_architecture

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def data():
    d = make_synthetic(900, seed=3, noise=3.0)
    return train_val_split(d, 300)


POP = 10


def _cfg(**kw):
    base = dict(seed=5, operator_set="tier1", population_size=POP, generations=3,
                epochs_per_individual=2, batch_size=64, elitism=2, early_kill_val_size=100)
    base.update(kw)
    return EvolutionConfig(**base)


def _run(data, ckpt, cfg, **kw):
    train, val = data
    return evolve_loss_for_architecture(TinyMLP, train, val, cfg, ckpt, log=lambda *a: None, **kw)


def _fingerprint(res):
    return ([round(t["best_raw"], 6) for t in res.trajectory],
            [t["gen_best_tree"] for t in res.trajectory],
            [str(i) for i in res.final_population], res.hall_of_fame)


def test_full_run_writes_everything(data, tmp_path):
    res = _run(data, str(tmp_path / "run"), _cfg())
    assert res.completed and len(res.trajectory) == 3
    assert res.best_tree is not None and res.best_fitness > 10
    for f in (STATE_FILE, CACHE_FILE, MANIFEST_FILE, RESULT_FILE):
        assert os.path.exists(tmp_path / "run" / f)
    manifest = json.load(open(tmp_path / "run" / MANIFEST_FILE))
    assert manifest["checkpoint_dir"] == str(tmp_path / "run")
    assert all(ind.height <= 5 for ind in res.final_population)


def test_resume_after_crash_mid_generation_is_identical(data, tmp_path):
    ref = _run(data, str(tmp_path / "ref"), _cfg())
    ckpt = str(tmp_path / "crash")
    # "disconnect" after POP + 2 fresh evaluations (gen 1 evaluates all POP,
    # so this stops inside generation 2)
    part = _run(data, ckpt, _cfg(), _stop_after_evaluations=POP + 2)
    assert not part.completed and len(part.trajectory) == 1
    resumed = _run(data, ckpt, _cfg())
    assert resumed.completed
    assert _fingerprint(resumed) == _fingerprint(ref)


def test_resume_does_not_rerun_generation_or_duplicate_archive(data, tmp_path):
    ckpt = str(tmp_path / "run")
    _run(data, ckpt, _cfg(generations=2))
    state = dill.load(open(os.path.join(ckpt, STATE_FILE), "rb"))
    assert state["generation"] == 2 and len(state["archive"]) == 2 * POP
    # nothing left to do: resuming must not add a generation or archive entries
    res = _run(data, ckpt, _cfg(generations=2))
    state = dill.load(open(os.path.join(ckpt, STATE_FILE), "rb"))
    assert len(res.trajectory) == 2 and len(state["archive"]) == 2 * POP


def test_extending_generations_matches_a_longer_fresh_run(data, tmp_path):
    ref = _run(data, str(tmp_path / "ref"), _cfg(generations=3))
    ckpt = str(tmp_path / "ext")
    _run(data, ckpt, _cfg(generations=2))
    ext = _run(data, ckpt, _cfg(generations=3))
    assert _fingerprint(ext) == _fingerprint(ref)


def test_config_mismatch_on_resume_raises(data, tmp_path):
    ckpt = str(tmp_path / "run")
    _run(data, ckpt, _cfg(generations=1))
    with pytest.raises(ValueError):
        _run(data, ckpt, _cfg(generations=1, population_size=8))


def test_checkpoint_loads_in_fresh_process(data, tmp_path):
    """Colab restart: dill.load must work after only importing src."""
    ckpt = str(tmp_path / "run")
    _run(data, ckpt, _cfg(generations=1))
    code = ("import dill, src.gp_core; "
            f"s = dill.load(open({os.path.join(ckpt, STATE_FILE)!r}, 'rb')); "
            "print(len(s['population']))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(POP)


def test_baseline_arm_never_uses_p_target(data, tmp_path):
    res = _run(data, str(tmp_path / "b"), _cfg(operator_set="baseline", generations=2))
    assert all("p_target" not in t for t, _ in res.hall_of_fame)
