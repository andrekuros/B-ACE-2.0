"""Shot log, missile-warning switch, paper studies and the lean PPO."""

from __future__ import annotations

import json

import pytest

from bace.experiment import run_experiment
from bace.studies import CONDITIONS, pilot, scenario, summarize


def _duels(env=None, n=6):
    cfgs = [
        scenario("duel", pilot("ref_balanced"), pilot("ref_balanced"), s, env=env)
        for s in range(100, 100 + n)
    ]
    return run_experiment(cfgs, max_parallel=4)


def test_shot_log_covers_both_teams():
    outs = _duels()
    shots = [s for o in outs for s in o["shots"]]
    assert {s["team"] for s in shots} == {0, 1}
    assert sum(o["missiles_fired"] for o in outs) == sum(1 for s in shots if s["team"] == 0)
    for s in shots:
        assert 0 < s["range_nm"] <= 1.2 * s["rmax_nm"]
        assert s["result"]
        if s["result"] == "hit":
            assert s["tof_s"] > 0 and s["f_pole_nm"] >= 0


def test_missile_warning_switch_changes_outcomes():
    on = _duels(n=12)
    off = _duels(env=CONDITIONS["no_warning"], n=12)
    hits_on = sum(o["missile_hits"] for o in on)
    hits_off = sum(o["missile_hits"] for o in off)
    assert hits_off > hits_on, "pilots that cannot see missiles coming should be hit more"


def test_summarize_reports_cis_and_raw_shots():
    sm = summarize(_duels(), raw=True)
    lo, mid, hi = sm["kills"][1], sm["kills"][0], sm["kills"][2]
    assert lo <= mid <= hi
    assert sm["raw_shots"] and len(sm["raw_shots"][0]) == 10
    assert 0.0 <= sm["blue_shots"]["hit_rate"] <= 1.0


def test_study_runs_and_writes_json(tmp_path):
    from bace.studies import run

    res = run("defend", tmp_path, n=2)
    data = json.loads((tmp_path / "defend.json").read_text())
    assert data["study"] == "defend"
    assert "ref_balanced|duck" in res["1v1"]


torch = pytest.importorskip("torch")


def test_ppo_trains_checkpoints_and_evaluates(tmp_path):
    from bace.ppo import PPOConfig, evaluate_vec, load_policy, select_checkpoint, train

    cfg = PPOConfig(
        opponents=["duck", "ref_balanced"],
        envs_per_opponent=2,
        steps=2 * 2 * 64 * 4,
        rollout=64,
        minibatch=128,
        hidden=32,
        save_every=2,
    )
    res = train(cfg, out=tmp_path, log=lambda _m: None)
    assert (tmp_path / "policy.pt").is_file()
    assert len(list((tmp_path / "ckpt").glob("*.pt"))) >= 2
    assert res["env_steps"] > 0 and res["history"]
    ev = evaluate_vec(res["checkpoint"], "duck", [5, 6, 7])
    assert ev["n"] == 3
    sel = select_checkpoint(tmp_path, opponents=["duck"], n=2)
    assert (tmp_path / "best.pt").is_file() and sel["best"]
    act = load_policy(tmp_path / "best.pt")
    import numpy as np

    a = act(np.zeros(res["checkpoint"]["obs_dim"]))
    assert a.shape == (3,)
