"""Experiment API, recipes, discrete actions, and make_env."""

from __future__ import annotations

import numpy as np

from bace import BaceEnv, make_env, run_experiment
from bace.experiment import run_fsm, run_wez


def test_run_experiment_roundtrip():
    configs = [
        {
            "env": {"max_cycles": 8, "seed": 1},
            "blue": {"num_agents": 1, "behavior": "duck"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    ]
    out = run_experiment(configs, max_parallel=2)
    assert len(out) == 1
    assert "end" in out[0]
    assert "blue_kills" in out[0]
    assert "missiles_fired" in out[0]
    assert int(out[0]["steps"]) > 0


def test_wez_one_cell_closer_hits_more():
    report = run_wez(smoke=True, max_parallel=4)
    cells = {c["range_nm"]: c for c in report["cells"]}
    assert 16.0 in cells and 40.0 in cells
    assert cells[16.0]["hit_rate"] >= cells[40.0]["hit_rate"]


def _genome_l1(a: dict, b: dict) -> float:
    return (
        abs(float(a.get("d_shot", 0)) - float(b.get("d_shot", 0)))
        + abs(float(a.get("l_crank", 0)) - float(b.get("l_crank", 0)))
        + abs(float(a.get("l_break", 0)) - float(b.get("l_break", 0)))
    )


def test_fsm_one_generation():
    report = run_fsm(smoke=True, max_parallel=4)
    assert report["history"]
    assert report["elites"]
    assert "frozen_fitness" in report["history"][0]
    assert "train_fitness" in report["history"][0]
    assert report["last_generation"]
    row = report["last_generation"][0]
    assert "mean_kills" in row and "mean_deaths" in row


def test_fsm_elites_distinct_when_multiple():
    report = run_fsm(smoke=True, max_parallel=4)
    elites = report["elites"]
    if len(elites) < 2:
        return
    for i, a in enumerate(elites):
        for b in elites[i + 1 :]:
            assert _genome_l1(a["genome"], b["genome"]) > 0.0


def test_fsm_frozen_archive_nondecreasing():
    report = run_fsm(
        params={
            "pop": 6,
            "generations": 3,
            "episodes": 4,
            "max_cycles": 40,
            "num_agents": 1,
            "eval_agents": 0,
            "seed": 1,
            "pool_interval": 2,
        },
        max_parallel=4,
        smoke=False,
        profile="default",
    )
    hist = report["history"]
    assert len(hist) == 3
    for a, b in zip(hist, hist[1:]):
        assert b["frozen_fitness"] + 1e-9 >= a["frozen_fitness"]


def test_make_env_factory():
    env = make_env(opponent="duck", agents=1, max_cycles=10, seed=2)
    obs, _ = env.reset(seed=2)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    env.step(actions)
    env.close()
    env2 = make_env(opponent="baseline", agents=2, max_cycles=10)
    obs, _ = env2.reset()
    assert len(obs) == 2
    env2.close()


def test_make_env_4v4_and_pettingzoo():
    from pettingzoo.test import parallel_api_test

    env = make_env(opponent="duck", agents=4, max_cycles=12, seed=4)
    obs, _ = env.reset(seed=4)
    assert len(obs) == 4
    assert env._obs_size == 9 + 13 * 4 + 6 * 3
    env.close()
    env_b = BaceEnv(
        {
            "env": {"max_cycles": 16, "seed": 2, "action_repeat": 5},
            "blue": {"num_agents": 4, "behavior": "external"},
            "red": {"num_agents": 4, "behavior": "duck"},
        }
    )
    parallel_api_test(env_b, num_cycles=6)
    env_b.close()


def test_4v4_baseline_vs_duck_terminates():
    out = run_experiment(
        [
            {
                "env": {"max_cycles": 20, "seed": 5, "action_repeat": 10},
                "blue": {"num_agents": 4, "behavior": "baseline1"},
                "red": {"num_agents": 4, "behavior": "duck"},
            }
        ],
        max_parallel=2,
    )
    assert len(out) == 1
    assert int(out[0]["steps"]) > 0


def test_wez_traces_and_fits_on_smoke():
    report = run_wez(smoke=True, max_parallel=4)
    assert "fits" in report
    assert "traces" in report
    assert any(abs(t["range_nm"] - 16.0) < 0.4 or abs(t["range_nm"] - 40.0) < 0.4 for t in report["traces"])


def test_fsm_mission_fitness_fields():
    report = run_fsm(smoke=True, max_parallel=4)
    row = report["last_generation"][0]
    assert "mean_shots" in row
    assert "mission_rate" in row


def test_marl_smoke_ppo():
    from bace.experiment import run_marl

    report = run_marl(steps=40, seed=0, opponent="duck", agents=1, algo="ppo", profile="smoke")
    if report.get("skipped"):
        import pytest

        pytest.skip(str(report.get("reason", "tianshou missing")))
    assert report.get("algo") in {"ppo", "ippo"}
    assert "eval" in report
    assert report.get("agents") == 1
    assert "eval_fire_once_16nm" in report or report.get("skipped")


def test_marl_selfplay_smoke():
    from bace.marl import run_selfplay

    report = run_selfplay(profile="smoke")
    if report.get("jobs") and report["jobs"][0].get("skipped"):
        import pytest

        pytest.skip(str(report["jobs"][0].get("reason", "tianshou missing")))
    assert report.get("recipe") == "marl_selfplay"
    assert report.get("self_play") is True
    jobs = report["jobs"]
    assert len(jobs) == 4
    modes = {(j.get("algo"), j.get("sp_mode")) for j in jobs}
    assert ("ippo", "mixed") in modes
    assert ("ippo", "pfsp") in modes
    assert ("mappo", "mixed") in modes
    assert ("maddpg", "mixed") in modes
    for job in jobs:
        if job.get("skipped"):
            continue
        assert job.get("self_play") is True
        assert job.get("agents") == 2
        assert "eval" in job
        assert "eval_vs_fsm" in job


def test_make_benchmarl_env_factory():
    from bace.experiment import make_benchmarl_env

    env = make_benchmarl_env(opponent="duck", agents=1, max_cycles=4, seed=1)
    assert env.possible_agents == ["agent_0"]
    env.close()


def test_bench_parallel_native():
    import json

    from bace._native import bench_parallel_py

    raw = json.loads(bench_parallel_py(None, 2, 8))
    assert raw["n_envs"] == 2
    assert raw["decision_hz"] > 0
    assert raw["realtime_factor"] > 0


def test_greedy_fire_from_marginal():
    from bace.marl import greedy_from_probs

    p = np.zeros(50, dtype=float)
    for turn in range(5):
        for level in range(5):
            a0 = 0 + 2 * (level + 5 * turn)
            p[a0] = 0.01
            p[a0 + 1] = 0.02
    p[2 * (2 + 5 * 1) + 1] = 0.4
    p = p / p.sum()
    out = greedy_from_probs(p)
    assert out[0] == 1
    assert out[1] == 2
    assert out[2] == 1


def test_factored_joint_probs_greedy():
    from bace.marl import factored_joint_probs, greedy_from_probs, _unflatten_discrete

    fire = np.array([0.2, 0.8])
    level = np.zeros(5)
    level[2] = 1.0
    turn = np.zeros(5)
    turn[1] = 1.0
    p = factored_joint_probs(fire, level, turn)
    assert abs(p.sum() - 1.0) < 1e-9
    assert abs(p[1::2].sum() - 0.8) < 1e-9
    out = greedy_from_probs(p)
    assert out.tolist() == [1, 2, 1]
    a = 1 + 2 * (2 + 5 * 1)
    assert _unflatten_discrete(a).tolist() == [1, 2, 1]
    p_hold = factored_joint_probs(np.array([0.7, 0.3]), level, turn)
    assert greedy_from_probs(p_hold)[0] == 0


def test_factored_actor_init_fire_bias():
    import pytest

    try:
        import torch
        from tianshou.utils.net.common import Net
    except ImportError:
        pytest.skip("torch/tianshou missing")
    from bace.marl import N_FLAT, _factored_discrete_actor_cls, greedy_from_probs

    actor = _factored_discrete_actor_cls()(
        preprocess_net=Net(state_shape=8, hidden_sizes=[16, 16]),
        fire_logit_bias=0.25,
    )
    actor.eval()
    with torch.no_grad():
        probs, _ = actor(np.zeros((4, 8), dtype=np.float32))
    p = probs.cpu().numpy()
    assert p.shape == (4, N_FLAT)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    marg = p[:, 1::2].sum(axis=1)
    assert (marg > 0.5).all()
    assert greedy_from_probs(p[0])[0] == 1


def test_discrete_action_space():
    env = BaceEnv(
        {
            "env": {"max_cycles": 12, "seed": 3, "action_type": "discrete", "action_repeat": 5},
            "blue": {"num_agents": 1, "behavior": "external"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    )
    assert env.action_space("agent_0").nvec.tolist() == [2, 5, 5]
    env.reset(seed=3)
    actions = {a: np.array([0, 2, 2], dtype=np.int64) for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert "agent_0" in rewards
    env.close()
