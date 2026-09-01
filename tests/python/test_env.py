"""Python integration tests for B-ACE 2.0."""

from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from bace import BaceEnv


@pytest.fixture
def env():
    e = BaceEnv(
        {
            "env": {"max_cycles": 40, "seed": 1, "action_repeat": 10},
            "blue": {"num_agents": 1, "behavior": "external"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    )
    yield e
    e.close()


def test_reset_shapes(env):
    obs, infos = env.reset(seed=3)
    assert set(obs) == {"agent_0"}
    assert obs["agent_0"].shape == (env._obs_size,)
    assert "agent_0" in infos


def test_step_contract(env):
    obs, _ = env.reset(seed=3)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert set(rewards) == set(obs)
    assert isinstance(rewards["agent_0"], float)
    assert isinstance(terms["agent_0"], bool)
    assert isinstance(truncs["agent_0"], bool)


def test_pettingzoo_api():
    env = BaceEnv(
        {
            "env": {"max_cycles": 20, "seed": 2, "action_repeat": 5},
            "blue": {"num_agents": 1, "behavior": "external"},
            "red": {"num_agents": 1, "behavior": "baseline1"},
        }
    )
    parallel_api_test(env, num_cycles=10)
    env.close()


def test_multi_blue_obs_size():
    env = BaceEnv(
        {
            "env": {"max_cycles": 10, "seed": 1},
            "blue": {"num_agents": 2, "behavior": "external"},
            "red": {"num_agents": 2, "behavior": "duck"},
        }
    )
    # own 9 + enemies 2*13 + allies 1*6 = 9+26+6 = 41
    assert env._obs_size == 41
    obs, _ = env.reset()
    assert len(obs) == 2
    env.close()


def test_episode_terminates():
    env = BaceEnv(
        {
            "env": {"max_cycles": 5, "seed": 9, "action_repeat": 5},
            "blue": {"num_agents": 1, "behavior": "external"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    )
    env.reset(seed=9)
    done = False
    for _ in range(20):
        if not env.agents:
            done = True
            break
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, _, terms, truncs, _ = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            done = True
            break
    assert done
    env.close()


def test_vec_env_parallel_1v1_duck():
    from bace import BaceVecEnv

    env = BaceVecEnv(num_envs=8, opponent="duck", agents=1, max_cycles=12, seed=1)
    assert env.num_envs == 8
    steps = env.reset(seed=1)
    assert len(steps) == 8
    assert "agent_0" in steps[0]["obs"]
    idle = {"agent_0": np.zeros(4, dtype=np.float32)}
    out = env.step([idle] * 8)
    assert len(out) == 8
    assert "agent_0" in out[0]["rewards"]
    env.close()


def test_ippo_native_vec_2v2():
    from bace.marl import IppoNativeVec, TrainSpec, _unflatten_discrete

    spec = TrainSpec(agents=2, n_envs=2, steps=8, max_cycles=8, eval_episodes=1)
    env = IppoNativeVec(spec, n_games=2)
    assert env.env_num == 4
    obs, infos = env.reset(seed=1)
    assert obs.shape[0] == 4
    acts = np.array([0, 1, 2, 3])
    o, r, t, tr, inf = env.step(acts)
    assert o.shape[0] == 4
    assert r.shape[0] == 4
    env.close()
    decoded = _unflatten_discrete(13)
    assert decoded.shape == (3,)


def test_make_env_rl_rewards():
    from bace import make_env

    env = make_env(opponent="duck", agents=1, max_cycles=4, seed=1, rewards="rl")
    assert env._config["env"]["rewards"]["missile_no_fire_factor"] == 0.0
    assert env._config["env"]["rewards"]["detect_loss_factor"] == -0.01
    env.close()
