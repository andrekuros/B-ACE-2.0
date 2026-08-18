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


def test_snapshot_has_fighters():
    env = BaceEnv(
        {
            "env": {"max_cycles": 8, "seed": 1, "action_repeat": 5},
            "blue": {"num_agents": 1, "behavior": "duck"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    )
    env.reset(seed=1)
    state = env.state()
    assert len(state["fighters"]) == 2
    assert state["end"] in {"ongoing", "max_cycles"}
    env.close()


def test_four_v_four_obs_and_episode():
    env = BaceEnv(
        {
            "env": {"max_cycles": 400, "seed": 1, "action_repeat": 20},
            "blue": {
                "num_agents": 4,
                "behavior": "baseline1",
                "offset_pos": {"x": 2.0, "y": 0.0, "z": 0.0},
            },
            "red": {
                "num_agents": 4,
                "behavior": "duck",
                "offset_pos": {"x": 2.0, "y": 0.0, "z": 0.0},
            },
        }
    )
    obs, _ = env.reset(seed=1)
    assert env._obs_size == 79
    assert set(obs) == {f"agent_{i}" for i in range(4)}
    assert len(env.state()["fighters"]) == 8
    last_end = None
    for _ in range(420):
        if not env.agents:
            break
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, _, terms, truncs, _ = env.step(actions)
        last_end = env.state()["end"]
        if all(terms.values()) or all(truncs.values()):
            break
    assert last_end == "red_killed"
    env.close()


def test_baseline_episode_runs():
    env = BaceEnv(
        {
            "env": {"max_cycles": 40, "seed": 4, "action_repeat": 10},
            "blue": {"num_agents": 1, "behavior": "baseline1"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    )
    env.reset(seed=4)
    last_end = None
    for _ in range(50):
        if not env.agents:
            break
        actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        _, _, terms, truncs, _ = env.step(actions)
        last_end = env.state()["end"]
        if all(terms.values()) or all(truncs.values()):
            break
    assert last_end is not None
    env.close()
