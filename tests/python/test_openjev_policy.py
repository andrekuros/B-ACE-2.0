"""Unit tests for the OpenJev discrete-tactic mapping (no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "examples" / "openjev"
sys.path.insert(0, str(ROOT))

from policy import (  # noqa: E402
    TACTICS,
    OpenJevUnavailable,
    TacticPolicy,
    cuda_available,
    discrete_to_continuous,
    load_openjev_scorer,
    random_scorer,
    serialize_state,
    tactic_to_action,
)


def _snap(*, aspect=0.0, dist=400.0, r_max=500.0, missiles=6, supporting=False, detected=True):
    return {
        "end": "ongoing",
        "action_step": 1,
        "missiles": [],
        "fighters": [
            {
                "id": 101,
                "team": 0,
                "agent_name": "agent_0",
                "alive": True,
                "missiles": missiles,
                "fsm": "Engage",
                "hpt_id": 201,
                "supporting_missile": supporting,
                "tracks": [
                    {
                        "id": 201,
                        "detected": detected,
                        "dist": dist,
                        "aspect": aspect,
                        "own_r_max": r_max,
                        "is_missile_support": supporting,
                    }
                ],
            }
        ],
    }


def test_discrete_bins_match_rust():
    a = discrete_to_continuous(1, 2, 0)
    assert a.shape == (4,)
    assert abs(a[0] - (-0.5)) < 1e-6  # -90/180
    assert a[3] > 0


def test_tactics_cover_fire_and_break():
    snap = _snap(aspect=5.0, dist=200.0, r_max=400.0)
    shoot = tactic_to_action("shoot", snap)
    brk = tactic_to_action("break_left", snap)
    search = tactic_to_action("search", snap)
    assert shoot[3] > 0
    assert brk[0] < -0.4
    assert search[3] < 0
    assert set(TACTICS) >= {"shoot", "support", "break_left"}


def test_serialize_mentions_range():
    text = serialize_state(_snap(dist=185.2, r_max=370.4))
    assert "NM" in text
    assert "missiles" in text


def test_openjev_not_forced_on_cpu():
    if cuda_available():
        return
    try:
        load_openjev_scorer()
    except OpenJevUnavailable as exc:
        assert "CUDA" in str(exc) or "torch" in str(exc).lower()
        return
    raise AssertionError("0.8B OpenJev must not load on CPU")


def test_random_tactics_step_against_duck():
    from bace import BaceEnv

    env = BaceEnv(
        {
            "env": {"max_cycles": 8, "seed": 1, "action_repeat": 5},
            "blue": {"num_agents": 1, "behavior": "external"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    )
    env.reset(seed=1)
    policy = TacticPolicy(random_scorer(np.random.default_rng(1)), "agent_0")
    text = serialize_state(env.state())
    assert "NM" in text or "detected" in text
    act = policy.act(env.state())
    assert act.shape == (4,)
    env.step({"agent_0": act})
    env.close()
    assert policy.last_tactic in TACTICS
