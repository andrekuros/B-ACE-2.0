"""PettingZoo ParallelEnv for B-ACE 2.0."""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from gymnasium import Env, spaces
from pettingzoo import ParallelEnv

try:
    from bace._native import NativeEnv, NativeVecEnv
except ImportError as exc:  # pragma: no cover
    NativeEnv = None  # type: ignore
    NativeVecEnv = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _default_config() -> dict[str, Any]:
    return {
        "env": {
            "phy_fps": 20,
            "max_cycles": 3600,
            "action_repeat": 20,
            "action_type": "continuous",
            "stop_mission": True,
            "seed": 1,
            "rewards": {},
        },
        "blue": {
            "num_agents": 1,
            "behavior": "external",
            "mission": "dca",
            "share_tracks": True,
        },
        "red": {
            "num_agents": 1,
            "behavior": "baseline1",
            "mission": "dca",
            "share_tracks": True,
            "init_position": {"x": 0.0, "y": 25000.0, "z": -30.0},
            "init_hdg": 180.0,
            "target_position": {"x": 0.0, "y": 25000.0, "z": 30.0},
        },
    }


class BaceEnv(ParallelEnv):
    """Multi-agent BVR environment backed by the Rust native core."""

    metadata = {"name": "bace_v2", "render_modes": []}

    def __init__(self, config: Optional[dict[str, Any]] = None, **kwargs: Any):
        if NativeEnv is None:
            raise ImportError(
                "bace native extension is not built. Run: pip install -e . "
                f"(maturin). Original error: {_IMPORT_ERROR}"
            )
        cfg = _default_config()
        if config:
            cfg = _deep_merge(cfg, config)
        if kwargs:
            cfg = _deep_merge(cfg, kwargs)
        for side in ("blue", "red"):
            if side in cfg and isinstance(cfg[side], dict):
                n = int(cfg[side].get("num_agents", 1))
                cfg[side]["num_agents"] = max(1, min(4, n))
        self._config = cfg
        self._native = NativeEnv(json.dumps(cfg))
        self.possible_agents = list(self._native.agent_ids())
        self.agents = list(self.possible_agents)
        self._obs_size = int(self._native.obs_size())
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float32
        )
        self._discrete = str(cfg.get("env", {}).get("action_type", "continuous")).lower() == "discrete"
        if self._discrete:
            # fire, level, turn — mapped through DiscreteAction::to_continuous
            self._action_space = spaces.MultiDiscrete([2, 5, 5])
        else:
            self._action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
        self._last: dict[str, Any] = {}

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        del options
        raw = self._native.reset(seed)
        self.agents = list(self.possible_agents)
        obs = {a: np.asarray(raw["obs"][a], dtype=np.float32) for a in self.agents}
        infos = {a: _clean_info(raw["infos"].get(a, {})) for a in self.agents}
        self._last = raw
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray | list[float]]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        payload = {a: _action_payload(v) for a, v in actions.items()}
        raw = self._native.step(payload)
        obs = {a: np.asarray(raw["obs"][a], dtype=np.float32) for a in raw["obs"]}
        rewards = {a: float(raw["rewards"][a]) for a in raw["rewards"]}
        terminations = {a: bool(raw["terminations"][a]) for a in raw["terminations"]}
        truncations = {a: bool(raw["truncations"][a]) for a in raw["truncations"]}
        infos = {}
        for a, raw_info in raw["infos"].items():
            infos[a] = {
                k: v
                for k, v in dict(raw_info).items()
                if not isinstance(v, str)
            }
        self.agents = [
            a
            for a in self.agents
            if not (terminations.get(a, False) or truncations.get(a, False))
        ]
        self._last = raw
        return obs, rewards, terminations, truncations, infos

    def state(self) -> dict[str, Any]:
        return json.loads(self._native.snapshot_json())

    def outcome(self) -> dict[str, Any]:
        return json.loads(self._native.outcome_json())

    def close(self) -> None:
        self.agents = []


def _clean_info(raw_info: dict) -> dict:
    return {k: v for k, v in dict(raw_info).items() if not isinstance(v, str)}


def _action_payload(v: np.ndarray | list[float] | list[int]) -> list[float]:
    arr = np.asarray(v).reshape(-1)
    return arr.astype(np.float64).tolist()


def _baseline_beh(name: str) -> dict[str, Any]:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "configs" / "baselines" / f"{name}.json"
    if root.is_file():
        import json as _json

        data = _json.loads(root.read_text())
        return data.get("red", data.get("blue", data))
    defaults = {
        "aggressive": {"d_shot": [1.2], "l_crank": [0.5], "l_break": [1.3]},
        "balanced": {"d_shot": [0.85], "l_crank": [0.6], "l_break": [0.95]},
        "cautious": {"d_shot": [0.55], "l_crank": [0.9], "l_break": [0.7]},
    }
    return {
        "behavior": "baseline1",
        "beh_config": defaults.get(name, defaults["balanced"]),
    }


_RL_REWARDS = {
    "mission_factor": 0.001,
    "missile_fire_factor": -0.1,
    "missile_no_fire_factor": 0.0,
    "missile_miss_factor": -0.5,
    "detect_loss_factor": -0.01,
    "keep_track_factor": 0.001,
    "hit_enemy_factor": 3.0,
    "hit_own_factor": -5.0,
    "mission_accomplished_factor": 10.0,
}


def _clamp_agents(n: int) -> int:
    return max(1, min(4, int(n)))


def _formation(n: int) -> dict[str, Any]:
    n = _clamp_agents(n)
    if n <= 1:
        return {}
    if n == 2:
        return {"offset_pos": {"x": 4.0, "y": 0.0, "z": 0.0}}
    return {"offset_pos": {"x": 4.0, "y": 0.0, "z": 4.0}}


def wez_close_overlay(range_nm: float = 16.0, altitude_ft: float = 25000.0) -> dict[str, Any]:
    """Head-on spawn at the WEZ close cell (default 16 NM)."""
    half = float(range_nm) / 2.0
    pos_b = {"x": 0.0, "y": float(altitude_ft), "z": half}
    pos_r = {"x": 0.0, "y": float(altitude_ft), "z": -half}
    return {
        "blue": {
            "init_position": pos_b,
            "init_hdg": 0.0,
            "target_position": pos_r,
        },
        "red": {
            "init_position": pos_r,
            "init_hdg": 180.0,
            "target_position": pos_b,
        },
    }


def make_env(
    opponent: str = "duck",
    agents: int = 1,
    max_cycles: int = 400,
    seed: int = 1,
    share_tracks: bool = True,
    red_mission: str = "dca",
    rewards: str = "default",
    action_type: str = "continuous",
    **kwargs: Any,
) -> BaceEnv:
    """PettingZoo factory. Only blue is `external` this release. Caps at 4v4."""
    agents = _clamp_agents(agents)
    form = _formation(agents)
    opp = opponent.lower()
    if opp == "duck":
        red: dict[str, Any] = {"num_agents": agents, "behavior": "duck", **form}
    elif opp == "baseline":
        red = {"num_agents": agents, "behavior": "baseline1", **form}
    elif opp in {"aggressive", "balanced", "cautious", "fsm", "elite"}:
        label = "balanced" if opp in {"fsm", "elite"} else opp
        red = {"num_agents": agents, **_baseline_beh(label), **form}
        red["behavior"] = "baseline1"
        red["num_agents"] = agents
    else:
        raise ValueError(f"unknown opponent {opponent!r}")
    red["share_tracks"] = share_tracks
    red["mission"] = red_mission
    env_cfg: dict[str, Any] = {
        "max_cycles": max_cycles,
        "seed": seed,
        "action_type": action_type,
    }
    if str(rewards).lower() == "rl":
        env_cfg["rewards"] = dict(_RL_REWARDS)
    cfg: dict[str, Any] = {
        "env": env_cfg,
        "blue": {
            "num_agents": agents,
            "behavior": "external",
            "share_tracks": share_tracks,
            **form,
        },
        "red": red,
    }
    cfg = _deep_merge(cfg, kwargs)
    return BaceEnv(cfg)


class BaceGymEnv(Env):
    """Single-agent Gymnasium wrapper (blue `agent_0`)."""

    metadata = {"render_modes": []}

    def __init__(self, env: Optional[BaceEnv] = None, **kwargs: Any):
        super().__init__()
        self._pz = env if env is not None else make_env(**kwargs)
        self.observation_space = self._pz.observation_space("agent_0")
        self.action_space = self._pz.action_space("agent_0")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, infos = self._pz.reset(seed=seed, options=options)
        return obs["agent_0"], infos.get("agent_0", {})

    def step(self, action):
        obs, rewards, terms, truncs, infos = self._pz.step({"agent_0": action})
        a = "agent_0"
        terminated = bool(terms.get(a, True))
        truncated = bool(truncs.get(a, False))
        return (
            obs.get(a, np.zeros(self._pz._obs_size, dtype=np.float32)),
            float(rewards.get(a, 0.0)),
            terminated,
            truncated,
            infos.get(a, {}),
        )

    def close(self):
        self._pz.close()


class BaceVecEnv:
    """n parallel PettingZoo-style envs in one rayon batch (`record=false`)."""

    def __init__(
        self,
        num_envs: int = 8,
        auto_reset: bool = True,
        **kwargs: Any,
    ):
        if NativeVecEnv is None:
            raise ImportError(
                "bace native extension is not built. Run: pip install -e . "
                f"(maturin). Original error: {_IMPORT_ERROR}"
            )
        inner = make_env(**kwargs)
        cfg = inner._config
        inner.close()
        self._native = NativeVecEnv(json.dumps(cfg), int(num_envs), auto_reset)
        self.num_envs = int(self._native.num_envs())
        self.possible_agents = list(self._native.agent_ids())
        self.agents = list(self.possible_agents)
        self._obs_size = int(self._native.obs_size())
        self._discrete = bool(self._native.is_discrete())
        if self._discrete:
            self._action_space = spaces.MultiDiscrete([2, 5, 5])
        else:
            self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float32
        )

    def observation_space(self, agent: str = "agent_0"):
        del agent
        return self._observation_space

    def action_space(self, agent: str = "agent_0"):
        del agent
        return self._action_space

    def reset(self, seed: Optional[int] = None):
        raw = self._native.reset(seed)
        return [_decode_vec_step(r, self._obs_size) for r in raw]

    def reset_at(self, index: int, seed: Optional[int] = None):
        raw = self._native.reset_at(int(index), seed)
        return _decode_vec_step(raw, self._obs_size)

    def step(self, actions: list[dict[str, np.ndarray | list[float]]]):
        payload = [{a: _action_payload(v) for a, v in d.items()} for d in actions]
        raw = self._native.step(payload)
        return [_decode_vec_step(r, self._obs_size) for r in raw]

    def close(self) -> None:
        self.agents = []


class BaceVecGym(Env):
    """Stacked single-agent (`agent_0`) vector env for Tianshou / gymnasium."""

    metadata = {"render_modes": []}

    def __init__(self, num_envs: int = 8, **kwargs: Any):
        super().__init__()
        self._vec = BaceVecEnv(num_envs=num_envs, auto_reset=True, **kwargs)
        self.num_envs = self._vec.num_envs
        self.observation_space = self._vec.observation_space()
        self.action_space = self._vec.action_space()
        self._obs_size = self._vec._obs_size

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        steps = self._vec.reset(seed=seed)
        obs = np.stack([s["obs"]["agent_0"] for s in steps], axis=0)
        infos = [s["infos"].get("agent_0", {}) for s in steps]
        return obs, {"infos": infos}

    def step(self, actions: np.ndarray):
        acts = []
        for i in range(self.num_envs):
            acts.append({"agent_0": np.asarray(actions[i])})
        steps = self._vec.step(acts)
        obs = np.stack([s["obs"].get("agent_0", np.zeros(self._obs_size, dtype=np.float32)) for s in steps])
        rew = np.asarray([s["rewards"].get("agent_0", 0.0) for s in steps], dtype=np.float32)
        term = np.asarray([s["terminations"].get("agent_0", False) for s in steps])
        trunc = np.asarray([s["truncations"].get("agent_0", False) for s in steps])
        infos = [s["infos"].get("agent_0", {}) for s in steps]
        return obs, rew, term, trunc, {"infos": infos}

    def close(self):
        self._vec.close()


def _decode_vec_step(raw: dict[str, Any], obs_size: int) -> dict[str, Any]:
    obs = {
        a: np.asarray(raw["obs"][a], dtype=np.float32)
        for a in raw.get("obs", {})
    }
    for a in list(obs):
        if obs[a].shape[0] != obs_size:
            obs[a] = np.zeros(obs_size, dtype=np.float32)
    return {
        "obs": obs,
        "rewards": {a: float(v) for a, v in raw.get("rewards", {}).items()},
        "terminations": {a: bool(v) for a, v in raw.get("terminations", {}).items()},
        "truncations": {a: bool(v) for a, v in raw.get("truncations", {}).items()},
        "infos": {a: dict(v) for a, v in raw.get("infos", {}).items()},
        "end": raw.get("end"),
    }


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
