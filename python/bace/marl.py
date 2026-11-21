"""Independent PPO (shared weights) on NativeVecEnv. Skip if extras missing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from gymnasium import spaces

from bace.env import BaceVecEnv, make_env, wez_close_overlay
from bace.experiment import run_experiment

N_FLAT = 50
_NVEC = (2, 5, 5)


def _tianshou_ok() -> bool:
    try:
        import tianshou  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return {"skipped": True, "reason": reason, **extra}


@dataclass
class TrainSpec:
    algo: str = "ippo"
    opponent: str = "duck"
    agents: int = 1
    seed: int = 0
    steps: int = 400
    max_cycles: int = 200
    share_tracks: bool = True
    red_mission: str = "dca"
    eval_episodes: int = 50
    n_envs: int = 8
    rewards: str = "rl"
    action_type: str = "discrete"
    self_play: bool = False
    sp_mode: str = ""

    def __post_init__(self) -> None:
        mode = (self.sp_mode or "").lower()
        if mode in {"naive", "mixed", "pfsp"}:
            object.__setattr__(self, "self_play", True)
            object.__setattr__(self, "sp_mode", mode)
        elif self.self_play and not mode:
            object.__setattr__(self, "sp_mode", "naive")
        if self.algo.lower() == "maddpg":
            object.__setattr__(self, "action_type", "continuous")


def _is_red(name: str) -> bool:
    return str(name).startswith("red_")


def _team_names(agent: str, names: list[str]) -> list[str]:
    red = _is_red(agent)
    return [n for n in names if _is_red(n) == red]


def _pack_obs(spec: TrainSpec, agent: str, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    own = np.asarray(obs_dict[agent], dtype=np.float32)
    if spec.algo.lower() not in {"mappo", "maddpg"}:
        return own
    mates = [
        np.asarray(obs_dict[n], dtype=np.float32)
        for n in sorted(_team_names(agent, list(obs_dict)))
        if n != agent
    ]
    need = max(0, spec.agents - 1)
    while len(mates) < need:
        mates.append(np.zeros_like(own))
    return np.concatenate([own] + mates[:need])


def _unflatten_discrete(a: int) -> np.ndarray:
    a = int(a)
    fire = a % 2
    rest = a // 2
    level = rest % 5
    turn = rest // 5
    return np.array([fire, level, turn], dtype=np.int64)


def greedy_from_probs(p: np.ndarray) -> np.ndarray:
    """Argmax kinematics; fire from marginal P(fire) so the 50-way flatten cannot hide it."""
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size != N_FLAT:
        return _unflatten_discrete(int(np.argmax(p)))
    fire = 1 if float(p[1::2].sum()) >= 0.5 else 0
    kin = p.reshape(5, 5, 2).sum(axis=2)
    turn, level = np.unravel_index(int(kin.argmax()), kin.shape)
    return np.array([fire, int(level), int(turn)], dtype=np.int64)


def factored_joint_probs(fire: np.ndarray, level: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """Outer product in Discrete(50) order: a = fire + 2*(level + 5*turn)."""
    fire = np.asarray(fire, dtype=float).reshape(2)
    level = np.asarray(level, dtype=float).reshape(5)
    turn = np.asarray(turn, dtype=float).reshape(5)
    joint = turn[:, None, None] * level[None, :, None] * fire[None, None, :]
    return joint.reshape(N_FLAT)


def _mean_ci(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = float(np.mean(xs))
    if n < 2:
        return m, 0.0
    se = float(np.std(xs, ddof=1) / np.sqrt(n))
    return m, 1.96 * se


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    empty = {
        "return": 0.0,
        "mission_rate": 0.0,
        "kills": 0.0,
        "deaths": 0.0,
        "kill_ratio": 0.0,
        "shots": 0.0,
        "hits": 0.0,
        "hit_rate": 0.0,
        "n": 0.0,
        "kills_ci": 0.0,
        "mission_ci": 0.0,
        "return_ci": 0.0,
    }
    if not rows:
        return empty
    keys = ("return", "mission", "kills", "deaths", "shots", "hits")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    kills_m, kills_ci = _mean_ci([r["kills"] for r in rows])
    miss_m, miss_ci = _mean_ci([r["mission"] for r in rows])
    ret_m, ret_ci = _mean_ci([r["return"] for r in rows])
    shots = mean["shots"]
    deaths = mean["deaths"]
    return {
        "return": ret_m,
        "mission_rate": miss_m,
        "kills": kills_m,
        "deaths": deaths,
        "kill_ratio": mean["kills"] / max(deaths, 1e-6),
        "shots": shots,
        "hits": mean["hits"],
        "hit_rate": mean["hits"] / max(shots, 1e-6),
        "n": float(len(rows)),
        "kills_ci": kills_ci,
        "mission_ci": miss_ci,
        "return_ci": ret_ci,
    }


def _rl_env_block(max_cycles: int, seed: int) -> dict[str, Any]:
    return {
        "max_cycles": max_cycles,
        "seed": seed,
        "rewards": {
            "missile_no_fire_factor": 0.0,
            "detect_loss_factor": -0.01,
            "mission_factor": 0.001,
            "missile_fire_factor": -0.1,
            "missile_miss_factor": -0.5,
            "keep_track_factor": 0.001,
            "hit_enemy_factor": 3.0,
            "hit_own_factor": -5.0,
            "mission_accomplished_factor": 10.0,
        },
    }


def eval_scripted(
    behavior: str,
    opponent: str = "duck",
    agents: int = 1,
    n: int = 50,
    max_cycles: int = 200,
    seed: int = 1,
    spawn: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    form: dict[str, Any] = {}
    if agents >= 2:
        form = {"offset_pos": {"x": 4.0, "y": 0.0, "z": 0.0 if agents == 2 else 4.0}}
    red_beh = "duck" if opponent == "duck" else "baseline1"
    configs = []
    for i in range(max(1, n)):
        cfg: dict[str, Any] = {
            "env": _rl_env_block(max_cycles, seed + i),
            "blue": {"num_agents": agents, "behavior": behavior, **form},
            "red": {"num_agents": agents, "behavior": red_beh, **form},
        }
        if spawn:
            from bace.env import _deep_merge

            cfg = _deep_merge(cfg, spawn)
        configs.append(cfg)
    out = run_experiment(configs, max_parallel=min(8, max(1, n)))
    rows = []
    for o in out:
        rows.append(
            {
                "return": float(o.get("episode_return", 0.0)),
                "mission": 1.0 if o.get("mission_success") else 0.0,
                "kills": float(o.get("blue_kills", 0)),
                "deaths": float(o.get("blue_deaths", 0)),
                "shots": float(o.get("missiles_fired", 0)),
                "hits": float(o.get("missile_hits", 0)),
            }
        )
    return _summarize(rows)


def eval_random(spec: TrainSpec, n: Optional[int] = None) -> dict[str, float]:
    rng = np.random.default_rng(spec.seed + 9_001)
    n = int(n or spec.eval_episodes)
    rows = []
    for i in range(max(1, n)):
        env = make_env(
            opponent=spec.opponent,
            agents=spec.agents,
            max_cycles=spec.max_cycles,
            seed=spec.seed + 80_000 + i,
            share_tracks=spec.share_tracks,
            red_mission=spec.red_mission,
            rewards=spec.rewards,
            action_type=spec.action_type,
            self_play=spec.self_play,
        )
        obs, _ = env.reset(seed=spec.seed + 80_000 + i)
        ep = 0.0
        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            for a in actions:
                if isinstance(env.action_space(a), spaces.Box):
                    actions[a] = rng.uniform(-1, 1, size=4).astype(np.float32)
            obs, rew, term, trunc, _ = env.step(actions)
            ep += float(sum(rew.values()))
            if all(term.get(a, True) or trunc.get(a, True) for a in term):
                break
        oc = env.outcome()
        env.close()
        rows.append(
            {
                "return": ep,
                "mission": 1.0 if oc.get("mission_success") else 0.0,
                "kills": float(oc.get("blue_kills", 0)),
                "deaths": float(oc.get("blue_deaths", 0)),
                "shots": float(oc.get("missiles_fired", 0)),
                "hits": float(oc.get("missile_hits", 0)),
            }
        )
    return _summarize(rows)


def eval_policy_fn(
    spec: TrainSpec,
    act_fn: Callable[[np.ndarray], np.ndarray],
    n: Optional[int] = None,
) -> dict[str, float]:
    n = int(n or spec.eval_episodes)
    rows = []
    for i in range(max(1, n)):
        env = make_env(
            opponent=spec.opponent,
            agents=spec.agents,
            max_cycles=spec.max_cycles,
            seed=spec.seed + 50_000 + i,
            share_tracks=spec.share_tracks,
            red_mission=spec.red_mission,
            rewards=spec.rewards,
            action_type=spec.action_type,
            self_play=spec.self_play,
        )
        obs, _ = env.reset(seed=spec.seed + 50_000 + i)
        ep = 0.0
        while env.agents:
            actions = {a: act_fn(_pack_obs(spec, a, obs)) for a in env.agents}
            obs, rew, term, trunc, _ = env.step(actions)
            ep += float(sum(rew.values()))
            if all(term.get(a, True) or trunc.get(a, True) for a in term):
                break
        oc = env.outcome()
        env.close()
        rows.append(
            {
                "return": ep,
                "mission": 1.0 if oc.get("mission_success") else 0.0,
                "kills": float(oc.get("blue_kills", 0)),
                "deaths": float(oc.get("blue_deaths", 0)),
                "shots": float(oc.get("missiles_fired", 0)),
                "hits": float(oc.get("missile_hits", 0)),
            }
        )
    return _summarize(rows)


class IppoNativeVec:
    """Tianshou vector env: NativeVecEnv games, independent per-agent slots."""

    is_async = False

    def __init__(
        self,
        spec: TrainSpec,
        n_games: int,
        *,
        blue_only: bool = False,
        opponent: str | None = None,
        self_play: bool | None = None,
    ):
        self.waiting_id: list[int] = []
        self.is_closed = False
        self._spec = spec
        self._n_games = max(1, n_games)
        self._continuous = str(spec.action_type).lower() == "continuous"
        self._team_concat = spec.algo.lower() in {"mappo", "maddpg"}
        self._blue_only = bool(blue_only)
        use_self_play = spec.self_play if self_play is None else bool(self_play)
        use_opp = spec.opponent if opponent is None else opponent
        self._vec = BaceVecEnv(
            num_envs=self._n_games,
            auto_reset=False,
            opponent=use_opp,
            agents=max(1, spec.agents),
            seed=spec.seed,
            max_cycles=spec.max_cycles,
            share_tracks=spec.share_tracks,
            red_mission=spec.red_mission,
            rewards=spec.rewards,
            action_type="continuous" if self._continuous else "discrete",
            self_play=use_self_play,
        )
        self._agent_ids = list(self._vec.possible_agents)
        self._n_agents = max(1, len(self._agent_ids))
        self._blue_ids = [n for n in self._agent_ids if not _is_red(n)]
        self._red_ids = [n for n in self._agent_ids if _is_red(n)]
        self._n_blue = max(1, len(self._blue_ids))
        self._own_dim = int(np.prod(self._vec.observation_space().shape))
        self._n_team = max(1, spec.agents)
        obs_dim = self._own_dim * self._n_team if self._team_concat else self._own_dim
        self._learner_ids = list(self._blue_ids) if self._blue_only else list(self._agent_ids)
        self._n_learn = max(1, len(self._learner_ids))
        self.env_num = self._n_games * self._n_learn
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        if self._continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(N_FLAT)
        self.workers: list[Any] = [None] * self.env_num
        self._zeros = np.zeros(self._own_dim, dtype=np.float32)
        self._own = np.zeros((self._n_games, self._n_agents, self._own_dim), dtype=np.float32)
        self._seed0 = spec.seed
        self.own_dim = self._own_dim
        self._game_frozen_act: list[Callable[[np.ndarray], np.ndarray] | None] = [
            None
        ] * self._n_games
        self._pfsp_pick: Callable[[int], Callable[[np.ndarray], np.ndarray]] | None = None
        self._game_blue_ret = np.zeros(self._n_games, dtype=np.float64)

    def set_pfsp_picker(self, fn: Callable[[int], Callable[[np.ndarray], np.ndarray]]) -> None:
        self._pfsp_pick = fn

    def __len__(self) -> int:
        return self.env_num

    def _wrap_id(self, env_id: Any) -> list[int]:
        if env_id is None:
            return list(range(self.env_num))
        if isinstance(env_id, (int, np.integer)):
            return [int(env_id)]
        return [int(i) for i in np.asarray(env_id).reshape(-1)]

    def _obs_for(self, game: int, agent: str) -> np.ndarray:
        names = list(self._agent_ids)
        obs_dict = {
            n: self._own[game, i]
            for i, n in enumerate(names)
        }
        return _pack_obs(self._spec, agent, obs_dict)

    def _fill_game(self, game: int, step: dict[str, Any]) -> None:
        for a, name in enumerate(self._agent_ids):
            self._own[game, a] = np.asarray(step["obs"].get(name, self._zeros), dtype=np.float32)

    def _decode_act(self, action: np.ndarray, local: int) -> np.ndarray:
        if self._continuous:
            arr = np.asarray(action, dtype=np.float32).reshape(-1, 4)
            return arr[local]
        return _unflatten_discrete(int(np.asarray(action).reshape(-1)[local]))

    def reset(self, env_id: Any = None, **kwargs: Any):
        ids = self._wrap_id(env_id)
        seed = kwargs.get("seed", self._seed0)
        games = sorted({i // self._n_learn for i in ids})
        for g in games:
            s = None if seed is None else int(seed) + g
            self._fill_game(g, self._vec.reset_at(g, seed=s))
            self._game_blue_ret[g] = 0.0
            if self._pfsp_pick is not None:
                self._game_frozen_act[g] = self._pfsp_pick(g)
        obs = np.stack(
            [self._obs_for(i // self._n_learn, self._learner_ids[i % self._n_learn]) for i in ids],
            axis=0,
        )
        infos = np.array([{"env_id": i} for i in ids])
        return obs, infos

    def step(self, action: Any, id: Any = None):
        ids = self._wrap_id(id)
        id_set = set(ids)
        acts: list[dict[str, np.ndarray]] = []
        for g in range(self._n_games):
            d: dict[str, np.ndarray] = {}
            for li, name in enumerate(self._learner_ids):
                sid = g * self._n_learn + li
                if sid in id_set:
                    local = ids.index(sid)
                    d[name] = self._decode_act(action, local)
                elif self._continuous:
                    d[name] = np.zeros(4, dtype=np.float32)
                else:
                    d[name] = np.array([0, 2, 2], dtype=np.int64)
            if self._blue_only:
                frozen = self._game_frozen_act[g]
                for name in self._red_ids:
                    o = self._obs_for(g, name)
                    if frozen is not None:
                        raw = frozen(o)
                        d[name] = (
                            np.asarray(raw, dtype=np.float32).reshape(-1)[:4]
                            if self._continuous
                            else np.asarray(raw, dtype=np.int64).reshape(-1)[:3]
                        )
                    elif self._continuous:
                        d[name] = np.zeros(4, dtype=np.float32)
                    else:
                        d[name] = np.array([0, 2, 2], dtype=np.int64)
            acts.append(d)
        steps = self._vec.step(acts)
        obs_l, rew_l, term_l, trunc_l, info_l = [], [], [], [], []
        for sid in ids:
            g = sid // self._n_learn
            name = self._learner_ids[sid % self._n_learn]
            st = steps[g]
            self._fill_game(g, st)
            ended = str(st.get("end", "Ongoing")) not in {"Ongoing", "ongoing"}
            o = self._obs_for(g, name)
            rew = float(st["rewards"].get(name, 0.0))
            if name in self._blue_ids:
                self._game_blue_ret[g] += rew
            obs_l.append(o)
            rew_l.append(rew)
            term_l.append(bool(ended))
            trunc_l.append(False)
            info_l.append({"env_id": sid, "blue_ret": float(self._game_blue_ret[g])})
        return (
            np.stack(obs_l),
            np.asarray(rew_l, dtype=np.float64),
            np.asarray(term_l),
            np.asarray(trunc_l),
            np.array(info_l),
        )

    def seed(self, seed: int | list[int] | None = None) -> list[int]:
        if seed is None:
            self._seed0 = 0
        elif isinstance(seed, int):
            self._seed0 = seed
        else:
            self._seed0 = int(seed[0])
        return [self._seed0]

    def render(self, **kwargs: Any) -> list[Any]:
        del kwargs
        return []

    def close(self) -> None:
        self._vec.close()
        self.is_closed = True


def _ippo_act_fn(policy, greedy: bool):
    def act(obs: np.ndarray) -> np.ndarray:
        import torch
        from tianshou.data import Batch

        was = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                batch = policy(
                    Batch(obs=np.asarray(obs, dtype=np.float32)[None], info={})
                )
                dist = batch.dist
                probs = dist.probs.detach().cpu().numpy()[0]
            if greedy:
                return greedy_from_probs(probs)
            a = int(dist.sample().detach().cpu().numpy().reshape(-1)[0])
            return _unflatten_discrete(a)
        finally:
            policy.train(was)

    return act


_HYPER = {
    "lr": 3e-4,
    "gae_lambda": 0.95,
    "eps_clip": 0.2,
    "gamma": 0.99,
    "ent_coef": 0.02,
    "vf_coef": 0.5,
    "hidden": [128, 128],
    "update_repeat": 4,
    "batch_size": 256,
    "fire_logit_bias": 0.25,
    "discrete_map": "factored MultiDiscrete[2,5,5] (outer-product Discrete(50)); greedy fire = marginal P(fire)",
    "reward_preset": "rl",
    "algorithm": "IPPO (shared weights, independent per-agent actions)",
}


def _factored_discrete_actor_cls():
    import torch.nn.functional as F
    from torch import nn
    from tianshou.utils.net.common import ModuleWithVectorOutput

    class FactoredDiscreteActor(ModuleWithVectorOutput):
        """Independent fire / altitude / turn heads; joint P is the outer product."""

        def __init__(self, preprocess_net, fire_logit_bias: float = 0.25, own_dim: int | None = None):
            super().__init__(N_FLAT)
            self.preprocess = preprocess_net
            self.own_dim = own_dim
            dim = preprocess_net.get_output_dim()
            self.fire = nn.Linear(dim, 2)
            self.level = nn.Linear(dim, 5)
            self.turn = nn.Linear(dim, 5)
            nn.init.zeros_(self.fire.bias)
            self.fire.bias.data[1] = float(fire_logit_bias)

        def forward(self, obs, state=None, info=None):
            del info
            import torch as _torch

            if not _torch.is_tensor(obs):
                obs = _torch.as_tensor(obs, dtype=_torch.float32)
            if self.own_dim is not None and obs.shape[-1] > self.own_dim:
                obs = obs[..., : self.own_dim]
            x, hidden = self.preprocess(obs, state)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            log_f = F.log_softmax(self.fire(x), dim=-1)
            log_l = F.log_softmax(self.level(x), dim=-1)
            log_t = F.log_softmax(self.turn(x), dim=-1)
            log_joint = log_t[:, :, None, None] + log_l[:, None, :, None] + log_f[:, None, None, :]
            probs = log_joint.reshape(x.shape[0], N_FLAT).exp()
            return probs, hidden

    return FactoredDiscreteActor


def _own_slice_net(net: Any, own_dim: int) -> Any:
    from tianshou.utils.net.common import ModuleWithVectorOutput

    class OwnSlice(ModuleWithVectorOutput):
        def __init__(self, inner: Any, dim: int):
            super().__init__(inner.get_output_dim())
            self.inner = inner
            self.dim = dim

        def forward(self, obs, state=None):
            import torch as _torch

            if not _torch.is_tensor(obs):
                obs = _torch.as_tensor(obs, dtype=_torch.float32)
            if obs.shape[-1] > self.dim:
                obs = obs[..., : self.dim]
            return self.inner(obs, state)

    return OwnSlice(net, own_dim)


def _clone_sd(module: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _smoke_schedule(spec: TrainSpec, env_num: int) -> dict[str, Any]:
    smoke = spec.steps <= 80
    if smoke:
        return {
            "smoke": True,
            "max_epochs": 2,
            "epoch_num_steps": max(env_num, spec.steps),
            "collect_steps": max(env_num, spec.steps),
            "batch_size": 32,
            "eval_n": 2,
            "verbose": False,
        }
    collect_steps = max(env_num * 32, min(2048, spec.steps))
    epoch_num_steps = max(collect_steps, min(10_000, spec.steps))
    return {
        "smoke": False,
        "max_epochs": max(1, spec.steps // epoch_num_steps),
        "epoch_num_steps": epoch_num_steps,
        "collect_steps": collect_steps,
        "batch_size": int(_HYPER["batch_size"]),
        "eval_n": spec.eval_episodes,
        "verbose": True,
    }


def _eval_bundle(spec: TrainSpec, act_g: Any, act_s: Any, eval_n: int) -> dict[str, Any]:
    ev = eval_policy_fn(spec, act_g, n=eval_n)
    ev_stoch = eval_policy_fn(spec, act_s, n=eval_n)
    rnd = eval_random(spec, n=eval_n)
    fire = eval_scripted(
        "fire_once",
        opponent=spec.opponent if not spec.self_play else "duck",
        agents=spec.agents,
        n=eval_n,
        max_cycles=spec.max_cycles,
        seed=spec.seed,
    )
    fire16 = eval_scripted(
        "fire_once",
        opponent=spec.opponent if not spec.self_play else "duck",
        agents=spec.agents,
        n=eval_n,
        max_cycles=spec.max_cycles,
        seed=spec.seed,
        spawn=wez_close_overlay(16.0),
    )
    transfer: dict[str, Any] = {}
    if spec.self_play:
        vs_fsm = replace(spec, self_play=False, opponent="fsm", sp_mode="")
        vs_duck = replace(spec, self_play=False, opponent="duck", sp_mode="")
        transfer = {
            "eval_vs_fsm": eval_policy_fn(vs_fsm, act_g, n=eval_n),
            "eval_vs_fsm_stochastic": eval_policy_fn(vs_fsm, act_s, n=eval_n),
            "eval_vs_duck": eval_policy_fn(vs_duck, act_g, n=eval_n),
            "eval_vs_duck_stochastic": eval_policy_fn(vs_duck, act_s, n=eval_n),
            "eval_vs_self": ev,
            "eval_random_vs_duck": eval_random(vs_duck, n=eval_n),
        }
    beat_random = ev["kills"] > rnd["kills"] or (
        ev["kills"] >= rnd["kills"] and ev["return"] > rnd["return"]
    )
    return {
        "eval": ev,
        "eval_greedy": ev,
        "eval_stochastic": ev_stoch,
        "eval_random": rnd,
        "eval_fire_once": fire,
        "eval_fire_once_16nm": fire16,
        "beat_random": beat_random,
        **transfer,
    }


def _gate_4v4(rep: dict[str, Any]) -> bool:
    if rep.get("skipped") or rep.get("failed"):
        return False
    duck = rep.get("eval_vs_duck") or {}
    rnd = rep.get("eval_random_vs_duck") or rep.get("eval_random") or {}
    shots = float((rep.get("eval") or {}).get("shots", 0.0))
    duck_shots = float(duck.get("shots", 0.0))
    duck_k = float(duck.get("kills", 0.0))
    rnd_k = float(rnd.get("kills", 0.0))
    return (shots > 0.0 or duck_shots > 0.0) and duck_k > rnd_k


def _maddpg_act_fn(policy: Any, greedy: bool):
    def act(obs: np.ndarray) -> np.ndarray:
        import torch
        from tianshou.data import Batch

        was = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                batch = policy(
                    Batch(obs=np.asarray(obs, dtype=np.float32)[None], info={})
                )
                a = np.asarray(batch.act.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            if not greedy:
                a = a + np.random.normal(0.0, 0.1, size=a.shape).astype(np.float32)
            return np.clip(a, -1.0, 1.0).astype(np.float32)
        finally:
            policy.train(was)

    return act


def train_ippo(spec: TrainSpec) -> dict[str, Any]:
    if not _tianshou_ok():
        return _skipped("tianshou/torch not installed (pip install -e '.[train]')", algo=spec.algo)

    import torch
    from tianshou.algorithm import PPO
    from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.trainer import OnPolicyTrainerParams
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import DiscreteCritic

    n_games = max(1, spec.n_envs)
    team_concat = spec.algo.lower() == "mappo"
    probe = IppoNativeVec(spec, n_games)
    obs_shape = probe.observation_space.shape
    act_space = probe.action_space
    own_dim = probe.own_dim
    hidden = list(_HYPER["hidden"])
    actor_shape = (own_dim,) if team_concat else obs_shape
    actor = _factored_discrete_actor_cls()(
        preprocess_net=Net(state_shape=actor_shape, hidden_sizes=hidden),
        fire_logit_bias=float(_HYPER["fire_logit_bias"]),
        own_dim=own_dim if team_concat else None,
    )
    critic = DiscreteCritic(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=hidden),
    )
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=torch.distributions.Categorical,
        action_space=act_space,
        deterministic_eval=False,
        action_scaling=False,
        action_bound_method=None,
    )
    algo = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=float(_HYPER["lr"])),
        eps_clip=float(_HYPER["eps_clip"]),
        ent_coef=float(_HYPER["ent_coef"]),
        vf_coef=float(_HYPER["vf_coef"]),
        gae_lambda=float(_HYPER["gae_lambda"]),
        gamma=float(_HYPER["gamma"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo.to(device)
    sched = _smoke_schedule(spec, probe.env_num)
    smoke = bool(sched["smoke"])
    max_epochs = int(sched["max_epochs"])
    epoch_num_steps = int(sched["epoch_num_steps"])
    collect_steps = int(sched["collect_steps"])
    batch_size = int(sched["batch_size"])
    eval_n = int(sched["eval_n"])
    verbose = bool(sched["verbose"])
    repeat = int(_HYPER["update_repeat"])
    curve: list[dict[str, float]] = []
    env_step = 0
    n_slots = probe.env_num
    result: Any = "manual"

    def training_fn(epoch: int, step: int) -> None:
        n_mid = min(8, eval_n) if not smoke else 1
        ev_i = eval_policy_fn(spec, _ippo_act_fn(policy, greedy=True), n=n_mid)
        curve.append(
            {
                "epoch": float(epoch),
                "env_step": float(step),
                "return": ev_i["return"],
                "kills": ev_i["kills"],
                "mission_rate": ev_i["mission_rate"],
            }
        )

    def _ppo_update(buf: Any) -> None:
        if len(buf) < 8:
            return
        from tianshou.utils.torch_utils import policy_within_training_step

        with policy_within_training_step(policy):
            algo.update(buf, batch_size=min(batch_size, len(buf)), repeat=repeat)
        buf.reset()

    def _collect_update(v: IppoNativeVec, steps: int) -> int:
        buf = VectorReplayBuffer(max(20_000, spec.steps), buffer_num=v.env_num)
        col = Collector(algo, v, buf)
        col.reset()
        col.collect(n_step=v.env_num)
        got = col.collect(n_step=max(steps, v.env_num))
        _ppo_update(buf)
        n = int(getattr(got, "n_collected_steps", steps) or steps)
        return n if n > 0 else steps

    def _make_frozen_discrete(sd: dict[str, Any]):
        frozen = _factored_discrete_actor_cls()(
            preprocess_net=Net(state_shape=actor_shape, hidden_sizes=hidden),
            fire_logit_bias=float(_HYPER["fire_logit_bias"]),
            own_dim=own_dim if team_concat else None,
        )
        frozen.load_state_dict(sd)
        frozen.eval()

        def act(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                o = torch.as_tensor(np.asarray(obs, dtype=np.float32))
                if o.ndim == 1:
                    o = o.unsqueeze(0)
                probs, _ = frozen(o)
                a = int(probs.argmax(dim=-1).reshape(-1)[0].item())
            return _unflatten_discrete(a)

        return act

    if spec.sp_mode == "mixed":
        probe.close()
        vec_sp = IppoNativeVec(spec, n_games, self_play=True)
        n_slots = vec_sp.env_num
        half = max(epoch_num_steps // 2, vec_sp.env_num)
        opp_cycle = ("duck", "fsm")
        vec_script = IppoNativeVec(spec, n_games, self_play=False, opponent="duck")
        for epoch in range(1, max_epochs + 1):
            env_step += _collect_update(vec_sp, half)
            opp = opp_cycle[(epoch - 1) % 2]
            vec_script.close()
            vec_script = IppoNativeVec(spec, n_games, self_play=False, opponent=opp)
            env_step += _collect_update(vec_script, half)
            if not smoke:
                training_fn(epoch, env_step)
        vec_sp.close()
        vec_script.close()
    elif spec.sp_mode == "pfsp":
        probe.close()
        vec_pf = IppoNativeVec(spec, n_games, self_play=True, blue_only=True)
        n_slots = vec_pf.env_num
        pool: list[dict[str, Any]] = []
        wins: list[float] = []
        rng = np.random.default_rng(spec.seed + 17)

        def _pick(_g: int):
            if not pool:
                return _ippo_act_fn(policy, greedy=True)
            w = np.maximum(np.asarray(wins, dtype=np.float64), 0.1)
            idx = int(rng.choice(len(pool), p=w / w.sum()))
            return _make_frozen_discrete(pool[idx])

        vec_pf.set_pfsp_picker(_pick)
        buf = VectorReplayBuffer(max(20_000, spec.steps), buffer_num=vec_pf.env_num)
        col = Collector(algo, vec_pf, buf)
        col.reset()
        col.collect(n_step=vec_pf.env_num)
        for epoch in range(1, max_epochs + 1):
            got = col.collect(n_step=max(epoch_num_steps, vec_pf.env_num))
            env_step += int(getattr(got, "n_collected_steps", epoch_num_steps) or epoch_num_steps)
            _ppo_update(buf)
            pool.append(_clone_sd(actor))
            wins.append(max(float(np.mean(vec_pf._game_blue_ret)), 0.1))
            if len(pool) > 8:
                pool.pop(0)
                wins.pop(0)
            if not smoke:
                training_fn(epoch, env_step)
        vec_pf.close()
    else:
        buf = VectorReplayBuffer(max(20_000, spec.steps), buffer_num=probe.env_num)
        collector = Collector(algo, probe, buf)
        result = algo.run_training(
            OnPolicyTrainerParams(
                training_collector=collector,
                max_epochs=max_epochs,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collect_steps,
                update_step_num_repetitions=repeat,
                batch_size=batch_size,
                test_in_training=False,
                test_step_num_episodes=0,
                training_fn=None if smoke else training_fn,
                verbose=verbose,
                show_progress=verbose,
            )
        )
        env_step = int(getattr(result, "collect_step", spec.steps))
        probe.close()

    algo.eval()
    policy.eval()
    training_fn(max_epochs, env_step or spec.steps)
    act_g = _ippo_act_fn(policy, greedy=True)
    act_s = _ippo_act_fn(policy, greedy=False)
    packed = _eval_bundle(spec, act_g, act_s, eval_n)
    ev = packed["eval"]
    returns = [c["return"] for c in curve] or [ev["return"]]
    kills_c = [c["kills"] for c in curve] or [ev["kills"]]
    first = float(returns[0])
    last = float(returns[-1])
    name = "MAPPO (team-concat critic)" if team_concat else "IPPO (shared weights)"
    return {
        "skipped": False,
        "algo": spec.algo.lower(),
        "algorithm": name,
        "sp_mode": spec.sp_mode,
        "opponent": spec.opponent,
        "agents": spec.agents,
        "seed": spec.seed,
        "steps": spec.steps,
        "episodes": len(curve),
        "returns": [float(x) for x in returns],
        "kills_curve": [float(x) for x in kills_c],
        "curve": curve,
        "first_mean": first,
        "last_mean": last,
        "improved": last > first or ev["kills"] > packed["eval_random"]["kills"],
        "self_play": spec.self_play,
        "share_tracks": spec.share_tracks,
        "red_mission": spec.red_mission,
        "rewards": spec.rewards,
        "n_envs": n_games,
        "env_num_slots": n_slots,
        "vec_env": "NativeVecEnv",
        "device": str(device),
        "hyper": {
            **_HYPER,
            "n_envs": n_games,
            "n_games": n_games,
            "max_cycles": spec.max_cycles,
            "sp_mode": spec.sp_mode,
            "team_concat": team_concat,
        },
        "trainer": str(type(result)),
        **packed,
    }


def train_maddpg(spec: TrainSpec) -> dict[str, Any]:
    if not _tianshou_ok():
        return _skipped("tianshou/torch not installed (pip install -e '.[train]')", algo="maddpg")

    import torch
    from tianshou.algorithm import DDPG
    from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.trainer import OffPolicyTrainerParams
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic

    n_games = max(1, spec.n_envs)
    probe = IppoNativeVec(spec, n_games)
    obs_shape = probe.observation_space.shape
    act_space = probe.action_space
    own_dim = probe.own_dim
    hidden = list(_HYPER["hidden"])
    actor_pre = _own_slice_net(Net(state_shape=(own_dim,), hidden_sizes=hidden), own_dim)
    actor = ContinuousActorDeterministic(
        preprocess_net=actor_pre,
        action_shape=4,
        max_action=1.0,
    )
    critic = ContinuousCritic(
        preprocess_net=Net(
            state_shape=obs_shape,
            action_shape=4,
            hidden_sizes=hidden,
            concat=True,
        ),
    )
    policy = ContinuousDeterministicPolicy(
        actor=actor,
        action_space=act_space,
        exploration_noise="default",
        action_scaling=True,
        action_bound_method="clip",
    )
    algo = DDPG(
        policy=policy,
        policy_optim=AdamOptimizerFactory(lr=float(_HYPER["lr"])),
        critic=critic,
        critic_optim=AdamOptimizerFactory(lr=float(_HYPER["lr"])),
        gamma=float(_HYPER["gamma"]),
        tau=0.005,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo.to(device)
    sched = _smoke_schedule(spec, probe.env_num)
    smoke = bool(sched["smoke"])
    max_epochs = int(sched["max_epochs"])
    epoch_num_steps = int(sched["epoch_num_steps"])
    collect_steps = int(sched["collect_steps"])
    batch_size = int(sched["batch_size"])
    eval_n = int(sched["eval_n"])
    verbose = bool(sched["verbose"])
    curve: list[dict[str, float]] = []
    env_step = 0
    n_slots = probe.env_num
    result: Any = "manual"

    def training_fn(epoch: int, step: int) -> None:
        n_mid = min(8, eval_n) if not smoke else 1
        ev_i = eval_policy_fn(spec, _maddpg_act_fn(policy, greedy=True), n=n_mid)
        curve.append(
            {
                "epoch": float(epoch),
                "env_step": float(step),
                "return": ev_i["return"],
                "kills": ev_i["kills"],
                "mission_rate": ev_i["mission_rate"],
            }
        )

    def _ddpg_update(buf: Any, n_grad: int) -> None:
        if len(buf) < batch_size:
            return
        from tianshou.utils.torch_utils import policy_within_training_step

        with policy_within_training_step(policy):
            for _ in range(max(1, n_grad)):
                algo.update(buf, sample_size=batch_size)

    def _make_frozen_cont(sd: dict[str, Any]):
        pre = _own_slice_net(Net(state_shape=(own_dim,), hidden_sizes=hidden), own_dim)
        frozen = ContinuousActorDeterministic(
            preprocess_net=pre, action_shape=4, max_action=1.0
        )
        frozen.load_state_dict(sd)
        frozen.eval()

        def act(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                o = torch.as_tensor(np.asarray(obs, dtype=np.float32))
                if o.ndim == 1:
                    o = o.unsqueeze(0)
                a, _ = frozen(o)
            return np.clip(a.detach().cpu().numpy().reshape(-1)[:4], -1.0, 1.0).astype(
                np.float32
            )

        return act

    if spec.sp_mode == "mixed":
        probe.close()
        vec_sp = IppoNativeVec(spec, n_games, self_play=True)
        n_slots = vec_sp.env_num
        buf = VectorReplayBuffer(max(50_000, spec.steps), buffer_num=vec_sp.env_num)
        col_sp = Collector(algo, vec_sp, buf)
        col_sp.reset()
        col_sp.collect(n_step=vec_sp.env_num)
        half = max(epoch_num_steps // 2, vec_sp.env_num)
        opp_cycle = ("duck", "fsm")
        vec_script = IppoNativeVec(spec, n_games, self_play=False, opponent="duck")
        for epoch in range(1, max_epochs + 1):
            got = col_sp.collect(n_step=half)
            env_step += int(getattr(got, "n_collected_steps", half) or half)
            opp = opp_cycle[(epoch - 1) % 2]
            vec_script.close()
            vec_script = IppoNativeVec(spec, n_games, self_play=False, opponent=opp)
            buf_s = VectorReplayBuffer(max(20_000, spec.steps), buffer_num=vec_script.env_num)
            col_s = Collector(algo, vec_script, buf_s)
            col_s.reset()
            got_s = col_s.collect(n_step=half)
            env_step += int(getattr(got_s, "n_collected_steps", half) or half)
            _ddpg_update(buf, max(1, half // max(batch_size, 1)))
            _ddpg_update(buf_s, max(1, half // max(batch_size, 1)))
            if not smoke:
                training_fn(epoch, env_step)
        vec_sp.close()
        vec_script.close()
    elif spec.sp_mode == "pfsp":
        probe.close()
        vec_pf = IppoNativeVec(spec, n_games, self_play=True, blue_only=True)
        n_slots = vec_pf.env_num
        pool: list[dict[str, Any]] = []
        wins: list[float] = []
        rng = np.random.default_rng(spec.seed + 19)

        def _pick(_g: int):
            if not pool:
                return _maddpg_act_fn(policy, greedy=True)
            w = np.maximum(np.asarray(wins, dtype=np.float64), 0.1)
            idx = int(rng.choice(len(pool), p=w / w.sum()))
            return _make_frozen_cont(pool[idx])

        vec_pf.set_pfsp_picker(_pick)
        buf = VectorReplayBuffer(max(50_000, spec.steps), buffer_num=vec_pf.env_num)
        col = Collector(algo, vec_pf, buf)
        col.reset()
        col.collect(n_step=vec_pf.env_num)
        for epoch in range(1, max_epochs + 1):
            got = col.collect(n_step=max(epoch_num_steps, vec_pf.env_num))
            n = int(getattr(got, "n_collected_steps", epoch_num_steps) or epoch_num_steps)
            env_step += n
            _ddpg_update(buf, max(1, n // max(batch_size, 1)))
            pool.append(_clone_sd(actor))
            wins.append(max(float(np.mean(vec_pf._game_blue_ret)), 0.1))
            if len(pool) > 8:
                pool.pop(0)
                wins.pop(0)
            if not smoke:
                training_fn(epoch, env_step)
        vec_pf.close()
    else:
        buf = VectorReplayBuffer(max(50_000, spec.steps), buffer_num=probe.env_num)
        collector = Collector(algo, probe, buf)
        result = algo.run_training(
            OffPolicyTrainerParams(
                training_collector=collector,
                max_epochs=max_epochs,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collect_steps,
                batch_size=batch_size,
                update_step_num_gradient_steps_per_sample=1.0,
                test_in_training=False,
                test_step_num_episodes=0,
                training_fn=None if smoke else training_fn,
                verbose=verbose,
                show_progress=verbose,
            )
        )
        env_step = int(getattr(result, "collect_step", spec.steps))
        probe.close()

    algo.eval()
    policy.eval()
    training_fn(max_epochs, env_step or spec.steps)
    act_g = _maddpg_act_fn(policy, greedy=True)
    act_s = _maddpg_act_fn(policy, greedy=False)
    packed = _eval_bundle(spec, act_g, act_s, eval_n)
    ev = packed["eval"]
    returns = [c["return"] for c in curve] or [ev["return"]]
    kills_c = [c["kills"] for c in curve] or [ev["kills"]]
    first = float(returns[0])
    last = float(returns[-1])
    return {
        "skipped": False,
        "algo": "maddpg",
        "algorithm": "MADDPG (shared actor, team-concat critic, own action)",
        "sp_mode": spec.sp_mode,
        "opponent": spec.opponent,
        "agents": spec.agents,
        "seed": spec.seed,
        "steps": spec.steps,
        "episodes": len(curve),
        "returns": [float(x) for x in returns],
        "kills_curve": [float(x) for x in kills_c],
        "curve": curve,
        "first_mean": first,
        "last_mean": last,
        "improved": last > first or ev["kills"] > packed["eval_random"]["kills"],
        "self_play": spec.self_play,
        "share_tracks": spec.share_tracks,
        "red_mission": spec.red_mission,
        "rewards": spec.rewards,
        "n_envs": n_games,
        "env_num_slots": n_slots,
        "vec_env": "NativeVecEnv",
        "device": str(device),
        "hyper": {
            **_HYPER,
            "n_envs": n_games,
            "n_games": n_games,
            "max_cycles": spec.max_cycles,
            "sp_mode": spec.sp_mode,
            "team_concat": True,
            "action_type": "continuous",
            "critic": "Q(team-concat obs, own action)",
        },
        "trainer": str(type(result)),
        **packed,
    }


def train(spec: TrainSpec) -> dict[str, Any]:
    algo = spec.algo.lower()
    if algo in {"ppo", "ippo", "reinforce", "mappo"}:
        return train_ippo(spec)
    if algo == "maddpg":
        return train_maddpg(spec)
    raise ValueError(f"unknown algo {spec.algo}")


def train_ppo(spec: TrainSpec) -> dict[str, Any]:
    return train_ippo(spec)


def train_benchmarl(spec: TrainSpec) -> dict[str, Any]:
    """CLI compatibility. Trains IPPO/MAPPO/MADDPG on Tianshou, not BenchMARL."""
    return train(spec)


def _strip_actor(rep: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rep.items() if k != "actor"}


def run_core(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
    max_jobs: Optional[int] = None,
    on_job: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    del out_dir
    smoke = profile == "smoke"
    eval_n = 2 if smoke else 50
    seeds = (0,) if smoke else (0, 1, 2)
    n_envs = 2 if smoke else 8
    jobs: list[TrainSpec] = []
    for seed in seeds:
        jobs.append(
            TrainSpec(
                algo="ippo",
                opponent="duck",
                agents=1,
                seed=seed,
                steps=80 if smoke else 200_000,
                eval_episodes=eval_n,
                n_envs=n_envs,
            )
        )
    if not smoke:
        for seed in seeds:
            jobs.append(
                TrainSpec(
                    algo="ippo",
                    opponent="fsm",
                    agents=1,
                    seed=seed,
                    steps=200_000,
                    eval_episodes=eval_n,
                    n_envs=n_envs,
                )
            )
        for seed in seeds:
            jobs.append(
                TrainSpec(
                    algo="ippo",
                    opponent="duck",
                    agents=2,
                    seed=seed,
                    steps=200_000,
                    eval_episodes=eval_n,
                    n_envs=n_envs,
                )
            )
        for seed in seeds:
            jobs.append(
                TrainSpec(
                    algo="ippo",
                    opponent="fsm",
                    agents=2,
                    seed=seed,
                    steps=200_000,
                    eval_episodes=eval_n,
                    n_envs=n_envs,
                )
            )
    else:
        jobs.append(
            TrainSpec(
                algo="ippo",
                opponent="duck",
                agents=2,
                seed=0,
                steps=80,
                eval_episodes=2,
                n_envs=2,
            )
        )
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    results = []
    duck_2v2_ok = False
    for spec in jobs:
        if spec.agents >= 4:
            results.append(
                {
                    **_skipped("4v4 gated", algo=spec.algo),
                    "agents": spec.agents,
                    "opponent": spec.opponent,
                    "seed": spec.seed,
                    "transfer": True,
                }
            )
            continue
        if spec.agents == 2 and spec.opponent == "fsm" and not duck_2v2_ok:
            results.append(
                {
                    **_skipped(
                        "2v2 FSM gated: 2v2 vs duck greedy kill rate did not exceed random",
                        algo=spec.algo,
                    ),
                    "agents": spec.agents,
                    "opponent": spec.opponent,
                    "seed": spec.seed,
                    "gated": True,
                }
            )
            continue
        try:
            rep = _strip_actor(train(spec))
        except Exception as exc:
            rep = {
                **_skipped(str(exc), algo=spec.algo),
                "failed": True,
                "agents": spec.agents,
                "opponent": spec.opponent,
                "seed": spec.seed,
            }
        results.append(rep)
        if (
            spec.agents == 2
            and spec.opponent == "duck"
            and not rep.get("skipped")
            and float((rep.get("eval") or {}).get("kills", 0))
            > float((rep.get("eval_random") or {}).get("kills", 0))
        ):
            duck_2v2_ok = True
        if on_job:
            on_job(rep)
    return {
        "recipe": "marl_core",
        "profile": profile,
        "n_jobs": len(results),
        "jobs": results,
        "duck_2v2_learned": duck_2v2_ok,
        "algorithm": "IPPO (shared weights)",
    }


def run_ablations(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    del out_dir
    smoke = profile == "smoke"
    spec = TrainSpec(
        algo="ippo",
        opponent="fsm",
        agents=2,
        seed=0,
        steps=80 if smoke else 10_000,
        eval_episodes=2 if smoke else 20,
        red_mission="striker",
        n_envs=2,
    )
    return {
        "recipe": "marl_ablations",
        "profile": profile,
        "n_jobs": 1,
        "jobs": [_strip_actor(train(spec))],
    }


def run_selfplay(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
    on_job: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    del out_dir
    smoke = profile == "smoke"
    eval_n = 2 if smoke else 50
    seeds = (0,) if smoke else (0, 1, 2)
    n_envs = 2 if smoke else 8
    if smoke:
        grid = (
            ("ippo", "mixed", 2, 80),
            ("ippo", "pfsp", 2, 80),
            ("mappo", "mixed", 2, 80),
            ("maddpg", "mixed", 2, 80),
        )
    else:
        grid = tuple(
            (algo, mode, 2, 100_000)
            for algo in ("ippo", "mappo", "maddpg")
            for mode in ("mixed", "pfsp")
        )
    results: list[dict[str, Any]] = []
    gate: dict[tuple[str, str], bool] = {}

    def _run_job(algo: str, mode: str, agents: int, steps: int, seed: int) -> dict[str, Any]:
        spec = TrainSpec(
            algo=algo,
            opponent="duck",
            agents=agents,
            seed=seed,
            steps=steps,
            eval_episodes=eval_n,
            n_envs=n_envs,
            sp_mode=mode,
        )
        try:
            rep = _strip_actor(train(spec))
        except Exception as exc:
            rep = {
                **_skipped(str(exc), algo=algo),
                "failed": True,
                "agents": agents,
                "opponent": "self",
                "seed": seed,
                "self_play": True,
                "sp_mode": mode,
            }
        ev = rep.get("eval") or {}
        duck = rep.get("eval_vs_duck") or {}
        print(
            f"selfplay {algo} {mode} {agents}v{agents} seed={seed} "
            f"skip={bool(rep.get('skipped'))} "
            f"K={ev.get('kills')} S={ev.get('shots')} duckK={duck.get('kills')}",
            flush=True,
        )
        return rep

    for algo, mode, agents, steps in grid:
        for seed in seeds:
            rep = _run_job(algo, mode, agents, steps, seed)
            results.append(rep)
            if on_job:
                on_job(rep)
            key = (algo, mode)
            gate[key] = bool(gate.get(key)) or _gate_4v4(rep)

    if not smoke:
        for algo, mode in (("ippo", "mixed"), ("ippo", "pfsp"), ("mappo", "mixed"), ("mappo", "pfsp"), ("maddpg", "mixed"), ("maddpg", "pfsp")):
            if not gate.get((algo, mode)):
                continue
            for seed in seeds:
                rep = _run_job(algo, mode, 4, 50_000, seed)
                results.append(rep)
                if on_job:
                    on_job(rep)

    return {
        "recipe": "marl_selfplay",
        "profile": profile,
        "n_jobs": len(results),
        "jobs": results,
        "algorithm": "IPPO/MAPPO/MADDPG mixed+PFSP self-play",
        "self_play": True,
        "gate_4v4": {f"{a}_{m}": v for (a, m), v in gate.items()},
        "naive_ippo_baseline": "reuse disk artifact; not re-run",
    }
