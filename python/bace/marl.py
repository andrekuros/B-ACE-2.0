"""Independent PPO (shared weights) on NativeVecEnv. Skip if extras missing."""

from __future__ import annotations

from dataclasses import dataclass
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
        )
        obs, _ = env.reset(seed=spec.seed + 50_000 + i)
        ep = 0.0
        while env.agents:
            actions = {a: act_fn(obs[a]) for a in env.agents}
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
    """Tianshou vector env: NativeVecEnv games, independent per-agent slots (IPPO)."""

    is_async = False

    def __init__(self, spec: TrainSpec, n_games: int):
        self.waiting_id: list[int] = []
        self.is_closed = False
        self._n_agents = max(1, spec.agents)
        self._n_games = max(1, n_games)
        self.env_num = self._n_games * self._n_agents
        self._vec = BaceVecEnv(
            num_envs=self._n_games,
            auto_reset=False,
            opponent=spec.opponent,
            agents=self._n_agents,
            seed=spec.seed,
            max_cycles=spec.max_cycles,
            share_tracks=spec.share_tracks,
            red_mission=spec.red_mission,
            rewards=spec.rewards,
            action_type="discrete",
        )
        self._agent_ids = list(self._vec.possible_agents)
        self.observation_space = self._vec.observation_space()
        self.action_space = spaces.Discrete(N_FLAT)
        self.workers: list[Any] = [None] * self.env_num
        self._zeros = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._obs = np.zeros((self.env_num, *self.observation_space.shape), dtype=np.float32)
        self._seed0 = spec.seed

    def __len__(self) -> int:
        return self.env_num

    def _wrap_id(self, env_id: Any) -> list[int]:
        if env_id is None:
            return list(range(self.env_num))
        if isinstance(env_id, (int, np.integer)):
            return [int(env_id)]
        return [int(i) for i in np.asarray(env_id).reshape(-1)]

    def _fill_game(self, game: int, step: dict[str, Any]) -> None:
        for a, name in enumerate(self._agent_ids):
            sid = game * self._n_agents + a
            self._obs[sid] = np.asarray(step["obs"].get(name, self._zeros), dtype=np.float32)

    def reset(self, env_id: Any = None, **kwargs: Any):
        ids = self._wrap_id(env_id)
        seed = kwargs.get("seed", self._seed0)
        games = sorted({i // self._n_agents for i in ids})
        for g in games:
            s = None if seed is None else int(seed) + g
            self._fill_game(g, self._vec.reset_at(g, seed=s))
        obs = np.stack([self._obs[i] for i in ids], axis=0)
        infos = np.array([{"env_id": i} for i in ids])
        return obs, infos

    def step(self, action: Any, id: Any = None):
        ids = self._wrap_id(id)
        action = np.asarray(action).reshape(-1)
        acts: list[dict[str, np.ndarray]] = []
        id_set = set(ids)
        for g in range(self._n_games):
            d: dict[str, np.ndarray] = {}
            for a, name in enumerate(self._agent_ids):
                sid = g * self._n_agents + a
                if sid in id_set:
                    local = ids.index(sid)
                    d[name] = _unflatten_discrete(int(action[local]))
                else:
                    d[name] = np.array([0, 2, 2], dtype=np.int64)
            acts.append(d)
        steps = self._vec.step(acts)
        obs_l, rew_l, term_l, trunc_l, info_l = [], [], [], [], []
        for sid in ids:
            g = sid // self._n_agents
            name = self._agent_ids[sid % self._n_agents]
            st = steps[g]
            ended = str(st.get("end", "Ongoing")) not in {"Ongoing", "ongoing"}
            o = np.asarray(st["obs"].get(name, self._zeros), dtype=np.float32)
            self._obs[sid] = o
            obs_l.append(o)
            rew_l.append(float(st["rewards"].get(name, 0.0)))
            term_l.append(bool(ended))
            trunc_l.append(False)
            info_l.append({"env_id": sid})
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
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "hidden": [128, 128],
    "update_repeat": 4,
    "batch_size": 256,
    "discrete_map": "MultiDiscrete([2,5,5]) flatten Discrete(50); greedy fire = marginal P(fire)",
    "reward_preset": "rl",
    "algorithm": "IPPO (shared weights, independent per-agent actions)",
}


def train_ippo(spec: TrainSpec) -> dict[str, Any]:
    if not _tianshou_ok():
        return _skipped("tianshou/torch not installed (pip install -e '.[train]')", algo="ippo")

    import torch
    from tianshou.algorithm import PPO
    from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.trainer import OnPolicyTrainerParams
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic

    n_games = max(1, spec.n_envs)
    train_env = IppoNativeVec(spec, n_games)
    obs_shape = train_env.observation_space.shape
    act_space = train_env.action_space
    hidden = list(_HYPER["hidden"])
    actor = DiscreteActor(
        preprocess_net=Net(state_shape=obs_shape, hidden_sizes=hidden),
        action_shape=act_space.n,
        softmax_output=True,
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
    buf = VectorReplayBuffer(max(20_000, spec.steps), buffer_num=train_env.env_num)
    collector = Collector(algo, train_env, buf)

    smoke = spec.steps <= 80
    if smoke:
        max_epochs = 2
        epoch_num_steps = max(train_env.env_num, spec.steps)
        collect_steps = max(train_env.env_num, spec.steps)
        batch_size = 32
        eval_n = 2
        verbose = False
    else:
        collect_steps = max(train_env.env_num * 32, min(2048, spec.steps))
        epoch_num_steps = max(collect_steps, min(10_000, spec.steps))
        max_epochs = max(1, spec.steps // epoch_num_steps)
        batch_size = int(_HYPER["batch_size"])
        eval_n = spec.eval_episodes
        verbose = True

    curve: list[dict[str, float]] = []

    def training_fn(epoch: int, env_step: int) -> None:
        n_mid = min(8, eval_n) if not smoke else 1
        ev_i = eval_policy_fn(spec, _ippo_act_fn(policy, greedy=True), n=n_mid)
        curve.append(
            {
                "epoch": float(epoch),
                "env_step": float(env_step),
                "return": ev_i["return"],
                "kills": ev_i["kills"],
                "mission_rate": ev_i["mission_rate"],
            }
        )

    result = algo.run_training(
        OnPolicyTrainerParams(
            training_collector=collector,
            max_epochs=max_epochs,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collect_steps,
            update_step_num_repetitions=int(_HYPER["update_repeat"]),
            batch_size=batch_size,
            test_in_training=False,
            test_step_num_episodes=0,
            training_fn=None if smoke else training_fn,
            verbose=verbose,
            show_progress=verbose,
        )
    )
    algo.eval()
    policy.eval()
    training_fn(max_epochs, spec.steps)

    act_g = _ippo_act_fn(policy, greedy=True)
    act_s = _ippo_act_fn(policy, greedy=False)
    ev = eval_policy_fn(spec, act_g, n=eval_n)
    ev_stoch = eval_policy_fn(spec, act_s, n=eval_n)
    rnd = eval_random(spec, n=eval_n)
    fire = eval_scripted(
        "fire_once",
        opponent=spec.opponent,
        agents=spec.agents,
        n=eval_n,
        max_cycles=spec.max_cycles,
        seed=spec.seed,
    )
    fire16 = eval_scripted(
        "fire_once",
        opponent=spec.opponent,
        agents=spec.agents,
        n=eval_n,
        max_cycles=spec.max_cycles,
        seed=spec.seed,
        spawn=wez_close_overlay(16.0),
    )
    n_slots = train_env.env_num
    train_env.close()
    returns = [c["return"] for c in curve] or [ev["return"]]
    kills_c = [c["kills"] for c in curve] or [ev["kills"]]
    first = float(returns[0])
    last = float(returns[-1])
    beat_random = ev["kills"] > rnd["kills"] or (
        ev["kills"] >= rnd["kills"] and ev["return"] > rnd["return"]
    )
    return {
        "skipped": False,
        "algo": "ippo",
        "algorithm": "IPPO (shared weights)",
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
        "improved": last > first or ev["kills"] > rnd["kills"],
        "eval": ev,
        "eval_greedy": ev,
        "eval_stochastic": ev_stoch,
        "eval_random": rnd,
        "eval_fire_once": fire,
        "eval_fire_once_16nm": fire16,
        "beat_random": beat_random,
        "share_tracks": spec.share_tracks,
        "red_mission": spec.red_mission,
        "rewards": spec.rewards,
        "n_envs": n_games,
        "env_num_slots": n_slots,
        "vec_env": "NativeVecEnv",
        "device": str(device),
        "hyper": {**_HYPER, "n_envs": n_games, "n_games": n_games, "max_cycles": spec.max_cycles},
        "trainer": str(type(result)),
    }


def train(spec: TrainSpec) -> dict[str, Any]:
    algo = spec.algo.lower()
    if algo in {"ppo", "ippo", "reinforce"}:
        return train_ippo(spec)
    if algo in {"mappo", "maddpg"}:
        return _skipped(
            "MAPPO/MADDPG is not trained in this release; use IPPO (shared weights).",
            algo=algo,
        )
    raise ValueError(f"unknown algo {spec.algo}")


def train_ppo(spec: TrainSpec) -> dict[str, Any]:
    return train_ippo(spec)


def train_benchmarl(spec: TrainSpec) -> dict[str, Any]:
    """CLI compatibility. MAPPO/MADDPG are skipped; IPPO is the trained algorithm."""
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
                    steps=100_000,
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
                    steps=100_000,
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
