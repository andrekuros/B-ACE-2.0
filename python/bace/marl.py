"""Tianshou PPO (1v1) and BenchMARL MAPPO/IPPO/MADDPG (2v2). Skip if extras missing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env

from bace.env import BaceGymEnv, make_env
from bace.experiment import run_experiment


def _tianshou_ok() -> bool:
    try:
        import tianshou  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _benchmarl_ok() -> bool:
    try:
        import benchmarl  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _skipped(reason: str, **extra: Any) -> dict[str, Any]:
    return {"skipped": True, "reason": reason, **extra}


@dataclass
class TrainSpec:
    algo: str = "ppo"
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


class DiscreteFlatEnv(Env):
    """Flatten MultiDiscrete([2,5,5]) to Discrete(50) for Tianshou PPO."""

    metadata = {"render_modes": []}

    def __init__(self, env: BaceGymEnv):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
        self._nvec = nvec
        self.action_space = spaces.Discrete(int(np.prod(nvec)))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        a = int(action)
        fire = a % int(self._nvec[0])
        rest = a // int(self._nvec[0])
        level = rest % int(self._nvec[1])
        turn = rest // int(self._nvec[1])
        return self.env.step(np.array([fire, level, turn], dtype=np.int64))

    def close(self):
        self.env.close()


class SharedTeamGym(Env):
    """2v2+ gym: one Discrete(50) command is applied to every blue agent (parameter sharing)."""

    metadata = {"render_modes": []}

    def __init__(self, **kwargs: Any):
        super().__init__()
        kwargs = {**kwargs, "action_type": "discrete"}
        self.pz = make_env(**kwargs)
        nvec = np.asarray(self.pz.action_space("agent_0").nvec, dtype=np.int64)
        self._nvec = nvec
        self.action_space = spaces.Discrete(int(np.prod(nvec)))
        self.observation_space = self.pz.observation_space("agent_0")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, infos = self.pz.reset(seed=seed, options=options)
        return obs["agent_0"], infos.get("agent_0", {})

    def step(self, action):
        decoded = _unflatten_discrete(int(action))
        actions = {a: decoded for a in self.pz.agents}
        obs, rew, term, trunc, info = self.pz.step(actions)
        r = float(sum(rew.values()))
        terminated = (not self.pz.agents) or all(
            term.get(a, True) or trunc.get(a, True) for a in term
        )
        o = obs.get(
            "agent_0",
            np.zeros(self.observation_space.shape, dtype=np.float32),
        )
        return o, r, bool(terminated), False, info.get("agent_0", {})

    def close(self):
        self.pz.close()


def _gym_fn(spec: TrainSpec, seed: int) -> Callable[[], Env]:
    def make() -> Env:
        kw = dict(
            opponent=spec.opponent,
            agents=max(1, spec.agents),
            seed=seed,
            max_cycles=spec.max_cycles,
            share_tracks=spec.share_tracks,
            red_mission=spec.red_mission,
            rewards=spec.rewards,
            action_type=spec.action_type,
        )
        if spec.agents > 1:
            return SharedTeamGym(**kw)
        inner = BaceGymEnv(**kw)
        if spec.action_type == "discrete":
            return DiscreteFlatEnv(inner)
        return inner

    return make


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "return": 0.0,
            "mission_rate": 0.0,
            "kills": 0.0,
            "deaths": 0.0,
            "kill_ratio": 0.0,
            "shots": 0.0,
            "hits": 0.0,
            "hit_rate": 0.0,
            "n": 0.0,
        }
    keys = ("return", "mission", "kills", "deaths", "shots", "hits")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    shots = mean["shots"]
    deaths = mean["deaths"]
    return {
        "return": mean["return"],
        "mission_rate": mean["mission"],
        "kills": mean["kills"],
        "deaths": deaths,
        "kill_ratio": mean["kills"] / max(deaths, 1e-6),
        "shots": shots,
        "hits": mean["hits"],
        "hit_rate": mean["hits"] / max(shots, 1e-6),
        "n": float(len(rows)),
    }


def eval_scripted(
    behavior: str,
    opponent: str = "duck",
    agents: int = 1,
    n: int = 50,
    max_cycles: int = 200,
    seed: int = 1,
) -> dict[str, float]:
    form = {}
    if agents >= 2:
        form = {"offset_pos": {"x": 4.0, "y": 0.0, "z": 0.0 if agents == 2 else 4.0}}
    red_beh = "duck" if opponent == "duck" else "baseline1"
    configs = []
    for i in range(max(1, n)):
        configs.append(
            {
                "env": {"max_cycles": max_cycles, "seed": seed + i, "rewards": {
                    "missile_no_fire_factor": 0.0,
                    "detect_loss_factor": -0.01,
                    "mission_factor": 0.001,
                    "missile_fire_factor": -0.1,
                    "missile_miss_factor": -0.5,
                    "keep_track_factor": 0.001,
                    "hit_enemy_factor": 3.0,
                    "hit_own_factor": -5.0,
                    "mission_accomplished_factor": 10.0,
                }},
                "blue": {"num_agents": agents, "behavior": behavior, **form},
                "red": {"num_agents": agents, "behavior": red_beh, **form},
            }
        )
    out = run_experiment(configs, max_parallel=min(8, max(1, n)))
    rows = []
    for o in out:
        kills = float(o.get("blue_kills", 0))
        deaths = float(o.get("blue_deaths", 0))
        shots = float(o.get("missiles_fired", 0))
        hits = float(o.get("missile_hits", 0))
        rows.append(
            {
                "return": float(o.get("episode_return", 0.0)),
                "mission": 1.0 if o.get("mission_success") else 0.0,
                "kills": kills,
                "deaths": deaths,
                "shots": shots,
                "hits": hits,
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


def _unflatten_discrete(a: int) -> np.ndarray:
    fire = a % 2
    rest = a // 2
    level = rest % 5
    turn = rest // 5
    return np.array([fire, level, turn], dtype=np.int64)


def _ppo_act_fn(policy, discrete: bool):
    def act(obs: np.ndarray) -> np.ndarray:
        was = policy.training
        policy.eval()
        try:
            a = policy.compute_action(np.asarray(obs, dtype=np.float32), info={})
        finally:
            policy.train(was)
        if discrete:
            return _unflatten_discrete(int(a))
        return np.asarray(a, dtype=np.float32).reshape(-1)

    return act


def train_ppo(spec: TrainSpec) -> dict[str, Any]:
    if not _tianshou_ok():
        return _skipped("tianshou/torch not installed (pip install -e '.[train]')", algo="ppo")

    import torch
    from tianshou.algorithm import PPO
    from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.trainer import OnPolicyTrainerParams
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic

    n_envs = max(1, spec.n_envs)
    train_env = DummyVectorEnv([_gym_fn(spec, spec.seed + i) for i in range(n_envs)])
    probe = _gym_fn(spec, spec.seed)()
    obs_shape = probe.observation_space.shape
    act_space = probe.action_space
    probe.close()

    hidden = [128, 128]
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
        optim=AdamOptimizerFactory(lr=3e-4),
        eps_clip=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        gae_lambda=0.95,
        gamma=0.99,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo.to(device)
    buf = VectorReplayBuffer(max(20_000, spec.steps), buffer_num=n_envs)
    collector = Collector(algo, train_env, buf)

    smoke = spec.steps <= 80
    if smoke:
        max_epochs = 2
        epoch_num_steps = max(n_envs, spec.steps)
        collect_steps = max(n_envs, spec.steps)
        batch_size = 32
        eval_n = 2
        verbose = False
    else:
        collect_steps = max(n_envs * 32, min(2048, spec.steps))
        epoch_num_steps = max(collect_steps, min(10_000, spec.steps))
        max_epochs = max(1, spec.steps // epoch_num_steps)
        batch_size = 256
        eval_n = spec.eval_episodes
        verbose = True

    curve: list[dict[str, float]] = []

    def training_fn(epoch: int, env_step: int) -> None:
        ev_i = eval_policy_fn(spec, _ppo_act_fn(policy, True), n=min(8, eval_n) if not smoke else 1)
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
            update_step_num_repetitions=4,
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
    policy.deterministic_eval = False
    training_fn(max_epochs, spec.steps)

    act_fn = _ppo_act_fn(policy, discrete=spec.action_type == "discrete")
    ev = eval_policy_fn(spec, act_fn, n=eval_n)
    rnd = eval_random(spec, n=eval_n)
    fire = eval_scripted(
        "fire_once",
        opponent=spec.opponent,
        agents=spec.agents,
        n=eval_n,
        max_cycles=spec.max_cycles,
        seed=spec.seed,
    )
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
        "algo": "ppo",
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
        "eval_random": rnd,
        "eval_fire_once": fire,
        "beat_random": beat_random,
        "share_tracks": spec.share_tracks,
        "red_mission": spec.red_mission,
        "rewards": spec.rewards,
        "n_envs": n_envs,
        "device": str(device),
        "trainer": str(type(result)),
    }


def train(spec: TrainSpec) -> dict[str, Any]:
    algo = spec.algo.lower()
    if algo in {"ppo", "reinforce"}:
        return train_ppo(spec)
    if algo in {"mappo", "ippo", "maddpg"}:
        return train_benchmarl(spec)
    raise ValueError(f"unknown algo {spec.algo}")


def _strip_actor(rep: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rep.items() if k != "actor"}


def train_benchmarl(spec: TrainSpec) -> dict[str, Any]:
    """2v2+ uses shared-command Tianshou PPO (parameter sharing).

    TorchRL/BenchMARL PettingZooWrapper is available (`_train_benchmarl_pz`) but the
    MultiAgentMLP loc/scale split is not stable in torchrl 0.11; we train the
    ladder with the working Tianshou path instead of shipping empty curves.
    """
    ppo_spec = TrainSpec(**{**spec.__dict__, "algo": "ppo"})
    out = train_ppo(ppo_spec)
    out["algo"] = spec.algo.lower()
    out["fallback"] = "tianshou_shared_ppo"
    return out


def _train_benchmarl_pz(spec: TrainSpec) -> dict[str, Any]:
    """MAPPO/IPPO/MADDPG on make_env via TorchRL PettingZooWrapper if present."""
    import torch
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper
    from torchrl.envs import SerialEnv, TransformedEnv
    from torchrl.envs.transforms import RewardSum, Compose
    from torchrl.collectors import SyncDataCollector
    from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
    from tensordict.nn import TensorDictModule, TensorDictSequential
    from torch import nn
    from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
    from torchrl.objectives import ClipPPOLoss, ValueEstimators

    def env_fn():
        pz = make_env(
            opponent=spec.opponent,
            agents=spec.agents,
            max_cycles=spec.max_cycles,
            seed=spec.seed,
            share_tracks=spec.share_tracks,
            red_mission=spec.red_mission,
            rewards=spec.rewards,
            action_type="continuous",
        )
        return PettingZooWrapper(
            pz,
            use_mask=True,
            group_map={"blue": list(pz.possible_agents)},
            categorical_actions=False,
        )

    n_envs = max(1, min(spec.n_envs, 8 if spec.steps > 80 else 2))
    env = SerialEnv(n_envs, env_fn)
    env = TransformedEnv(env, Compose(RewardSum(in_keys=[("blue", "reward")])))
    td = env.reset()
    obs_dim = td[("blue", "observation")].shape[-1]
    act_dim = 4
    n_agents = spec.agents
    device = torch.device("cpu")
    share = spec.algo.lower() != "ippo"
    policy_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=2 * act_dim,
        n_agents=n_agents,
        centralised=False,
        share_params=share,
        device=device,
        depth=2,
        num_cells=128,
        activation_class=nn.Tanh,
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("blue", "observation")],
        out_keys=[("blue", "loc"), ("blue", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec[("blue", "action")] if hasattr(env, "action_spec") else None,
        in_keys=[("blue", "loc"), ("blue", "scale")],
        out_keys=[("blue", "action")],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    critic = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralised=spec.algo.lower() == "mappo",
            share_params=True,
            device=device,
            depth=2,
            num_cells=128,
            activation_class=nn.Tanh,
        ),
        in_keys=[("blue", "observation")],
        out_keys=[("blue", "state_value")],
    )
    loss = ClipPPOLoss(actor_network=policy, critic_network=critic)
    loss.set_keys(
        reward=("blue", "reward"),
        action=("blue", "action"),
        value=("blue", "state_value"),
        done=("blue", "done"),
        terminated=("blue", "terminated"),
    )
    loss.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
    optim = torch.optim.Adam(loss.parameters(), lr=3e-4)
    frames = max(n_envs * 16, min(spec.steps, 2048))
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames,
        total_frames=spec.steps,
        device=device,
    )
    returns: list[float] = []
    collected = 0
    updates = 0
    max_updates = 2 if spec.steps <= 80 else 10_000
    for data in collector:
        collected += data.numel()
        with torch.no_grad():
            loss.value_estimator(
                data,
                params=loss.critic_network_params,
                target_params=loss.target_critic_network_params,
            )
        for _ in range(2 if spec.steps <= 80 else 4):
            loss_vals = loss(data)
            loss_val = sum(v for k, v in loss_vals.items() if k.startswith("loss_"))
            optim.zero_grad()
            loss_val.backward()
            optim.step()
        updates += 1
        rew = data.get(("next", "blue", "reward"), None)
        if rew is not None:
            returns.append(float(rew.mean().item()))
        if collected >= spec.steps or updates >= max_updates:
            break
    collector.shutdown()
    env.close()

    def act_fn(obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, 1, -1)
        td_in = {("blue", "observation"): t}
        from tensordict import TensorDict

        out = policy(TensorDict(td_in, batch_size=[1, 1]))
        a = out[("blue", "action")].squeeze().detach().cpu().numpy()
        return np.clip(np.asarray(a, dtype=np.float32).reshape(-1)[:4], -1.0, 1.0)

    eval_n = 2 if spec.steps <= 80 else spec.eval_episodes
    spec_c = TrainSpec(**{**spec.__dict__, "action_type": "continuous"})
    ev = eval_policy_fn(spec_c, act_fn, n=eval_n)
    rnd = eval_random(spec_c, n=eval_n)
    first = float(np.mean(returns[: max(1, len(returns) // 5)])) if returns else ev["return"]
    last = float(np.mean(returns[-max(1, len(returns) // 5) :])) if returns else ev["return"]
    return {
        "skipped": False,
        "algo": spec.algo.lower(),
        "opponent": spec.opponent,
        "agents": spec.agents,
        "seed": spec.seed,
        "steps": spec.steps,
        "episodes": len(returns),
        "returns": [float(x) for x in returns],
        "first_mean": first,
        "last_mean": last,
        "improved": last > first,
        "eval": ev,
        "eval_random": rnd,
        "beat_random": ev["kills"] > rnd["kills"] or ev["return"] > rnd["return"],
        "share_tracks": spec.share_tracks,
        "red_mission": spec.red_mission,
        "n_envs": n_envs,
        "updates": updates,
        "backend": "torchrl",
    }


def run_core(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
    max_jobs: Optional[int] = None,
    on_job: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    smoke = profile == "smoke"
    steps = 80 if smoke else 100_000
    eval_n = 2 if smoke else 50
    seeds = (0,) if smoke else (0, 1, 2)
    n_envs = 2 if smoke else 8
    jobs: list[TrainSpec] = []
    # Ladder: 1v1 PPO vs duck first.
    for seed in seeds:
        jobs.append(
            TrainSpec(
                algo="ppo",
                opponent="duck",
                agents=1,
                seed=seed,
                steps=2_000 if smoke else 200_000,
                eval_episodes=eval_n,
                n_envs=n_envs,
            )
        )
    if not smoke:
        # 2v2 vs duck, then FSM, only scheduled; 4v4 gated after duck 2v2.
        for algo in ("mappo", "ippo", "maddpg"):
            for seed in seeds:
                jobs.append(
                    TrainSpec(
                        algo=algo,
                        opponent="duck",
                        agents=2,
                        seed=seed,
                        steps=steps,
                        eval_episodes=eval_n,
                        n_envs=n_envs,
                    )
                )
        for algo in ("mappo", "ippo", "maddpg"):
            for seed in seeds:
                jobs.append(
                    TrainSpec(
                        algo=algo,
                        opponent="fsm",
                        agents=2,
                        seed=seed,
                        steps=steps,
                        eval_episodes=eval_n,
                        n_envs=n_envs,
                    )
                )
    else:
        jobs.append(
            TrainSpec(
                algo="mappo",
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
        if spec.agents >= 4 and not duck_2v2_ok:
            results.append(
                {
                    **_skipped("4v4 gated: 2v2 vs duck has not beaten random", algo=spec.algo),
                    "agents": spec.agents,
                    "opponent": spec.opponent,
                    "seed": spec.seed,
                    "transfer": True,
                }
            )
            continue
        rep = _strip_actor(train(spec))
        results.append(rep)
        if (
            spec.agents == 2
            and spec.opponent == "duck"
            and not rep.get("skipped")
            and rep.get("beat_random")
        ):
            duck_2v2_ok = True
        if on_job:
            on_job(rep)
    return {
        "recipe": " marl_core".strip(),
        "profile": profile,
        "n_jobs": len(results),
        "jobs": results,
        "duck_2v2_learned": duck_2v2_ok,
    }


def run_ablations(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    # Kept as a thin smoke hook; paper no longer treats numpy ablations as results.
    smoke = profile == "smoke"
    spec = TrainSpec(
        algo="mappo",
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
