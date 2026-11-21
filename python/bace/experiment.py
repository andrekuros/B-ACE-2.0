"""Experiment CLI and shared runners for B-ACE 2.0 recipes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bace.env import BaceGymEnv, make_env


def _configs_root() -> Path:
    here = Path(__file__).resolve()
    repo = here.parents[2]
    if (repo / "configs").is_dir():
        return repo / "configs"
    cwd = Path.cwd() / "configs"
    if cwd.is_dir():
        return cwd
    return repo / "configs"


def _load_recipe_json(name: str) -> dict[str, Any]:
    path = _configs_root() / "experiments" / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def _pava_decreasing(y: Any) -> Any:
    np = __import__("numpy")
    vals = [float(v) for v in y]
    wts = [1.0] * len(vals)
    i = 0
    while i < len(vals) - 1:
        if vals[i] >= vals[i + 1] - 1e-15:
            i += 1
            continue
        merged = (wts[i] * vals[i] + wts[i + 1] * vals[i + 1]) / (wts[i] + wts[i + 1])
        wts[i] = wts[i] + wts[i + 1]
        vals[i] = merged
        del vals[i + 1]
        del wts[i + 1]
        if i:
            i -= 1
    out = np.empty(int(round(sum(wts))))
    k = 0
    for v, w in zip(vals, wts):
        c = int(round(w))
        out[k : k + c] = v
        k += c
    return out


def _crossing_p50(xs: Any, ys: Any) -> float:
    np = __import__("numpy")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0:
        return 0.0
    if ys[0] < 0.5:
        return float(xs[0])
    if ys[-1] >= 0.5:
        return float(xs[-1])
    for i in range(len(xs) - 1):
        if ys[i] >= 0.5 > ys[i + 1]:
            t = (ys[i] - 0.5) / (ys[i] - ys[i + 1] + 1e-9)
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return float(xs[-1])


def attach_wez_fits(report: dict[str, Any]) -> dict[str, Any]:
    """Logistic and isotonic P50 envelopes; polynomial kept as a diagnostic overlay."""
    np = __import__("numpy")
    cells = report.get("cells", [])
    fits = []
    for alt in sorted({c["altitude_ft"] for c in cells}):
        for asp in ("head", "beam", "tail"):
            rows = sorted(
                [
                    c
                    for c in cells
                    if c.get("aspect") == asp and abs(c.get("altitude_ft", 0) - alt) < 1
                ],
                key=lambda c: c["range_nm"],
            )
            if len(rows) < 4:
                continue
            xs = np.array([c["range_nm"] for c in rows], dtype=float)
            ys = np.array([c["hit_rate"] for c in rows], dtype=float)
            deg = int(min(4, len(xs) - 1))
            coef = np.polyfit(xs, ys, deg)
            iso = _pava_decreasing(ys)
            isotonic_p50 = _crossing_p50(xs, iso)
            best = (1e9, 0.4, float(xs.mean()))
            for k in np.linspace(0.05, 0.8, 16):
                for r0 in np.linspace(float(xs.min()), float(xs.max()), 16):
                    pred = 1.0 / (1.0 + np.exp(k * (xs - r0)))
                    err = float(np.mean((pred - ys) ** 2))
                    if err < best[0]:
                        best = (err, float(k), float(r0))
            logistic_p50 = float(best[2])
            rnez = float(xs[0])
            for r, p in zip(xs, iso):
                if p < 0.9:
                    rnez = float(r)
                    break
            fits.append(
                {
                    "altitude_ft": alt,
                    "aspect": asp,
                    "poly_coef": [float(c) for c in coef],
                    "logistic_k": best[1],
                    "logistic_r0": logistic_p50,
                    "logistic_p50_nm": logistic_p50,
                    "isotonic_p50_nm": isotonic_p50,
                    "empirical_rmax_nm": isotonic_p50,
                    "empirical_rnez_nm": rnez,
                    "analytic_rmax_nm": float(rows[0].get("analytic_rmax_nm", 0.0)),
                    "primary": "isotonic_p50",
                }
            )
    report["fits"] = fits
    traces = []
    for o in report.get("outcomes", []):
        cfg = o.get("config") or {}
        blue = (cfg.get("blue") or {}).get("init_position") or {}
        red = (cfg.get("red") or {}).get("init_position") or {}
        rng = abs(float(blue.get("z", 0)) - float(red.get("z", 0)))
        if abs(rng - 16.0) > 0.3 and abs(rng - 40.0) > 0.3:
            continue
        traces.append(
            {
                "range_nm": rng,
                "hit": int(o.get("missile_hits", 0)) > 0,
                "tof": float(o.get("missile_tof", 0.0)),
                "pitbull": bool(o.get("missile_pitbull", False)),
                "pitbull_time": float(o.get("missile_pitbull_time", 0.0)),
                "miss_cause": str(o.get("miss_cause", "")),
            }
        )
    report["traces"] = traces
    return report


def _native():
    try:
        from bace._native import run_experiment as _run
        from bace._native import run_fsm_search_py, run_wez_experiment
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "bace native extension is not built. Run: pip install -e . (maturin)."
        ) from exc
    return _run, run_wez_experiment, run_fsm_search_py


def run_experiment(
    configs: list[dict[str, Any]], max_parallel: int = 8
) -> list[dict[str, Any]]:
    """Run scenario dicts in the Rust batch runner. Returns CaseResult-like dicts."""
    run_fn, _, _ = _native()
    raw = run_fn(json.dumps(configs), max_parallel)
    return json.loads(raw)


@dataclass
class CaseResult:
    end: str
    steps: int
    seed: int
    blue_alive: int
    red_alive: int
    blue_kills: int
    blue_deaths: int
    mission_success: bool
    episode_return: float
    missiles_fired: int
    missile_hits: int
    missile_tof: float = 0.0
    missile_pitbull: bool = False
    miss_cause: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseResult:
        return cls(
            end=str(d.get("end", "")),
            steps=int(d.get("steps", 0)),
            seed=int(d.get("seed", 0)),
            blue_alive=int(d.get("blue_alive", 0)),
            red_alive=int(d.get("red_alive", 0)),
            blue_kills=int(d.get("blue_kills", 0)),
            blue_deaths=int(d.get("blue_deaths", 0)),
            mission_success=bool(d.get("mission_success", False)),
            episode_return=float(d.get("episode_return", 0.0)),
            missiles_fired=int(d.get("missiles_fired", 0)),
            missile_hits=int(d.get("missile_hits", 0)),
            missile_tof=float(d.get("missile_tof", 0.0)),
            missile_pitbull=bool(d.get("missile_pitbull", False)),
            miss_cause=str(d.get("miss_cause", "")),
        )


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_artifact(out_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{_stamp()}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_wez(
    params: Optional[dict[str, Any]] = None,
    max_parallel: int = 8,
    out_dir: Optional[Path] = None,
    smoke: bool = False,
    profile: str = "default",
) -> dict[str, Any]:
    _, wez_fn, _ = _native()
    if profile == "paper":
        merged = {**_load_recipe_json("wez_paper"), **(params or {})}
        if not merged:
            merged = {
                "ranges_nm": list(range(6, 42, 2)),
                "altitudes_ft": [10000, 25000, 40000],
                "aspects": ["head", "beam", "tail"],
                "repeats": 30,
                "max_cycles": 150,
                "seed": 1,
            }
    else:
        merged = {**_load_recipe_json("wez"), **(params or {})}
    if smoke or profile == "smoke":
        merged.update(
            {
                "ranges_nm": [16.0, 40.0],
                "altitudes_ft": [25000.0],
                "aspects": ["head"],
                "repeats": 2,
                "max_cycles": 80,
            }
        )
    report = json.loads(wez_fn(json.dumps(merged) if merged else None, max_parallel))
    attach_wez_fits(report)
    dest = out_dir or Path("runs/experiments")
    dest.mkdir(parents=True, exist_ok=True)
    json_path = write_artifact(dest, "wez", report)
    csv_path = dest / f"wez_{_stamp()}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "range_nm",
                "altitude_ft",
                "aspect",
                "n",
                "hits",
                "hit_rate",
                "fired",
                "analytic_rmax_nm",
                "analytic_rnez_nm",
            ],
        )
        w.writeheader()
        for row in report.get("cells", []):
            w.writerow({k: row.get(k) for k in w.fieldnames})
    try:
        import matplotlib.pyplot as plt

        cells = report.get("cells", [])
        if cells:
            fig, ax = plt.subplots(figsize=(6, 4))
            for asp in sorted({c["aspect"] for c in cells}):
                xs = [c["range_nm"] for c in cells if c["aspect"] == asp]
                ys = [c["hit_rate"] for c in cells if c["aspect"] == asp]
                ax.plot(xs, ys, marker="o", label=asp)
            ax.set_xlabel("range (NM)")
            ax.set_ylabel("hit rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(dest / f"wez_{_stamp()}.png")
            plt.close(fig)
    except Exception:
        pass
    report["_artifact"] = str(json_path)
    report["_csv"] = str(csv_path)
    return report


def run_fsm(
    params: Optional[dict[str, Any]] = None,
    max_parallel: int = 8,
    out_dir: Optional[Path] = None,
    smoke: bool = False,
    profile: str = "default",
) -> dict[str, Any]:
    _, _, fsm_fn = _native()
    if profile == "paper":
        merged = {**_load_recipe_json("fsm_paper"), **(params or {})}
        if not merged:
            merged = {
                "pop": 32,
                "generations": 40,
                "episodes": 20,
                "max_cycles": 400,
                "seed": 1,
                "num_agents": 2,
                "eval_agents": 4,
                "pool_interval": 3,
            }
    else:
        merged = {**_load_recipe_json("fsm"), **(params or {})}
    if smoke or profile == "smoke":
        merged.update(
            {
                "pop": 4,
                "generations": 1,
                "episodes": 2,
                "max_cycles": 40,
                "num_agents": 1,
                "eval_agents": 0,
            }
        )
    report = json.loads(fsm_fn(json.dumps(merged) if merged else None, max_parallel))
    dest = out_dir or Path("runs/experiments")
    json_path = write_artifact(dest, "fsm", report)
    base_dir = _configs_root() / "baselines"
    base_dir.mkdir(parents=True, exist_ok=True)
    for elite in report.get("elites", []):
        label = elite.get("label", "elite")
        genome = elite.get("genome", {})
        n = int(merged.get("num_agents", 2) or 2)
        n = max(1, min(4, n))
        form = {"offset_pos": {"x": 4.0, "y": 0.0, "z": 0.0 if n <= 2 else 4.0}}
        scenario = {
            "env": {"max_cycles": 400, "seed": 1},
            "blue": {"num_agents": n, "behavior": "external", **form},
            "red": {
                "num_agents": n,
                "behavior": "baseline1",
                "beh_config": {
                    "d_shot": [genome.get("d_shot", 0.85)],
                    "l_crank": [genome.get("l_crank", 0.6)],
                    "l_break": [genome.get("l_break", 0.95)],
                },
                **form,
            },
        }
        (base_dir / f"{label}.json").write_text(json.dumps(scenario, indent=2) + "\n")
    report["_artifact"] = str(json_path)
    return report


def _reinforce_train(steps: int, seed: int, opponent: str, agents: int) -> dict[str, Any]:
    rng = __import__("numpy").random.default_rng(seed)
    np = __import__("numpy")
    env = BaceGymEnv(opponent=opponent, agents=agents, seed=seed, max_cycles=200)
    obs, _ = env.reset(seed=seed)
    obs_dim = int(obs.shape[0])
    act_dim = 4
    w = rng.normal(0, 0.05, size=(act_dim, obs_dim))
    b = np.zeros(act_dim)
    log_std = np.full(act_dim, -0.7)
    returns: list[float] = []
    ep_ret = 0.0
    traj: list[tuple] = []
    lr = 1e-3
    for t in range(steps):
        mean = np.tanh(w @ obs + b)
        std = np.exp(log_std)
        noise = rng.normal(size=act_dim)
        action = np.clip(mean + std * noise, -1.0, 1.0).astype(np.float32)
        logp = -0.5 * np.sum(((action - mean) / (std + 1e-6)) ** 2)
        next_obs, reward, term, trunc, _ = env.step(action)
        ep_ret += float(reward)
        traj.append((obs, action, logp, float(reward)))
        obs = next_obs
        if term or trunc:
            g = 0.0
            for _o, _a, lp, r in reversed(traj):
                g = r + 0.99 * g
                grad_scale = lr * g * lp
                # small parameter nudge along sampled noise direction
                w += grad_scale * 0.01 * rng.normal(size=w.shape)
            returns.append(ep_ret)
            ep_ret = 0.0
            traj = []
            obs, _ = env.reset(seed=seed + t + 1)
    env.close()
    first = float(np.mean(returns[: max(1, len(returns) // 5)])) if returns else 0.0
    last = float(np.mean(returns[-max(1, len(returns) // 5) :])) if returns else 0.0
    return {
        "recipe": "marl",
        "algo": "reinforce",
        "steps": steps,
        "episodes": len(returns),
        "returns": returns,
        "first_mean": first,
        "last_mean": last,
        "improved": last > first,
    }


def _ppo_train(steps: int, seed: int, opponent: str, agents: int) -> dict[str, Any]:
    import torch
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.policy import PPOPolicy
    from tianshou.trainer import OnpolicyTrainer
    from tianshou.utils.net.common import ActorCritic, Net
    from tianshou.utils.net.continuous import ActorProb, Critic

    def make():
        inner = BaceGymEnv(opponent=opponent, agents=1, seed=seed, max_cycles=200)
        return inner

    # Tianshou expects gymnasium.Env; wrap if needed
    train_env = DummyVectorEnv([make])
    env0 = make()
    obs_space = env0.observation_space
    act_space = env0.action_space
    env0.close()
    net = Net(state_shape=obs_space.shape, hidden_sizes=[64, 64])
    actor = ActorProb(net, act_space.shape, unbounded=False, conditioned_sigma=True)
    critic = Critic(Net(state_shape=obs_space.shape, hidden_sizes=[64, 64]))
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

    def dist_fn(*logits):
        return torch.distributions.Independent(
            torch.distributions.Normal(*logits), 1
        )

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        action_space=act_space,
        action_scaling=True,
    )
    buf = VectorReplayBuffer(20_000, buffer_num=1)
    collector = Collector(policy, train_env, buf)
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=collector,
        max_epoch=max(1, steps // 200),
        step_per_epoch=min(steps, 200),
        repeat_per_collect=4,
        episode_per_test=0,
        batch_size=64,
        step_per_collect=200,
    ).run()
    rews = list(result.get("returns", []) or [])
    if not rews:
        # collect a few eval episodes for a curve
        stats = collector.collect(n_episode=5)
        rews = list(stats.get("rews", [])) if isinstance(stats, dict) else []
        if hasattr(stats, "returns"):
            rews = list(stats.returns)
    train_env.close()
    first = float(sum(rews[:1])) if rews else 0.0
    last = float(sum(rews[-1:])) if rews else 0.0
    return {
        "recipe": "marl",
        "algo": "ppo",
        "steps": steps,
        "returns": [float(x) for x in rews],
        "first_mean": first,
        "last_mean": last,
        "improved": last >= first,
        "tianshou": True,
    }


def run_marl(
    steps: int = 400,
    seed: int = 0,
    opponent: str = "duck",
    agents: int = 1,
    out_dir: Optional[Path] = None,
    algo: str = "ippo",
    profile: str = "default",
    share_tracks: bool = True,
    red_mission: str = "dca",
) -> dict[str, Any]:
    if profile == "paper":
        recipe = _load_recipe_json("marl_core")
        algo = str(recipe.get("algo", "ippo"))
        steps = int(recipe.get("ppo_1v1_steps", 200_000))
        agents = 1
        opponent = "duck"
    elif profile == "smoke":
        steps = min(int(steps), 80)
        agents = min(agents, 2)
    from bace.marl import TrainSpec, train, _strip_actor

    spec = TrainSpec(
        algo=algo,
        opponent=opponent,
        agents=max(1, min(4, agents)),
        seed=seed,
        steps=steps,
        eval_episodes=2 if profile == "smoke" else 50,
        share_tracks=share_tracks,
        red_mission=red_mission,
        n_envs=2 if profile == "smoke" else 8,
    )
    report = _strip_actor(train(spec))
    report["recipe"] = "marl"
    dest = out_dir or Path("runs/experiments")
    path = write_artifact(dest, "marl", report)
    report["_artifact"] = str(path)
    return report


def run_marl_core(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    from bace.marl import run_core

    dest = out_dir or Path("runs/experiments")
    report = run_core(profile=profile, out_dir=dest)
    path = write_artifact(dest, "marl_core", report)
    report["_artifact"] = str(path)
    return report


def run_marl_ablations(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
    kind: str = "all",
) -> dict[str, Any]:
    from bace.marl import run_ablations

    dest = out_dir or Path("runs/experiments")
    report = run_ablations(profile=profile, out_dir=dest)
    if kind == "striker":
        report["jobs"] = [j for j in report.get("jobs", []) if j.get("red_mission") == "striker"]
        report["recipe"] = "marl_striker"
    elif kind == "tracks":
        report["jobs"] = [j for j in report.get("jobs", []) if not j.get("share_tracks", True)]
        report["recipe"] = "marl_tracks"
    path = write_artifact(dest, report.get("recipe", "marl_ablations"), report)
    report["_artifact"] = str(path)
    return report


def run_marl_selfplay(
    profile: str = "smoke",
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    from bace.marl import run_selfplay

    dest = out_dir or Path("runs/experiments")
    report = run_selfplay(profile=profile, out_dir=dest)
    path = write_artifact(dest, "marl_selfplay", report)
    report["_artifact"] = str(path)
    return report


def make_benchmarl_env(**kwargs: Any):
    """BenchMARL-oriented factory: returns a PettingZoo ParallelEnv."""
    return make_env(**kwargs)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="bace.experiment")
    p.add_argument(
        "recipe",
        choices=["wez", "fsm", "marl", "marl_core", "marl_selfplay", "marl_striker", "marl_tracks", "bench"],
    )
    p.add_argument("--out", type=Path, default=Path("runs/experiments"))
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="tiny grid/pop for tests")
    p.add_argument(
        "--profile",
        choices=["default", "smoke", "paper"],
        default="default",
        help="paper = dense WEZ / 2v2 FSM / MARL core grid; smoke = CI",
    )
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--vs", dest="opponent", default="duck")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--algo", default="ippo")
    args = p.parse_args(argv)
    profile = "smoke" if args.smoke else args.profile

    if args.recipe == "wez":
        report = run_wez(
            max_parallel=args.max_parallel, out_dir=args.out, smoke=profile == "smoke", profile=profile
        )
        print(report.get("summary", json.dumps(report, indent=2)))
        print("fits", len(report.get("fits", [])), "traces", len(report.get("traces", [])))
        print("wrote", report.get("_artifact"))
    elif args.recipe == "fsm":
        report = run_fsm(
            max_parallel=args.max_parallel, out_dir=args.out, smoke=profile == "smoke", profile=profile
        )
        print(report.get("summary", json.dumps(report, indent=2)))
        print("wrote", report.get("_artifact"))
        print("elites:", [e.get("label") for e in report.get("elites", [])])
    elif args.recipe == "marl_core":
        report = run_marl_core(profile=profile if profile != "default" else "paper", out_dir=args.out)
        print(f"marl_core jobs={report.get('n_jobs')} profile={report.get('profile')}")
        print("wrote", report.get("_artifact"))
    elif args.recipe == "marl_selfplay":
        report = run_marl_selfplay(profile=profile if profile != "default" else "paper", out_dir=args.out)
        print(f"marl_selfplay jobs={report.get('n_jobs')} profile={report.get('profile')}")
        print("wrote", report.get("_artifact"))
    elif args.recipe in {"marl_striker", "marl_tracks"}:
        kind = "striker" if args.recipe == "marl_striker" else "tracks"
        report = run_marl_ablations(
            profile=profile if profile != "default" else "paper", out_dir=args.out, kind=kind
        )
        print(f"{args.recipe} jobs={len(report.get('jobs', []))} profile={report.get('profile')}")
        print("wrote", report.get("_artifact"))
    elif args.recipe == "bench":
        from bace.bench import run_bench

        report = run_bench(
            steps=args.steps,
            out_dir=args.out,
            profile=profile,
            max_parallel=args.max_parallel,
        )
        print(json.dumps({k: report[k] for k in ("cpu_count", "scaling", "pyo3_tax") if k in report}, indent=2)[:2000])
        print("wrote", report.get("_artifact"))
    else:
        report = run_marl(
            steps=args.steps,
            seed=args.seed,
            opponent=args.opponent,
            agents=args.agents,
            out_dir=args.out,
            algo=args.algo,
            profile=profile,
        )
        if report.get("skipped"):
            print(f"marl skipped: {report.get('reason')}")
        else:
            ev = report.get("eval") or {}
            print(
                f"marl algo={report.get('algo')} episodes={report.get('episodes', 0)} "
                f"eval_kills={ev.get('kills')} eval_return={ev.get('return')} "
                f"beat_random={report.get('beat_random')}"
            )
        print("wrote", report.get("_artifact"))


if __name__ == "__main__":
    main()
