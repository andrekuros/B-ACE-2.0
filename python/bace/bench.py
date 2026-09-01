"""Throughput study: ParallelEnvs scaling, PyO3 tax, WEZ/FSM wall-clock."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bace.env import BaceEnv, BaceVecEnv, make_env
from bace.experiment import run_fsm, run_wez, write_artifact


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _scenario_json(agents: int = 1, max_cycles: int = 200, seed: int = 1) -> str:
    env = make_env(opponent="duck", agents=agents, max_cycles=max_cycles, seed=seed)
    raw = json.dumps(env._config)
    env.close()
    return raw


def _bench_native(config_json: str, n_envs: int, steps: int) -> dict[str, Any]:
    from bace._native import bench_parallel_py

    return json.loads(bench_parallel_py(config_json, int(n_envs), int(steps)))


def _bench_pyo3_vec(config_json: str, n_envs: int, steps: int) -> dict[str, Any]:
    from bace._native import NativeVecEnv

    env = NativeVecEnv(config_json, int(n_envs), True)
    env.reset(1)
    ids = list(env.agent_ids())
    idle = {a: [0.0, 0.0, 0.0, -1.0] for a in ids}
    payload = [idle for _ in range(int(n_envs))]
    t0 = time.perf_counter()
    for _ in range(int(steps)):
        env.step(payload)
    wall_s = max(time.perf_counter() - t0, 1e-9)
    decisions = float(n_envs) * float(steps)
    phy_fps = 20.0
    action_repeat = 20.0
    physics = decisions * action_repeat
    return {
        "n_envs": int(n_envs),
        "steps": int(steps),
        "wall_s": wall_s,
        "decision_hz": decisions / wall_s,
        "physics_hz": physics / wall_s,
        "realtime_factor": physics / (wall_s * phy_fps),
        "backend": "pyo3_vec",
    }


def _bench_bace_env(steps: int, seed: int = 1) -> dict[str, Any]:
    env = BaceEnv(
        {
            "env": {"max_cycles": 200, "seed": seed},
            "blue": {"num_agents": 1, "behavior": "external"},
            "red": {"num_agents": 1, "behavior": "duck"},
        }
    )
    env.reset(seed=seed)
    idle = {a: np.zeros(4, dtype=np.float32) for a in env.possible_agents}
    t0 = time.perf_counter()
    for i in range(int(steps)):
        if not env.agents:
            env.reset(seed=seed + i + 1)
        env.step(idle)
    wall_s = max(time.perf_counter() - t0, 1e-9)
    env.close()
    decisions = float(steps)
    phy_fps = 20.0
    action_repeat = 20.0
    physics = decisions * action_repeat
    return {
        "n_envs": 1,
        "steps": int(steps),
        "wall_s": wall_s,
        "decision_hz": decisions / wall_s,
        "physics_hz": physics / wall_s,
        "realtime_factor": physics / (wall_s * phy_fps),
        "backend": "bace_env",
    }


def _time_call(fn) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_bench(
    steps: int = 400,
    out_dir: Optional[Path] = None,
    profile: str = "default",
    max_parallel: int = 8,
) -> dict[str, Any]:
    cpu = _cpu_count()
    n_list = [n for n in (1, 2, 4, 8, 16, 32) if n <= cpu]
    if cpu not in n_list:
        n_list.append(cpu)
    if profile == "smoke":
        steps = min(steps, 40)
        n_list = [n for n in n_list if n <= 4] or [1]
    cfg = _scenario_json(agents=1, max_cycles=200, seed=1)

    scaling = []
    for n in n_list:
        row = _bench_native(cfg, n, steps)
        row["backend"] = "rust"
        row["efficiency"] = float(row["realtime_factor"]) / max(n, 1)
        scaling.append(row)

    pyo3_steps = min(steps, 80 if profile == "smoke" else steps)
    rust1 = _bench_native(cfg, 1, pyo3_steps)
    rust1["backend"] = "rust"
    pyo3_vec = _bench_pyo3_vec(cfg, 1, pyo3_steps)
    pyo3_env = _bench_bace_env(pyo3_steps)
    pyo3_tax = {
        "rust": rust1,
        "pyo3_vec": pyo3_vec,
        "bace_env": pyo3_env,
        "vec_vs_rust": float(rust1["decision_hz"]) / max(float(pyo3_vec["decision_hz"]), 1e-9),
        "env_vs_rust": float(rust1["decision_hz"]) / max(float(pyo3_env["decision_hz"]), 1e-9),
    }

    wez_profile = "smoke" if profile != "paper" else "paper"
    fsm_profile = "smoke"
    wez_times = []
    for mp in (1, min(max_parallel, cpu)):
        _, wall = _time_call(
            lambda m=mp: run_wez(
                max_parallel=m,
                out_dir=Path("/tmp/bace_bench_wez"),
                smoke=wez_profile == "smoke",
                profile=wez_profile,
            )
        )
        wez_times.append({"max_parallel": mp, "wall_s": wall, "profile": wez_profile})
    fsm_times = []
    for mp in (1, min(max_parallel, cpu)):
        _, wall = _time_call(
            lambda m=mp: run_fsm(
                max_parallel=m,
                out_dir=Path("/tmp/bace_bench_fsm"),
                smoke=True,
                profile="smoke",
            )
        )
        fsm_times.append({"max_parallel": mp, "wall_s": wall, "profile": "smoke"})

    report = {
        "recipe": "bench",
        "profile": profile,
        "cpu_count": cpu,
        "phy_fps": 20,
        "action_repeat": 20,
        "decision_hz_note": "1 Hz decisions in realtime (action_repeat=20, phy_fps=20)",
        "scaling": scaling,
        "pyo3_tax": pyo3_tax,
        "wez_wall": wez_times,
        "fsm_wall": fsm_times,
        "stamp": datetime.now(timezone.utc).isoformat(),
    }
    dest = out_dir or Path("runs/experiments")
    path = write_artifact(dest, "bench", report)
    report["_artifact"] = str(path)
    return report


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="bace.bench")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("runs/experiments"))
    p.add_argument("--profile", choices=["default", "smoke", "paper"], default="default")
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    profile = "smoke" if args.smoke else args.profile
    report = run_bench(
        steps=args.steps,
        out_dir=args.out,
        profile=profile,
        max_parallel=args.max_parallel,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "_artifact"}, indent=2))
    print("wrote", report.get("_artifact"))


if __name__ == "__main__":
    main()
