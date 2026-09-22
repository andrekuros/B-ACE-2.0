"""Eval discrete-tactic policies vs duck and baseline1.

Compares stock FSM, random tactics, and OpenJev (0.8B, CUDA only).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy import (  # noqa: E402
    TacticPolicy,
    cuda_available,
    load_openjev_scorer,
    OpenJevUnavailable,
    random_scorer,
)

from bace import BaceEnv


def _shots(snapshot: dict[str, Any]) -> dict[int, int]:
    out = {0: 0, 1: 0}
    for f in snapshot.get("fighters") or []:
        team = int(f.get("team") or 0)
        out[team] += 6 - int(f.get("missiles") or 6)
    return out


def run_episode(
    backend: str,
    red: str,
    seed: int,
    max_cycles: int,
    chooser: Optional[Callable] = None,
) -> dict[str, Any]:
    blue_behavior = "baseline1" if backend == "fsm" else "external"
    env = BaceEnv(
        {
            "env": {"max_cycles": max_cycles, "seed": seed, "action_repeat": 20},
            "blue": {"num_agents": 1, "behavior": blue_behavior},
            "red": {"num_agents": 1, "behavior": red},
        }
    )
    env.reset(seed=seed)
    policy = TacticPolicy(chooser, "agent_0") if chooser is not None else None
    steps = 0
    while env.agents and steps < max_cycles + 2:
        if policy is None:
            actions = {a: np.zeros(4, dtype=np.float32) for a in env.agents}
        else:
            snap = env.state()
            actions = {a: policy.act(snap) for a in env.agents}
        _, _, terms, truncs, _ = env.step(actions)
        steps += 1
        st = env.state()
        if st.get("end") != "ongoing":
            break
        if all(terms.get(a, False) or truncs.get(a, False) for a in terms):
            break
    st = env.state()
    alive = {0: 0, 1: 0}
    for f in st.get("fighters") or []:
        if f.get("alive"):
            alive[int(f.get("team") or 0)] += 1
    env.close()
    return {
        "seed": seed,
        "end": st.get("end"),
        "steps": steps,
        "blue_alive": alive[0],
        "red_alive": alive[1],
        "shots": _shots(st),
        "last_tactic": None if policy is None else policy.last_tactic,
    }


def summarize(name: str, red: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ends = Counter(r["end"] for r in rows)
    wins = ends.get("red_killed", 0)
    mean_steps = sum(r["steps"] for r in rows) / max(n, 1)
    return {
        "policy": name,
        "red": red,
        "cases": n,
        "win_rate": wins / max(n, 1),
        "red_killed": wins,
        "blue_killed": ends.get("blue_killed", 0),
        "mutual_kill": ends.get("mutual_kill", 0),
        "timeouts": ends.get("max_cycles", 0),
        "mean_steps": mean_steps,
        "results": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="OpenJev tactic-head eval vs B-ACE models")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--backend",
        choices=("auto", "openjev", "random", "fsm", "all"),
        default="auto",
        help="auto: FSM + random, plus OpenJev only if CUDA is available",
    )
    args = p.parse_args()

    reds = ("duck", "baseline1")
    want = {args.backend}
    if args.backend in ("auto", "all"):
        want = {"fsm", "random"}
        if args.backend == "all" or cuda_available():
            want.add("openjev")
        elif args.backend == "auto":
            print("OpenJev skipped: no CUDA GPU (0.8B checkpoint is not run on CPU).", file=sys.stderr)

    choosers: dict[str, Optional[Callable]] = {"fsm": None}
    if "random" in want:
        choosers["random"] = random_scorer(np.random.default_rng(args.seed))
    if "openjev" in want:
        try:
            choosers["openjev"] = load_openjev_scorer()
        except OpenJevUnavailable as exc:
            print(str(exc), file=sys.stderr)
            if args.backend == "openjev":
                sys.exit(1)
            want.discard("openjev")

    reports = []
    for red in reds:
        for name in ("fsm", "random", "openjev"):
            if name not in want:
                continue
            rows = [
                run_episode(name, red, args.seed + i, args.max_cycles, choosers.get(name))
                for i in range(args.seeds)
            ]
            rep = summarize(name, red, rows)
            reports.append(rep)
            print(
                f"{name:8} vs {red:10}  win={rep['win_rate']:.2f}  "
                f"rk={rep['red_killed']} bk={rep['blue_killed']} "
                f"mut={rep['mutual_kill']} to={rep['timeouts']}  "
                f"mean_steps={rep['mean_steps']:.1f}"
            )

    out = {"cuda": cuda_available(), "reports": reports}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
