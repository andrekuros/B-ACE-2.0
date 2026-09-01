"""FSM parameter search (wrapper around bace.experiment run_fsm)."""

from __future__ import annotations

import argparse
import json

from bace.experiment import run_fsm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    report = run_fsm(
        params={
            "generations": args.generations,
            "pop": args.pop,
            "episodes": args.episodes,
            "max_cycles": args.max_cycles,
            "seed": args.seed,
        },
        smoke=args.smoke,
    )
    print(json.dumps({k: report[k] for k in ("summary", "elites", "history") if k in report}, indent=2))


if __name__ == "__main__":
    main()
