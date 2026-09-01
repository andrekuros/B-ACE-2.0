"""Tianshou PPO 1v1 vs duck (real trainer, not the DDPG toy)."""

from __future__ import annotations

import argparse
import json

from bace.experiment import run_marl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vs", dest="opponent", default="duck")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--profile", choices=["default", "smoke", "paper"], default="default")
    args = p.parse_args()
    report = run_marl(
        steps=args.steps,
        seed=args.seed,
        opponent=args.opponent,
        agents=args.agents,
        algo="ppo",
        profile="smoke" if args.smoke else args.profile,
    )
    keys = (
        "skipped",
        "reason",
        "algo",
        "steps",
        "eval",
        "eval_random",
        "eval_fire_once",
        "beat_random",
        "first_mean",
        "last_mean",
    )
    print(json.dumps({k: report[k] for k in keys if k in report}, indent=2))


if __name__ == "__main__":
    main()
