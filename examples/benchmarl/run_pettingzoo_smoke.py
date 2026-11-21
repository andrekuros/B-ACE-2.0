"""BenchMARL experiment factory on make_env."""

from __future__ import annotations

import argparse
import json

from bace.env import make_env
from bace.experiment import run_marl
from bace.marl import TrainSpec, train_benchmarl


def make_benchmarl_env(**kwargs):
    return make_env(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vs", dest="opponent", default="duck")
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--algo", default="ippo")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    env = make_benchmarl_env(opponent=args.opponent, agents=args.agents, max_cycles=40, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    print("PettingZoo agents:", list(obs))
    env.close()
    spec = TrainSpec(
        algo=args.algo,
        opponent=args.opponent,
        agents=args.agents,
        seed=args.seed,
        steps=80 if args.smoke else args.steps,
        eval_episodes=2 if args.smoke else 20,
        n_envs=2,
    )
    report = train_benchmarl(spec)
    print(json.dumps({k: report[k] for k in report if k != "returns"}, indent=2))
    if args.smoke:
        return
    # paper-scale path goes through the experiment CLI
    _ = run_marl


if __name__ == "__main__":
    main()
