"""Batch experiment runner (thin wrapper around bace.experiment)."""

from __future__ import annotations

import argparse

from bace.experiment import run_fsm, run_wez


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=8, help="unused; kept for compatibility")
    p.add_argument("--max-cycles", type=int, default=100)
    p.add_argument("--recipe", choices=["wez", "fsm"], default="wez")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.recipe == "fsm":
        report = run_fsm(
            params={"max_cycles": args.max_cycles},
            smoke=args.smoke or args.cases <= 8,
        )
    else:
        report = run_wez(
            params={"max_cycles": args.max_cycles},
            smoke=args.smoke or args.cases <= 8,
        )
    print(report.get("summary", report))


if __name__ == "__main__":
    main()
