"""Paper studies: every number in the B-ACE 2.0 paper comes from one of these.

    python -m bace.studies all --out runs/paper            # scripted studies (minutes)
    python -m bace.studies rl_eval --policies runs/paper/rl/*/policy.pt

Each study writes ``<out>/<study>.json``. Scenarios:

* ``duel``: 1v1 (or 2v2) defensive counter-air on both sides, head-on 60 NM apart at
  25,000 ft, 700 decisions. Symmetric; used for scripted cross-play.
* ``defend``: blue defends its point against red strikers that fly to it (red wins on
  arrival), 400 decisions. The RL task.

Both add spawn jitter (±5 NM in x and z, ±3,000 ft) and 10% threshold noise on every
scripted pilot, so episodes differ by seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from bace.env import _baseline_beh, _deep_merge
from bace.experiment import _native, run_experiment

ROOT = Path(__file__).resolve().parents[2]
JITTER = {"x": 5.0, "y": 3000.0, "z": 5.0}
NOISE = 0.1
REF = ["ref_aggressive", "ref_balanced", "ref_conservative"]
PILOTS = REF + ["fsm_b1"]

# Physics conditions for the ablation study.
CONDITIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_datalink": {"missile_support": "launch"},
    "no_warning": {"missile_warning": False},
    "constant_speed": {"aircraft_model": "constant_speed"},
    "legacy_missile": {"missile_model": "legacy"},
    "bace_2_0": {
        "missile_model": "legacy",
        "missile_support": "launch",
        "aircraft_model": "constant_speed",
        "missile_warning": False,
    },
}


# ---------------------------------------------------------------- scenarios


def pilot(name: str) -> dict[str, Any]:
    """Team block for a scripted pilot."""
    if name in REF:
        b = _baseline_beh(name[4:], folder="reference")
    elif name == "fsm_b1":
        b = {"behavior": "baseline1"}
    elif name.startswith("fsm:"):
        d, c, k = (float(v) for v in name[4:].split(","))
        b = {"behavior": "baseline1", "beh_config": {"d_shot": [d], "l_crank": [c], "l_break": [k]}}
    elif name in ("duck", "fire_once"):
        return {"behavior": name}
    else:
        raise ValueError(name)
    out = {k: v for k, v in b.items() if k in ("behavior", "beh_config")}
    out["behavior"] = "baseline1"
    out["beh_noise"] = NOISE
    return out


def _formation(n: int) -> dict[str, Any]:
    if n <= 1:
        return {}
    return {"offset_pos": {"x": 4.0, "y": 0.0, "z": 0.0 if n == 2 else 4.0}}


def scenario(
    kind: str,
    blue: dict[str, Any],
    red: dict[str, Any],
    seed: int,
    agents: int = 1,
    env: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    form = _formation(agents)
    base_b = {"num_agents": agents, "share_tracks": True, "rnd_offset_range": dict(JITTER), **form}
    base_r = {"num_agents": agents, "share_tracks": True, "rnd_offset_range": dict(JITTER), **form}
    if kind == "duel":
        e = {"max_cycles": 700, "seed": seed}
        base_b["mission"] = "dca"
        base_r["mission"] = "dca"
    elif kind == "defend":
        e = {"max_cycles": 400, "seed": seed}
        base_b["mission"] = "dca"
        base_r["mission"] = "striker"
    else:
        raise ValueError(kind)
    cfg = {"env": {**e, **(env or {})}, "blue": {**base_b, **blue}, "red": {**base_r, **red}}
    return cfg


# ---------------------------------------------------------------- metrics


def _boot(xs: Iterable[float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(list(xs), dtype=float)
    if a.size == 0:
        return (float("nan"),) * 3
    if a.size == 1:
        return float(a[0]), float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n, a.size))].mean(1)
    return float(a.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(outs: list[dict[str, Any]], agents: int = 1, raw: bool = False) -> dict[str, Any]:
    """Blue-side metrics with 95% bootstrap CIs, plus both teams' shot statistics."""
    k = [o["blue_kills"] for o in outs]
    d = [o["blue_deaths"] for o in outs]
    win = [1.0 if o["blue_kills"] == agents and o["blue_deaths"] < agents else 0.0 for o in outs]
    lose = [
        1.0
        if (o["blue_deaths"] == agents and o["blue_kills"] < agents) or o["end"] == "red_mission"
        else 0.0
        for o in outs
    ]
    net = [a - b for a, b in zip(k, d)]
    res: dict[str, Any] = {
        "n": len(outs),
        "kills": _boot(k),
        "deaths": _boot(d),
        "net": _boot(net),
        "win": _boot(win),
        "lose": _boot(lose),
        "draw": float(1 - np.mean(win) - np.mean(lose)),
        "steps": float(np.mean([o["steps"] for o in outs])),
        "ends": _count(o["end"] for o in outs),
    }
    for team, name in ((0, "blue"), (1, "red")):
        shots = [s for o in outs for s in o.get("shots", []) if s["team"] == team]
        res[f"{name}_shots"] = shot_stats(shots, len(outs))
    if raw:
        # [team, range_nm, rmax_nm, result (1 hit, 0 miss, -1 in flight), f_pole_nm,
        #  target_aspect_deg, shooter_alt_ft, target_alt_ft, tof_s, episode index]
        res["raw_shots"] = [
            [
                s["team"],
                round(s["range_nm"], 2),
                round(s["rmax_nm"], 2),
                1 if s["result"] == "hit" else (-1 if s["result"] == "in_flight" else 0),
                round(s["f_pole_nm"], 2),
                round(s["target_aspect_deg"], 1),
                round(s["shooter_alt_ft"]),
                round(s["target_alt_ft"]),
                round(s["tof_s"], 1),
                ep,
            ]
            for ep, o in enumerate(outs)
            for s in o.get("shots", [])
        ]
    return res


def shot_stats(shots: list[dict[str, Any]], episodes: int) -> dict[str, Any]:
    done = [s for s in shots if s["result"] != "in_flight"]
    hits = [s for s in done if s["result"] == "hit"]
    ratio = [s["range_nm"] / s["rmax_nm"] for s in shots if s["rmax_nm"] > 0]
    return {
        "per_episode": len(shots) / max(episodes, 1),
        "hit_rate": len(hits) / max(len(done), 1),
        "range_nm": float(np.mean([s["range_nm"] for s in shots])) if shots else float("nan"),
        "range_over_rmax": float(np.mean(ratio)) if ratio else float("nan"),
        "range_over_rmax_q": [float(q) for q in np.quantile(ratio, [0.1, 0.5, 0.9])]
        if ratio
        else [],
        "f_pole_hit_nm": float(np.mean([s["f_pole_nm"] for s in hits if s["f_pole_nm"] > 0]))
        if hits
        else float("nan"),
        "tof_hit_s": float(np.mean([s["tof_s"] for s in hits])) if hits else float("nan"),
        "causes": _count(s["result"] for s in shots),
    }


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _run(cfgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outs = run_experiment(cfgs, max_parallel=8)
    for o in outs:
        o.pop("config", None)
    return outs


def _seeds(n: int, base: int) -> range:
    return range(base, base + n)


# ---------------------------------------------------------------- studies


def study_missile(out: Path, **_: Any) -> dict[str, Any]:
    """Fly-out traces (cargo example) and the missile parameter card."""
    raw = subprocess.run(
        ["cargo", "run", "--release", "-q", "-p", "bace-core", "--example", "missile_traces"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    info = json.loads(_wasm_free_info())
    return {"traces": json.loads(raw), "params": info}


def _wasm_free_info() -> str:
    """Missile parameters as the core reports them (via the WEZ table header)."""
    table = json.loads((ROOT / "crates/bace-core/data/wez_generic_mraam.json").read_text())
    return json.dumps({k: v for k, v in table.items() if k != "rmax_nm" and k != "rnez_nm"})


def study_wez(out: Path, repeats: int = 30, **_: Any) -> dict[str, Any]:
    """WEZ table plus fire-once validation for both missile models."""
    table = json.loads((ROOT / "crates/bace-core/data/wez_generic_mraam.json").read_text())
    _, wez_fn, _ = _native()
    val = {}
    for model in ("generic_mraam", "legacy"):
        params = {
            "ranges_nm": [float(r) for r in range(4, 50, 2)],
            "altitudes_ft": [10000.0, 25000.0, 40000.0],
            "aspects": ["head", "beam", "tail"],
            "repeats": repeats,
            "max_cycles": 200,
            "seed": 1,
            "missile_model": model,
        }
        val[model] = json.loads(wez_fn(json.dumps(params), 8))
    return {"table": table, "validation": val}


def study_crossplay(out: Path, n: int = 200, **_: Any) -> dict[str, Any]:
    """Scripted pilots against each other in the duel, 1v1 and 2v2."""
    res: dict[str, Any] = {"pilots": PILOTS, "n": n}
    for agents in (1, 2):
        cells = {}
        for b in PILOTS:
            for r in PILOTS:
                cfgs = [
                    scenario("duel", pilot(b), pilot(r), s, agents)
                    for s in _seeds(n, 10_000)
                ]
                cells[f"{b}|{r}"] = summarize(_run(cfgs), agents, raw=True)
        res[f"{agents}v{agents}"] = cells
    return res


def study_ablation(out: Path, n: int = 200, **_: Any) -> dict[str, Any]:
    """The same pilot matrix under different physics: does fidelity change the ranking?"""
    res: dict[str, Any] = {"pilots": PILOTS, "conditions": CONDITIONS, "n": n}
    for cname, env in CONDITIONS.items():
        cells = {}
        for b in PILOTS:
            for r in PILOTS:
                cfgs = [
                    scenario("duel", pilot(b), pilot(r), s, 1, env=env)
                    for s in _seeds(n, 20_000)
                ]
                cells[f"{b}|{r}"] = summarize(_run(cfgs))
        res[cname] = cells
    return res


D_SHOT = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
L_BREAK = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]


def study_landscape(out: Path, n: int = 100, **_: Any) -> dict[str, Any]:
    """Shot-range x break-threshold grid against the balanced reference pilot."""
    res: dict[str, Any] = {"d_shot": D_SHOT, "l_break": L_BREAK, "l_crank": 1.5, "n": n}
    for cname in ("full", "bace_2_0"):
        grid = []
        for d in D_SHOT:
            row = []
            for k in L_BREAK:
                cfgs = [
                    scenario(
                        "duel",
                        pilot(f"fsm:{d},1.5,{k}"),
                        pilot("ref_balanced"),
                        s,
                        1,
                        env=CONDITIONS[cname],
                    )
                    for s in _seeds(n, 30_000)
                ]
                sm = summarize(_run(cfgs))
                row.append(
                    {
                        "net": sm["net"][0],
                        "win": sm["win"][0],
                        "lose": sm["lose"][0],
                        "kills": sm["kills"][0],
                        "deaths": sm["deaths"][0],
                        "hit_rate": sm["blue_shots"]["hit_rate"],
                        "shots": sm["blue_shots"]["per_episode"],
                    }
                )
            grid.append(row)
        res[cname] = grid
    return res


DEFENDERS = REF + ["fsm_b1", "fire_once"]
STRIKERS = ["duck"] + REF


def study_defend(out: Path, n: int = 200, **_: Any) -> dict[str, Any]:
    """Scripted defenders against strikers: the baselines for the RL task."""
    res: dict[str, Any] = {"defenders": DEFENDERS, "strikers": STRIKERS, "n": n}
    for agents in (1, 2):
        cells = {}
        for b in DEFENDERS:
            for r in STRIKERS:
                cfgs = [
                    scenario("defend", pilot(b), pilot(r), s, agents)
                    for s in _seeds(n, 40_000)
                ]
                cells[f"{b}|{r}"] = summarize(_run(cfgs), agents, raw=True)
        res[f"{agents}v{agents}"] = cells
    return res


def play(
    act: Callable[[np.ndarray], np.ndarray],
    red: str,
    seed: int,
    agents: int = 1,
    action_type: str = "discrete",
) -> dict[str, Any]:
    """One `defend` episode with a learned blue policy; returns the outcome with shots."""
    from bace.env import BaceEnv

    cfg = scenario("defend", {"behavior": "external"}, pilot(red), seed, agents)
    cfg["env"]["action_type"] = action_type
    cfg["env"]["rewards"] = {
        "missile_no_fire_factor": 0.0,
        "detect_loss_factor": -0.01,
        "mission_factor": 0.001,
        "missile_fire_factor": -0.1,
        "missile_miss_factor": -0.5,
        "keep_track_factor": 0.001,
        "hit_enemy_factor": 3.0,
        "hit_own_factor": -5.0,
        "mission_accomplished_factor": 10.0,
    }
    env = BaceEnv(cfg)
    obs, _ = env.reset(seed=seed)
    while env.agents:
        obs, *_ = env.step({a: act(obs[a]) for a in env.agents})
    oc = env.outcome()
    oc.pop("config", None)
    env.close()
    return oc


def study_rl_eval(
    out: Path, policies: Optional[list[str]] = None, n: int = 200, **_: Any
) -> dict[str, Any]:
    """Learned policies against every striker, same seeds as the scripted baselines.

    Batched on the vector env with the training scenario (``defend``: same spawn jitter,
    pilots and rewards); the policy is sampled as trained, torch seeded per opponent.
    """
    import torch

    from bace.ppo import evaluate_vec

    res: dict[str, Any] = {"strikers": STRIKERS, "n": n, "policies": {}}
    seeds = list(_seeds(n, 40_000))
    for p in policies or []:
        ck = torch.load(p, weights_only=False)
        agents = int(ck["config"]["agents"])
        row = {}
        for r in STRIKERS:
            outs: list[dict[str, Any]] = []
            evaluate_vec(ck, r, seeds, outcomes=outs)
            row[r] = summarize(outs, agents, raw=True)
        res["policies"][p] = {
            "agents": agents,
            "seed": ck["config"]["seed"],
            "opponents": ck["config"]["opponents"],
            "env_steps": ck["env_steps"],
            "train_time_s": ck["train_time_s"],
            "eval": row,
        }
    return res


def study_rl_train(out: Path, runs: Optional[list[str]] = None, **_: Any) -> dict[str, Any]:
    """Training histories and checkpoint selections of PPO run directories."""
    res: dict[str, Any] = {"runs": []}
    for d in runs or []:
        d = Path(d)
        tr = json.loads((d / "train.json").read_text())
        sel_p = d / "selection.json"
        sel = json.loads(sel_p.read_text()) if sel_p.is_file() else {}
        best = next((r for r in sel.get("rows", []) if r["path"] == sel.get("best")), None)
        res["runs"].append(
            {
                "name": d.name,
                "agents": tr["config"]["agents"],
                "seed": tr["config"]["seed"],
                "config": tr["config"],
                "env_steps": tr["env_steps"],
                "train_time_s": tr["history"][-1]["time_s"],
                "history": tr["history"],
                "selected_steps": best["env_steps"] if best else None,
                "selection": [
                    {"env_steps": r["env_steps"], "score": r["score"]} for r in sel.get("rows", [])
                ],
            }
        )
    return res


def study_bench(out: Path, **_: Any) -> dict[str, Any]:
    """Decision throughput of the Rust batch runner per physics condition."""
    from bace._native import bench_parallel_py

    res: dict[str, Any] = {"host_threads": __import__("os").cpu_count()}
    for cname in ("full", "bace_2_0"):
        cfg = scenario("duel", pilot("ref_balanced"), pilot("ref_balanced"), 1, 1, CONDITIONS[cname])
        rows = []
        for n in (1, 2, 4, 8):
            r = json.loads(bench_parallel_py(json.dumps(cfg), n, 20_000))
            rows.append({"n_envs": n, **r})
        res[cname] = rows
    return res


STUDIES: dict[str, Callable[..., dict[str, Any]]] = {
    "missile": study_missile,
    "wez": study_wez,
    "crossplay": study_crossplay,
    "ablation": study_ablation,
    "landscape": study_landscape,
    "defend": study_defend,
    "bench": study_bench,
    "rl_eval": study_rl_eval,
    "rl_train": study_rl_train,
}


def run(name: str, out: Path, **kw: Any) -> dict[str, Any]:
    t = time.time()
    res = STUDIES[name](out, **kw)
    res["study"] = name
    res["seconds"] = time.time() - t
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res) + "\n")
    return res


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="bace.studies")
    p.add_argument("study", choices=list(STUDIES) + ["all"])
    p.add_argument("--out", type=Path, default=Path("runs/paper"))
    p.add_argument("--n", type=int, default=None, help="episodes per cell")
    p.add_argument("--policies", nargs="*", default=None)
    p.add_argument("--runs", nargs="*", default=None, help="PPO run directories (rl_train)")
    a = p.parse_args(argv)
    names = [s for s in STUDIES if s not in ("rl_eval", "rl_train")] if a.study == "all" else [a.study]
    kw: dict[str, Any] = {}
    if a.n:
        kw["n"] = a.n
    if a.policies:
        kw["policies"] = a.policies
    if a.runs:
        kw["runs"] = a.runs
    for s in names:
        r = run(s, a.out, **kw)
        print(f"{s}: {r['seconds']:.0f} s -> {a.out / (s + '.json')}", flush=True)


if __name__ == "__main__":
    main()
