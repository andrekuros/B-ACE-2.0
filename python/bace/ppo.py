"""Lean PPO (IPPO with shared weights) on the Rust vector env.

One file, no RL framework: factored categorical heads for the discrete action
``MultiDiscrete[2, 5, 5]`` (fire, altitude level, turn), running observation
normalisation, many parallel games, and a mixture of opponents trained at the
same time (each opponent gets its own group of games). Every blue aircraft is a
learner with the same weights and its own observation (independent PPO).

    python -m bace.ppo --opponents duck ref_aggressive ref_balanced ref_conservative \\
        --agents 1 --steps 5000000 --seed 0 --out runs/paper/rl/ippo_1v1_s0

Requires ``torch``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from bace.env import BaceVecEnv

NVEC = (2, 5, 5)


@dataclass
class PPOConfig:
    opponents: list[str] = field(
        default_factory=lambda: ["duck", "ref_aggressive", "ref_balanced", "ref_conservative"]
    )
    agents: int = 1
    red_mission: str = "striker"
    max_cycles: int = 400
    envs_per_opponent: int = 16
    steps: int = 2_000_000
    rollout: int = 64
    epochs: int = 4
    minibatch: int = 2048
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    hidden: int = 256
    seed: int = 0
    threads: int = 2
    save_every: int = 25
    env_overrides: dict[str, Any] = field(default_factory=dict)


class RunningNorm:
    def __init__(self, dim: int, clip: float = 10.0):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> "RunningNorm":
        n = cls(len(s["mean"]))
        n.mean = np.asarray(s["mean"])
        n.var = np.asarray(s["var"])
        n.count = float(s["count"])
        return n


def _build_nets(obs_dim: int, hidden: int):
    import torch.nn as nn

    def mlp(out: int, gain: float):
        layers = [
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out),
        ]
        for m in layers:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 2**0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(layers[-1].weight, gain)
        return nn.Sequential(*layers)

    actor = mlp(sum(NVEC), 0.01)
    critic = mlp(1, 1.0)
    return actor, critic


def _split(logits):
    return logits.split(list(NVEC), dim=-1)


def _dist(logits):
    import torch

    return [torch.distributions.Categorical(logits=lg) for lg in _split(logits)]


class Policy:
    """Greedy (or sampling) act function from a saved checkpoint."""

    def __init__(self, ckpt: dict[str, Any], greedy: bool = False):
        import torch

        self.norm = RunningNorm.from_state(ckpt["obs_norm"])
        self.actor, _ = _build_nets(int(ckpt["obs_dim"]), int(ckpt["config"]["hidden"]))
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.greedy = greedy
        self._torch = torch

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.as_tensor(self.norm(np.asarray(obs, dtype=np.float64)[None]))
        with torch.no_grad():
            logits = self.actor(x)
        if self.greedy:
            return np.array([int(lg.argmax(-1)) for lg in _split(logits)], dtype=np.int64)
        return np.array([int(d.sample()) for d in _dist(logits)], dtype=np.int64)


def load_policy(path: str | Path, greedy: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    import torch

    return Policy(torch.load(path, weights_only=False), greedy=greedy)


class _Group:
    """Games against one opponent; slots are (game, blue agent)."""

    def __init__(self, cfg: PPOConfig, opponent: str, seed_base: int):
        self.opponent = opponent
        self.vec = BaceVecEnv(
            num_envs=cfg.envs_per_opponent,
            auto_reset=False,
            opponent=opponent,
            agents=cfg.agents,
            red_mission=cfg.red_mission,
            max_cycles=cfg.max_cycles,
            rewards="rl",
            action_type="discrete",
            seed=seed_base,
            **cfg.env_overrides,
        )
        self.n = self.vec.num_envs
        self.blue = [a for a in self.vec.possible_agents if not a.startswith("red_")]
        self.seed_next = seed_base
        self.steps = self.vec.reset(seed=None)
        self.cur_seed = [0] * self.n
        for i in range(self.n):
            self.cur_seed[i] = self._seed()
            self.steps[i] = self.vec.reset_at(i, seed=self.cur_seed[i])
        self.alive = np.ones((self.n, len(self.blue)), dtype=bool)
        self.ep_stats = [self._blank() for _ in range(self.n)]
        self.finished: list[dict[str, float]] = []
        self.keep_outcomes = False

    def _seed(self) -> int:
        self.seed_next += 1
        return self.seed_next

    @staticmethod
    def _blank() -> dict[str, float]:
        return {"return": 0.0, "kills": 0.0, "deaths": 0.0, "shots": 0.0, "len": 0.0}

    def obs(self, dim: int) -> np.ndarray:
        out = np.zeros((self.n, len(self.blue), dim), dtype=np.float64)
        for i, s in enumerate(self.steps):
            for j, a in enumerate(self.blue):
                o = s["obs"].get(a)
                if o is not None and self.alive[i, j]:
                    out[i, j] = o
        return out

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns reward, done (slot finished) and the mask of slots that acted."""
        acted = self.alive.copy()
        payload = []
        for i in range(self.n):
            d = {}
            for j, a in enumerate(self.blue):
                if acted[i, j]:
                    d[a] = actions[i, j]
            payload.append(d)
        self.steps = self.vec.step(payload)
        rew = np.zeros_like(acted, dtype=np.float64)
        done = np.zeros_like(acted)
        for i, s in enumerate(self.steps):
            end = str(s["end"]).replace("_", "").lower()
            game_over = end != "ongoing"
            st = self.ep_stats[i]
            st["len"] += 1
            for j, a in enumerate(self.blue):
                if not acted[i, j]:
                    continue
                r = float(s["rewards"].get(a, 0.0))
                rew[i, j] = r
                st["return"] += r
                info = s["infos"].get(a, {})
                bd = info.get("reward_breakdown")
                if bd:
                    b = json.loads(bd) if isinstance(bd, str) else bd
                    if b.get("hit_enemy", 0.0) > 0:
                        st["kills"] += 1
                    if b.get("missile_fire", 0.0) < 0:
                        st["shots"] += 1
                    if b.get("hit_own", 0.0) < 0:
                        st["deaths"] += 1
                term = bool(s["terminations"].get(a, False)) or bool(
                    s["truncations"].get(a, False)
                )
                if term or game_over:
                    done[i, j] = True
                    self.alive[i, j] = False
            if game_over:
                st["end"] = end
                st["seed"] = self.cur_seed[i]
                if self.keep_outcomes:
                    st["outcome"] = self.vec.outcome_at(i)
                self.finished.append(st)
                self.ep_stats[i] = self._blank()
                self.cur_seed[i] = self._seed()
                self.steps[i] = self.vec.reset_at(i, seed=self.cur_seed[i])
                self.alive[i] = True
        return rew, done, acted


def train(cfg: PPOConfig, out: Optional[Path] = None, log: Callable[[str], None] = print) -> dict:
    import torch

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, cfg.threads))
    rng = np.random.default_rng(cfg.seed)
    groups = [
        _Group(cfg, opp, seed_base=1_000_000 * (cfg.seed + 1) + 100_000 * k)
        for k, opp in enumerate(cfg.opponents)
    ]
    obs_dim = int(groups[0].vec._obs_size)
    n_agents = len(groups[0].blue)
    actor, critic = _build_nets(obs_dim, cfg.hidden)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
    norm = RunningNorm(obs_dim)

    n_slots = sum(g.n for g in groups) * n_agents
    per_update = n_slots * cfg.rollout
    n_updates = max(1, cfg.steps // per_update)
    log(
        f"PPO: {len(groups)} opponents x {cfg.envs_per_opponent} games x {n_agents} agents, "
        f"{n_updates} updates of {per_update} steps"
    )
    history: list[dict[str, Any]] = []
    t0 = time.time()
    env_steps = 0

    def snapshot() -> dict[str, Any]:
        return {
            "config": asdict(cfg),
            "obs_dim": obs_dim,
            "actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.detach().clone() for k, v in critic.state_dict().items()},
            "obs_norm": norm.state(),
            "env_steps": env_steps,
            "train_time_s": time.time() - t0,
        }

    for upd in range(1, n_updates + 1):
        frac = 1.0 - (upd - 1) / n_updates
        for pg in opt.param_groups:
            pg["lr"] = cfg.lr * frac
        T = cfg.rollout
        buf_obs = np.zeros((T, n_slots, obs_dim), dtype=np.float32)
        buf_act = np.zeros((T, n_slots, 3), dtype=np.int64)
        buf_logp = np.zeros((T, n_slots), dtype=np.float32)
        buf_val = np.zeros((T + 1, n_slots), dtype=np.float32)
        buf_rew = np.zeros((T, n_slots), dtype=np.float32)
        buf_done = np.zeros((T, n_slots), dtype=np.float32)
        buf_mask = np.zeros((T, n_slots), dtype=np.float32)

        for t in range(T):
            raw = np.concatenate([g.obs(obs_dim).reshape(-1, obs_dim) for g in groups], 0)
            alive = np.concatenate([g.alive.reshape(-1) for g in groups], 0)
            norm.update(raw[alive])
            x = norm(raw)
            with torch.no_grad():
                xt = torch.as_tensor(x)
                dists = _dist(actor(xt))
                acts = torch.stack([d.sample() for d in dists], -1)
                logp = sum(d.log_prob(acts[:, k]) for k, d in enumerate(dists))
                val = critic(xt).squeeze(-1)
            a_np = acts.numpy()
            rew_all, done_all, acted_all = [], [], []
            off = 0
            for g in groups:
                k = g.n * n_agents
                r, d, m = g.step(a_np[off : off + k].reshape(g.n, n_agents, 3))
                rew_all.append(r.reshape(-1))
                done_all.append(d.reshape(-1))
                acted_all.append(m.reshape(-1))
                off += k
            buf_obs[t] = x
            buf_act[t] = a_np
            buf_logp[t] = logp.numpy()
            buf_val[t] = val.numpy()
            buf_rew[t] = np.concatenate(rew_all) * cfg.reward_scale
            buf_done[t] = np.concatenate(done_all)
            buf_mask[t] = np.concatenate(acted_all)
            env_steps += int(buf_mask[t].sum())

        raw = np.concatenate([g.obs(obs_dim).reshape(-1, obs_dim) for g in groups], 0)
        with torch.no_grad():
            buf_val[T] = critic(torch.as_tensor(norm(raw))).squeeze(-1).numpy()

        # GAE. A slot that did not act at step t contributes nothing; a done cuts bootstrap.
        adv = np.zeros((T, n_slots), dtype=np.float32)
        last = np.zeros(n_slots, dtype=np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - buf_done[t]
            nxt = buf_val[t + 1] * nonterm
            delta = buf_rew[t] + cfg.gamma * nxt - buf_val[t]
            last = delta + cfg.gamma * cfg.lam * nonterm * last
            last = last * buf_mask[t]
            adv[t] = last
        ret = adv + buf_val[:T]

        sel = buf_mask.reshape(-1) > 0
        b_obs = torch.as_tensor(buf_obs.reshape(-1, obs_dim)[sel])
        b_act = torch.as_tensor(buf_act.reshape(-1, 3)[sel])
        b_logp = torch.as_tensor(buf_logp.reshape(-1)[sel])
        b_adv = torch.as_tensor(adv.reshape(-1)[sel])
        b_ret = torch.as_tensor(ret.reshape(-1)[sel])
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_obs.shape[0]
        clipfracs, ents, vlosses = [], [], []
        for _ in range(cfg.epochs):
            perm = torch.as_tensor(rng.permutation(n))
            for s in range(0, n, cfg.minibatch):
                idx = perm[s : s + cfg.minibatch]
                dists = _dist(actor(b_obs[idx]))
                newlogp = sum(d.log_prob(b_act[idx, k]) for k, d in enumerate(dists))
                ent = sum(d.entropy() for d in dists).mean()
                ratio = (newlogp - b_logp[idx]).exp()
                a = b_adv[idx]
                pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v = critic(b_obs[idx]).squeeze(-1)
                vl = 0.5 * ((v - b_ret[idx]) ** 2).mean()
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                clipfracs.append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
                ents.append(float(ent))
                vlosses.append(float(vl))

        row: dict[str, Any] = {
            "update": upd,
            "env_steps": env_steps,
            "time_s": time.time() - t0,
            "entropy": float(np.mean(ents)),
            "value_loss": float(np.mean(vlosses)),
            "clipfrac": float(np.mean(clipfracs)),
            "by_opponent": {},
        }
        for g in groups:
            fin = g.finished
            g.finished = []
            if fin:
                row["by_opponent"][g.opponent] = {
                    "episodes": len(fin),
                    "return": float(np.mean([f["return"] for f in fin])),
                    "kills": float(np.mean([f["kills"] for f in fin])),
                    "shots": float(np.mean([f["shots"] for f in fin])),
                    "deaths": float(np.mean([f["deaths"] for f in fin])),
                    "win": float(np.mean([f.get("end") == "redkilled" for f in fin])),
                    "lose": float(
                        np.mean(
                            [f.get("end") in ("bluekilled", "redmission", "mutualkill") for f in fin]
                        )
                    ),
                }
        history.append(row)
        if out is not None and cfg.save_every and (upd % cfg.save_every == 0 or upd == n_updates):
            (out / "ckpt").mkdir(parents=True, exist_ok=True)
            torch.save(snapshot(), out / "ckpt" / f"{env_steps:010d}.pt")
        if upd % 10 == 0 or upd == n_updates:
            parts = " ".join(
                f"{k}:K{v['kills']:.2f}/W{v['win']:.2f}"
                for k, v in row["by_opponent"].items()
            )
            log(
                f"[{upd}/{n_updates}] {env_steps/1e6:.2f}M steps "
                f"{env_steps/max(row['time_s'],1e-6):.0f}/s ent {row['entropy']:.2f} {parts}"
            )

    ckpt = snapshot()
    result = {"config": asdict(cfg), "history": history, "env_steps": env_steps}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out / "policy.pt")
        (out / "train.json").write_text(json.dumps(result) + "\n")
    result["checkpoint"] = ckpt
    return result


def evaluate_vec(
    ckpt: dict[str, Any],
    opponent: str,
    seeds: list[int],
    greedy: bool = False,
    outcomes: Optional[list[dict[str, Any]]] = None,
) -> dict[str, float]:
    """Fast batched evaluation of a checkpoint (blue win / loss / kills per episode).

    The policy fires by sampling, so the default evaluates it as trained (sampled
    actions, torch seeded from the first episode seed for reproducibility).
    """
    import torch

    torch.manual_seed(int(seeds[0]))
    cfg = PPOConfig(**{k: v for k, v in ckpt["config"].items() if k in PPOConfig.__dataclass_fields__})
    pol = Policy(ckpt, greedy=greedy)
    n_env = min(len(seeds), 32)
    cfg.envs_per_opponent = n_env
    g = _Group(cfg, opponent, seed_base=0)
    g.keep_outcomes = outcomes is not None
    queue = list(seeds)
    dummy = 10**12
    g._seed = lambda: queue.pop(0) if queue else dummy  # type: ignore[method-assign]
    for i in range(n_env):
        g.cur_seed[i] = g._seed()
        g.steps[i] = g.vec.reset_at(i, seed=g.cur_seed[i])
        g.alive[i] = True
        g.ep_stats[i] = g._blank()
    obs_dim = int(ckpt["obs_dim"])
    n_agents = len(g.blue)
    want = set(seeds)
    results: list[dict[str, float]] = []
    while len(results) < len(seeds):
        x = pol.norm(g.obs(obs_dim).reshape(-1, obs_dim))
        with torch.no_grad():
            logits = pol.actor(torch.as_tensor(x))
        if greedy:
            acts = np.stack([lg.argmax(-1).numpy() for lg in _split(logits)], -1)
        else:
            acts = np.stack([d.sample().numpy() for d in _dist(logits)], -1)
        g.step(acts.reshape(n_env, n_agents, 3))
        for f in g.finished:
            if f["seed"] in want:
                results.append(f)
        g.finished = []
    results = results[: len(seeds)]
    if outcomes is not None:
        by_seed = {r["seed"]: r["outcome"] for r in results}
        outcomes.extend(by_seed[s] for s in seeds)
    return {
        "n": float(len(results)),
        "win": float(np.mean([r.get("end") == "redkilled" for r in results])),
        "lose": float(
            np.mean([r.get("end") in ("bluekilled", "redmission", "mutualkill") for r in results])
        ),
        "kills": float(np.mean([r["kills"] for r in results])),
        "deaths": float(np.mean([r["deaths"] for r in results])),
        "shots": float(np.mean([r["shots"] for r in results])),
    }


def select_checkpoint(
    run_dir: Path, opponents: Optional[list[str]] = None, n: int = 48, seed0: int = 900_000
) -> dict[str, Any]:
    """Pick the checkpoint with the best mean (win - loss) on validation seeds.

    Validation seeds (900,000+) are disjoint from training (1,000,000+ per seed) and from
    the paper's evaluation seeds (40,000+).
    """
    import shutil

    import torch

    opponents = opponents or ["ref_aggressive", "ref_balanced", "ref_conservative"]
    seeds = list(range(seed0, seed0 + n))
    rows = []
    for path in sorted((run_dir / "ckpt").glob("*.pt")):
        ck = torch.load(path, weights_only=False)
        ev = {o: evaluate_vec(ck, o, seeds) for o in opponents}
        score = float(np.mean([e["win"] - e["lose"] for e in ev.values()]))
        rows.append({"path": str(path), "env_steps": ck["env_steps"], "score": score, "eval": ev})
    best = max(rows, key=lambda r: r["score"])
    shutil.copy(best["path"], run_dir / "best.pt")
    sel = {"opponents": opponents, "n": n, "seed0": seed0, "best": best["path"], "rows": rows}
    (run_dir / "selection.json").write_text(json.dumps(sel) + "\n")
    return sel


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="bace.ppo")
    p.add_argument("--opponents", nargs="+", default=PPOConfig().opponents)
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs-per-opponent", type=int, default=16)
    p.add_argument("--red-mission", default="striker")
    p.add_argument("--max-cycles", type=int, default=400)
    p.add_argument(
        "--env",
        default=json.dumps(
            {
                "blue": {"rnd_offset_range": {"x": 5.0, "y": 3000.0, "z": 5.0}},
                "red": {"rnd_offset_range": {"x": 5.0, "y": 3000.0, "z": 5.0}},
            }
        ),
        help="JSON merged into the scenario (default: the paper's spawn jitter)",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cfg = PPOConfig(
        opponents=a.opponents,
        agents=a.agents,
        steps=a.steps,
        seed=a.seed,
        envs_per_opponent=a.envs_per_opponent,
        red_mission=a.red_mission,
        max_cycles=a.max_cycles,
        env_overrides=json.loads(a.env),
    )
    train(cfg, out=a.out, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
