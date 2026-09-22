"""Discrete BVR tactic head for OpenJev (or random / heuristic fallback).

OpenJev scores named tactics with NLI entailment. Those tactics map onto
B-ACE's continuous action via the same bins as `DiscreteAction` in bace-core.
The 0.8B checkpoint is loaded only when CUDA is available; otherwise callers
should skip the OpenJev backend (CPU cannot run it live).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

TACTICS = (
    "search",
    "pursuit",
    "crank_left",
    "crank_right",
    "shoot",
    "support",
    "break_left",
    "break_right",
)

OPENJEV_REPO = "AlexWortega/openjev"
OPENJEV_SUBFOLDER = "qwen3.5-0.8b-nli-v2s-long"

GDM2NM = 100.0 / 1852.0


@dataclass
class CombatCue:
    alive: bool
    missiles: int
    supporting: bool
    hpt_detected: bool
    range_nm: float
    aspect_deg: float
    r_max_nm: float
    range_ratio: float
    in_wez_aspect: bool
    fsm: str


def discrete_to_continuous(fire: int, level: int, turn: int) -> np.ndarray:
    """Mirror `DiscreteAction::to_continuous` in crates/bace-core/src/obs.rs."""
    turn_deg = {0: -90.0, 1: -45.0, 2: 0.0, 3: 45.0}.get(int(turn), 90.0)
    level_v = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5}.get(int(level), 1.0)
    if turn in (0, 4):
        g = 1.0
    elif turn in (1, 3):
        g = 0.0
    else:
        g = -1.0
    return np.array(
        [turn_deg / 180.0, level_v, g, 1.0 if fire > 0 else -1.0],
        dtype=np.float32,
    )


def own_fighter(snapshot: dict[str, Any], agent: str = "agent_0") -> Optional[dict[str, Any]]:
    for f in snapshot.get("fighters") or []:
        if f.get("agent_name") == agent or (agent == "agent_0" and f.get("team") == 0):
            return f
    blues = [f for f in snapshot.get("fighters") or [] if f.get("team") == 0]
    return blues[0] if blues else None


def cues_from_snapshot(snapshot: dict[str, Any], agent: str = "agent_0") -> CombatCue:
    f = own_fighter(snapshot, agent) or {}
    tracks = f.get("tracks") or []
    hpt_id = f.get("hpt_id")
    hpt = next((t for t in tracks if t.get("detected") and t.get("id") == hpt_id), None)
    if hpt is None:
        hpt = next((t for t in tracks if t.get("detected")), None)
    r_max = float((hpt or {}).get("own_r_max") or 0.01)
    dist = float((hpt or {}).get("dist") or -1.0)
    aspect = float((hpt or {}).get("aspect") or 0.0)
    range_nm = dist * GDM2NM if dist > 0 else -1.0
    r_max_nm = r_max * GDM2NM
    ratio = dist / max(r_max, 1e-6) if dist > 0 else 99.0
    return CombatCue(
        alive=bool(f.get("alive", True)),
        missiles=int(f.get("missiles") or 0),
        supporting=bool(f.get("supporting_missile")),
        hpt_detected=hpt is not None,
        range_nm=range_nm,
        aspect_deg=aspect,
        r_max_nm=r_max_nm,
        range_ratio=ratio,
        in_wez_aspect=hpt is not None and ratio <= 1.04 and abs(aspect) < 30.0,
        fsm=str(f.get("fsm") or "Search"),
    )


def serialize_state(snapshot: dict[str, Any], agent: str = "agent_0") -> str:
    c = cues_from_snapshot(snapshot, agent)
    if not c.hpt_detected:
        track = "No hostile is currently detected on radar."
    else:
        track = (
            f"The highest-priority track is at {c.range_nm:.1f} NM, "
            f"aspect {c.aspect_deg:.0f} deg, own RMax {c.r_max_nm:.1f} NM "
            f"(range/RMax={c.range_ratio:.2f})."
        )
    support = "A missile is in flight and needs radar support." if c.supporting else "No missile is in flight."
    wez = "The target is inside WEZ and nearly on the nose." if c.in_wez_aspect else "The target is not a valid shot."
    return (
        f"Beyond-visual-range air combat. Own ship is {'alive' if c.alive else 'dead'} "
        f"with {c.missiles} missiles. FSM was {c.fsm}. {track} {support} {wez}"
    )


def tactic_hypothesis(tactic: str) -> str:
    return f"The correct next tactic is {tactic.replace('_', ' ')}."


def tactic_to_action(tactic: str, snapshot: dict[str, Any], agent: str = "agent_0") -> np.ndarray:
    """Map a named tactic onto continuous control, using DiscreteAction bins."""
    c = cues_from_snapshot(snapshot, agent)
    fire_ok = c.in_wez_aspect and c.missiles > 0
    if tactic == "search":
        return discrete_to_continuous(0, 2, 2)
    if tactic == "pursuit":
        dh = float(np.clip(c.aspect_deg / 180.0, -1.0, 1.0)) if c.hpt_detected else 0.0
        return np.array([dh, 0.0, 0.0, -1.0], dtype=np.float32)
    if tactic == "crank_left":
        return discrete_to_continuous(0, 2, 1)
    if tactic == "crank_right":
        return discrete_to_continuous(0, 2, 3)
    if tactic == "shoot":
        act = discrete_to_continuous(1 if fire_ok else 0, 2, 2)
        if c.hpt_detected:
            act[0] = float(np.clip(c.aspect_deg / 180.0, -1.0, 1.0))
        return act
    if tactic == "support":
        return discrete_to_continuous(0, 2, 1 if c.aspect_deg >= 0 else 3)
    if tactic == "break_left":
        return discrete_to_continuous(1 if fire_ok else 0, 2, 0)
    if tactic == "break_right":
        return discrete_to_continuous(1 if fire_ok else 0, 2, 4)
    return discrete_to_continuous(0, 2, 2)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class OpenJevUnavailable(RuntimeError):
    pass


def load_openjev_scorer(
    repo: str = OPENJEV_REPO,
    subfolder: str = OPENJEV_SUBFOLDER,
) -> Callable[[str, tuple[str, ...]], str]:
    """Return tactic chooser using OpenJev 0.8B. Requires CUDA + transformers."""
    if not cuda_available():
        raise OpenJevUnavailable("OpenJev skipped: no CUDA GPU (0.8B will not run live on CPU)")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError as exc:
        raise OpenJevUnavailable("OpenJev skipped: install torch and transformers") from exc

    tok = AutoTokenizer.from_pretrained(repo, subfolder=subfolder, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, subfolder=subfolder, trust_remote_code=True
    )
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    template = getattr(model.config, "nli_template", "{premise}\n{hypothesis}")
    # 3-way: contradiction, entailment, neutral
    labels = {str(v).lower(): int(k) for k, v in getattr(model.config, "id2label", {}).items()}
    ent_index = labels.get("entailment", 1)

    def choose(premise: str, tactics: tuple[str, ...] = TACTICS) -> str:
        hyps = [tactic_hypothesis(t) for t in tactics]
        texts = [template.format(premise=premise, hypothesis=h) for h in hyps]
        batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
            ent = logits[:, ent_index]
            idx = int(ent.argmax().item())
        return tactics[idx]

    return choose


def random_scorer(rng: np.random.Generator) -> Callable[[str, tuple[str, ...]], str]:
    def choose(_premise: str, tactics: tuple[str, ...] = TACTICS) -> str:
        return str(rng.choice(tactics))

    return choose


class TacticPolicy:
    def __init__(self, chooser: Callable[[str, tuple[str, ...]], str], agent: str = "agent_0"):
        self.chooser = chooser
        self.agent = agent
        self.last_tactic = "search"

    def act(self, snapshot: dict[str, Any]) -> np.ndarray:
        premise = serialize_state(snapshot, self.agent)
        self.last_tactic = self.chooser(premise, TACTICS)
        return tactic_to_action(self.last_tactic, snapshot, self.agent)
