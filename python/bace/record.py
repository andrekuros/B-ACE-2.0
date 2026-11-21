"""Record episodes from Python so they can be replayed in the B-ACE Lab.

Wrap any :class:`bace.BaceEnv`. Each finished episode is written as
``<runs_dir>/<run_id>/episode.json`` (+ ``meta.json``), the same layout the
dashboard server lists, so it appears under *Replay* when you run
``bace-server --runs-dir <runs_dir>``. The file can also be dropped onto the
Replay tab of the hosted Lab, no server needed.

Example::

    from bace import make_env
    from bace.record import EpisodeRecorder

    env = EpisodeRecorder(make_env(opponent="aggressive"), runs_dir="runs")
    obs, infos = env.reset(seed=3)
    while env.agents:
        actions = {a: my_policy(obs[a]) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
    print("saved", env.last_path)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

_ACTION_KEYS = ("d_heading", "d_altitude", "g_force", "fire")


def _action_record(a: Any) -> dict[str, float]:
    v = np.asarray(a, dtype=np.float64).reshape(-1).tolist()
    if len(v) == 3:  # discrete [fire, level, turn]: store the raw ints in the first slots
        v = v + [0.0]
    v = (v + [0.0, 0.0, -1.0, -1.0])[:4] if len(v) < 4 else v[:4]
    return dict(zip(_ACTION_KEYS, (float(x) for x in v)))


class EpisodeRecorder:
    """PettingZoo-compatible wrapper that records every episode to ``runs_dir``."""

    def __init__(self, env: Any, runs_dir: str | Path = "runs", every: int = 1):
        self.env = env
        self.runs_dir = Path(runs_dir)
        self.every = max(1, int(every))
        self.last_path: Optional[Path] = None
        self._episode = 0
        self._steps: list[dict[str, Any]] = []
        self._recording = False
        env._native.set_keep_frames(True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def agents(self) -> list[str]:
        return self.env.agents

    def _frame(self) -> dict[str, Any]:
        return json.loads(self.env._native.frame_json())

    def _push(self, frame: dict[str, Any], actions: dict[str, Any]) -> None:
        agents = frame.get("agents", {})
        self._steps.append(
            {
                "action_step": frame["action_step"],
                "actions": {k: _action_record(v) for k, v in actions.items()},
                "rewards": {k: v["reward"] for k, v in agents.items()},
                "reward_breakdowns": {k: v["reward_breakdown"] for k, v in agents.items()},
                "obs": {k: v["obs"] for k, v in agents.items()},
                "snapshot": frame["snapshot"],
                "events": frame.get("events", []),
            }
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        out = self.env.reset(seed=seed, options=options)
        self._episode += 1
        self._recording = (self._episode - 1) % self.every == 0
        self._steps = []
        if self._recording:
            self._push(self._frame(), {})
        return out

    def step(self, actions: dict[str, Any]):
        out = self.env.step(actions)
        if self._recording:
            frame = self._frame()
            self._push(frame, actions)
            if frame["end"] != "ongoing":
                self.save()
        return out

    def save(self) -> Optional[Path]:
        """Write the current episode (called automatically when it ends)."""
        if not self._steps:
            return None
        run_id = str(uuid.uuid4())
        folder = self.runs_dir / run_id
        folder.mkdir(parents=True, exist_ok=True)
        last = self._steps[-1]
        meta = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "config": json.loads(self.env._native.config_json()),
            "end": last["snapshot"]["end"],
            "total_steps": len(self._steps),
        }
        (folder / "episode.json").write_text(json.dumps({"meta": meta, "steps": self._steps}))
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        self.last_path = folder / "episode.json"
        self._steps = []
        self._recording = False
        return self.last_path

    def close(self) -> None:
        self.env.close()
