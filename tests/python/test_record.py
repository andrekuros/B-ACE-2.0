import json

import numpy as np
import pytest

bace = pytest.importorskip("bace")
from bace import make_env  # noqa: E402
from bace.record import EpisodeRecorder  # noqa: E402


def test_recorder_writes_server_and_lab_format(tmp_path):
    env = EpisodeRecorder(make_env(opponent="aggressive", agents=2, max_cycles=60, seed=2), runs_dir=tmp_path)
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(0)
    while env.agents:
        acts = {a: rng.uniform(-1, 1, size=4).astype(np.float32) for a in env.agents}
        obs, *_ = env.step(acts)
    path = env.last_path
    assert path is not None and path.exists()
    rec = json.loads(path.read_text())
    assert rec["meta"]["total_steps"] == len(rec["steps"])
    assert (path.parent / "meta.json").exists()
    first, last = rec["steps"][0], rec["steps"][-1]
    assert first["action_step"] == 0 and last["snapshot"]["end"] != "ongoing"
    assert set(first["obs"]) == {"agent_0", "agent_1"}
    assert "enemies" in first["obs"]["agent_0"]
    assert set(rec["steps"][1]["actions"]["agent_0"]) == {"d_heading", "d_altitude", "g_force", "fire"}
    assert rec["meta"]["config"]["red"]["num_agents"] == 2


def test_recorder_every_n(tmp_path):
    env = EpisodeRecorder(make_env(opponent="duck", max_cycles=5), runs_dir=tmp_path, every=2)
    for ep in range(4):
        env.reset(seed=ep)
        while env.agents:
            env.step({a: [0, 0, -1, -1] for a in env.agents})
    assert len(list(tmp_path.iterdir())) == 2
