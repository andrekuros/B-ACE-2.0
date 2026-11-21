# B-ACE 2.0

Beyond Visual Range Air Combat Environment — a **Rust-native** simulator with PettingZoo Python bindings and **B-ACE Lab**, a browser workbench that runs the same simulator compiled to WebAssembly.

Successor to [B-ACE 1.0](https://github.com/andrekuros/B-ACE) (Godot + TCP/JSON). This repository is standalone.

## Highlights

- High-performance **Rust** sim core (`bace-core`) with in-process parallel envs (`bace-vec`)
- **Physics from open sources only**: a notional generic medium-range missile (energy, drag, proportional navigation, datalink and seeker), US Standard Atmosphere, aircraft energy, and a weapon engagement zone computed by flying the missile. Every element is traced in [docs/PUBLIC_REFERENCES.md](docs/PUBLIC_REFERENCES.md); no doctrine or system data.
- Clean **PettingZoo ParallelEnv** API via PyO3 (`from bace import BaceEnv, make_env`)
- **B-ACE Lab** in the browser, no install: scenario builder, 2D tactical display (radar, WEZ/NEZ, tracks, missile guidance state) with an optional 3D view, playback and scrubbing, engagement / reward / altitude telemetry, an event log, and a named observation inspector
- **In-browser experiments**: cross-play matrices with the paper's metrics and 95% bootstrap CIs, WEZ validation against the missile model, FSM opponent search, and an environment self-test
- `bace.record.EpisodeRecorder` to watch your trained Python policies in the Lab
- Three study-case recipes: WEZ characterization, FSM search, IPPO (shared weights) on NativeVecEnv
- Vectorized `BaceVecEnv` (`NativeVecEnv`, rayon, `record=false`) and `python -m bace.bench`
- Optional OpenJev tactic head (`examples/openjev`) as a discrete comparison, not a paper result

Blue and red can both be `external` (self-play). Scripted red behaviors are `baseline1`, `duck`, and `fire_once`. Public recipes cap force size at 4v4.

Scripted episodes are deterministic unless you add randomness: set `beh_noise` (relative jitter on the FSM thresholds per decision, e.g. `0.1`) and/or `rnd_offset_range` (spawn jitter) on a team. Without them every seed produces the same scripted episode. See [docs/MODEL.md](docs/MODEL.md).

## Quick start

```bash
# Rust tests
cargo test -p bace-core -p bace-record -p bace-vec

# Python env
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest pettingzoo gymnasium numpy
maturin develop -m crates/bace-py/Cargo.toml
pytest -q tests/python
python -m bace.examples.random_policy
```

### B-ACE Lab (browser)

```bash
./scripts/build_web.sh                      # builds web/dist/engine/bace.wasm
cargo run -p bace-server -- --port 8787 --static-dir web/dist --runs-dir runs
# open http://127.0.0.1:8787
```

Any static file server works too (`python -m http.server -d web/dist`); the server only adds the
list of recordings under `runs/`. The `pages` workflow publishes the same folder to GitHub Pages.

| Tab | What it does |
|-----|--------------|
| Simulate | Presets or your own scenario; pilots are scripted FSMs or RL slots (random, idle, or flown with the keyboard). Copy the scenario as Python or JSON. |
| Evaluate | Every blue pilot against every red pilot, N episodes each, in Web Workers. F/B/R-score, mission, missile efficiency and return, each with a 95% bootstrap CI. Export CSV. |
| WEZ | Polar plot of the analytic envelope the agents observe, and the WEZ recipe measuring real hit rates against it. |
| FSM search | The GA recipe for scripted opponents; promote elites to pilots. |
| Diagnostics | Reproducibility, seed variety, observation contract and bounds, no information on undetected enemies, reward accounting, termination flags, event log, WEZ consistency, speed. |
| Replay | Open episodes saved from the Lab, from Python, or listed by the server. |

Watch a Python policy:

```python
from bace import make_env
from bace.record import EpisodeRecorder

env = EpisodeRecorder(make_env(opponent="aggressive"), runs_dir="runs")
obs, infos = env.reset(seed=3)
while env.agents:
    obs, rewards, terms, truncs, infos = env.step({a: policy(obs[a]) for a in env.agents})
# runs/<id>/episode.json now shows under Replay
```

Web tests: `cd tests/web && npm ci && npx playwright install chromium && npm test`.

## Demo path (framework release)

0. `cargo test -p bace-core -p bace-vec` and `pytest -q tests/python`, then `python -m bace.examples.random_policy`
1. Dashboard live: 4 envs, baseline vs duck
2. `python -m bace.experiment wez`
3. `python -m bace.experiment fsm` (writes `configs/baselines/{aggressive,balanced,cautious}.json`)
4. `python -m bace.experiment marl_core --profile paper` (IPPO vs duck/FSM). `python -m bace.experiment marl_selfplay --profile paper` (2v2 shared-policy self-play).
5. `python -m bace.bench` (throughput vs `n_envs`, PyO3 tax, WEZ/FSM wall-clock)
6. Optional, not a paper result: `python examples/openjev/eval.py --backend auto --seeds 16` (0.8B only if CUDA is available)

Tiny CI-sized runs: add `--smoke` (or `--profile smoke`) to wez/fsm/marl.
Paper-scale CLI profiles (`--profile paper`) stay in this repo; the draft and figures live in [B_ACE_2_Paper](https://github.com/andrekuros/B_ACE_2_Paper).

## Python API

```python
from bace import BaceEnv, make_env

env = make_env(opponent="duck", agents=1, max_cycles=500, seed=1)
obs, infos = env.reset(seed=1)
actions = {a: env.action_space(a).sample() for a in env.agents}
obs, rewards, terms, truncs, infos = env.step(actions)
```

`make_env(opponent=...)` accepts `duck`, `baseline`, the reference pilots `ref_aggressive`, `ref_balanced`, `ref_conservative`, and the searched profiles `aggressive`, `balanced`, `cautious`. Caps at **4v4** with a 2×2 box spawn (4 NM). `share_tracks` and `red_mission="striker"` are keyword args.

Observations are exported as a flat `float32` vector: own(9) + enemies(13 each) + allies(6 each).
Continuous actions are `[d_heading, d_altitude, g_force, fire]` in `[-1, 1]`.
Set `"action_type": "discrete"` for `MultiDiscrete([2, 5, 5])` (fire, level, turn).

Behaviors: `external` (RL, blue and red), `baseline1` (FSM), `duck`, `fire_once`.

Batch experiments:

```python
from bace import run_experiment
results = run_experiment([{
    "env": {"max_cycles": 80, "seed": 1},
    "blue": {"behavior": "baseline1"},
    "red": {"behavior": "duck"},
}])
```

## Layout

| Path | Role |
|------|------|
| `crates/bace-core` | Physics, WEZ, missiles, radar, FSM, rewards |
| `crates/bace-vec` | Batched parallel envs + WEZ/FSM recipes |
| `crates/bace-record` | Episode JSON recordings under `runs/` |
| `crates/bace-py` | PyO3 native module |
| `crates/bace-server` | Axum server for the Lab and recordings |
| `crates/bace-wasm` | WebAssembly build of the simulator (plain JSON C ABI) |
| `python/bace` | PettingZoo wrapper, experiment CLI, episode recorder |
| `web/dist` | B-ACE Lab (vanilla ES modules, no build step) |
| `tests/web` | Node tests for the engine and a headless smoke test of every tab |
| `docs/MODEL.md` | What is simulated, and known limitations |
| `docs/PUBLIC_REFERENCES.md` | Open-literature source of every model element |
| `configs/reference/` | Reference scripted pilots |
| `examples/openjev` | Optional discrete tactic head vs duck and baseline1 |
| `configs/experiments/` | WEZ, FSM, MARL recipe JSON |
| `examples/` | Thin wrappers around the package CLI |

## Related

- [B-ACE 1.0](https://github.com/andrekuros/B-ACE) — original Godot-based environment
- [B_ACE_2_Paper](https://github.com/andrekuros/B_ACE_2_Paper) — working paper, figures, and reproduction script

## License

MIT — see [LICENSE](LICENSE).
