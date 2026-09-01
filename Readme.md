# B-ACE 2.0

Beyond Visual Range Air Combat Environment — a **Rust-native** simulator with PettingZoo Python bindings and a local web dashboard.

Successor to [B-ACE 1.0](https://github.com/andrekuros/B-ACE) (Godot + TCP/JSON). This repository is standalone.

## Highlights

- High-performance **Rust** sim core (`bace-core`) with in-process parallel envs (`bace-vec`)
- Clean **PettingZoo ParallelEnv** API via PyO3 (`from bace import BaceEnv, make_env`)
- **Web dashboard** for live parallel runs, episode replay, and WEZ / FSM recipes
- Three study-case recipes: WEZ characterization, FSM search, IPPO (shared weights) on NativeVecEnv
- Vectorized `BaceVecEnv` (`NativeVecEnv`, rayon, `record=false`) and `python -m bace.bench`

Only **blue** can be `external` (learning) in this release. Red uses `baseline1`, `duck`, or `fire_once`.

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

### Dashboard

```bash
cargo run -p bace-server -- --port 8787 --static-dir web/dist --runs-dir runs
# open http://127.0.0.1:8787
```

Start a live job (4 envs, baseline vs duck), then use Experiment → WEZ or FSM.

## Demo path (framework release)

0. `cargo test -p bace-core -p bace-vec` and `pytest -q tests/python`, then `python -m bace.examples.random_policy`
1. Dashboard live: 4 envs, baseline vs duck
2. `python -m bace.experiment wez`
3. `python -m bace.experiment fsm` (writes `configs/baselines/{aggressive,balanced,cautious}.json`)
4. `python -m bace.experiment marl_core --profile paper` (IPPO, NativeVecEnv; 1v1 duck/FSM then 2v2 duck; 2v2 FSM gated). `--vs duck|fsm --agents 1|2`
5. `python -m bace.bench` (throughput vs `n_envs`, PyO3 tax, WEZ/FSM wall-clock)

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

`make_env(opponent=...)` accepts `duck`, `baseline`, `aggressive`, `balanced`, `cautious`, `fsm`. Caps at **4v4** with a 2×2 box spawn (4 NM). `share_tracks` and `red_mission="striker"` are keyword args.

Observations are exported as a flat `float32` vector: own(9) + enemies(13 each) + allies(6 each).
Continuous actions are `[d_heading, d_altitude, g_force, fire]` in `[-1, 1]`.
Set `"action_type": "discrete"` for `MultiDiscrete([2, 5, 5])` (fire, level, turn).

Behaviors: `external` (RL, blue only), `baseline1` (FSM), `duck`, `fire_once`.

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
| `crates/bace-server` | Axum HTTP/WS dashboard API |
| `python/bace` | PettingZoo wrapper + experiment CLI |
| `web/dist` | Live + replay + experiment UI |
| `configs/experiments/` | WEZ, FSM, MARL recipe JSON |
| `examples/` | Thin wrappers around the package CLI |

## Related

- [B-ACE 1.0](https://github.com/andrekuros/B-ACE) — original Godot-based environment
- [B_ACE_2_Paper](https://github.com/andrekuros/B_ACE_2_Paper) — working paper, figures, and reproduction script

## License

MIT — see [LICENSE](LICENSE).
