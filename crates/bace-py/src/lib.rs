//! Python bindings for B-ACE 2.0 (PyO3).
//! Built with maturin; the pure-Python PettingZoo wrapper lives in `python/bace`.

use bace_core::config::ActionType;
use bace_core::{Action, DiscreteAction, ScenarioConfig, Simulation};
use bace_vec::recipes::{run_fsm_search, run_wez, FsmParams, WezParams};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::HashMap;

#[pyclass]
struct NativeEnv {
    sim: Simulation,
    discrete: bool,
}

fn parse_action(v: &Bound<'_, PyAny>, discrete: bool) -> PyResult<Action> {
    if let Ok(arr) = v.extract::<Vec<f64>>() {
        if discrete || arr.len() == 3 {
            let ints: Vec<i64> = arr.iter().map(|x| x.round() as i64).collect();
            return Ok(DiscreteAction::from_ints(&ints).to_continuous());
        }
        return Ok(Action::from_slice(&arr));
    }
    if let Ok(arr) = v.extract::<Vec<i64>>() {
        return Ok(DiscreteAction::from_ints(&arr).to_continuous());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "action must be a sequence of floats or ints",
    ))
}

#[pymethods]
impl NativeEnv {
    #[new]
    #[pyo3(signature = (config_json=None))]
    fn new(config_json: Option<&str>) -> PyResult<Self> {
        let mut cfg = ScenarioConfig::default();
        if let Some(s) = config_json {
            cfg.merge_json(s).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("config error: {e}"))
            })?;
        }
        let discrete = cfg.env.action_type == ActionType::Discrete;
        Ok(Self {
            sim: Simulation::new(cfg),
            discrete,
        })
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) -> PyResult<PyObject> {
        let result = self.sim.reset(seed);
        Python::with_gil(|py| step_to_py(py, &result))
    }

    fn step(&mut self, actions: Bound<'_, PyDict>) -> PyResult<PyObject> {
        let mut map = HashMap::new();
        for (k, v) in actions.iter() {
            let name: String = k.extract()?;
            map.insert(name, parse_action(&v, self.discrete)?);
        }
        let result = self.sim.step(&map);
        Python::with_gil(|py| step_to_py(py, &result))
    }

    fn obs_size(&self) -> usize {
        self.sim.obs_size()
    }

    fn agent_ids(&self) -> Vec<String> {
        self.sim.blue_agent_ids.clone()
    }

    fn is_discrete(&self) -> bool {
        self.discrete
    }

    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.sim.snapshot())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn config_json(&self) -> PyResult<String> {
        self.sim
            .config
            .to_json_pretty()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn outcome_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.sim.outcome())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
struct NativeVecEnv {
    envs: bace_vec::ParallelEnvs,
    discrete: bool,
    auto_reset: bool,
}

#[pymethods]
impl NativeVecEnv {
    #[new]
    #[pyo3(signature = (config_json=None, num_envs=8, auto_reset=true))]
    fn new(config_json: Option<&str>, num_envs: usize, auto_reset: bool) -> PyResult<Self> {
        let mut cfg = ScenarioConfig::default();
        if let Some(s) = config_json {
            cfg.merge_json(s).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("config error: {e}"))
            })?;
        }
        let discrete = cfg.env.action_type == ActionType::Discrete;
        let envs = bace_vec::ParallelEnvs::new(bace_vec::VecEnvConfig {
            num_envs: num_envs.max(1),
            scenario: cfg,
            record: false,
            runs_dir: std::path::PathBuf::from("runs"),
        });
        Ok(Self {
            envs,
            discrete,
            auto_reset,
        })
    }

    fn num_envs(&self) -> usize {
        self.envs.num_envs()
    }

    fn obs_size(&self) -> usize {
        self.envs.envs[0].obs_size()
    }

    fn agent_ids(&self) -> Vec<String> {
        self.envs.envs[0].blue_agent_ids.clone()
    }

    fn is_discrete(&self) -> bool {
        self.discrete
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, py: Python<'_>, seed: Option<u64>) -> PyResult<PyObject> {
        let results = py.allow_threads(|| self.envs.reset_all(seed));
        steps_to_py(py, &results)
    }

    fn step(&mut self, py: Python<'_>, actions: Bound<'_, PyList>) -> PyResult<PyObject> {
        if actions.len() != self.envs.num_envs() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} action dicts, got {}",
                self.envs.num_envs(),
                actions.len()
            )));
        }
        let mut parsed: Vec<HashMap<String, Action>> = Vec::with_capacity(actions.len());
        for item in actions.iter() {
            let dict = item.downcast::<PyDict>()?;
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                let name: String = k.extract()?;
                map.insert(name, parse_action(&v, self.discrete)?);
            }
            parsed.push(map);
        }
        let results = py.allow_threads(|| self.envs.step_all(&parsed));
        if self.auto_reset {
            for (i, r) in results.iter().enumerate() {
                if r.end != bace_core::EndCondition::Ongoing {
                    let seed = Some(1u64.wrapping_add(i as u64).wrapping_add(r.action_step as u64));
                    let _ = self.envs.envs[i].reset(seed);
                }
            }
        }
        steps_to_py(py, &results)
    }
}

fn steps_to_py(py: Python<'_>, results: &[bace_core::StepResult]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for r in results {
        list.append(step_to_py(py, r)?)?;
    }
    Ok(list.into())
}

fn step_to_py(py: Python<'_>, result: &bace_core::StepResult) -> PyResult<PyObject> {
    let obs = PyDict::new_bound(py);
    let rewards = PyDict::new_bound(py);
    let terms = PyDict::new_bound(py);
    let truncs = PyDict::new_bound(py);
    let infos = PyDict::new_bound(py);
    for (k, v) in &result.agents {
        obs.set_item(k, PyList::new_bound(py, &v.flat_obs))?;
        rewards.set_item(k, v.reward)?;
        terms.set_item(k, v.terminated)?;
        truncs.set_item(k, v.truncated)?;
        let info = PyDict::new_bound(py);
        info.set_item(
            "reward_breakdown",
            serde_json::to_string(&v.reward_breakdown).unwrap_or_default(),
        )?;
        infos.set_item(k, info)?;
    }
    let out = PyDict::new_bound(py);
    out.set_item("obs", obs)?;
    out.set_item("rewards", rewards)?;
    out.set_item("terminations", terms)?;
    out.set_item("truncations", truncs)?;
    out.set_item("infos", infos)?;
    out.set_item("end", format!("{:?}", result.end))?;
    out.set_item("action_step", result.action_step)?;
    Ok(out.into())
}

#[pyfunction]
#[pyo3(signature = (configs_json, max_parallel=8))]
fn run_experiment(configs_json: &str, max_parallel: usize) -> PyResult<String> {
    let configs: Vec<ScenarioConfig> = serde_json::from_str(configs_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("configs json: {e}"))
    })?;
    let results = bace_vec::run_experiment(configs, max_parallel.max(1));
    serde_json::to_string(&results)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (params_json=None, max_parallel=8))]
fn run_wez_experiment(params_json: Option<&str>, max_parallel: usize) -> PyResult<String> {
    let params: WezParams = match params_json {
        Some(s) if !s.is_empty() => serde_json::from_str(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("wez params: {e}")))?,
        _ => WezParams::default(),
    };
    let report = run_wez(params, max_parallel.max(1));
    serde_json::to_string(&report)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (params_json=None, max_parallel=8))]
fn run_fsm_search_py(params_json: Option<&str>, max_parallel: usize) -> PyResult<String> {
    let params: FsmParams = match params_json {
        Some(s) if !s.is_empty() => serde_json::from_str(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fsm params: {e}")))?,
        _ => FsmParams::default(),
    };
    let report = run_fsm_search(params, max_parallel.max(1));
    serde_json::to_string(&report)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (config_json=None, n_envs=8, steps=200))]
fn bench_parallel_py(config_json: Option<&str>, n_envs: usize, steps: u32) -> PyResult<String> {
    let mut cfg = ScenarioConfig::default();
    if let Some(s) = config_json {
        if !s.is_empty() {
            cfg.merge_json(s).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("config error: {e}"))
            })?;
        }
    }
    let report = bace_vec::bench_parallel(cfg, n_envs.max(1), steps.max(1));
    serde_json::to_string(&report)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeEnv>()?;
    m.add_class::<NativeVecEnv>()?;
    m.add_function(wrap_pyfunction!(run_experiment, m)?)?;
    m.add_function(wrap_pyfunction!(run_wez_experiment, m)?)?;
    m.add_function(wrap_pyfunction!(run_fsm_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_parallel_py, m)?)?;
    m.add("__version__", "2.0.0")?;
    Ok(())
}
