//! WebAssembly bridge for the B-ACE Lab.
//!
//! A deliberately tiny C ABI so the browser needs no generated glue:
//! inputs are UTF-8 JSON written into memory obtained from [`alloc`], and every
//! call that produces data writes JSON into a shared result buffer read through
//! [`result_ptr`] / [`result_len`]. Calls return `0` (or a handle `>= 0`) on
//! success and `-1` on error, in which case the result buffer holds
//! `{"error": "..."}`.
//!
//! The same functions are plain Rust and are unit-tested natively.

use bace_core::{Action, EndCondition, ScenarioConfig, Simulation};
use bace_vec::recipes::{self, FsmParams, WezParams};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const REFERENCE: [(&str, &str); 3] = [
    (
        "aggressive",
        include_str!("../../../configs/reference/aggressive.json"),
    ),
    (
        "balanced",
        include_str!("../../../configs/reference/balanced.json"),
    ),
    (
        "conservative",
        include_str!("../../../configs/reference/conservative.json"),
    ),
];

const BASELINES: [(&str, &str); 3] = [
    (
        "aggressive",
        include_str!("../../../configs/baselines/aggressive.json"),
    ),
    (
        "balanced",
        include_str!("../../../configs/baselines/balanced.json"),
    ),
    (
        "cautious",
        include_str!("../../../configs/baselines/cautious.json"),
    ),
];

thread_local! {
    static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static SIMS: RefCell<Vec<Option<Simulation>>> = const { RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// Pure Rust API (tested natively)
// ---------------------------------------------------------------------------

/// Policy used for `external` agents when the browser does not send actions.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExternalPolicy {
    /// Fly straight and level, never fire (`Action::default()`).
    #[default]
    Idle,
    /// Uniform random continuous actions; fires with probability 0.05 per decision.
    Random,
}

pub fn random_action(rng: &mut StdRng) -> Action {
    Action {
        d_heading: rng.gen_range(-1.0..1.0),
        d_altitude: rng.gen_range(-1.0..1.0),
        g_force: rng.gen_range(-1.0..1.0),
        fire: if rng.gen_bool(0.05) { 1.0 } else { -1.0 },
    }
}

/// Frame sent to the browser after reset/step: everything the Lab draws.
pub fn frame(sim: &Simulation, step: &bace_core::StepResult) -> Value {
    json!({
        "snapshot": sim.snapshot(),
        "agents": step.agents,
        "events": step.events,
        "end": step.end,
        "action_step": step.action_step,
    })
}

/// Run one episode to the end; external agents follow `policy`.
pub fn run_episode(cfg: &ScenarioConfig, policy: ExternalPolicy) -> Value {
    let mut sim = Simulation::new(cfg.clone());
    let seed = cfg.env.seed;
    sim.reset(Some(seed));
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let ids = sim.agent_ids();
    let mut acts: HashMap<String, Action> = HashMap::new();
    while sim.end == EndCondition::Ongoing {
        acts.clear();
        for id in &ids {
            let a = match policy {
                ExternalPolicy::Idle => Action::default(),
                ExternalPolicy::Random => random_action(&mut rng),
            };
            acts.insert(id.clone(), a);
        }
        sim.step(&acts);
    }
    let mut v = serde_json::to_value(sim.outcome()).unwrap_or(Value::Null);
    if let Value::Object(m) = &mut v {
        m.remove("config");
    }
    v
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    /// Partial scenarios, each merged over `ScenarioConfig::default()`.
    pub cases: Vec<Value>,
    #[serde(default)]
    pub policy: ExternalPolicy,
}

pub fn run_batch(req: &BatchRequest) -> Result<Value, serde_json::Error> {
    let mut out = Vec::with_capacity(req.cases.len());
    for c in &req.cases {
        out.push(run_episode(
            &ScenarioConfig::from_partial_value(c)?,
            req.policy,
        ));
    }
    Ok(Value::Array(out))
}

pub fn baselines() -> Value {
    profiles(&BASELINES)
}

/// Reference FSM profiles from open BVR principles (`configs/reference`).
pub fn reference() -> Value {
    profiles(&REFERENCE)
}

fn profiles(list: &[(&str, &str)]) -> Value {
    let mut m = serde_json::Map::new();
    for &(name, src) in list {
        let v: Value = serde_json::from_str(src).unwrap_or(Value::Null);
        // Profiles are stored as full scenarios; the opponent lives in `red`.
        let team = v.get("red").cloned().unwrap_or(Value::Null);
        m.insert(name.to_string(), team);
    }
    Value::Object(m)
}

pub fn info() -> Value {
    json!({
        "version": VERSION,
        "default_config": ScenarioConfig::default(),
        "baselines": baselines(),
        "reference": reference(),
        "wez_defaults": WezParams::default(),
        "wez_smoke": WezParams::smoke(),
        "fsm_defaults": FsmParams::default(),
        "fsm_smoke": FsmParams::smoke(),
        "missile_params": bace_core::missile::MissileParams::default(),
        "wez_table": bace_core::wez::table(),
    })
}

/// WEZ the agents observe. `target_aspect_deg`: target nose vs its line of sight to the
/// shooter (0 = hot, 180 = cold); `off_bore_deg`: target bearing off the shooter's nose.
pub fn wez_eval(
    legacy: bool,
    shooter_ft: f64,
    target_ft: f64,
    off_bore_deg: f64,
    target_aspect_deg: f64,
) -> Value {
    use bace_core::{MissileModel, SConv};
    let model = if legacy {
        MissileModel::Legacy
    } else {
        MissileModel::GenericMraam
    };
    // For the legacy envelope the heading difference with the target on the nose.
    let angle_off = 180.0 - target_aspect_deg.abs().min(180.0);
    let w = bace_core::wez::evaluate_model(
        model,
        shooter_ft * SConv::FT2GDM,
        target_ft * SConv::FT2GDM,
        off_bore_deg,
        target_aspect_deg,
        angle_off,
    );
    json!({
        "r_max_nm": w.r_max * SConv::GDM2NM,
        "r_nez_nm": w.r_nez * SConv::GDM2NM,
    })
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

fn set_result(v: &Value) {
    let bytes = serde_json::to_vec(v).unwrap_or_else(|_| b"null".to_vec());
    RESULT.with(|r| *r.borrow_mut() = bytes);
}

fn fail(msg: impl std::fmt::Display) -> i32 {
    set_result(&json!({ "error": msg.to_string() }));
    -1
}

/// # Safety
/// `ptr`/`len` must describe memory previously returned by [`alloc`].
unsafe fn read_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, std::str::Utf8Error> {
    if len == 0 {
        return Ok("");
    }
    std::str::from_utf8(std::slice::from_raw_parts(ptr, len))
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len.max(1));
    let p = buf.as_mut_ptr();
    std::mem::forget(buf);
    p
}

/// # Safety
/// Must pair with a prior [`alloc`] of the same length.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
}

#[no_mangle]
pub extern "C" fn result_ptr() -> *const u8 {
    RESULT.with(|r| r.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}

#[no_mangle]
pub extern "C" fn lab_info() -> i32 {
    set_result(&info());
    0
}

/// Create a simulation from a (possibly partial) scenario JSON merged over defaults.
///
/// # Safety
/// See [`read_str`].
#[no_mangle]
pub unsafe extern "C" fn sim_create(ptr: *const u8, len: usize) -> i32 {
    let src = match read_str(ptr, len) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let mut cfg = ScenarioConfig::default();
    if !src.trim().is_empty() {
        if let Err(e) = cfg.merge_json(src) {
            return fail(format!("bad scenario: {e}"));
        }
    }
    let sim = Simulation::new(cfg);
    SIMS.with(|s| {
        let mut sims = s.borrow_mut();
        if let Some(i) = sims.iter().position(Option::is_none) {
            sims[i] = Some(sim);
            i as i32
        } else {
            sims.push(Some(sim));
            (sims.len() - 1) as i32
        }
    })
}

#[no_mangle]
pub extern "C" fn sim_free(h: i32) {
    SIMS.with(|s| {
        if let Some(slot) = s.borrow_mut().get_mut(h as usize) {
            *slot = None;
        }
    });
}

fn with_sim<R>(h: i32, f: impl FnOnce(&mut Simulation) -> R) -> Option<R> {
    SIMS.with(|s| s.borrow_mut().get_mut(h as usize)?.as_mut().map(f))
}

/// Reset with `seed` (pass a negative number to keep the configured seed).
#[no_mangle]
pub extern "C" fn sim_reset(h: i32, seed: f64) -> i32 {
    let seed = if seed >= 0.0 { Some(seed as u64) } else { None };
    match with_sim(h, |sim| {
        let r = sim.reset(seed);
        frame(sim, &r)
    }) {
        Some(v) => {
            set_result(&v);
            0
        }
        None => fail("unknown sim handle"),
    }
}

/// Step with `{"agent_0": [d_hdg, d_alt, g, fire], ...}`. Missing agents hold the default action.
///
/// # Safety
/// See [`read_str`].
#[no_mangle]
pub unsafe extern "C" fn sim_step(h: i32, ptr: *const u8, len: usize) -> i32 {
    let src = match read_str(ptr, len) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let raw: HashMap<String, Vec<f64>> = if src.trim().is_empty() {
        HashMap::new()
    } else {
        match serde_json::from_str(src) {
            Ok(v) => v,
            Err(e) => return fail(format!("bad actions: {e}")),
        }
    };
    let acts: HashMap<String, Action> = raw
        .into_iter()
        .map(|(k, v)| (k, Action::from_slice(&v)))
        .collect();
    match with_sim(h, |sim| {
        let r = sim.step(&acts);
        frame(sim, &r)
    }) {
        Some(v) => {
            set_result(&v);
            0
        }
        None => fail("unknown sim handle"),
    }
}

#[no_mangle]
pub extern "C" fn sim_outcome(h: i32) -> i32 {
    match with_sim(h, |sim| {
        serde_json::to_value(sim.outcome()).unwrap_or(Value::Null)
    }) {
        Some(v) => {
            set_result(&v);
            0
        }
        None => fail("unknown sim handle"),
    }
}

/// `{"cases": [scenario...], "policy": "idle" | "random"}` -> `[outcome...]` (config stripped).
///
/// # Safety
/// See [`read_str`].
#[no_mangle]
pub unsafe extern "C" fn batch_run(ptr: *const u8, len: usize) -> i32 {
    let src = match read_str(ptr, len) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    match serde_json::from_str::<BatchRequest>(src) {
        Ok(req) => match run_batch(&req) {
            Ok(v) => {
                set_result(&v);
                0
            }
            Err(e) => fail(format!("bad scenario in batch: {e}")),
        },
        Err(e) => fail(format!("bad batch: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn wez_evaluate(
    legacy: i32,
    shooter_ft: f64,
    target_ft: f64,
    off_bore_deg: f64,
    target_aspect_deg: f64,
) -> i32 {
    set_result(&wez_eval(
        legacy != 0,
        shooter_ft,
        target_ft,
        off_bore_deg,
        target_aspect_deg,
    ));
    0
}

/// Run the WEZ characterization recipe (same code as `python -m bace.experiment wez`).
///
/// # Safety
/// See [`read_str`].
#[no_mangle]
pub unsafe extern "C" fn recipe_wez(ptr: *const u8, len: usize) -> i32 {
    let src = match read_str(ptr, len) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let params: WezParams = if src.trim().is_empty() {
        WezParams::default()
    } else {
        match serde_json::from_str(src) {
            Ok(p) => p,
            Err(e) => return fail(format!("bad wez params: {e}")),
        }
    };
    let mut rep = serde_json::to_value(recipes::run_wez(params, 1)).unwrap_or(Value::Null);
    if let Value::Object(m) = &mut rep {
        m.remove("outcomes");
    }
    set_result(&rep);
    0
}

/// Run the FSM parameter search recipe (same code as `python -m bace.experiment fsm`).
///
/// # Safety
/// See [`read_str`].
#[no_mangle]
pub unsafe extern "C" fn recipe_fsm(ptr: *const u8, len: usize) -> i32 {
    let src = match read_str(ptr, len) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let params: FsmParams = if src.trim().is_empty() {
        FsmParams::default()
    } else {
        match serde_json::from_str(src) {
            Ok(p) => p,
            Err(e) => return fail(format!("bad fsm params: {e}")),
        }
    };
    set_result(&serde_json::to_value(recipes::run_fsm_search(params, 1)).unwrap_or(Value::Null));
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result() -> Value {
        RESULT.with(|r| serde_json::from_slice(&r.borrow()).unwrap())
    }

    #[test]
    fn info_has_profiles_and_defaults() {
        let v = info();
        for k in ["aggressive", "balanced", "cautious"] {
            assert_eq!(v["baselines"][k]["behavior"], "baseline1", "{k}");
        }
        for k in ["aggressive", "balanced", "conservative"] {
            assert_eq!(v["reference"][k]["behavior"], "baseline1", "{k}");
        }
        assert!(v["wez_table"]["rmax_nm"].as_array().unwrap().len() > 100);
        assert_eq!(v["default_config"]["env"]["phy_fps"], 20);
    }

    #[test]
    fn create_reset_step_roundtrip() {
        let cfg = br#"{"env":{"max_cycles":30},"red":{"behavior":"duck"}}"#;
        let h = unsafe { sim_create(cfg.as_ptr(), cfg.len()) };
        assert!(h >= 0);
        assert_eq!(sim_reset(h, 3.0), 0);
        let f = result();
        assert_eq!(f["action_step"], 0);
        assert!(f["agents"]["agent_0"]["flat_obs"].is_array());
        let acts = br#"{"agent_0":[0.2,0.0,0.0,-1.0]}"#;
        assert_eq!(unsafe { sim_step(h, acts.as_ptr(), acts.len()) }, 0);
        let f = result();
        assert_eq!(f["action_step"], 1);
        assert!(f["snapshot"]["time_s"].as_f64().unwrap() > 0.0);
        sim_free(h);
        assert_eq!(sim_reset(h, 1.0), -1);
    }

    #[test]
    fn bad_json_reports_error() {
        let bad = b"{nope";
        assert_eq!(unsafe { sim_create(bad.as_ptr(), bad.len()) }, -1);
        assert!(result()["error"].as_str().unwrap().contains("bad scenario"));
    }

    #[test]
    fn batch_is_deterministic_and_random_policy_differs() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 120;
        cfg.red.behavior = bace_core::Behavior::Baseline1;
        let cfg = serde_json::to_value(cfg).unwrap();
        let req = BatchRequest {
            cases: vec![cfg.clone(), cfg.clone()],
            policy: ExternalPolicy::Random,
        };
        let a = run_batch(&req).unwrap();
        let b = run_batch(&req).unwrap();
        assert_eq!(a, b);
        assert!(a[0].get("config").is_none());
        let idle = run_batch(&BatchRequest {
            cases: vec![cfg],
            policy: ExternalPolicy::Idle,
        })
        .unwrap();
        assert_eq!(idle.as_array().unwrap().len(), 1);
    }

    #[test]
    fn batch_merges_partial_cases_over_defaults() {
        // A bare red block must not spawn red on top of blue.
        let req: BatchRequest = serde_json::from_str(
            r#"{"cases":[{"env":{"max_cycles":3},"red":{"behavior":"duck"}}]}"#,
        )
        .unwrap();
        let out = run_batch(&req).unwrap();
        assert_eq!(out[0]["steps"], 3);
        let c = ScenarioConfig::from_partial_value(&req.cases[0]).unwrap();
        assert!(c.red.init_position.z < 0.0);
    }

    #[test]
    fn wez_eval_head_longer_than_tail() {
        for legacy in [false, true] {
            let head = wez_eval(legacy, 25000.0, 25000.0, 0.0, 0.0);
            let tail = wez_eval(legacy, 25000.0, 25000.0, 0.0, 180.0);
            assert!(
                head["r_max_nm"].as_f64() > tail["r_max_nm"].as_f64(),
                "legacy={legacy}"
            );
        }
    }
}
