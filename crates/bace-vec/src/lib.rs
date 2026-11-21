//! Batched parallel B-ACE environments.

pub mod recipes;

#[cfg(feature = "native")]
mod native;
#[cfg(feature = "native")]
pub use native::*;

use bace_core::{EndCondition, EpisodeOutcome, ScenarioConfig, Simulation};
use std::collections::HashMap;

/// Run one scripted episode to completion. External agents hold `Action::default()`.
pub fn run_episode(cfg: &ScenarioConfig) -> EpisodeOutcome {
    let mut sim = Simulation::new(cfg.clone());
    sim.reset(Some(cfg.env.seed));
    let empty = HashMap::new();
    while sim.end == EndCondition::Ongoing {
        sim.step(&empty);
    }
    sim.outcome()
}

/// Run a grid of scenario configs (experiment mode).
#[cfg(feature = "native")]
pub fn run_experiment(cases: Vec<ScenarioConfig>, max_parallel: usize) -> Vec<EpisodeOutcome> {
    use rayon::prelude::*;
    let chunk = max_parallel.max(1);
    cases
        .par_chunks(chunk)
        .flat_map(|chunk| chunk.iter().map(run_episode).collect::<Vec<_>>())
        .collect()
}

/// Sequential fallback used by WebAssembly builds (no threads).
#[cfg(not(feature = "native"))]
pub fn run_experiment(cases: Vec<ScenarioConfig>, _max_parallel: usize) -> Vec<EpisodeOutcome> {
    cases.iter().map(run_episode).collect()
}
