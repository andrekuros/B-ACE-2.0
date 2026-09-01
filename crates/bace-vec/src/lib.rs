//! Batched parallel B-ACE environments.

pub mod recipes;

use bace_core::{
    Action, AgentId, Behavior, EndCondition, EpisodeOutcome, ScenarioConfig, SimSnapshot,
    Simulation, StepResult,
};
use bace_record::{EpisodeRecord, Recorder, StepRecord};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecEnvConfig {
    pub num_envs: usize,
    pub scenario: ScenarioConfig,
    pub record: bool,
    pub runs_dir: PathBuf,
}

impl Default for VecEnvConfig {
    fn default() -> Self {
        Self {
            num_envs: 1,
            scenario: ScenarioConfig::default(),
            record: false,
            runs_dir: PathBuf::from("runs"),
        }
    }
}

pub struct ParallelEnvs {
    pub envs: Vec<Simulation>,
    recorders: Vec<Option<Recorder>>,
    cfg: VecEnvConfig,
}

impl ParallelEnvs {
    pub fn new(cfg: VecEnvConfig) -> Self {
        let envs: Vec<_> = (0..cfg.num_envs)
            .map(|i| {
                let mut sc = cfg.scenario.clone();
                sc.env.seed = cfg.scenario.env.seed.wrapping_add(i as u64);
                Simulation::new(sc)
            })
            .collect();
        let recorders = envs
            .iter()
            .map(|e| {
                if cfg.record {
                    Recorder::new(&cfg.runs_dir, e.config.clone()).ok()
                } else {
                    None
                }
            })
            .collect();
        Self {
            envs,
            recorders,
            cfg,
        }
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn reset_all(&mut self, seed: Option<u64>) -> Vec<StepResult> {
        self.envs
            .iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let s = seed.map(|base| base.wrapping_add(i as u64));
                env.reset(s)
            })
            .collect()
    }

    pub fn step_all(&mut self, actions: &[HashMap<AgentId, Action>]) -> Vec<StepResult> {
        assert_eq!(actions.len(), self.envs.len());
        // Sequential recording bookkeeping; parallelize physics via rayon on owned chunks if needed.
        // For correctness with recorders, step sequentially per env but can parallelize without record.
        if self.cfg.record {
            self.envs
                .iter_mut()
                .zip(actions.iter())
                .zip(self.recorders.iter_mut())
                .map(|((env, acts), rec)| {
                    let result = env.step(acts);
                    if let Some(r) = rec {
                        r.push(StepRecord {
                            action_step: result.action_step,
                            actions: acts.clone(),
                            rewards: result
                                .agents
                                .iter()
                                .map(|(k, v)| (k.clone(), v.reward))
                                .collect(),
                            reward_breakdowns: result
                                .agents
                                .iter()
                                .map(|(k, v)| (k.clone(), v.reward_breakdown.clone()))
                                .collect(),
                            obs: result
                                .agents
                                .iter()
                                .map(|(k, v)| (k.clone(), v.obs.clone()))
                                .collect(),
                            snapshot: env.snapshot(),
                        });
                    }
                    result
                })
                .collect()
        } else {
            self.envs
                .par_iter_mut()
                .zip(actions.par_iter())
                .map(|(env, acts)| env.step(acts))
                .collect()
        }
    }

    pub fn snapshots(&self) -> Vec<SimSnapshot> {
        self.envs.iter().map(|e| e.snapshot()).collect()
    }

    pub fn finish_recordings(&mut self) -> Vec<Option<EpisodeRecord>> {
        let ends: Vec<EndCondition> = self.envs.iter().map(|e| e.end).collect();
        self.recorders
            .drain(..)
            .zip(ends)
            .map(|(rec, end)| rec.and_then(|r| r.finish(end).ok()))
            .collect()
    }
}

/// Run a grid of scenario configs (experiment mode).
pub fn run_experiment(cases: Vec<ScenarioConfig>, max_parallel: usize) -> Vec<EpisodeOutcome> {
    let chunk = max_parallel.max(1);
    cases
        .into_iter()
        .collect::<Vec<_>>()
        .par_chunks(chunk)
        .flat_map(|chunk| {
            chunk
                .iter()
                .map(|cfg| {
                    let mut sim = Simulation::new(cfg.clone());
                    sim.reset(Some(cfg.env.seed));
                    let empty = HashMap::new();
                    while sim.end == EndCondition::Ongoing {
                        sim.step(&empty);
                    }
                    sim.outcome()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputReport {
    pub n_envs: usize,
    pub steps: u32,
    pub wall_s: f64,
    pub decision_hz: f64,
    pub physics_hz: f64,
    pub realtime_factor: f64,
    pub action_repeat: u32,
    pub phy_fps: u32,
}

/// Duck-vs-duck (or given scenario) parallel step throughput. `record=false` uses rayon.
pub fn bench_parallel(mut scenario: ScenarioConfig, n_envs: usize, steps: u32) -> ThroughputReport {
    scenario.blue.behavior = if scenario.blue.behavior == Behavior::External {
        Behavior::Duck
    } else {
        scenario.blue.behavior
    };
    let n_envs = n_envs.max(1);
    let steps = steps.max(1);
    let action_repeat = scenario.env.action_repeat.max(1);
    let phy_fps = scenario.env.phy_fps.max(1);
    let mut pe = ParallelEnvs::new(VecEnvConfig {
        num_envs: n_envs,
        scenario,
        record: false,
        runs_dir: PathBuf::from("/tmp/bace_bench"),
    });
    pe.reset_all(Some(1));
    let actions: Vec<HashMap<AgentId, Action>> = (0..n_envs).map(|_| HashMap::new()).collect();
    let t0 = Instant::now();
    for _ in 0..steps {
        let results = pe.step_all(&actions);
        for (i, r) in results.iter().enumerate() {
            if r.end != EndCondition::Ongoing {
                pe.envs[i].reset(Some(1 + i as u64 + r.action_step as u64));
            }
        }
    }
    let wall_s = t0.elapsed().as_secs_f64().max(1e-9);
    let decisions = n_envs as f64 * steps as f64;
    let physics = decisions * action_repeat as f64;
    ThroughputReport {
        n_envs,
        steps,
        wall_s,
        decision_hz: decisions / wall_s,
        physics_hz: physics / wall_s,
        realtime_factor: physics / (wall_s * phy_fps as f64),
        action_repeat,
        phy_fps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bace_core::Behavior;

    #[test]
    fn parallel_duck_episode() {
        let mut scenario = ScenarioConfig::default();
        scenario.env.max_cycles = 20;
        scenario.blue.behavior = Behavior::Duck;
        scenario.red.behavior = Behavior::Duck;
        let mut pe = ParallelEnvs::new(VecEnvConfig {
            num_envs: 4,
            scenario,
            record: false,
            runs_dir: PathBuf::from("/tmp/bace_runs"),
        });
        pe.reset_all(Some(1));
        let actions: Vec<_> = (0..4).map(|_| HashMap::new()).collect();
        for _ in 0..5 {
            let results = pe.step_all(&actions);
            assert_eq!(results.len(), 4);
        }
    }

    #[test]
    fn bench_parallel_reports_hz() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 40;
        cfg.blue.behavior = Behavior::Duck;
        cfg.red.behavior = Behavior::Duck;
        let r = bench_parallel(cfg, 4, 20);
        assert_eq!(r.n_envs, 4);
        assert!(r.decision_hz > 0.0);
        assert!(r.realtime_factor > 0.0);
    }

    #[test]
    fn run_experiment_returns_outcomes() {
        let mut cfg = bace_core::ScenarioConfig::default();
        cfg.env.max_cycles = 8;
        cfg.blue.behavior = Behavior::Duck;
        cfg.red.behavior = Behavior::Duck;
        let out = run_experiment(vec![cfg.clone(), cfg], 2);
        assert_eq!(out.len(), 2);
        assert!(out[0].steps > 0);
    }
}
