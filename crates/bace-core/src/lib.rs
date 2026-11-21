//! B-ACE 2.0 simulation core.
//!
//! Redesigned BVR air-combat environment with kinematic fighters,
//! WEZ-based missiles, geometric radar, and FSM baselines.

pub mod atmosphere;
pub mod config;
pub mod fighter;
pub mod geometry;
pub mod missile;
pub mod obs;
pub mod rewards;
pub mod sim;
pub mod units;
pub mod wez;

pub use config::{
    ActionType, AircraftModel, Behavior, EnvConfig, MissileModel, MissileSupport, Mission,
    RewardsConfig, ScenarioConfig, TeamConfig,
};
pub use obs::{Action, AllyObs, DiscreteAction, EnemyObs, OwnObs, StructuredObs};
pub use rewards::RewardBreakdown;
pub use sim::{
    AgentId, EndCondition, EpisodeOutcome, EventKind, SimEvent, SimSnapshot, Simulation,
    StepResult, Team,
};
pub use units::SConv;
