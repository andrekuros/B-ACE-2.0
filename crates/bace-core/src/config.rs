//! Typed scenario configuration for B-ACE 2.0.

pub use crate::missile::MissileModel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Behavior {
    External,
    Baseline1,
    Duck,
    /// Fire a single missile when a track is detected, then fly toward the target.
    FireOnce,
}

impl Default for Behavior {
    fn default() -> Self {
        Self::Baseline1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mission {
    Dca,
    Striker,
}

impl Default for Mission {
    fn default() -> Self {
        Self::Dca
    }
}

/// Aircraft speed model (`env.aircraft_model`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AircraftModel {
    /// Point-mass energy: speed recovers toward the altitude's maximum, bleeds with
    /// load factor (induced drag) and trades with climb.
    #[default]
    Energy,
    /// B-ACE 2.0: constant speed for the altitude; manoeuvring is free.
    ConstantSpeed,
}

/// How missile datalink guidance depends on the shooter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MissileSupport {
    /// The missile receives perfect target updates from launch until its seeker goes
    /// active (or the shooter fires another missile), whatever the shooter does.
    /// B-ACE 2.0 behaviour.
    Launch,
    /// Datalink updates only while the shooter is alive and tracks the target (radar or
    /// shared datalink); regained when the track returns. Unsupported, the generic
    /// missile flies inertially to the last predicted point and can still acquire with
    /// its seeker; the legacy missile flies straight and cannot go active (B-ACE 1.0).
    #[default]
    Track,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    Continuous,
    Discrete,
}

impl Default for ActionType {
    fn default() -> Self {
        Self::Continuous
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Vec3Config {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Vec3Config {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 25000.0, // ft (converted at spawn)
            z: 30.0,    // NM
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BehConfig {
    /// Shoot when range_ratio <= d_shot (range / RMax). The turn-in starts at 1.15 x d_shot.
    pub d_shot: Vec<f64>,
    /// Crank when the threat factor of the target exceeds l_crank (B-ACE 1.0 semantics).
    pub l_crank: Vec<f64>,
    /// Break (turn away, held 8 s) when threat_factor >= l_break (+0.5 while supporting a missile).
    pub l_break: Vec<f64>,
}

impl Default for BehConfig {
    fn default() -> Self {
        Self {
            d_shot: vec![1.04, 0.50, 1.09],
            l_crank: vec![1.06, 0.98, 0.98],
            l_break: vec![1.05, 1.17, 0.45],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TeamConfig {
    pub num_agents: usize,
    pub share_tracks: bool,
    pub behavior: Behavior,
    pub mission: Mission,
    pub beh_config: BehConfig,
    pub init_position: Vec3Config,
    pub offset_pos: Vec3Config,
    pub init_hdg: f64,
    pub target_position: Vec3Config,
    pub rnd_offset_range: Vec3Config,
    /// Relative jitter applied to scripted FSM thresholds (`d_shot`, `l_crank`,
    /// `l_break`) at every decision, drawn uniformly from `[-beh_noise, beh_noise]`.
    /// `0.0` (default) keeps scripted agents fully deterministic.
    pub beh_noise: f64,
}

impl TeamConfig {
    pub const MAX_AGENTS: usize = 4;

    pub fn clamp_agents(&mut self) {
        self.num_agents = self.num_agents.clamp(1, Self::MAX_AGENTS);
    }

    /// 2-ship line or 2×2 block at 4 NM if the caller did not set `offset_pos`.
    pub fn apply_box_formation(&mut self) {
        self.clamp_agents();
        if self.num_agents <= 1 {
            return;
        }
        if self.offset_pos.x.abs() < 1e-9 && self.offset_pos.z.abs() < 1e-9 {
            self.offset_pos.x = 4.0;
            if self.num_agents >= 3 {
                self.offset_pos.z = 4.0;
            }
        }
    }
}

impl Default for TeamConfig {
    fn default() -> Self {
        Self {
            num_agents: 1,
            share_tracks: true,
            behavior: Behavior::Baseline1,
            mission: Mission::Dca,
            beh_config: BehConfig::default(),
            init_position: Vec3Config::default(),
            offset_pos: Vec3Config {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            init_hdg: 0.0,
            target_position: Vec3Config {
                x: 0.0,
                y: 25000.0,
                z: -30.0,
            },
            rnd_offset_range: Vec3Config {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            beh_noise: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardsConfig {
    pub mission_factor: f64,
    pub missile_fire_factor: f64,
    pub missile_no_fire_factor: f64,
    pub missile_miss_factor: f64,
    pub detect_loss_factor: f64,
    pub keep_track_factor: f64,
    pub hit_enemy_factor: f64,
    pub hit_own_factor: f64,
    pub mission_accomplished_factor: f64,
}

impl RewardsConfig {
    /// Denser combat signal: drop no-fire spam, shrink detect-loss.
    pub fn rl() -> Self {
        Self {
            mission_factor: 0.001,
            missile_fire_factor: -0.1,
            missile_no_fire_factor: 0.0,
            missile_miss_factor: -0.5,
            detect_loss_factor: -0.01,
            keep_track_factor: 0.001,
            hit_enemy_factor: 3.0,
            hit_own_factor: -5.0,
            mission_accomplished_factor: 10.0,
        }
    }
}

impl Default for RewardsConfig {
    fn default() -> Self {
        Self {
            mission_factor: 0.001,
            missile_fire_factor: -0.1,
            missile_no_fire_factor: -0.001,
            missile_miss_factor: -0.5,
            detect_loss_factor: -0.1,
            keep_track_factor: 0.001,
            hit_enemy_factor: 3.0,
            hit_own_factor: -5.0,
            mission_accomplished_factor: 10.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnvConfig {
    pub phy_fps: u32,
    pub speed_up: u32,
    pub max_cycles: u32,
    pub action_repeat: u32,
    pub action_type: ActionType,
    pub stop_mission: bool,
    pub seed: u64,
    pub rewards: RewardsConfig,
    /// Missile datalink model (see [`MissileSupport`]).
    pub missile_support: MissileSupport,
    /// Missile physics (see [`MissileModel`]).
    pub missile_model: MissileModel,
    /// Aircraft speed model (see [`AircraftModel`]).
    pub aircraft_model: AircraftModel,
    /// Missile warning: aircraft learn the bearing of an enemy missile whose seeker
    /// is active against them (scripted pilots defend on it).
    #[serde(default = "default_true")]
    pub missile_warning: bool,
}

fn default_true() -> bool {
    true
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            phy_fps: 20,
            speed_up: 1,
            max_cycles: 3600,
            action_repeat: 20,
            action_type: ActionType::Continuous,
            stop_mission: true,
            seed: 1,
            rewards: RewardsConfig::default(),
            missile_support: MissileSupport::default(),
            missile_model: MissileModel::default(),
            aircraft_model: AircraftModel::default(),
            missile_warning: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScenarioConfig {
    pub env: EnvConfig,
    pub blue: TeamConfig,
    pub red: TeamConfig,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        let mut red = TeamConfig::default();
        red.init_position.z = -30.0;
        red.init_hdg = 180.0;
        red.target_position.z = 30.0;
        red.behavior = Behavior::Baseline1;
        Self {
            env: EnvConfig::default(),
            blue: TeamConfig {
                behavior: Behavior::External,
                ..TeamConfig::default()
            },
            red,
        }
    }
}

impl ScenarioConfig {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deep-merge overrides from a partial JSON object.
    pub fn merge_json(&mut self, partial: &str) -> Result<(), serde_json::Error> {
        let patch: serde_json::Value = serde_json::from_str(partial)?;
        self.merge_value(&patch)
    }

    /// Deep-merge overrides from a partial JSON value.
    pub fn merge_value(&mut self, patch: &serde_json::Value) -> Result<(), serde_json::Error> {
        let mut base = serde_json::to_value(&*self)?;
        merge_values(&mut base, patch);
        *self = serde_json::from_value(base)?;
        Ok(())
    }

    /// Build a scenario from a partial JSON value merged over [`ScenarioConfig::default`].
    ///
    /// Prefer this over `serde_json::from_value` for user input: plain
    /// deserialization fills a missing `red` field from `TeamConfig::default()`,
    /// which is the *blue* spawn point, so red would start on top of blue.
    pub fn from_partial_value(patch: &serde_json::Value) -> Result<Self, serde_json::Error> {
        let mut c = Self::default();
        c.merge_value(patch)?;
        Ok(c)
    }
}

fn merge_values(base: &mut serde_json::Value, patch: &serde_json::Value) {
    match (base, patch) {
        (serde_json::Value::Object(b), serde_json::Value::Object(p)) => {
            for (k, v) in p {
                let entry = b.entry(k.clone()).or_insert(serde_json::Value::Null);
                merge_values(entry, v);
            }
        }
        (b, p) => *b = p.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roundtrip() {
        let c = ScenarioConfig::default();
        let s = c.to_json_pretty().unwrap();
        let c2 = ScenarioConfig::from_json(&s).unwrap();
        assert_eq!(c2.blue.num_agents, 1);
        assert_eq!(c2.env.phy_fps, 20);
    }

    #[test]
    fn partial_red_keeps_red_side_defaults() {
        let v = serde_json::json!({"red": {"behavior": "duck"}});
        let c = ScenarioConfig::from_partial_value(&v).unwrap();
        assert!(
            (c.red.init_position.z + 30.0).abs() < 1e-9,
            "red must spawn north"
        );
        assert!((c.red.init_hdg - 180.0).abs() < 1e-9);
        assert_eq!(c.red.behavior, Behavior::Duck);
    }

    #[test]
    fn deep_merge_preserves_rewards() {
        let mut c = ScenarioConfig::default();
        c.merge_json(r#"{"env":{"max_cycles":100}}"#).unwrap();
        assert_eq!(c.env.max_cycles, 100);
        assert!((c.env.rewards.hit_enemy_factor - 3.0).abs() < 1e-9);
    }
}
