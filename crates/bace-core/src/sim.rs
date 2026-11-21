//! Episode / simulation manager.

use crate::config::{ScenarioConfig, TeamConfig};
use crate::fighter::{AircraftView, Fighter, FsmState};
use crate::geometry::{distance2d, heading_to};
use crate::missile::{velocity_ms, Missile, TargetState};
use crate::obs::{Action, StructuredObs};
use crate::rewards::RewardBreakdown;
use crate::units::SConv;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type AgentId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Team {
    Blue,
    Red,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndCondition {
    Ongoing,
    BlueKilled,
    RedKilled,
    MutualKill,
    MaxCycles,
    RedMission,
}

/// Discrete things that happened during a step, for logs, replays and analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// `actor` launched missile `missile` at `target`; `value` = range in NM.
    Launch,
    /// Missile `missile` went active (own seeker); `value` = time of flight in s.
    Pitbull,
    /// Missile `missile` from `actor` destroyed `target`; `value` = time of flight in s.
    Hit,
    /// Missile `missile` from `actor` missed `target`; `detail` = cause.
    Miss,
    /// `actor` gained a track on `target` (local radar or datalink).
    TrackGained,
    /// `actor` lost the track on a live `target`.
    TrackLost,
    /// Episode ended; `detail` = end condition.
    End,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimEvent {
    /// Simulated time in seconds.
    pub t: f64,
    pub step: u32,
    pub kind: EventKind,
    pub actor: u32,
    #[serde(default)]
    pub target: Option<u32>,
    #[serde(default)]
    pub missile: Option<u32>,
    #[serde(default)]
    pub value: f64,
    #[serde(default)]
    pub detail: String,
    /// Shooter-to-target range in NM at this moment: the A-pole on `pitbull`,
    /// the F-pole on `hit` / `miss`. 0 when not applicable.
    #[serde(default)]
    pub pole_nm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub obs: StructuredObs,
    pub flat_obs: Vec<f64>,
    pub reward: f64,
    pub reward_breakdown: RewardBreakdown,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub agents: HashMap<AgentId, AgentStep>,
    pub end: EndCondition,
    pub action_step: u32,
    /// Events emitted since the previous step (or reset).
    #[serde(default)]
    pub events: Vec<SimEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackSnapshot {
    pub id: u32,
    pub detected: bool,
    pub dist: f64,
    pub aspect: f64,
    pub own_r_max: f64,
    pub is_missile_support: bool,
    #[serde(default)]
    pub own_r_nez: f64,
    #[serde(default)]
    pub enemy_r_max: f64,
    #[serde(default)]
    pub enemy_r_nez: f64,
    #[serde(default)]
    pub angle_off: f64,
    #[serde(default)]
    pub offensive_factor: f64,
    #[serde(default)]
    pub threat_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FighterSnapshot {
    pub id: u32,
    pub team: u8,
    pub pos: [f64; 3],
    pub hdg: f64,
    pub pitch: f64,
    pub speed: f64,
    pub alive: bool,
    pub missiles: u32,
    pub fsm: FsmState,
    pub agent_name: Option<String>,
    pub hpt_id: Option<u32>,
    pub supporting_missile: bool,
    pub radar_range: f64,
    pub radar_hfov: f64,
    pub radar_vfov_up: f64,
    pub radar_vfov_down: f64,
    pub tracks: Vec<TrackSnapshot>,
    #[serde(default)]
    pub behavior: Option<crate::config::Behavior>,
    #[serde(default)]
    pub mission: Option<crate::config::Mission>,
    #[serde(default)]
    pub target_pos: Option<[f64; 3]>,
    #[serde(default)]
    pub hdg_cmd: f64,
    #[serde(default)]
    pub g_cmd: f64,
    /// Cumulative reward breakdown for this aircraft (computed for scripted agents too).
    #[serde(default)]
    pub reward: RewardBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissileSnapshot {
    pub id: u32,
    pub pos: [f64; 3],
    pub hdg: f64,
    pub team: u8,
    pub pitbull: bool,
    pub target_id: u32,
    pub has_support: bool,
    #[serde(default)]
    pub shooter_id: u32,
    #[serde(default)]
    pub time_alive: f64,
    /// Mach number (generic model; 0 for the legacy missile).
    #[serde(default)]
    pub mach: f64,
    /// `boost`, `sustain`, `coast` or `legacy`.
    #[serde(default)]
    pub phase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimSnapshot {
    pub action_step: u32,
    pub end: EndCondition,
    /// Simulated time in seconds.
    #[serde(default)]
    pub time_s: f64,
    pub fighters: Vec<FighterSnapshot>,
    pub missiles: Vec<MissileSnapshot>,
}

/// Compact end-of-episode metrics for experiment mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeOutcome {
    pub config: ScenarioConfig,
    pub end: EndCondition,
    pub steps: u32,
    pub seed: u64,
    pub blue_alive: usize,
    pub red_alive: usize,
    pub blue_kills: usize,
    pub blue_deaths: usize,
    pub mission_success: bool,
    pub episode_return: f64,
    pub missiles_fired: u32,
    pub missile_hits: u32,
    pub missile_tof: f64,
    pub missile_pitbull: bool,
    pub missile_pitbull_time: f64,
    pub miss_cause: String,
    pub fsm_search: u32,
    pub fsm_engage: u32,
    pub fsm_support: u32,
    pub fsm_evade: u32,
    pub mean_ally_spacing_nm: f64,
    pub mean_fire_range_nm: f64,
    pub tracks_on_frac: f64,
}

pub struct Simulation {
    pub config: ScenarioConfig,
    pub fighters: Vec<Fighter>,
    pub missiles: Vec<Missile>,
    pub blue_agent_ids: Vec<AgentId>,
    pub red_agent_ids: Vec<AgentId>,
    pub action_step: u32,
    pub physics_step: u32,
    pub end: EndCondition,
    pub next_missile_id: u32,
    pub missiles_fired: u32,
    pub missile_hits: u32,
    rng: StdRng,
    last_missile_tof: f64,
    last_missile_pitbull: bool,
    last_missile_pitbull_time: f64,
    last_miss_cause: String,
    fsm_search: u32,
    fsm_engage: u32,
    fsm_support: u32,
    fsm_evade: u32,
    ally_spacing_sum: f64,
    ally_spacing_n: u32,
    fire_range_sum: f64,
    fire_range_n: u32,
    tracks_on_sum: f64,
    tracks_on_n: u32,
    events: Vec<SimEvent>,
}

impl Simulation {
    pub fn new(config: ScenarioConfig) -> Self {
        let mut config = config;
        config.blue.apply_box_formation();
        config.red.apply_box_formation();
        let mut sim = Self {
            rng: StdRng::seed_from_u64(config.env.seed),
            config,
            fighters: Vec::new(),
            missiles: Vec::new(),
            blue_agent_ids: Vec::new(),
            red_agent_ids: Vec::new(),
            action_step: 0,
            physics_step: 0,
            end: EndCondition::Ongoing,
            next_missile_id: 1,
            missiles_fired: 0,
            missile_hits: 0,
            last_missile_tof: 0.0,
            last_missile_pitbull: false,
            last_missile_pitbull_time: 0.0,
            last_miss_cause: String::new(),
            fsm_search: 0,
            fsm_engage: 0,
            fsm_support: 0,
            fsm_evade: 0,
            ally_spacing_sum: 0.0,
            ally_spacing_n: 0,
            fire_range_sum: 0.0,
            fire_range_n: 0,
            tracks_on_sum: 0.0,
            tracks_on_n: 0,
            events: Vec::new(),
        };
        sim.spawn_teams();
        sim
    }

    fn formation_offset_nm(team: &TeamConfig, idx: usize) -> (f64, f64) {
        let n = team.num_agents.clamp(1, TeamConfig::MAX_AGENTS);
        let sx = team.offset_pos.x;
        let sz = team.offset_pos.z;
        if n <= 2 {
            return (idx as f64 * sx, 0.0);
        }
        let col = (idx % 2) as f64;
        let row = (idx / 2) as f64;
        (col * sx, row * if sz.abs() > 1e-9 { sz } else { sx })
    }

    fn pos_from_cfg(team: &TeamConfig, idx: usize, rng: &mut StdRng) -> ([f64; 3], [f64; 3], f64) {
        let (dx, dz) = Self::formation_offset_nm(team, idx);
        let base = [
            team.init_position.x * SConv::NM2GDM
                + dx * SConv::NM2GDM
                + rng.gen_range(-1.0..1.0) * team.rnd_offset_range.x * SConv::NM2GDM,
            team.init_position.y * SConv::FT2GDM
                + idx as f64 * team.offset_pos.y * SConv::FT2GDM
                + rng.gen_range(-1.0..1.0) * team.rnd_offset_range.y * SConv::FT2GDM,
            team.init_position.z * SConv::NM2GDM
                + dz * SConv::NM2GDM
                + rng.gen_range(-1.0..1.0) * team.rnd_offset_range.z * SConv::NM2GDM,
        ];
        let target = [
            team.target_position.x * SConv::NM2GDM,
            team.target_position.y * SConv::FT2GDM,
            team.target_position.z * SConv::NM2GDM,
        ];
        (base, target, team.init_hdg)
    }

    /// Per-aircraft FSM noise stream derived from the episode seed without
    /// consuming the spawn RNG (keeps pre-existing spawn sequences unchanged).
    fn noise_seed(seed: u64, id: u32) -> u64 {
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    }

    fn spawn_teams(&mut self) {
        self.fighters.clear();
        self.missiles.clear();
        self.blue_agent_ids.clear();
        self.red_agent_ids.clear();
        self.next_missile_id = 1;

        let n_blue = self.config.blue.num_agents;
        let n_red = self.config.red.num_agents;
        let blue_ids: Vec<u32> = (0..n_blue).map(|i| 101 + i as u32).collect();
        let red_ids: Vec<u32> = (0..n_red).map(|i| 201 + i as u32).collect();

        for (i, &id) in blue_ids.iter().enumerate() {
            let (pos, target, hdg) = Self::pos_from_cfg(&self.config.blue, i, &mut self.rng);
            let allies: Vec<u32> = blue_ids.iter().copied().filter(|x| *x != id).collect();
            let f = Fighter::new(
                id,
                0,
                pos,
                hdg,
                target,
                self.config.blue.behavior,
                self.config.blue.mission,
                self.config.blue.beh_config.clone(),
                self.config.blue.share_tracks,
                self.config.env.rewards.clone(),
                &red_ids,
                allies,
            )
            .with_noise(
                self.config.blue.beh_noise,
                Self::noise_seed(self.config.env.seed, id),
            )
            .with_missile_model(self.config.env.missile_model)
            .with_aircraft_model(self.config.env.aircraft_model);
            let agent_name = format!("agent_{i}");
            self.blue_agent_ids.push(agent_name);
            self.fighters.push(f);
        }

        for (i, &id) in red_ids.iter().enumerate() {
            let (pos, target, hdg) = Self::pos_from_cfg(&self.config.red, i, &mut self.rng);
            let allies: Vec<u32> = red_ids.iter().copied().filter(|x| *x != id).collect();
            let f = Fighter::new(
                id,
                1,
                pos,
                hdg,
                target,
                self.config.red.behavior,
                self.config.red.mission,
                self.config.red.beh_config.clone(),
                self.config.red.share_tracks,
                self.config.env.rewards.clone(),
                &blue_ids,
                allies,
            )
            .with_noise(
                self.config.red.beh_noise,
                Self::noise_seed(self.config.env.seed, id),
            )
            .with_missile_model(self.config.env.missile_model)
            .with_aircraft_model(self.config.env.aircraft_model);
            self.red_agent_ids.push(format!("red_{i}"));
            self.fighters.push(f);
        }
    }

    /// Current kinematic state of every aircraft, as seen by missiles.
    fn target_states(&self) -> HashMap<u32, TargetState> {
        self.fighters
            .iter()
            .map(|f| {
                (
                    f.id,
                    TargetState {
                        pos: f.pos,
                        vel: if f.alive {
                            velocity_ms(f.hdg, f.pitch, f.speed)
                        } else {
                            [0.0; 3]
                        },
                        alive: f.alive,
                    },
                )
            })
            .collect()
    }

    /// Simulated time in seconds.
    pub fn time_s(&self) -> f64 {
        self.physics_step as f64 / self.config.env.phy_fps.max(1) as f64
    }

    fn emit(
        &mut self,
        kind: EventKind,
        actor: u32,
        target: Option<u32>,
        missile: Option<u32>,
        value: f64,
        detail: &str,
    ) {
        let pole_nm = match (kind, target) {
            (EventKind::Pitbull | EventKind::Hit | EventKind::Miss, Some(tid)) => {
                let a = self.fighters.iter().find(|f| f.id == actor && f.alive);
                let b = self.fighters.iter().find(|f| f.id == tid);
                match (a, b) {
                    (Some(a), Some(b)) => distance2d(a.pos, b.pos) * SConv::GDM2NM,
                    _ => 0.0,
                }
            }
            _ => 0.0,
        };
        self.events.push(SimEvent {
            t: self.time_s(),
            step: self.action_step,
            kind,
            actor,
            target,
            missile,
            value,
            detail: detail.to_string(),
            pole_nm,
        });
    }

    pub fn red_is_external(&self) -> bool {
        self.config.red.behavior == crate::config::Behavior::External
    }

    pub fn agent_ids(&self) -> Vec<AgentId> {
        let mut ids = self.blue_agent_ids.clone();
        if self.red_is_external() {
            ids.extend(self.red_agent_ids.iter().cloned());
        }
        ids
    }

    pub fn reset(&mut self, seed: Option<u64>) -> StepResult {
        if let Some(s) = seed {
            self.config.env.seed = s;
            self.rng = StdRng::seed_from_u64(s);
        }
        self.action_step = 0;
        self.physics_step = 0;
        self.end = EndCondition::Ongoing;
        self.missiles_fired = 0;
        self.missile_hits = 0;
        self.last_missile_tof = 0.0;
        self.last_missile_pitbull = false;
        self.last_missile_pitbull_time = 0.0;
        self.last_miss_cause = String::new();
        self.fsm_search = 0;
        self.fsm_engage = 0;
        self.fsm_support = 0;
        self.fsm_evade = 0;
        self.ally_spacing_sum = 0.0;
        self.ally_spacing_n = 0;
        self.fire_range_sum = 0.0;
        self.fire_range_n = 0;
        self.tracks_on_sum = 0.0;
        self.tracks_on_n = 0;
        self.events.clear();
        self.spawn_teams();
        self.sense_and_observe(true)
    }

    fn views(&self) -> Vec<AircraftView> {
        self.fighters
            .iter()
            .map(|f| AircraftView {
                id: f.id,
                pos: f.pos,
                hdg: f.hdg,
                alive: f.alive,
                dist2go: f.dist2go,
            })
            .collect()
    }

    fn team_dl(&self, team: u8) -> Vec<u32> {
        let mut ids = Vec::new();
        for f in self.fighters.iter().filter(|f| f.team == team && f.alive) {
            for t in &f.enemy_tracks {
                if t.detected && !ids.contains(&t.id) {
                    ids.push(t.id);
                }
            }
        }
        ids
    }

    fn sense_all(&mut self) {
        let before: Vec<(u32, Vec<(u32, bool)>)> = self
            .fighters
            .iter()
            .map(|f| {
                (
                    f.id,
                    f.enemy_tracks.iter().map(|t| (t.id, t.detected)).collect(),
                )
            })
            .collect();
        self.sense_all_inner();
        self.update_missile_support();
        let alive: HashMap<u32, bool> = self.fighters.iter().map(|f| (f.id, f.alive)).collect();
        let mut evs = Vec::new();
        for (f, (fid, prev)) in self.fighters.iter().zip(before.iter()) {
            debug_assert_eq!(f.id, *fid);
            if !f.alive {
                continue;
            }
            for (t, (tid, was)) in f.enemy_tracks.iter().zip(prev.iter()) {
                debug_assert_eq!(t.id, *tid);
                if t.detected && !*was {
                    evs.push((EventKind::TrackGained, f.id, t.id));
                } else if !t.detected && *was && *alive.get(&t.id).unwrap_or(&false) {
                    evs.push((EventKind::TrackLost, f.id, t.id));
                }
            }
        }
        for (k, a, t) in evs {
            self.emit(k, a, Some(t), None, 0.0, "");
        }
    }

    /// Datalink guidance follows the shooter's track (B-ACE 1.0): a missile keeps
    /// support only while its shooter is alive, still owns it, and tracks the target
    /// (own radar or shared datalink). Support is regained when the track returns.
    fn update_missile_support(&mut self) {
        if self.config.env.missile_support != crate::config::MissileSupport::Track {
            return;
        }
        for m in self.missiles.iter_mut().filter(|m| m.alive && !m.pitbull) {
            let supported = self.fighters.iter().any(|f| {
                f.id == m.shooter_id
                    && f.alive
                    && f.in_flight_missile_id == Some(m.id)
                    && f.enemy_tracks
                        .iter()
                        .any(|t| t.id == m.target_id && t.detected)
            });
            if supported {
                m.recover_support();
            } else {
                m.lost_support();
            }
        }
    }

    /// Radar-warning cue: each aircraft learns the bearing of the closest enemy
    /// missile whose seeker is active against it.
    fn update_missile_warnings(&mut self) {
        for f in self.fighters.iter_mut() {
            f.missile_warning = self
                .missiles
                .iter()
                .filter(|m| m.alive && m.pitbull && m.target_id == f.id && m.team != f.team)
                .min_by(|a, b| {
                    distance2d(a.pos, f.pos)
                        .partial_cmp(&distance2d(b.pos, f.pos))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|m| heading_to(f.pos, m.pos));
        }
    }

    fn sense_all_inner(&mut self) {
        let views = self.views();
        // First pass local detection without DL, then with DL from previous detections.
        // Approximate: compute local, build DL, update again.
        let blue_ids: Vec<u32> = self
            .fighters
            .iter()
            .filter(|f| f.team == 0)
            .map(|f| f.id)
            .collect();
        let red_ids: Vec<u32> = self
            .fighters
            .iter()
            .filter(|f| f.team == 1)
            .map(|f| f.id)
            .collect();

        for f in &mut self.fighters {
            let enemies: Vec<AircraftView> = views
                .iter()
                .copied()
                .filter(|v| {
                    if f.team == 0 {
                        red_ids.contains(&v.id)
                    } else {
                        blue_ids.contains(&v.id)
                    }
                })
                .collect();
            f.update_tracks(&enemies, &[]);
        }
        let blue_dl = self.team_dl(0);
        let red_dl = self.team_dl(1);
        for f in &mut self.fighters {
            let enemies: Vec<AircraftView> = views
                .iter()
                .copied()
                .filter(|v| {
                    if f.team == 0 {
                        red_ids.contains(&v.id)
                    } else {
                        blue_ids.contains(&v.id)
                    }
                })
                .collect();
            let dl = if f.team == 0 { &blue_dl } else { &red_dl };
            f.update_tracks(&enemies, dl);
        }
    }

    fn try_fire(
        fighter: &mut Fighter,
        missiles: &mut Vec<Missile>,
        next_id: &mut u32,
        model: crate::missile::MissileModel,
        targets: &HashMap<u32, TargetState>,
    ) -> Option<f64> {
        if !fighter.alive || !fighter.fire_cmd {
            return None;
        }
        if fighter.missiles == 0 {
            fighter.rewards.add_missile_no_fire();
            return None;
        }
        let Some(hpt) = fighter.hpt_id else {
            fighter.rewards.add_missile_no_fire();
            return None;
        };
        let can_fire = fighter
            .enemy_tracks
            .iter()
            .find(|t| t.id == hpt)
            .map(|t| t.detected && t.aspect.abs() < 30.0 && !t.is_missile_support)
            .unwrap_or(false);
        if !can_fire {
            fighter.rewards.add_missile_no_fire();
            return None;
        }
        let fire_range_nm = fighter
            .enemy_tracks
            .iter()
            .find(|t| t.id == hpt)
            .map(|t| t.dist * SConv::GDM2NM)
            .unwrap_or(0.0);
        if let Some(mid) = fighter.in_flight_missile_id {
            if let Some(m) = missiles.iter_mut().find(|m| m.id == mid && m.alive) {
                m.lost_support();
            }
            for t in &mut fighter.enemy_tracks {
                t.is_missile_support = false;
            }
        }
        let target = targets.get(&hpt).copied().unwrap_or(TargetState {
            pos: fighter.pos,
            vel: [0.0; 3],
            alive: false,
        });
        let m = Missile::launch_model(
            model,
            *next_id,
            fighter.id,
            hpt,
            fighter.team,
            fighter.pos,
            fighter.hdg,
            velocity_ms(fighter.hdg, fighter.pitch, fighter.speed),
            &target,
        );
        *next_id += 1;
        fighter.missiles -= 1;
        fighter.in_flight_missile_id = Some(m.id);
        fighter.supporting_missile = true;
        if let Some(track) = fighter.enemy_tracks.iter_mut().find(|t| t.id == hpt) {
            track.is_missile_support = true;
        }
        fighter.rewards.add_missile_fire();
        missiles.push(m);
        fighter.fire_cmd = false;
        Some(fire_range_nm)
    }

    fn integrate_physics(&mut self) {
        let dt = 1.0 / self.config.env.phy_fps as f64;
        let repeat = self.config.env.action_repeat.max(1);

        for _ in 0..repeat {
            if self.end != EndCondition::Ongoing {
                break;
            }

            // Missiles
            let targets = self.target_states();

            let mut hits: Vec<(u32, u32, u8)> = Vec::new(); // target_id, shooter_id, team
            let mut misses: Vec<u32> = Vec::new();
            let mut mevents: Vec<(EventKind, u32, u32, u32, f64, String)> = Vec::new();

            for m in &mut self.missiles {
                if !m.alive {
                    continue;
                }
                let tstate = targets.get(&m.target_id).copied().unwrap_or(TargetState {
                    pos: [0.0; 3],
                    vel: [0.0; 3],
                    alive: false,
                });
                let was_pitbull = m.pitbull;
                let hit = m.tick(dt, &tstate);
                if m.pitbull && !was_pitbull {
                    mevents.push((
                        EventKind::Pitbull,
                        m.shooter_id,
                        m.target_id,
                        m.id,
                        m.time_alive,
                        String::new(),
                    ));
                }
                if hit {
                    mevents.push((
                        EventKind::Hit,
                        m.shooter_id,
                        m.target_id,
                        m.id,
                        m.time_alive,
                        String::new(),
                    ));
                } else if !m.alive {
                    mevents.push((
                        EventKind::Miss,
                        m.shooter_id,
                        m.target_id,
                        m.id,
                        m.time_alive,
                        m.miss_cause.clone().unwrap_or_default(),
                    ));
                }
                if hit {
                    self.last_missile_tof = m.time_alive;
                    self.last_missile_pitbull = m.pitbull;
                    self.last_missile_pitbull_time = m.pitbull_time.unwrap_or(0.0);
                    self.last_miss_cause.clear();
                    hits.push((m.target_id, m.shooter_id, m.team));
                } else if !m.alive {
                    self.last_missile_tof = m.time_alive;
                    self.last_missile_pitbull = m.pitbull;
                    self.last_missile_pitbull_time = m.pitbull_time.unwrap_or(0.0);
                    self.last_miss_cause = m.miss_cause.clone().unwrap_or_default();
                    misses.push(m.shooter_id);
                }
            }

            for (target_id, shooter_id, team) in hits {
                if team == 0 {
                    self.missile_hits += 1;
                }
                if let Some(victim) = self.fighters.iter_mut().find(|f| f.id == target_id) {
                    victim.alive = false;
                    victim.done = true;
                    victim.rewards.add_hit_own();
                }
                if let Some(shooter) = self.fighters.iter_mut().find(|f| f.id == shooter_id) {
                    shooter.rewards.add_hit_enemy();
                    shooter.supporting_missile = false;
                    shooter.in_flight_missile_id = None;
                    for t in &mut shooter.enemy_tracks {
                        t.is_missile_support = false;
                    }
                }
            }
            for shooter_id in misses {
                if let Some(shooter) = self.fighters.iter_mut().find(|f| f.id == shooter_id) {
                    shooter.rewards.add_missile_miss();
                    shooter.supporting_missile = false;
                    shooter.in_flight_missile_id = None;
                    for t in &mut shooter.enemy_tracks {
                        t.is_missile_support = false;
                    }
                }
            }
            self.missiles.retain(|m| m.alive);

            // Sense periodically every action boundary handled outside; light update each tick optional
            for f in &mut self.fighters {
                f.physics_tick(dt);
            }

            self.physics_step += 1;
            // Missile events are stamped with the end time of the tick that produced them.
            for (k, a, t, mid, v, d) in mevents {
                self.emit(k, a, Some(t), Some(mid), v, &d);
            }
            self.check_terminals();
        }
        self.action_step += 1;
    }

    fn accumulate_behavior_stats(&mut self) {
        for f in self.fighters.iter().filter(|f| f.team == 0 && f.alive) {
            match f.fsm {
                FsmState::Search => self.fsm_search += 1,
                FsmState::Engage => self.fsm_engage += 1,
                FsmState::MissileSupport => self.fsm_support += 1,
                FsmState::Evade => self.fsm_evade += 1,
            }
            let tot = f.enemy_tracks.len();
            if tot > 0 {
                let det = f.enemy_tracks.iter().filter(|t| t.detected).count();
                self.tracks_on_sum += det as f64 / tot as f64;
                self.tracks_on_n += 1;
            }
        }
        let blues: Vec<[f64; 3]> = self
            .fighters
            .iter()
            .filter(|f| f.team == 0 && f.alive)
            .map(|f| f.pos)
            .collect();
        if blues.len() >= 2 {
            let mut s = 0.0;
            let mut n = 0u32;
            for i in 0..blues.len() {
                for j in (i + 1)..blues.len() {
                    s += distance2d(blues[i], blues[j]) * SConv::GDM2NM;
                    n += 1;
                }
            }
            if n > 0 {
                self.ally_spacing_sum += s / n as f64;
                self.ally_spacing_n += 1;
            }
        }
    }

    fn check_terminals(&mut self) {
        if self.end != EndCondition::Ongoing {
            return;
        }
        let blue_alive = self
            .fighters
            .iter()
            .filter(|f| f.team == 0 && f.alive)
            .count();
        let red_alive = self
            .fighters
            .iter()
            .filter(|f| f.team == 1 && f.alive)
            .count();
        let missiles_left = !self.missiles.is_empty();

        // Both sides dead is not a blue loss. K/D still come from alive counts.
        // mission_success stays false, matching the old BlueKilled label.
        if blue_alive == 0 && red_alive == 0 && !missiles_left {
            self.end = EndCondition::MutualKill;
            for f in self.fighters.iter_mut() {
                f.rewards.add_terminal("Team_Killed", f.missiles);
                f.done = true;
            }
            return;
        }

        if blue_alive == 0 && !missiles_left {
            self.end = EndCondition::BlueKilled;
            for f in self.fighters.iter_mut().filter(|f| f.team == 0) {
                f.rewards.add_terminal("Team_Killed", f.missiles);
                f.done = true;
            }
            for f in self.fighters.iter_mut().filter(|f| f.team == 1) {
                f.done = true;
            }
            return;
        }
        if red_alive == 0 && !missiles_left {
            self.end = EndCondition::RedKilled;
            for f in self.fighters.iter_mut().filter(|f| f.team == 0) {
                f.rewards.add_terminal("Enemies_Killed", f.missiles);
                f.done = true;
            }
            for f in self.fighters.iter_mut().filter(|f| f.team == 1) {
                f.done = true;
            }
            return;
        }

        if self.config.env.stop_mission {
            for f in &self.fighters {
                if f.team == 1
                    && f.alive
                    && f.mission == crate::config::Mission::Striker
                    && f.dist2go < 5.0
                {
                    self.end = EndCondition::RedMission;
                    break;
                }
            }
            if self.end == EndCondition::RedMission {
                for f in self.fighters.iter_mut() {
                    if f.team == 0 {
                        f.rewards.add_terminal("Enemy_Achieved_Target", f.missiles);
                    }
                    f.done = true;
                }
                return;
            }
        }

        if self.action_step + 1 >= self.config.env.max_cycles {
            self.end = EndCondition::MaxCycles;
            for f in self.fighters.iter_mut() {
                f.rewards.add_terminal("Max_Cycles", f.missiles);
                f.done = true;
            }
        }
    }

    fn mission_shape_learners(&mut self) {
        let red_ext = self.red_is_external();
        for f in self.fighters.iter_mut().filter(|f| f.alive) {
            if f.team == 0 || (f.team == 1 && red_ext) {
                let shaped = 1.0 + (-0.99) / (1.0 + (-0.02 * (f.dist2go - 370.4)).exp());
                f.rewards.add_mission(shaped);
            }
        }
    }

    fn collect_learner_steps(&mut self, take_reward: bool) -> HashMap<AgentId, AgentStep> {
        let views = self.views();
        let end = self.end;
        let mut names: Vec<(AgentId, u8, usize)> = self
            .blue_agent_ids
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), 0u8, i))
            .collect();
        if self.red_is_external() {
            names.extend(
                self.red_agent_ids
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), 1u8, i)),
            );
        }
        let mut pending = Vec::new();
        for (name, team, idx) in names {
            let f = self
                .fighters
                .iter()
                .filter(|f| f.team == team)
                .nth(idx)
                .expect("learner index");
            let allies: Vec<AircraftView> = views
                .iter()
                .copied()
                .filter(|v| f.ally_ids.contains(&v.id))
                .collect();
            let enemies: Vec<AircraftView> = views
                .iter()
                .copied()
                .filter(|v| f.enemy_tracks.iter().any(|t| t.id == v.id))
                .collect();
            let mut obs = f.build_obs(&allies);
            // Enemy distance-to-goal is only observable for detected tracks;
            // undetected enemies keep the neutral placeholder from `build_obs`.
            for e in &mut obs.enemies {
                if e.detected < 0.5 {
                    continue;
                }
                if let Some(ev) = enemies.iter().find(|x| x.id == e.id) {
                    e.dist_target = ev.dist2go / 3000.0;
                }
            }
            let flat = obs.to_flat();
            let terminated = f.done && end != EndCondition::MaxCycles;
            let truncated = end == EndCondition::MaxCycles;
            pending.push((name, team, idx, obs, flat, terminated, truncated));
        }
        let mut agents = HashMap::new();
        for (name, team, idx, obs, flat, terminated, truncated) in pending {
            let (reward, breakdown) = if take_reward {
                let fmut = self
                    .fighters
                    .iter_mut()
                    .filter(|f| f.team == team)
                    .nth(idx)
                    .expect("learner index");
                fmut.rewards.take_step()
            } else {
                (0.0, RewardBreakdown::default())
            };
            agents.insert(
                name,
                AgentStep {
                    obs,
                    flat_obs: flat,
                    reward,
                    reward_breakdown: breakdown,
                    terminated,
                    truncated,
                },
            );
        }
        agents
    }

    fn sense_and_observe(&mut self, _is_reset: bool) -> StepResult {
        self.sense_all();
        self.mission_shape_learners();
        let agents = self.collect_learner_steps(false);
        StepResult {
            agents,
            end: self.end,
            action_step: self.action_step,
            events: std::mem::take(&mut self.events),
        }
    }

    /// Apply blue external actions (by agent name), run FSM for others, step physics.
    pub fn step(&mut self, actions: &HashMap<AgentId, Action>) -> StepResult {
        if self.end != EndCondition::Ongoing {
            return self.sense_and_observe(false);
        }

        self.sense_all();
        self.update_missile_warnings();

        // Apply actions / behaviors
        let blue_ids = self.blue_agent_ids.clone();
        let red_ids = self.red_agent_ids.clone();
        let red_ext = self.red_is_external();
        for (i, name) in blue_ids.iter().enumerate() {
            if let Some(f) = self.fighters.iter_mut().filter(|f| f.team == 0).nth(i) {
                if f.behavior == crate::config::Behavior::External {
                    if let Some(a) = actions.get(name) {
                        f.apply_action(*a);
                    }
                } else {
                    f.run_behavior();
                }
            }
        }
        for (i, f) in self.fighters.iter_mut().filter(|f| f.team == 1).enumerate() {
            if red_ext {
                if let Some(a) = red_ids.get(i).and_then(|n| actions.get(n)) {
                    f.apply_action(*a);
                }
            } else {
                f.run_behavior();
            }
        }

        self.accumulate_behavior_stats();

        // Firing (needs mutable fighters + missiles)
        let mut next_id = self.next_missile_id;
        let mut missiles = std::mem::take(&mut self.missiles);
        let mut launches: Vec<(u32, u32, u32, f64)> = Vec::new();
        let model = self.config.env.missile_model;
        let targets = self.target_states();
        for f in &mut self.fighters {
            let before = f.missiles;
            if let Some(rng_nm) = Self::try_fire(f, &mut missiles, &mut next_id, model, &targets) {
                if let Some(m) = missiles.last() {
                    launches.push((f.id, m.target_id, m.id, rng_nm));
                }
                if f.team == 0 {
                    self.fire_range_sum += rng_nm;
                    self.fire_range_n += 1;
                }
            }
            if f.team == 0 && f.missiles < before {
                self.missiles_fired += before - f.missiles;
            }
        }
        self.missiles = missiles;
        self.next_missile_id = next_id;
        let track_mode = self.config.env.missile_support == crate::config::MissileSupport::Track;
        for m in self.missiles.iter_mut() {
            m.needs_support = track_mode;
        }
        for (shooter, target, mid, rng_nm) in launches {
            self.emit(
                EventKind::Launch,
                shooter,
                Some(target),
                Some(mid),
                rng_nm,
                "",
            );
        }

        self.integrate_physics();

        // Collect rewards into step result
        self.sense_all();
        self.mission_shape_learners();
        let agents = self.collect_learner_steps(true);
        if self.end != EndCondition::Ongoing {
            let detail = format!("{:?}", self.end);
            self.emit(EventKind::End, 0, None, None, 0.0, &detail);
        }
        StepResult {
            agents,
            end: self.end,
            action_step: self.action_step,
            events: std::mem::take(&mut self.events),
        }
    }

    pub fn snapshot(&self) -> SimSnapshot {
        SimSnapshot {
            action_step: self.action_step,
            end: self.end,
            time_s: self.time_s(),
            fighters: {
                let mut blue_i = 0usize;
                let mut red_i = 0usize;
                let red_ext = self.red_is_external();
                self.fighters
                    .iter()
                    .map(|f| {
                        let agent_name = if f.team == 0 {
                            let name = self.blue_agent_ids.get(blue_i).cloned();
                            blue_i += 1;
                            name
                        } else if red_ext {
                            let name = self.red_agent_ids.get(red_i).cloned();
                            red_i += 1;
                            name
                        } else {
                            None
                        };
                        FighterSnapshot {
                            id: f.id,
                            team: f.team,
                            pos: f.pos,
                            hdg: f.hdg,
                            pitch: f.pitch,
                            speed: f.speed,
                            alive: f.alive,
                            missiles: f.missiles,
                            fsm: f.fsm,
                            agent_name,
                            hpt_id: f.hpt_id,
                            supporting_missile: f.supporting_missile,
                            radar_range: f.radar_range,
                            radar_hfov: f.radar_hfov,
                            radar_vfov_up: f.radar_vfov_up,
                            radar_vfov_down: f.radar_vfov_down,
                            tracks: f
                                .enemy_tracks
                                .iter()
                                .map(|t| TrackSnapshot {
                                    id: t.id,
                                    detected: t.detected,
                                    dist: t.dist,
                                    aspect: t.aspect,
                                    own_r_max: t.own_wez.r_max,
                                    is_missile_support: t.is_missile_support,
                                    own_r_nez: t.own_wez.r_nez,
                                    enemy_r_max: t.enemy_wez.r_max,
                                    enemy_r_nez: t.enemy_wez.r_nez,
                                    angle_off: t.angle_off,
                                    offensive_factor: t.offensive_factor,
                                    threat_factor: t.threat_factor,
                                })
                                .collect(),
                            behavior: Some(f.behavior),
                            mission: Some(f.mission),
                            target_pos: Some(f.target_pos),
                            hdg_cmd: f.hdg_cmd,
                            g_cmd: f.g_cmd,
                            reward: f.rewards.cumulative().clone(),
                        }
                    })
                    .collect()
            },
            missiles: self
                .missiles
                .iter()
                .map(|m| MissileSnapshot {
                    id: m.id,
                    pos: m.pos,
                    hdg: m.hdg,
                    team: m.team,
                    pitbull: m.pitbull,
                    target_id: m.target_id,
                    has_support: m.has_support,
                    shooter_id: m.shooter_id,
                    time_alive: m.time_alive,
                    mach: m.mach,
                    phase: m.phase().to_string(),
                })
                .collect(),
        }
    }

    pub fn obs_size(&self) -> usize {
        let n_blue = self.config.blue.num_agents;
        let n_red = self.config.red.num_agents;
        StructuredObs::flat_size(n_blue.saturating_sub(1), n_red)
    }

    /// Snapshot experiment metrics for the current (usually finished) episode.
    pub fn outcome(&self) -> EpisodeOutcome {
        let n_blue = self.config.blue.num_agents;
        let n_red = self.config.red.num_agents;
        let blue_alive = self
            .fighters
            .iter()
            .filter(|f| f.team == 0 && f.alive)
            .count();
        let red_alive = self
            .fighters
            .iter()
            .filter(|f| f.team == 1 && f.alive)
            .count();
        let episode_return: f64 = self
            .fighters
            .iter()
            .filter(|f| f.team == 0)
            .map(|f| f.rewards.cumulative().total())
            .sum();
        let mission_success = !matches!(
            self.end,
            EndCondition::RedMission | EndCondition::BlueKilled | EndCondition::MutualKill
        );
        EpisodeOutcome {
            config: self.config.clone(),
            end: self.end,
            steps: self.action_step,
            seed: self.config.env.seed,
            blue_alive,
            red_alive,
            blue_kills: n_red.saturating_sub(red_alive),
            blue_deaths: n_blue.saturating_sub(blue_alive),
            mission_success,
            episode_return,
            missiles_fired: self.missiles_fired,
            missile_hits: self.missile_hits,
            missile_tof: self.last_missile_tof,
            missile_pitbull: self.last_missile_pitbull,
            missile_pitbull_time: self.last_missile_pitbull_time,
            miss_cause: self.last_miss_cause.clone(),
            fsm_search: self.fsm_search,
            fsm_engage: self.fsm_engage,
            fsm_support: self.fsm_support,
            fsm_evade: self.fsm_evade,
            mean_ally_spacing_nm: if self.ally_spacing_n > 0 {
                self.ally_spacing_sum / self.ally_spacing_n as f64
            } else {
                0.0
            },
            mean_fire_range_nm: if self.fire_range_n > 0 {
                self.fire_range_sum / self.fire_range_n as f64
            } else {
                0.0
            },
            tracks_on_frac: if self.tracks_on_n > 0 {
                self.tracks_on_sum / self.tracks_on_n as f64
            } else {
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Behavior, ScenarioConfig};
    use crate::obs::Action;

    #[test]
    fn reset_and_step_runs() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 50;
        cfg.blue.behavior = Behavior::Duck;
        cfg.red.behavior = Behavior::Duck;
        let mut sim = Simulation::new(cfg);
        let r = sim.reset(Some(42));
        assert_eq!(r.agents.len(), 1);
        assert_eq!(r.end, EndCondition::Ongoing);
        let actions = HashMap::new();
        for _ in 0..10 {
            let s = sim.step(&actions);
            assert!(!s.agents["agent_0"].flat_obs.is_empty());
        }
    }

    #[test]
    fn determinism_same_seed() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 30;
        cfg.blue.behavior = Behavior::Baseline1;
        cfg.red.behavior = Behavior::Duck;
        let mut a = Simulation::new(cfg.clone());
        let mut b = Simulation::new(cfg);
        a.reset(Some(7));
        b.reset(Some(7));
        let empty = HashMap::new();
        for _ in 0..20 {
            a.step(&empty);
            b.step(&empty);
        }
        let sa = a.snapshot();
        let sb = b.snapshot();
        for (fa, fb) in sa.fighters.iter().zip(sb.fighters.iter()) {
            assert!((fa.pos[0] - fb.pos[0]).abs() < 1e-9);
            assert!((fa.pos[2] - fb.pos[2]).abs() < 1e-9);
        }
    }

    #[test]
    fn fire_once_launches() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 40;
        cfg.env.action_repeat = 10;
        cfg.blue.behavior = Behavior::FireOnce;
        cfg.blue.init_position.z = 8.0;
        cfg.blue.target_position.z = -8.0;
        cfg.red.behavior = Behavior::Duck;
        cfg.red.init_position.z = -8.0;
        cfg.red.init_hdg = 180.0;
        let mut sim = Simulation::new(cfg);
        sim.reset(Some(1));
        let empty = HashMap::new();
        for _ in 0..15 {
            sim.step(&empty);
            if sim.missiles_fired > 0 {
                break;
            }
        }
        assert!(
            sim.missiles_fired > 0,
            "FireOnce should launch when a track is detected"
        );
        let out = sim.outcome();
        assert_eq!(out.missiles_fired, sim.missiles_fired);
    }

    #[test]
    fn four_v_four_box_and_terminates() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 30;
        cfg.env.action_repeat = 10;
        cfg.blue.num_agents = 4;
        cfg.blue.behavior = Behavior::Duck;
        cfg.red.num_agents = 4;
        cfg.red.behavior = Behavior::Duck;
        let mut sim = Simulation::new(cfg);
        assert_eq!(sim.fighters.len(), 8);
        let blues: Vec<_> = sim.fighters.iter().filter(|f| f.team == 0).collect();
        let dx = (blues[1].pos[0] - blues[0].pos[0]).abs() * SConv::GDM2NM;
        assert!(
            (dx - 4.0).abs() < 0.2,
            "2×2 box should space ~4 NM, got {dx}"
        );
        sim.reset(Some(3));
        let empty = HashMap::new();
        let mut ended = false;
        for _ in 0..40 {
            let s = sim.step(&empty);
            if s.end != EndCondition::Ongoing {
                ended = true;
                break;
            }
        }
        assert!(ended || sim.action_step >= 30);
        assert_eq!(sim.blue_agent_ids.len(), 4);
        assert_eq!(sim.obs_size(), 9 + 13 * 4 + 6 * 3);
    }

    #[test]
    fn self_play_2v2_exposes_red_learners() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 8;
        cfg.blue.num_agents = 2;
        cfg.red.num_agents = 2;
        cfg.blue.behavior = Behavior::External;
        cfg.red.behavior = Behavior::External;
        cfg.blue.apply_box_formation();
        cfg.red.apply_box_formation();
        let mut sim = Simulation::new(cfg);
        let r = sim.reset(Some(3));
        assert_eq!(sim.agent_ids().len(), 4);
        assert!(r.agents.contains_key("agent_0"));
        assert!(r.agents.contains_key("red_1"));
        assert_eq!(r.agents["agent_0"].flat_obs.len(), sim.obs_size());
        assert_eq!(r.agents["red_0"].flat_obs.len(), sim.obs_size());
        let mut acts = HashMap::new();
        for n in sim.agent_ids() {
            acts.insert(n, Action::default());
        }
        let s = sim.step(&acts);
        assert_eq!(s.agents.len(), 4);
        assert!(s.agents["red_0"].flat_obs.len() == sim.obs_size());
    }

    #[test]
    fn undetected_enemy_does_not_leak_goal_distance() {
        // Start far outside radar range so nothing is detected at reset.
        let mut cfg = ScenarioConfig::default();
        cfg.blue.init_position.z = 60.0;
        cfg.blue.target_position.z = -60.0;
        cfg.red.init_position.z = -60.0;
        cfg.red.target_position.z = 60.0;
        cfg.red.behavior = Behavior::Duck;
        let mut sim = Simulation::new(cfg);
        let r = sim.reset(Some(1));
        let e = &r.agents["agent_0"].obs.enemies[0];
        assert_eq!(e.detected, 0.0);
        assert_eq!(
            e.dist_target, 0.5,
            "undetected enemy must use the placeholder"
        );
        assert_eq!(e.aspect, 0.0);
    }

    #[test]
    fn engagement_emits_launch_and_terminal_events() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 400;
        cfg.blue.behavior = Behavior::Baseline1;
        cfg.red.behavior = Behavior::Duck;
        cfg.red.mission = crate::config::Mission::Striker;
        let mut sim = Simulation::new(cfg);
        sim.reset(Some(2));
        let mut kinds = Vec::new();
        let empty = HashMap::new();
        let mut last_t = 0.0;
        while sim.end == EndCondition::Ongoing {
            for e in sim.step(&empty).events {
                assert!(e.t >= last_t, "events must be time ordered");
                last_t = e.t;
                kinds.push(e.kind);
            }
        }
        assert!(kinds.contains(&EventKind::TrackGained));
        assert!(kinds.contains(&EventKind::Launch));
        assert_eq!(kinds.last(), Some(&EventKind::End));
        let launches = kinds.iter().filter(|k| **k == EventKind::Launch).count();
        assert_eq!(launches as u32, sim.missiles_fired);
    }

    #[test]
    fn snapshot_carries_reward_breakdown() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 20;
        let mut sim = Simulation::new(cfg);
        sim.reset(Some(1));
        let mut acts = HashMap::new();
        acts.insert("agent_0".to_string(), Action::default());
        let mut total = 0.0;
        for _ in 0..10 {
            total += sim.step(&acts).agents["agent_0"].reward;
        }
        let snap = sim.snapshot();
        let blue = snap.fighters.iter().find(|f| f.team == 0).unwrap();
        assert!((blue.reward.total() - total).abs() < 1e-9);
        assert!(snap.time_s > 0.0);
    }

    fn run_to_end(cfg: &ScenarioConfig, seed: u64) -> (u32, usize, usize) {
        let mut sim = Simulation::new(cfg.clone());
        sim.reset(Some(seed));
        let empty = HashMap::new();
        while sim.end == EndCondition::Ongoing {
            sim.step(&empty);
        }
        let o = sim.outcome();
        (o.steps, o.blue_kills, o.missiles_fired as usize)
    }

    #[test]
    fn beh_noise_makes_seeds_matter_and_stays_reproducible() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 500;
        cfg.blue.behavior = Behavior::Baseline1;
        cfg.red.behavior = Behavior::Baseline1;
        // Default: seeds do not change a scripted episode.
        assert_eq!(run_to_end(&cfg, 1), run_to_end(&cfg, 2));
        cfg.blue.beh_noise = 0.15;
        cfg.red.beh_noise = 0.15;
        cfg.blue.rnd_offset_range.x = 5.0;
        cfg.red.rnd_offset_range.x = 5.0;
        let runs: Vec<_> = (1..=6).map(|s| run_to_end(&cfg, s)).collect();
        assert!(
            runs.iter().any(|r| *r != runs[0]),
            "noise should vary outcomes: {runs:?}"
        );
        assert_eq!(run_to_end(&cfg, 4), runs[3], "same seed must reproduce");
    }

    /// Median |aspect| to the highest-priority threat while in `state`.
    fn median_aspect_in(state: FsmState, seeds: std::ops::Range<u64>) -> Option<f64> {
        let mut xs = Vec::new();
        for seed in seeds {
            let mut cfg = ScenarioConfig::default();
            cfg.env.max_cycles = 600;
            cfg.blue.behavior = Behavior::Baseline1;
            cfg.red.behavior = Behavior::Baseline1;
            cfg.red.beh_config.d_shot = vec![0.62];
            cfg.red.beh_config.l_crank = vec![0.58];
            cfg.red.beh_config.l_break = vec![1.28];
            for t in [&mut cfg.blue, &mut cfg.red] {
                t.beh_noise = 0.1;
                t.rnd_offset_range.x = 4.0;
            }
            let mut sim = Simulation::new(cfg);
            sim.reset(Some(seed));
            let empty = HashMap::new();
            while sim.end == EndCondition::Ongoing {
                sim.step(&empty);
                for f in sim.fighters.iter().filter(|f| f.alive && f.fsm == state) {
                    if let Some(t) = f
                        .enemy_tracks
                        .iter()
                        .find(|t| Some(t.id) == f.hpt_id && t.detected)
                    {
                        xs.push(t.aspect.abs());
                    }
                }
            }
        }
        if xs.is_empty() {
            return None;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(xs[xs.len() / 2])
    }

    #[test]
    fn fsm_support_holds_an_offset_instead_of_zigzagging() {
        // Regression: the side used to flip with the sign of the aspect every
        // decision, pinning the target on the nose (median |aspect| ~3 deg).
        let m = median_aspect_in(FsmState::MissileSupport, 1..7).expect("support observed");
        assert!(
            m > 25.0,
            "support should offset the target, median |aspect| = {m:.1}"
        );
    }

    #[test]
    fn fsm_evade_turns_away_from_the_threat() {
        // Regression: evade used to keep the threat within ~6 deg of the nose.
        // Turning away drops the track past the 60 deg radar gimbal, so only the
        // first part of the turn is observable here.
        if let Some(m) = median_aspect_in(FsmState::Evade, 1..7) {
            assert!(m > 30.0, "evade should turn away, median |aspect| = {m:.1}");
        }
    }

    #[test]
    fn legacy_missile_without_support_cannot_go_active() {
        // FireOnce launches at ~25 NM, then we kill the shooter: the missile must
        // lose datalink support and never pitbull.
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 200;
        cfg.env.stop_mission = false;
        cfg.env.missile_support = crate::config::MissileSupport::Track;
        cfg.env.missile_model = crate::missile::MissileModel::Legacy;
        cfg.blue.behavior = Behavior::FireOnce;
        cfg.blue.init_position.z = 12.5;
        cfg.blue.target_position.z = -12.5;
        cfg.red.behavior = Behavior::Duck;
        cfg.red.init_position.z = -12.5;
        cfg.red.init_hdg = 0.0;
        cfg.red.target_position.z = -80.0;
        let mut sim = Simulation::new(cfg);
        sim.reset(Some(1));
        let empty = HashMap::new();
        while sim.missiles_fired == 0 && sim.end == EndCondition::Ongoing {
            sim.step(&empty);
        }
        assert_eq!(sim.missiles_fired, 1);
        sim.fighters.iter_mut().find(|f| f.team == 0).unwrap().alive = false;
        let mut kinds = Vec::new();
        for _ in 0..140 {
            let r = sim.step(&empty);
            assert!(
                sim.missiles.iter().all(|m| !m.has_support),
                "dead shooter cannot support"
            );
            kinds.extend(r.events.iter().map(|e| (e.kind, e.detail.clone())));
            if sim.missiles.is_empty() {
                break;
            }
        }
        assert!(
            !kinds.iter().any(|(k, _)| *k == EventKind::Pitbull),
            "{kinds:?}"
        );
        assert!(
            kinds
                .iter()
                .any(|(k, d)| *k == EventKind::Miss && d == "lost_support"),
            "{kinds:?}"
        );
    }

    #[test]
    fn hard_turns_bleed_speed_and_it_recovers() {
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 200;
        cfg.red.behavior = Behavior::Duck;
        let mut sim = Simulation::new(cfg.clone());
        sim.reset(Some(1));
        let top = sim.fighters[0].top_speed();
        assert!((sim.fighters[0].speed - top).abs() < 1e-9);
        let turn = Action {
            d_heading: 0.5,
            d_altitude: 0.0,
            g_force: 1.0,
            fire: -1.0,
        };
        let mut acts = HashMap::new();
        for _ in 0..10 {
            acts.insert("agent_0".to_string(), turn);
            sim.step(&acts);
        }
        let slowed = sim.fighters[0].speed;
        assert!(
            slowed < 0.85 * top,
            "10 s at max g should bleed speed: {slowed} vs {top}"
        );
        for _ in 0..60 {
            acts.insert(
                "agent_0".to_string(),
                Action {
                    d_heading: 0.0,
                    ..turn
                },
            );
            sim.step(&acts);
        }
        assert!(
            sim.fighters[0].speed > 0.95 * top,
            "straight flight recovers speed"
        );
        // The legacy constant-speed model ignores manoeuvres.
        cfg.env.aircraft_model = crate::config::AircraftModel::ConstantSpeed;
        let mut legacy = Simulation::new(cfg);
        legacy.reset(Some(1));
        for _ in 0..10 {
            acts.insert("agent_0".to_string(), turn);
            legacy.step(&acts);
        }
        assert!((legacy.fighters[0].speed - legacy.fighters[0].top_speed()).abs() < 1e-9);
    }

    #[test]
    fn scripted_pilot_defends_against_an_active_missile() {
        // Red (balanced-like FSM) is shot at from inside 10 NM; once the missile is
        // active it must turn cold, regardless of its offensive state.
        let mut cfg = ScenarioConfig::default();
        cfg.env.max_cycles = 60;
        cfg.blue.behavior = Behavior::FireOnce;
        cfg.blue.init_position.z = 4.0;
        cfg.red.behavior = Behavior::Baseline1;
        cfg.red.beh_config.l_break = vec![2.5]; // never breaks on geometry alone
        cfg.red.init_position.z = -4.0;
        let mut sim = Simulation::new(cfg);
        sim.reset(Some(1));
        let empty = HashMap::new();
        let mut defended = false;
        for _ in 0..30 {
            sim.step(&empty);
            let red = sim.fighters.iter().find(|f| f.team == 1).unwrap();
            if let Some(brg) = red.missile_warning {
                assert_eq!(red.fsm, FsmState::Evade);
                assert!((crate::geometry::clamp_hdg(red.hdg_cmd - brg).abs() - 180.0).abs() < 1.0);
                defended = true;
                break;
            }
            if sim.end != EndCondition::Ongoing {
                break;
            }
        }
        assert!(defended, "red should get a missile warning and defend");
    }

    #[test]
    fn both_teams_dead_is_mutual_kill() {
        let mut sim = Simulation::new(ScenarioConfig::default());
        sim.reset(Some(1));
        for f in &mut sim.fighters {
            f.alive = false;
        }
        sim.missiles.clear();
        sim.check_terminals();
        assert_eq!(sim.end, EndCondition::MutualKill);
        let out = sim.outcome();
        assert!(!out.mission_success);
        assert_eq!(out.blue_kills, sim.config.red.num_agents);
        assert_eq!(out.blue_deaths, sim.config.blue.num_agents);
        let snap = sim.snapshot();
        assert!(!snap.fighters.is_empty());
        assert!(snap.fighters.iter().all(|f| f.radar_range > 0.0));
    }
}
