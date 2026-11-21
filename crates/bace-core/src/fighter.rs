//! Kinematic fighter agent with radar tracks and FSM baselines.

use crate::config::{BehConfig, Behavior, Mission, RewardsConfig};
use crate::geometry::{
    angle_off, aspect_angle, clamp_hdg, desired_heading, distance2d, heading_to,
};
use crate::obs::{Action, AllyObs, EnemyObs, OwnObs, StructuredObs};
use crate::rewards::Rewards;
use crate::units::SConv;
use crate::wez::{self, WezRanges};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FsmState {
    #[default]
    Search,
    Engage,
    MissileSupport,
    Evade,
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u32,
    pub detected: bool,
    pub is_missile_support: bool,
    pub aspect: f64,
    pub angle_off: f64,
    pub dist: f64,
    pub alt_diff: f64,
    pub own_wez: WezRanges,
    pub enemy_wez: WezRanges,
    pub threat_factor: f64,
    pub offensive_factor: f64,
}

impl Track {
    pub fn blank(id: u32) -> Self {
        Self {
            id,
            detected: false,
            is_missile_support: false,
            aspect: 0.0,
            angle_off: 0.0,
            dist: -1.0,
            alt_diff: 0.0,
            own_wez: WezRanges {
                r_max: 0.01,
                r_nez: 0.01,
            },
            enemy_wez: WezRanges {
                r_max: 0.01,
                r_nez: 0.01,
            },
            threat_factor: 0.0,
            offensive_factor: 0.0,
        }
    }
}

/// Snapshot of another aircraft for sensing.
#[derive(Debug, Clone, Copy)]
pub struct AircraftView {
    pub id: u32,
    pub pos: [f64; 3],
    pub hdg: f64,
    pub alive: bool,
    pub dist2go: f64,
}

#[derive(Debug, Clone)]
pub struct Fighter {
    pub id: u32,
    pub team: u8,
    pub pos: [f64; 3],
    pub hdg: f64,
    pub pitch: f64,
    pub speed: f64,
    pub alive: bool,
    pub missiles: u32,
    pub behavior: Behavior,
    pub mission: Mission,
    pub beh: BehConfig,
    pub target_pos: [f64; 3],
    pub dist2go: f64,
    pub fsm: FsmState,
    pub hpt_id: Option<u32>,
    pub supporting_missile: bool,
    pub in_flight_missile_id: Option<u32>,
    pub share_tracks: bool,
    pub rewards: Rewards,
    pub done: bool,
    pub hdg_cmd: f64,
    pub level_cmd: f64,
    pub g_cmd: f64,
    pub fire_cmd: bool,
    pub radar_range: f64,
    pub radar_hfov: f64,
    pub radar_vfov_up: f64,
    pub radar_vfov_down: f64,
    pub max_speed: f64,
    pub max_g: f64,
    pub max_level: f64,
    pub min_level: f64,
    pub enemy_tracks: Vec<Track>,
    pub ally_ids: Vec<u32>,
    /// Relative FSM threshold jitter (see `TeamConfig::beh_noise`).
    pub beh_noise: f64,
    /// Side (+1 right / -1 left) chosen when an engagement starts and kept for
    /// cranks and missile support, as in B-ACE 1.0.
    pub defense_side: f64,
    /// Decisions left in a committed break (hysteresis against Engage/Evade dithering).
    pub evade_hold: u32,
    /// Missile physics carried by this aircraft (drives its WEZ estimates).
    pub missile_model: crate::missile::MissileModel,
    pub aircraft_model: crate::config::AircraftModel,
    /// Bearing (deg) of the nearest active enemy missile tracking this aircraft,
    /// as a radar warning receiver would report it. `None` when no warning.
    pub missile_warning: Option<f64>,
    rng: StdRng,
}

impl Fighter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        team: u8,
        pos: [f64; 3],
        hdg: f64,
        target_pos: [f64; 3],
        behavior: Behavior,
        mission: Mission,
        beh: BehConfig,
        share_tracks: bool,
        rewards_cfg: RewardsConfig,
        enemy_ids: &[u32],
        ally_ids: Vec<u32>,
    ) -> Self {
        Self {
            id,
            team,
            pos,
            hdg: clamp_hdg(hdg),
            pitch: 0.0,
            speed: 650.0 * SConv::KNOT2GDM_S * Self::altitude_speed_factor(pos[1]),
            alive: true,
            missiles: 6,
            behavior,
            mission,
            beh,
            target_pos,
            dist2go: distance2d(pos, target_pos),
            fsm: FsmState::Search,
            hpt_id: None,
            supporting_missile: false,
            in_flight_missile_id: None,
            share_tracks,
            rewards: Rewards::new(rewards_cfg),
            done: false,
            hdg_cmd: clamp_hdg(hdg),
            level_cmd: pos[1],
            g_cmd: 1.0,
            fire_cmd: false,
            radar_range: 50.0 * SConv::NM2GDM,
            radar_hfov: 60.0,
            radar_vfov_up: 40.0,
            radar_vfov_down: 20.0,
            max_speed: 650.0 * SConv::KNOT2GDM_S,
            max_g: 9.0,
            max_level: 50000.0 * SConv::FT2GDM,
            min_level: 1000.0 * SConv::FT2GDM,
            enemy_tracks: enemy_ids.iter().map(|&eid| Track::blank(eid)).collect(),
            ally_ids,
            beh_noise: 0.0,
            defense_side: 0.0,
            evade_hold: 0,
            missile_model: crate::missile::MissileModel::default(),
            aircraft_model: crate::config::AircraftModel::default(),
            missile_warning: None,
            rng: StdRng::seed_from_u64(id as u64),
        }
    }

    pub fn with_missile_model(mut self, model: crate::missile::MissileModel) -> Self {
        self.missile_model = model;
        self
    }

    pub fn with_aircraft_model(mut self, model: crate::config::AircraftModel) -> Self {
        self.aircraft_model = model;
        self
    }

    /// Top speed at the current altitude, GDM/s.
    pub fn top_speed(&self) -> f64 {
        self.max_speed * Self::altitude_speed_factor(self.pos[1])
    }

    /// Enable stochastic FSM decisions with a per-aircraft stream.
    pub fn with_noise(mut self, beh_noise: f64, seed: u64) -> Self {
        self.beh_noise = beh_noise.max(0.0);
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    fn jitter(&mut self, v: f64) -> f64 {
        if self.beh_noise <= 0.0 {
            return v;
        }
        v * (1.0 + self.rng.gen_range(-self.beh_noise..=self.beh_noise))
    }

    pub fn altitude_speed_factor(alt: f64) -> f64 {
        0.2 * alt / 76.2 + 0.8
    }

    pub fn altitude_g_factor(alt: f64) -> f64 {
        (-0.5 * alt / 76.2 + 1.5).max(0.3)
    }

    pub fn apply_action(&mut self, action: Action) {
        self.hdg_cmd = desired_heading(self.hdg, action.d_heading.clamp(-1.0, 1.0));
        let level_ft = action.d_altitude.clamp(-1.0, 1.0) * 25000.0 + 25000.0;
        self.level_cmd = (level_ft * SConv::FT2GDM).clamp(self.min_level, self.max_level);
        self.g_cmd =
            (action.g_force.clamp(-1.0, 1.0) * (self.max_g - 1.0) + (self.max_g + 1.0)) / 2.0;
        self.fire_cmd = action.fire > 0.0;
    }

    pub fn is_detectable(&self, other_pos: [f64; 3]) -> bool {
        let dist = distance2d(self.pos, other_pos);
        if dist > self.radar_range || dist < 1e-6 {
            return false;
        }
        let brg = heading_to(self.pos, other_pos);
        if aspect_angle(self.hdg, brg).abs() > self.radar_hfov {
            return false;
        }
        let elev = ((other_pos[1] - self.pos[1]) / dist).atan().to_degrees();
        elev >= -self.radar_vfov_down && elev <= self.radar_vfov_up
    }

    pub fn update_tracks(&mut self, enemies: &[AircraftView], dl_detected: &[u32]) {
        let own_pos = self.pos;
        let own_hdg = self.hdg;
        let radar_range = self.radar_range;
        let radar_hfov = self.radar_hfov;
        let radar_vfov_up = self.radar_vfov_up;
        let radar_vfov_down = self.radar_vfov_down;
        let share_tracks = self.share_tracks;

        let detectable = |other_pos: [f64; 3]| -> bool {
            let dist = distance2d(own_pos, other_pos);
            if dist > radar_range || dist < 1e-6 {
                return false;
            }
            let brg = heading_to(own_pos, other_pos);
            if aspect_angle(own_hdg, brg).abs() > radar_hfov {
                return false;
            }
            let elev = ((other_pos[1] - own_pos[1]) / dist).atan().to_degrees();
            elev >= -radar_vfov_down && elev <= radar_vfov_up
        };

        for track in &mut self.enemy_tracks {
            let Some(enemy) = enemies.iter().find(|e| e.id == track.id) else {
                continue;
            };
            if !enemy.alive {
                if track.detected {
                    let mult = if track.is_missile_support { 5.0 } else { 1.0 };
                    self.rewards.add_detect_loss(mult);
                }
                track.detected = false;
                track.dist = -1.0;
                continue;
            }

            let was = track.detected;
            let local = detectable(enemy.pos);
            let via_dl = dl_detected.contains(&track.id);
            track.detected = local || (share_tracks && via_dl);

            if was && !track.detected {
                let mult = if track.is_missile_support { 5.0 } else { 1.0 };
                self.rewards.add_detect_loss(mult);
            }

            if track.detected {
                self.rewards.add_keep_track();
                let dist = distance2d(own_pos, enemy.pos);
                let brg = heading_to(own_pos, enemy.pos);
                track.aspect = aspect_angle(own_hdg, brg);
                track.angle_off = angle_off(own_hdg, enemy.hdg);
                track.dist = dist;
                track.alt_diff = (own_pos[1] - enemy.pos[1]) / 150.0;
                // Enemy's nose relative to its line of sight to us (0 = hot).
                let inv_asp = aspect_angle(enemy.hdg, heading_to(enemy.pos, own_pos));
                let inv_off = angle_off(enemy.hdg, own_hdg);
                let model = self.missile_model;
                track.own_wez = wez::evaluate_model(
                    model,
                    own_pos[1],
                    enemy.pos[1],
                    track.aspect,
                    inv_asp,
                    track.angle_off,
                );
                track.enemy_wez = wez::evaluate_model(
                    model,
                    enemy.pos[1],
                    own_pos[1],
                    inv_asp,
                    track.aspect,
                    inv_off,
                );
                track.offensive_factor =
                    wez::offensive_factor(dist, track.own_wez.r_max, track.own_wez.r_nez);
                track.threat_factor =
                    wez::threat_factor(dist, track.enemy_wez.r_max, track.enemy_wez.r_nez);
            } else {
                track.dist = -1.0;
            }
        }

        let mut best_off = -1.0;
        let mut hpt = None;
        for t in &self.enemy_tracks {
            if t.detected && t.offensive_factor >= best_off {
                best_off = t.offensive_factor;
                hpt = Some(t.id);
            }
        }
        if hpt.is_none() {
            let mut best_thr = -1.0;
            for t in &self.enemy_tracks {
                if t.detected && t.threat_factor >= best_thr {
                    best_thr = t.threat_factor;
                    hpt = Some(t.id);
                }
            }
        }
        self.hpt_id = hpt;
    }

    pub fn run_behavior(&mut self) {
        match self.behavior {
            Behavior::External => {}
            Behavior::Duck => {
                self.fsm = FsmState::Search;
                self.hdg_cmd = heading_to(self.pos, self.target_pos);
                self.level_cmd = self.target_pos[1];
                self.g_cmd = 1.0;
                self.fire_cmd = false;
            }
            Behavior::Baseline1 => self.run_baseline1(),
            Behavior::FireOnce => self.run_fire_once(),
        }
    }

    fn run_fire_once(&mut self) {
        let hpt = self
            .hpt_id
            .and_then(|id| self.enemy_tracks.iter().find(|t| t.id == id && t.detected))
            .cloned();

        if self.supporting_missile {
            self.fsm = FsmState::MissileSupport;
            if let Some(ref t) = hpt {
                self.hdg_cmd = clamp_hdg(self.hdg + t.aspect);
            } else {
                self.hdg_cmd = heading_to(self.pos, self.target_pos);
            }
            self.level_cmd = self.target_pos[1];
            self.g_cmd = 2.0;
            self.fire_cmd = false;
            return;
        }

        if let Some(ref t) = hpt {
            self.fsm = FsmState::Engage;
            self.hdg_cmd = clamp_hdg(self.hdg + t.aspect);
            self.level_cmd = self.pos[1];
            self.g_cmd = 2.0;
            // One shot only (spawn load is 6).
            self.fire_cmd = t.aspect.abs() < 30.0 && self.missiles == 6;
            return;
        }

        self.fsm = FsmState::Search;
        self.hdg_cmd = heading_to(self.pos, self.target_pos);
        self.level_cmd = self.target_pos[1];
        self.g_cmd = 1.0;
        self.fire_cmd = false;
    }

    fn run_baseline1(&mut self) {
        let d_shot = self.beh.d_shot.first().copied().unwrap_or(0.85);
        let l_crank = self.beh.l_crank.first().copied().unwrap_or(0.6);
        let l_break = self.beh.l_break.first().copied().unwrap_or(0.95);
        let (d_shot, l_crank, l_break) = (
            self.jitter(d_shot),
            self.jitter(l_crank),
            self.jitter(l_break),
        );

        let hpt = self
            .hpt_id
            .and_then(|id| self.enemy_tracks.iter().find(|t| t.id == id && t.detected))
            .cloned();

        // Latch a defensive side at the start of an engagement (random when
        // `beh_noise > 0`, otherwise from the current geometry) so cranks and
        // support offsets hold one side instead of flipping every decision.
        if hpt.is_some() && self.defense_side == 0.0 {
            self.defense_side = if self.beh_noise > 0.0 {
                if self.rng.gen_bool(0.5) {
                    1.0
                } else {
                    -1.0
                }
            } else if hpt.as_ref().map(|t| t.aspect).unwrap_or(0.0) >= 0.0 {
                -1.0
            } else {
                1.0
            };
        }
        if hpt.is_none() && !self.supporting_missile {
            self.defense_side = 0.0;
        }
        let side = if self.defense_side == 0.0 {
            1.0
        } else {
            self.defense_side
        };

        // Minimum time a break is held before re-engaging (decisions).
        const EVADE_HOLD: u32 = 8;
        // Missile warning: an active missile is tracking us. Defend by turning
        // cold (missile on the tail) at the break g; this overrides support.
        if let Some(brg) = self.missile_warning {
            self.fsm = FsmState::Evade;
            self.evade_hold = EVADE_HOLD;
            self.hdg_cmd = clamp_hdg(brg + 180.0);
            self.g_cmd = self.max_g * 0.8;
            self.fire_cmd = false;
            return;
        }
        // Start the turn-in this far outside the shot range (fraction of d_shot).
        const COMMIT_MARGIN: f64 = 1.15;
        if let Some(ref t) = hpt {
            let bearing = clamp_hdg(self.hdg + t.aspect);
            let range_ratio = t.dist / t.own_wez.r_max.max(1e-6);
            // Commit: from just outside shot range, turn in and accept the risk
            // until the missile is off. A valid shot is taken before any break
            // (B-ACE 1.0 checks the launch before the break).
            let committed = !self.supporting_missile
                && self.evade_hold == 0
                && self.missiles > 0
                && range_ratio <= d_shot * COMMIT_MARGIN
                && t.aspect.abs() < 45.0;
            let can_shoot_now = committed && range_ratio <= d_shot && t.aspect.abs() < 30.0;
            // Supporting a missile tolerates more threat (B-ACE 1.0: +0.5).
            let break_at = l_break + if self.supporting_missile { 0.5 } else { 0.0 };
            let in_break = self.evade_hold > 0 && t.threat_factor >= 0.8 * l_break;
            if !committed && !can_shoot_now && (t.threat_factor >= break_at || in_break) {
                // Break: turn away from the threat (B-ACE 1.0: radial + 180) and hold it.
                if self.fsm != FsmState::Evade {
                    self.evade_hold = EVADE_HOLD;
                }
                self.evade_hold = self.evade_hold.saturating_sub(1).max(1);
                self.fsm = FsmState::Evade;
                self.hdg_cmd = clamp_hdg(bearing + 180.0);
                self.g_cmd = self.max_g * 0.8;
                self.fire_cmd = false;
                return;
            }
        }
        self.evade_hold = self.evade_hold.saturating_sub(1);
        if self.fsm == FsmState::Evade && self.evade_hold > 0 {
            // Threat dropped or track lost while still in the hold: keep running.
            self.g_cmd = self.max_g * 0.8;
            self.fire_cmd = false;
            return;
        }

        if self.supporting_missile {
            // Offset to keep the target inside the radar while the missile flies.
            self.fsm = FsmState::MissileSupport;
            if let Some(ref t) = hpt {
                let bearing = clamp_hdg(self.hdg + t.aspect);
                self.hdg_cmd = clamp_hdg(bearing + 50.0 * side);
                self.g_cmd = 3.0;
            }
            self.fire_cmd = false;
            return;
        }

        if let Some(ref t) = hpt {
            self.fsm = FsmState::Engage;
            let bearing = clamp_hdg(self.hdg + t.aspect);
            let range_ratio = t.dist / t.own_wez.r_max.max(1e-6);
            let shot_valid = range_ratio <= d_shot && self.missiles > 0;
            if range_ratio <= d_shot * COMMIT_MARGIN && self.missiles > 0 {
                // Commit: turn in to put the target inside the launch cone.
                self.hdg_cmd = bearing;
            } else if t.threat_factor > l_crank {
                // Crank once threatened (B-ACE 1.0: threat_factor > lCrank): hold the
                // target 40 deg off the nose on the latched side to slow the closure.
                self.hdg_cmd = clamp_hdg(bearing + 40.0 * side);
            } else {
                self.hdg_cmd = bearing;
            }
            self.g_cmd = 4.0;
            self.fire_cmd = shot_valid && t.aspect.abs() < 30.0;
            return;
        }

        self.fsm = FsmState::Search;
        self.hdg_cmd = heading_to(self.pos, self.target_pos);
        self.level_cmd = self.target_pos[1];
        self.g_cmd = 1.5;
        self.fire_cmd = false;
    }

    pub fn physics_tick(&mut self, dt: f64) {
        if !self.alive {
            return;
        }
        self.dist2go = distance2d(self.pos, self.target_pos);

        let hdg_diff = clamp_hdg(self.hdg_cmd - self.hdg);
        let g_lim = self
            .g_cmd
            .clamp(1.0, self.max_g * Self::altitude_g_factor(self.pos[1]));
        let turn_rate_deg = (g_lim * SConv::GRAVITY_GDM / self.speed.max(1e-3)).to_degrees();
        let step = hdg_diff.clamp(-turn_rate_deg * dt, turn_rate_deg * dt);
        self.hdg = clamp_hdg(self.hdg + step);

        let level_diff = self.level_cmd - self.pos[1];
        let desired_pitch = level_diff.clamp(-15.0, 35.0);
        self.pitch += (desired_pitch - self.pitch).clamp(-0.5, 0.5);

        let v_top = self.top_speed();
        match self.aircraft_model {
            crate::config::AircraftModel::ConstantSpeed => self.speed = v_top,
            crate::config::AircraftModel::Energy => {
                // Point-mass energy balance: thrust excess recovers speed toward the
                // altitude's maximum; induced drag grows with load factor (n² for the
                // turn component); climbing trades speed for height.
                const RECOVER_TAU_S: f64 = 25.0;
                const BLEED_PER_G2: f64 = 0.19 * SConv::REAL2GD; // m/s² per g² -> GDM/s²
                let omega = (step / dt.max(1e-9)).to_radians().abs();
                let n_turn = omega * self.speed / SConv::GRAVITY_GDM;
                let accel = (v_top - self.speed) / RECOVER_TAU_S
                    - BLEED_PER_G2 * n_turn * n_turn
                    - SConv::GRAVITY_GDM * self.pitch.to_radians().sin();
                self.speed = (self.speed + accel * dt).clamp(0.45 * v_top, 1.05 * v_top);
            }
        }
        let rad = self.hdg.to_radians();
        self.pos[0] += rad.sin() * self.speed * dt;
        self.pos[1] = (self.pos[1] + self.pitch.to_radians().sin() * self.speed * dt)
            .clamp(self.min_level, self.max_level);
        self.pos[2] += -rad.cos() * self.speed * dt;
    }

    pub fn build_obs(&self, allies: &[AircraftView]) -> StructuredObs {
        let brg_tgt = heading_to(self.pos, self.target_pos);
        let own = OwnObs {
            x: self.pos[0] / 3000.0,
            z: self.pos[2] / 3000.0,
            altitude: self.pos[1] / 150.0,
            dist_target: self.dist2go / 3000.0,
            aspect_to_target: aspect_angle(self.hdg, brg_tgt) / 180.0,
            heading: self.hdg / 180.0,
            speed: self.speed / self.max_speed,
            missiles: self.missiles as f64 / 6.0,
            supporting_missile: if self.supporting_missile { 1.0 } else { 0.0 },
        };

        let enemies = self
            .enemy_tracks
            .iter()
            .map(|t| {
                if t.detected {
                    let dist_target = allies
                        .iter()
                        .find(|a| a.id == t.id)
                        .map(|a| a.dist2go / 3000.0)
                        .unwrap_or(0.5);
                    // enemies are not in allies — use track only
                    let _ = dist_target;
                    EnemyObs {
                        id: t.id,
                        alt_diff: t.alt_diff,
                        aspect: t.aspect / 180.0,
                        angle_off: t.angle_off / 180.0,
                        dist: t.dist / 3000.0,
                        dist_target: 0.5,
                        own_r_max: t.own_wez.r_max / 926.0,
                        own_r_nez: t.own_wez.r_nez / 926.0,
                        enemy_r_max: t.enemy_wez.r_max / 926.0,
                        enemy_r_nez: t.enemy_wez.r_nez / 926.0,
                        threat_factor: t.threat_factor - 1.0,
                        offensive_factor: t.offensive_factor - 1.0,
                        is_missile_support: if t.is_missile_support { 1.0 } else { 0.0 },
                        detected: 1.0,
                    }
                } else {
                    EnemyObs {
                        id: t.id,
                        alt_diff: 0.0,
                        aspect: 0.0,
                        angle_off: 0.0,
                        dist: -1.0,
                        dist_target: 0.5,
                        own_r_max: 0.0,
                        own_r_nez: 0.0,
                        enemy_r_max: 0.0,
                        enemy_r_nez: 0.0,
                        threat_factor: 0.0,
                        offensive_factor: 0.0,
                        is_missile_support: 0.0,
                        detected: 0.0,
                    }
                }
            })
            .collect();

        let ally_obs = self
            .ally_ids
            .iter()
            .map(|aid| {
                if let Some(a) = allies.iter().find(|x| x.id == *aid) {
                    if a.alive && self.share_tracks {
                        let dist = distance2d(self.pos, a.pos);
                        let brg = heading_to(self.pos, a.pos);
                        AllyObs {
                            id: *aid,
                            alt_diff: (self.pos[1] - a.pos[1]) / 150.0,
                            aspect: aspect_angle(self.hdg, brg) / 180.0,
                            angle_off: angle_off(self.hdg, a.hdg) / 180.0,
                            dist: dist / 3000.0,
                            dist_target: a.dist2go / 3000.0,
                            detected: 1.0,
                        }
                    } else {
                        AllyObs {
                            id: *aid,
                            alt_diff: 0.0,
                            aspect: 0.0,
                            angle_off: 0.0,
                            dist: -1.0,
                            dist_target: 0.5,
                            detected: 0.0,
                        }
                    }
                } else {
                    AllyObs {
                        id: *aid,
                        alt_diff: 0.0,
                        aspect: 0.0,
                        angle_off: 0.0,
                        dist: -1.0,
                        dist_target: 0.5,
                        detected: 0.0,
                    }
                }
            })
            .collect();

        StructuredObs {
            own,
            allies: ally_obs,
            enemies,
        }
    }
}
