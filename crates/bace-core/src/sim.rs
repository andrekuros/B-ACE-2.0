//! Episode / simulation manager.

use crate::config::{ScenarioConfig, TeamConfig};
use crate::fighter::{AircraftView, Fighter, FsmState};
use crate::geometry::distance2d;
use crate::missile::Missile;
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
    MaxCycles,
    RedMission,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FighterSnapshot {
    pub id: u32,
    pub team: u8,
    pub pos: [f64; 3],
    pub hdg: f64,
    pub alive: bool,
    pub missiles: u32,
    pub fsm: FsmState,
    pub agent_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissileSnapshot {
    pub id: u32,
    pub pos: [f64; 3],
    pub hdg: f64,
    pub team: u8,
    pub pitbull: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimSnapshot {
    pub action_step: u32,
    pub end: EndCondition,
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

    fn spawn_teams(&mut self) {
        self.fighters.clear();
        self.missiles.clear();
        self.blue_agent_ids.clear();
        self.next_missile_id = 1;

        let n_blue = self.config.blue.num_agents;
        let n_red = self.config.red.num_agents;
        let blue_ids: Vec<u32> = (0..n_blue).map(|i| 101 + i as u32).collect();
        let red_ids: Vec<u32> = (0..n_red).map(|i| 201 + i as u32).collect();

        for (i, &id) in blue_ids.iter().enumerate() {
            let (pos, target, hdg) =
                Self::pos_from_cfg(&self.config.blue, i, &mut self.rng);
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
            );
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
            );
            self.fighters.push(f);
        }
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
        let views = self.views();
        // First pass local detection without DL, then with DL from previous detections.
        // Approximate: compute local, build DL, update again.
        let blue_ids: Vec<u32> = self.fighters.iter().filter(|f| f.team == 0).map(|f| f.id).collect();
        let red_ids: Vec<u32> = self.fighters.iter().filter(|f| f.team == 1).map(|f| f.id).collect();

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

    fn try_fire(fighter: &mut Fighter, missiles: &mut Vec<Missile>, next_id: &mut u32) -> Option<f64> {
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
        let m = Missile::launch(
            *next_id,
            fighter.id,
            hpt,
            fighter.team,
            fighter.pos,
            fighter.hdg,
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
            let targets: HashMap<u32, ( [f64; 3], bool)> = self
                .fighters
                .iter()
                .map(|f| (f.id, (f.pos, f.alive)))
                .collect();

            let mut hits: Vec<(u32, u32, u8)> = Vec::new(); // target_id, shooter_id, team
            let mut misses: Vec<u32> = Vec::new();

            for m in &mut self.missiles {
                if !m.alive {
                    continue;
                }
                let (tpos, talive) = targets
                    .get(&m.target_id)
                    .copied()
                    .unwrap_or(([0.0; 3], false));
                let hit = m.tick(dt, Some(tpos), talive);
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
        let blue_alive = self.fighters.iter().filter(|f| f.team == 0 && f.alive).count();
        let red_alive = self.fighters.iter().filter(|f| f.team == 1 && f.alive).count();
        let missiles_left = !self.missiles.is_empty();

        if blue_alive == 0 && !missiles_left {
            self.end = EndCondition::BlueKilled;
            for f in self.fighters.iter_mut().filter(|f| f.team == 0) {
                f.rewards
                    .add_terminal("Team_Killed", f.missiles);
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
                f.rewards
                    .add_terminal("Enemies_Killed", f.missiles);
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
                        f.rewards
                            .add_terminal("Enemy_Achieved_Target", f.missiles);
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

    fn sense_and_observe(&mut self, _is_reset: bool) -> StepResult {
        self.sense_all();

        // Mission shaping for blue
        for f in self.fighters.iter_mut().filter(|f| f.team == 0 && f.alive) {
            let shaped = 1.0 + (-0.99) / (1.0 + (-0.02 * (f.dist2go - 370.4)).exp());
            f.rewards.add_mission(shaped);
        }

        let views = self.views();
        let mut agents = HashMap::new();
        let blue: Vec<&Fighter> = self.fighters.iter().filter(|f| f.team == 0).collect();
        for (i, f) in blue.iter().enumerate() {
            let name = self
                .blue_agent_ids
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("agent_{i}"));
            let allies: Vec<AircraftView> = views
                .iter()
                .copied()
                .filter(|v| f.ally_ids.contains(&v.id))
                .collect();
            // Also pass enemy dist2go via views of enemies
            let enemies: Vec<AircraftView> = views
                .iter()
                .copied()
                .filter(|v| f.enemy_tracks.iter().any(|t| t.id == v.id))
                .collect();
            let mut obs = f.build_obs(&allies);
            for e in &mut obs.enemies {
                if let Some(ev) = enemies.iter().find(|x| x.id == e.id) {
                    e.dist_target = ev.dist2go / 3000.0;
                }
            }
            let flat = obs.to_flat();
            // rewards taken later in step; on reset zero
            agents.insert(
                name,
                AgentStep {
                    obs,
                    flat_obs: flat,
                    reward: 0.0,
                    reward_breakdown: RewardBreakdown::default(),
                    terminated: f.done && self.end != EndCondition::MaxCycles,
                    truncated: self.end == EndCondition::MaxCycles,
                },
            );
        }

        StepResult {
            agents,
            end: self.end,
            action_step: self.action_step,
        }
    }

    /// Apply blue external actions (by agent name), run FSM for others, step physics.
    pub fn step(&mut self, actions: &HashMap<AgentId, Action>) -> StepResult {
        if self.end != EndCondition::Ongoing {
            return self.sense_and_observe(false);
        }

        self.sense_all();

        // Apply actions / behaviors
        let blue_ids = self.blue_agent_ids.clone();
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
        for f in self.fighters.iter_mut().filter(|f| f.team == 1) {
            f.run_behavior();
        }

        self.accumulate_behavior_stats();

        // Firing (needs mutable fighters + missiles)
        let mut next_id = self.next_missile_id;
        let mut missiles = std::mem::take(&mut self.missiles);
        for f in &mut self.fighters {
            let before = f.missiles;
            if let Some(rng_nm) = Self::try_fire(f, &mut missiles, &mut next_id) {
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

        self.integrate_physics();

        // Collect rewards into step result
        self.sense_all();
        for f in self.fighters.iter_mut().filter(|f| f.team == 0 && f.alive) {
            let shaped = 1.0 + (-0.99) / (1.0 + (-0.02 * (f.dist2go - 370.4)).exp());
            f.rewards.add_mission(shaped);
        }

        let views = self.views();
        let mut agents = HashMap::new();
        let n_blue = self.fighters.iter().filter(|f| f.team == 0).count();
        for i in 0..n_blue {
            let name = self.blue_agent_ids[i].clone();
            let f = self.fighters.iter_mut().filter(|f| f.team == 0).nth(i).unwrap();
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
            for e in &mut obs.enemies {
                if let Some(ev) = enemies.iter().find(|x| x.id == e.id) {
                    e.dist_target = ev.dist2go / 3000.0;
                }
            }
            let flat = obs.to_flat();
            let (reward, breakdown) = f.rewards.take_step();
            let terminated = f.done && self.end != EndCondition::MaxCycles;
            let truncated = self.end == EndCondition::MaxCycles;
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

        StepResult {
            agents,
            end: self.end,
            action_step: self.action_step,
        }
    }

    pub fn snapshot(&self) -> SimSnapshot {
        SimSnapshot {
            action_step: self.action_step,
            end: self.end,
            fighters: {
                let mut blue_i = 0usize;
                self.fighters
                    .iter()
                    .map(|f| {
                        let agent_name = if f.team == 0 {
                            let name = self.blue_agent_ids.get(blue_i).cloned();
                            blue_i += 1;
                            name
                        } else {
                            None
                        };
                        FighterSnapshot {
                            id: f.id,
                            team: f.team,
                            pos: f.pos,
                            hdg: f.hdg,
                            alive: f.alive,
                            missiles: f.missiles,
                            fsm: f.fsm,
                            agent_name,
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
        let mission_success =
            self.end != EndCondition::RedMission && self.end != EndCondition::BlueKilled;
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
}
