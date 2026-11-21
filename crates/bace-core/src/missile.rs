//! Air-to-air missile models.
//!
//! * [`MissileModel::GenericMraam`] (default): a notional, generic medium-range
//!   air-to-air missile built only from open literature. 3-D point mass with a
//!   boost–sustain motor and mass loss, Mach-dependent zero-lift drag plus
//!   induced drag, a manoeuvre limit set by dynamic pressure, pure proportional
//!   navigation with a first-order autopilot lag, an optional loft, inertial
//!   midcourse with datalink updates and an active seeker for the endgame.
//!   See `docs/PUBLIC_REFERENCES.md` for the source of every element.
//! * [`MissileModel::Legacy`]: the B-ACE 2.0 kinematic missile (constant
//!   900 kt, horizontal pure pursuit), kept to reproduce earlier results.
//!
//! Positions are in GDM (metres / 100) like the rest of the simulator; the
//! generic model integrates internally in SI units.

use crate::atmosphere;
use crate::geometry::{aspect_angle, clamp_hdg, distance2d, heading_to};
use crate::units::SConv;
use serde::{Deserialize, Serialize};

/// Which missile physics the simulation uses (`env.missile_model`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MissileModel {
    /// Notional generic medium-range AAM (energy, drag, PN, seeker).
    #[default]
    GenericMraam,
    /// B-ACE 2.0 kinematic missile: constant 900 kt, horizontal pursuit.
    Legacy,
}

/// Parameters of the notional generic medium-range missile.
///
/// Values are of the order publicly reported for the medium-range AAM class
/// (launch mass ~150 kg, 178 mm body, Mach ~4) and follow textbook rocket and
/// aerodynamics relations; they describe no specific weapon.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MissileParams {
    /// Launch mass, kg.
    pub mass_launch: f64,
    /// Propellant burned in the boost phase, kg.
    pub prop_boost: f64,
    /// Propellant burned in the sustain phase, kg.
    pub prop_sustain: f64,
    /// Boost duration, s.
    pub t_boost: f64,
    /// Sustain duration, s.
    pub t_sustain: f64,
    /// Specific impulse, s (thrust = Isp · g0 · ṁ).
    pub isp: f64,
    /// Body diameter, m (reference area = π d² / 4).
    pub diameter: f64,
    /// Zero-lift drag multiplier while the motor burns (reduced base drag).
    pub power_on_drag_factor: f64,
    /// Induced-drag factor `k` in `C_Di = k · C_N²`.
    pub k_induced: f64,
    /// Maximum normal-force coefficient (aerodynamic manoeuvre limit).
    pub cn_max: f64,
    /// Structural lateral acceleration limit, g.
    pub g_max: f64,
    /// Proportional-navigation constant.
    pub nav_const: f64,
    /// Autopilot / airframe time constant, s.
    pub tau: f64,
    /// Maximum loft height above the target estimate, m.
    pub loft_max_m: f64,
    /// Active seeker acquisition range, m (the "A-pole" happens here).
    pub seeker_range_m: f64,
    /// Seeker gimbal limit, degrees off the missile velocity.
    pub seeker_fov_deg: f64,
    /// Warhead lethal radius, m.
    pub lethal_radius_m: f64,
    /// Guidance/battery life, s.
    pub max_tof: f64,
    /// After burnout the missile is spent below this speed, m/s.
    pub min_speed: f64,
}

impl Default for MissileParams {
    fn default() -> Self {
        Self {
            mass_launch: 150.0,
            prop_boost: 30.0,
            prop_sustain: 20.0,
            t_boost: 4.0,
            t_sustain: 8.0,
            isp: 245.0,
            diameter: 0.178,
            power_on_drag_factor: 0.8,
            k_induced: 0.12,
            cn_max: 25.0,
            g_max: 40.0,
            nav_const: 4.0,
            tau: 0.25,
            loft_max_m: 8_000.0,
            seeker_range_m: 10.0 * 1852.0,
            seeker_fov_deg: 45.0,
            lethal_radius_m: 12.0,
            max_tof: 100.0,
            min_speed: 300.0,
        }
    }
}

impl MissileParams {
    pub fn ref_area(&self) -> f64 {
        std::f64::consts::PI * self.diameter * self.diameter / 4.0
    }

    /// Zero-lift drag coefficient vs Mach for a slender finned body: flat
    /// subsonic, transonic rise to a peak just above Mach 1, then supersonic
    /// decay (the generic shape in missile aerodynamics texts).
    pub fn cd0(&self, mach: f64) -> f64 {
        let m = mach.max(0.0);
        if m < 0.8 {
            0.30
        } else if m < 1.1 {
            0.30 + (m - 0.8) / 0.3 * 0.40
        } else {
            0.70 * (1.1 / m).powf(0.65)
        }
    }

    /// Thrust (N) and mass flow (kg/s) at time `t` since launch.
    pub fn motor(&self, t: f64) -> (f64, f64) {
        let g0 = SConv::G0;
        if t < self.t_boost {
            let mdot = self.prop_boost / self.t_boost;
            (self.isp * g0 * mdot, mdot)
        } else if t < self.t_boost + self.t_sustain {
            let mdot = self.prop_sustain / self.t_sustain;
            (self.isp * g0 * mdot, mdot)
        } else {
            (0.0, 0.0)
        }
    }

    pub fn burn_time(&self) -> f64 {
        self.t_boost + self.t_sustain
    }
}

/// What the simulation knows about the missile's target this tick.
#[derive(Debug, Clone, Copy)]
pub struct TargetState {
    /// Position, GDM.
    pub pos: [f64; 3],
    /// Velocity, m/s.
    pub vel: [f64; 3],
    pub alive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Missile {
    pub id: u32,
    pub shooter_id: u32,
    pub target_id: u32,
    pub team: u8,
    /// Position, GDM.
    pub pos: [f64; 3],
    /// Heading of the velocity vector, degrees (0 = north / −z).
    pub hdg: f64,
    /// Speed, GDM/s.
    pub speed: f64,
    pub alive: bool,
    /// Own seeker has the target (A-pole passed).
    pub pitbull: bool,
    /// Receiving datalink updates from the shooter.
    pub has_support: bool,
    pub time_alive: f64,
    pub max_time: f64,
    pub hit_radius: f64,
    /// Seconds after launch when pitbull first engaged.
    pub pitbull_time: Option<f64>,
    /// Set when the missile dies without a hit.
    pub miss_cause: Option<String>,
    /// Legacy model, `MissileSupport::Track`: going active requires datalink support.
    #[serde(default)]
    pub needs_support: bool,
    #[serde(default)]
    pub model: MissileModel,
    /// Velocity, m/s (generic model).
    #[serde(default)]
    pub vel: [f64; 3],
    /// Mass, kg (generic model).
    #[serde(default)]
    pub mass: f64,
    /// Achieved lateral acceleration, m/s² (generic model, after autopilot lag).
    #[serde(default)]
    pub acc: [f64; 3],
    /// Target estimate used for guidance: position (GDM) and velocity (m/s).
    #[serde(default)]
    pub est_pos: [f64; 3],
    #[serde(default)]
    pub est_vel: [f64; 3],
    /// Mach number (generic model).
    #[serde(default)]
    pub mach: f64,
    /// Time spent opening on the target after pitbull, s.
    #[serde(default)]
    opening_time: f64,
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn to_m(p: [f64; 3]) -> [f64; 3] {
    scale(p, SConv::GDM2M)
}
fn to_gdm(p: [f64; 3]) -> [f64; 3] {
    scale(p, 1.0 / SConv::GDM2M)
}

/// Velocity vector (m/s) from heading (deg), pitch (deg) and speed (GDM/s).
pub fn velocity_ms(hdg_deg: f64, pitch_deg: f64, speed_gdm: f64) -> [f64; 3] {
    let v = speed_gdm * SConv::GDM2M;
    let (h, p) = (hdg_deg.to_radians(), pitch_deg.to_radians());
    [h.sin() * p.cos() * v, p.sin() * v, -h.cos() * p.cos() * v]
}

/// Minimum distance between two points moving linearly over one step.
fn closest_approach(r0: [f64; 3], r1: [f64; 3]) -> f64 {
    let d = sub(r1, r0);
    let dd = dot(d, d);
    if dd < 1e-12 {
        return norm(r0);
    }
    let t = (-dot(r0, d) / dd).clamp(0.0, 1.0);
    norm(add(r0, scale(d, t)))
}

/// Integration step for the generic model (s). Physics ticks are split to this.
pub const GENERIC_DT: f64 = 0.01;

impl Missile {
    /// Legacy constructor (B-ACE 2.0 kinematic missile).
    pub fn launch(
        id: u32,
        shooter_id: u32,
        target_id: u32,
        team: u8,
        pos: [f64; 3],
        hdg: f64,
    ) -> Self {
        Self {
            id,
            shooter_id,
            target_id,
            team,
            pos,
            hdg,
            speed: 900.0 * SConv::KNOT2GDM_S,
            alive: true,
            pitbull: false,
            has_support: true,
            time_alive: 0.0,
            max_time: 120.0,
            hit_radius: 0.5, // 50 m
            pitbull_time: None,
            miss_cause: None,
            needs_support: false,
            model: MissileModel::Legacy,
            vel: [0.0; 3],
            mass: 0.0,
            acc: [0.0; 3],
            est_pos: pos,
            est_vel: [0.0; 3],
            mach: 0.0,
            opening_time: 0.0,
        }
    }

    /// Launch with the selected model. `shooter_vel` in m/s; `target` seeds the estimate.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_model(
        model: MissileModel,
        id: u32,
        shooter_id: u32,
        target_id: u32,
        team: u8,
        pos: [f64; 3],
        hdg: f64,
        shooter_vel: [f64; 3],
        target: &TargetState,
    ) -> Self {
        let mut m = Self::launch(id, shooter_id, target_id, team, pos, hdg);
        if model == MissileModel::GenericMraam {
            let p = MissileParams::default();
            m.model = model;
            m.vel = shooter_vel;
            m.mass = p.mass_launch;
            m.max_time = p.max_tof;
            m.hit_radius = p.lethal_radius_m / SConv::GDM2M;
            m.speed = norm(shooter_vel) / SConv::GDM2M;
            m.est_pos = target.pos;
            m.est_vel = target.vel;
            m.mach = norm(shooter_vel) / atmosphere::at(pos[1] * SConv::GDM2M).sound;
        }
        m
    }

    pub fn lost_support(&mut self) {
        self.has_support = false;
    }

    pub fn recover_support(&mut self) {
        if !self.pitbull {
            self.has_support = true;
        }
    }

    /// Flight phase label for displays.
    pub fn phase(&self) -> &'static str {
        if self.model == MissileModel::Legacy {
            return "legacy";
        }
        let p = MissileParams::default();
        if self.time_alive < p.t_boost {
            "boost"
        } else if self.time_alive < p.burn_time() {
            "sustain"
        } else {
            "coast"
        }
    }

    /// Integrate one physics tick. Returns true if the target was hit.
    pub fn tick(&mut self, dt: f64, target: &TargetState) -> bool {
        if !self.alive {
            return false;
        }
        match self.model {
            MissileModel::Legacy => self.tick_legacy(dt, Some(target.pos), target.alive),
            MissileModel::GenericMraam => {
                let n = (dt / GENERIC_DT).ceil().max(1.0) as usize;
                let h = dt / n as f64;
                let mut tgt = *target;
                for _ in 0..n {
                    if self.step_generic(h, &tgt) {
                        return true;
                    }
                    if !self.alive {
                        return false;
                    }
                    // Advance the target linearly inside the tick for sub-step accuracy.
                    tgt.pos = add(tgt.pos, scale(tgt.vel, h / SConv::GDM2M));
                }
                false
            }
        }
    }

    fn die(&mut self, cause: &str) {
        self.alive = false;
        self.miss_cause = Some(cause.into());
    }

    fn step_generic(&mut self, h: f64, target: &TargetState) -> bool {
        let p = MissileParams::default();
        self.time_alive += h;
        if self.time_alive > p.max_tof {
            self.die("timeout");
            return false;
        }
        if !target.alive {
            self.die("target_gone");
            return false;
        }

        let pos_m = to_m(self.pos);
        let tgt_m = to_m(target.pos);
        let speed = norm(self.vel).max(1.0);
        let vhat = scale(self.vel, 1.0 / speed);

        // --- What the missile knows about the target -----------------------
        let true_rel = sub(tgt_m, pos_m);
        let true_range = norm(true_rel);
        let off_bore = (dot(true_rel, vhat) / true_range.max(1e-6))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let seeker_sees = true_range < p.seeker_range_m * 1.2 && off_bore <= p.seeker_fov_deg;
        if self.pitbull {
            if seeker_sees {
                self.est_pos = target.pos;
                self.est_vel = target.vel;
            } else {
                self.extrapolate_estimate(h);
            }
        } else {
            if self.has_support {
                // Datalink midcourse update from the shooter's track.
                self.est_pos = target.pos;
                self.est_vel = target.vel;
            } else {
                // Inertial midcourse: fly to where the target was predicted to go.
                self.extrapolate_estimate(h);
            }
            let est_range = norm(sub(to_m(self.est_pos), pos_m));
            if est_range < p.seeker_range_m && true_range < p.seeker_range_m && seeker_sees {
                self.pitbull = true;
                self.pitbull_time = Some(self.time_alive);
            }
        }

        // --- Guidance: pure PN on the estimate (lofted in midcourse) --------
        let est_m = to_m(self.est_pos);
        let mut aim = est_m;
        let horiz = ((est_m[0] - pos_m[0]).powi(2) + (est_m[2] - pos_m[2]).powi(2)).sqrt();
        if !self.pitbull {
            aim[1] += (0.25 * (horiz - 15_000.0)).clamp(0.0, p.loft_max_m);
        }
        let r = sub(aim, pos_m);
        let vrel = sub(self.est_vel, self.vel);
        let r2 = dot(r, r).max(1.0);
        let omega = scale(cross(r, vrel), 1.0 / r2);
        let mut a_cmd = scale(cross(omega, self.vel), p.nav_const);
        // Gravity compensation (lateral part of +g).
        let up = [0.0, SConv::G0, 0.0];
        a_cmd = add(a_cmd, sub(up, scale(vhat, dot(up, vhat))));

        // --- Airframe: lag, then aerodynamic and structural limits ---------
        let air = atmosphere::at(pos_m[1]);
        let q = 0.5 * air.density * speed * speed;
        let s = p.ref_area();
        self.acc = add(self.acc, scale(sub(a_cmd, self.acc), (h / p.tau).min(1.0)));
        self.acc = sub(self.acc, scale(vhat, dot(self.acc, vhat)));
        let a_lim = (p.cn_max * q * s / self.mass).min(p.g_max * SConv::G0);
        let a_mag = norm(self.acc);
        if a_mag > a_lim {
            self.acc = scale(self.acc, a_lim / a_mag);
        }

        // --- Forces: thrust, zero-lift + induced drag, gravity -------------
        let (thrust, mdot) = p.motor(self.time_alive);
        let mach = speed / air.sound;
        let mut cd0 = p.cd0(mach);
        if thrust > 0.0 {
            cd0 *= p.power_on_drag_factor;
        }
        let lat_force = self.mass * norm(self.acc);
        let drag = q * s * cd0 + p.k_induced * lat_force * lat_force / (q * s).max(1.0);
        let axial = (thrust - drag) / self.mass;
        let accel = add(add(scale(vhat, axial), self.acc), [0.0, -SConv::G0, 0.0]);

        let tgt_next = add(tgt_m, scale(target.vel, h));
        self.vel = add(self.vel, scale(accel, h));
        let new_pos_m = add(pos_m, scale(self.vel, h));
        self.mass = (self.mass - mdot * h).max(p.mass_launch - p.prop_boost - p.prop_sustain);

        // --- Fuze: closest approach over the step ---------------------------
        if closest_approach(sub(tgt_m, pos_m), sub(tgt_next, new_pos_m)) <= p.lethal_radius_m {
            self.pos = to_gdm(new_pos_m);
            self.alive = false;
            self.miss_cause = None;
            return true;
        }

        self.pos = to_gdm(new_pos_m);
        let v = norm(self.vel);
        self.speed = v / SConv::GDM2M;
        self.mach = v / air.sound;
        self.hdg = clamp_hdg(self.vel[0].atan2(-self.vel[2]).to_degrees());

        // --- Termination ----------------------------------------------------
        if new_pos_m[1] <= 0.0 {
            self.die("ground");
        } else if self.time_alive > p.burn_time() && v < p.min_speed.max(norm(target.vel) * 1.05) {
            self.die("energy");
        } else if self.pitbull {
            let closing =
                -dot(sub(tgt_next, new_pos_m), sub(target.vel, self.vel)) / true_range.max(1.0);
            self.opening_time = if closing < 0.0 {
                self.opening_time + h
            } else {
                0.0
            };
            if self.opening_time > 1.0 {
                self.die("overshoot");
            }
        }
        false
    }

    fn extrapolate_estimate(&mut self, h: f64) {
        self.est_pos = add(self.est_pos, scale(self.est_vel, h / SConv::GDM2M));
    }

    /// B-ACE 2.0 kinematic missile, unchanged.
    fn tick_legacy(&mut self, dt: f64, target_pos: Option<[f64; 3]>, target_alive: bool) -> bool {
        self.time_alive += dt;
        if self.time_alive > self.max_time {
            self.alive = false;
            let cause = if self.needs_support && !self.pitbull && !self.has_support {
                "lost_support"
            } else {
                "timeout"
            };
            self.miss_cause = Some(cause.into());
            return false;
        }

        if let Some(tp) = target_pos {
            let range = distance2d(self.pos, tp);
            if range < 10.0 * SConv::NM2GDM
                && !self.pitbull
                && (self.has_support || !self.needs_support)
            {
                self.pitbull = true;
                self.pitbull_time = Some(self.time_alive);
            }

            let can_guide = self.pitbull || self.has_support;
            if can_guide && target_alive {
                let desired = heading_to(self.pos, tp);
                let err = clamp_hdg(desired - self.hdg);
                let turn_rate = 120.0_f64.to_radians();
                let max_turn = turn_rate * dt;
                let turn = err.to_radians().clamp(-max_turn, max_turn);
                self.hdg = clamp_hdg(self.hdg + turn.to_degrees());

                if range < self.hit_radius {
                    self.alive = false;
                    self.miss_cause = None;
                    return true;
                }
            }
        }

        let rad = self.hdg.to_radians();
        self.pos[0] += rad.sin() * self.speed * dt;
        self.pos[2] += -rad.cos() * self.speed * dt;
        false
    }

    pub fn aspect_to_target(&self, target_pos: [f64; 3]) -> f64 {
        let brg = heading_to(self.pos, target_pos);
        aspect_angle(self.hdg, brg)
    }
}

/// Fly one generic missile against a target that follows `target_step`
/// (called every physics tick with dt, returns the new target state).
/// Used by tests and the WEZ table generator. Returns (hit, time of flight).
pub fn flyout(
    shooter_pos: [f64; 3],
    shooter_hdg: f64,
    shooter_vel: [f64; 3],
    target0: TargetState,
    target_step: impl FnMut(f64, &TargetState) -> TargetState,
) -> (bool, f64, String) {
    flyout_supported(
        shooter_pos,
        shooter_hdg,
        shooter_vel,
        target0,
        true,
        target_step,
    )
}

/// [`flyout`] with control over datalink support (`false`: fire and forget).
pub fn flyout_supported(
    shooter_pos: [f64; 3],
    shooter_hdg: f64,
    shooter_vel: [f64; 3],
    target0: TargetState,
    supported: bool,
    mut target_step: impl FnMut(f64, &TargetState) -> TargetState,
) -> (bool, f64, String) {
    let mut m = Missile::launch_model(
        MissileModel::GenericMraam,
        1,
        0,
        1,
        0,
        shooter_pos,
        shooter_hdg,
        shooter_vel,
        &target0,
    );
    if !supported {
        m.lost_support();
    }
    let mut tgt = target0;
    let dt = 0.05;
    while m.alive {
        if m.tick(dt, &tgt) {
            return (true, m.time_alive, String::new());
        }
        tgt = target_step(dt, &tgt);
    }
    (false, m.time_alive, m.miss_cause.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NM_GDM: f64 = SConv::NM2GDM;

    fn level(alt_ft: f64, z_nm: f64, hdg: f64, kt: f64) -> TargetState {
        TargetState {
            pos: [0.0, alt_ft * SConv::FT2GDM, z_nm * NM_GDM],
            vel: velocity_ms(hdg, 0.0, kt * SConv::KNOT2GDM_S),
            alive: true,
        }
    }

    fn straight(dt: f64, t: &TargetState) -> TargetState {
        TargetState {
            pos: add(t.pos, scale(t.vel, dt / SConv::GDM2M)),
            ..*t
        }
    }

    fn shot(alt_ft: f64, range_nm: f64, target_hdg: f64) -> (bool, f64, String) {
        let s = level(alt_ft, 0.0, 0.0, 600.0);
        let t = level(alt_ft, -range_nm, target_hdg, 600.0);
        flyout(s.pos, 0.0, s.vel, t, straight)
    }

    #[test]
    fn motor_impulse_matches_propellant() {
        let p = MissileParams::default();
        let mut impulse = 0.0;
        let mut burned = 0.0;
        let dt = 0.001;
        let mut t = 0.0;
        while t < p.burn_time() + 1.0 {
            let (f, mdot) = p.motor(t);
            impulse += f * dt;
            burned += mdot * dt;
            t += dt;
        }
        let expected = p.isp * SConv::G0 * (p.prop_boost + p.prop_sustain);
        assert!((impulse - expected).abs() / expected < 1e-2);
        assert!((burned - p.prop_boost - p.prop_sustain).abs() < 0.1);
    }

    #[test]
    fn peaks_near_mach_four_and_decays() {
        let s = level(30_000.0, 0.0, 0.0, 600.0);
        let t = level(30_000.0, -60.0, 180.0, 600.0);
        let mut m = Missile::launch_model(
            MissileModel::GenericMraam,
            1,
            0,
            1,
            0,
            s.pos,
            0.0,
            s.vel,
            &t,
        );
        let mut peak: f64 = 0.0;
        let mut tgt = t;
        for _ in 0..(20.0 / 0.05) as usize {
            m.tick(0.05, &tgt);
            tgt = straight(0.05, &tgt);
            peak = peak.max(m.mach);
        }
        assert!((3.2..4.6).contains(&peak), "peak Mach {peak:.2}");
        let at_20 = m.mach;
        for _ in 0..(20.0 / 0.05) as usize {
            m.tick(0.05, &tgt);
            tgt = straight(0.05, &tgt);
        }
        assert!(m.mach < at_20, "coasting missile must slow down");
    }

    #[test]
    fn hits_head_on_and_altitude_extends_range() {
        assert!(shot(25_000.0, 20.0, 180.0).0);
        let max_range = |alt| {
            (10..90)
                .rev()
                .find(|&r| shot(alt, r as f64, 180.0).0)
                .unwrap_or(0)
        };
        let low = max_range(10_000.0);
        let high = max_range(40_000.0);
        assert!(
            high as f64 > low as f64 * 1.4,
            "head-on Rmax low {low} high {high} NM"
        );
    }

    #[test]
    fn tail_chase_is_much_shorter_than_head_on() {
        let max_range = |hdg| {
            (2..90)
                .rev()
                .find(|&r| shot(25_000.0, r as f64, hdg).0)
                .unwrap_or(0)
        };
        let head = max_range(180.0);
        let tail = max_range(0.0);
        assert!(tail * 2 < head, "head {head} tail {tail}");
        assert!(tail >= 5, "tail {tail}");
    }

    #[test]
    fn running_away_late_defeats_a_long_shot() {
        // A long head-on shot hits a non-manoeuvring target but misses one that
        // turns cold at launch and runs (the no-escape-zone idea).
        let s = level(25_000.0, 0.0, 0.0, 600.0);
        let t0 = level(25_000.0, -30.0, 180.0, 600.0);
        assert!(flyout(s.pos, 0.0, s.vel, t0, straight).0);
        let run = |dt: f64, t: &TargetState| {
            let v = velocity_ms(0.0, 0.0, 600.0 * SConv::KNOT2GDM_S);
            TargetState {
                pos: add(t.pos, scale(v, dt / SConv::GDM2M)),
                vel: v,
                alive: true,
            }
        };
        let (hit, _, cause) = flyout(s.pos, 0.0, s.vel, t0, run);
        assert!(!hit, "a target running cold from 30 NM should escape");
        assert!(cause == "energy" || cause == "timeout", "{cause}");
    }
}

#[cfg(test)]
mod support_tests {
    use super::*;
    use crate::wez::gen::aircraft;

    /// Head-on shot; the target turns to the beam `turn_at` seconds after launch.
    fn beam_after(range_nm: f64, turn_at: f64, supported: bool) -> (bool, String) {
        let shooter = aircraft(25_000.0, [0.0; 3], 0.0);
        let mut target = aircraft(25_000.0, [0.0, 0.0, -range_nm * SConv::NM2GDM], 180.0);
        let state = |f: &crate::fighter::Fighter| TargetState {
            pos: f.pos,
            vel: velocity_ms(f.hdg, f.pitch, f.speed),
            alive: true,
        };
        let mut t = 0.0;
        let s_vel = velocity_ms(0.0, 0.0, shooter.speed);
        let (hit, _, cause) = flyout_supported(
            shooter.pos,
            0.0,
            s_vel,
            state(&target),
            supported,
            |dt, _| {
                t += dt;
                if t >= turn_at {
                    target.hdg_cmd = 90.0;
                    target.g_cmd = 7.0;
                }
                target.physics_tick(dt);
                state(&target)
            },
        );
        (hit, cause)
    }

    #[test]
    fn datalink_support_beats_fire_and_forget_against_a_late_turn() {
        // Inside kinematic reach a beam turn after launch is followed by a
        // supported missile but leaves a fire-and-forget shot chasing a stale
        // prediction. (Beyond ~25 NM the beam defeats both on energy.)
        let cases = [(20.0, 3.0), (25.0, 15.0), (16.0, 3.0), (12.0, 8.0)];
        let supported: u32 = cases
            .iter()
            .map(|&(r, t)| beam_after(r, t, true).0 as u32)
            .sum();
        let unsupported: u32 = cases
            .iter()
            .map(|&(r, t)| beam_after(r, t, false).0 as u32)
            .sum();
        assert_eq!(
            supported, 4,
            "supported shots inside the envelope should hit"
        );
        assert!(
            unsupported < supported,
            "fire-and-forget should do worse ({unsupported}/4)"
        );
        let (hit, cause) = beam_after(30.0, 3.0, true);
        assert!(
            !hit && cause == "energy",
            "a 30 NM shot is defeated by an early beam turn"
        );
    }

    #[test]
    #[ignore]
    fn print_beam_matrix() {
        for r in [12.0, 16.0, 20.0, 25.0, 30.0] {
            for turn in [3.0, 8.0, 15.0] {
                let a = beam_after(r, turn, true);
                let b = beam_after(r, turn, false);
                println!(
                    "R {r:>4} turn@{turn:>4}  supported {:?}  unsupported {:?}",
                    a, b
                );
            }
        }
    }

    #[test]
    fn fire_and_forget_still_finds_a_non_manoeuvring_target() {
        assert!(
            beam_after(25.0, 1e9, false).0,
            "inertial midcourse + seeker vs a straight target"
        );
    }
}
