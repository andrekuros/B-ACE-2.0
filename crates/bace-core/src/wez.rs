//! Weapon engagement zone (WEZ): maximum launch range `Rmax` and no-escape
//! range `Rnez` for a shooter–target geometry.
//!
//! * Generic missile (default): looked up in a table produced by flying the
//!   missile model itself (`examples/gen_wez.rs`, stored in
//!   `data/wez_generic_mraam.json`) and interpolated multilinearly. `Rmax` is
//!   the longest range that hits a target holding course and speed; `Rnez` the
//!   longest range that still hits a target turning cold at launch and running.
//!   These follow the open-literature definitions (launch acceptability region,
//!   "no escape" range) cited in `docs/PUBLIC_REFERENCES.md`.
//! * Legacy: the B-ACE 2.0 analytic envelope, kept for the legacy missile.

use crate::missile::MissileModel;
use crate::units::SConv;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Weapon engagement zone estimate in GDM.
#[derive(Debug, Clone, Copy)]
pub struct WezRanges {
    pub r_max: f64,
    pub r_nez: f64,
}

/// Evaluate WEZ for a shooter against a target geometry.
///
/// Inputs:
/// - `shooter_alt_gdm`: altitude in GDM
/// - `aspect_deg`: aspect angle of the target from the shooter (0 = on the nose)
/// - `angle_off_deg`: heading difference. Reciprocal headings (`|angle_off| ≈ 180`)
///   are a closing-speed proxy and **increase** \(R_{\max}\); co-speed/tail
///   (`|angle_off| ≈ 0`) is the shortest envelope.
pub fn evaluate(shooter_alt_gdm: f64, aspect_deg: f64, angle_off_deg: f64) -> WezRanges {
    // Base ranges at 25k ft reference (~76.2 GDM).
    let alt_ft = shooter_alt_gdm / SConv::FT2GDM;
    let alt_factor = (0.6 + 0.4 * (alt_ft / 25000.0).clamp(0.2, 1.5)).clamp(0.4, 1.4);

    let aspect = aspect_deg.abs().min(180.0);
    let aoff = angle_off_deg.abs().min(180.0);
    // 0 = co-speed / tail chase, 1 = reciprocal / head-on.
    let closing = (aoff / 180.0).clamp(0.0, 1.0);
    // Mild on-the-nose bonus; recipe analytic args use aspect = 0.
    let nose = 1.0 - 0.12 * (aspect / 180.0).clamp(0.0, 1.0);

    // At 25 kft, aspect=0: head (aoff=180) ≈ 40 NM, beam (90) ≈ 17 NM, tail (0) ≈ 9 NM.
    let r_max_nm = (9.0 + 31.0 * closing * closing) * alt_factor * nose;
    let r_nez_nm = r_max_nm * (0.35 + 0.15 * closing);

    WezRanges {
        r_max: (r_max_nm * SConv::NM2GDM).max(0.01),
        r_nez: (r_nez_nm * SConv::NM2GDM).max(0.01),
    }
}

/// Flyout-derived WEZ lookup table (ranges in NM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WezTable {
    pub model: String,
    pub shooter_alt_ft: Vec<f64>,
    pub target_alt_ft: Vec<f64>,
    /// Target aspect: angle between the target's heading and its line of sight
    /// to the shooter (0 = pointing at the shooter, 180 = cold).
    pub target_aspect_deg: Vec<f64>,
    /// Off-boresight: target bearing from the shooter's nose.
    pub off_boresight_deg: Vec<f64>,
    /// Indexed `[shooter_alt][target_alt][target_aspect][off_boresight]`, row-major.
    pub rmax_nm: Vec<f64>,
    pub rnez_nm: Vec<f64>,
}

static TABLE: OnceLock<WezTable> = OnceLock::new();

/// The embedded table for the generic missile.
pub fn table() -> &'static WezTable {
    TABLE.get_or_init(|| {
        serde_json::from_str(include_str!("../data/wez_generic_mraam.json"))
            .expect("embedded WEZ table is valid JSON")
    })
}

/// Bracketing index and weight for `x` on a sorted axis (clamped at the ends).
fn bracket(axis: &[f64], x: f64) -> (usize, usize, f64) {
    if axis.len() == 1 || x <= axis[0] {
        return (0, 0, 0.0);
    }
    let last = axis.len() - 1;
    if x >= axis[last] {
        return (last, last, 0.0);
    }
    let i = axis
        .windows(2)
        .position(|w| x >= w[0] && x <= w[1])
        .unwrap_or(0);
    let w = (x - axis[i]) / (axis[i + 1] - axis[i]);
    (i, i + 1, w)
}

impl WezTable {
    pub fn idx(&self, s: usize, t: usize, a: usize, o: usize) -> usize {
        ((s * self.target_alt_ft.len() + t) * self.target_aspect_deg.len() + a)
            * self.off_boresight_deg.len()
            + o
    }

    /// Multilinear interpolation of (Rmax, Rnez) in NM.
    pub fn lookup(
        &self,
        shooter_ft: f64,
        target_ft: f64,
        aspect: f64,
        off_bore: f64,
    ) -> (f64, f64) {
        let bs = bracket(&self.shooter_alt_ft, shooter_ft);
        let bt = bracket(&self.target_alt_ft, target_ft);
        let ba = bracket(&self.target_aspect_deg, aspect.abs().min(180.0));
        let bo = bracket(&self.off_boresight_deg, off_bore.abs().min(180.0));
        let (mut rmax, mut rnez) = (0.0, 0.0);
        for (si, sw) in [(bs.0, 1.0 - bs.2), (bs.1, bs.2)] {
            for (ti, tw) in [(bt.0, 1.0 - bt.2), (bt.1, bt.2)] {
                for (ai, aw) in [(ba.0, 1.0 - ba.2), (ba.1, ba.2)] {
                    for (oi, ow) in [(bo.0, 1.0 - bo.2), (bo.1, bo.2)] {
                        let w = sw * tw * aw * ow;
                        if w == 0.0 {
                            continue;
                        }
                        let k = self.idx(si, ti, ai, oi);
                        rmax += w * self.rmax_nm[k];
                        rnez += w * self.rnez_nm[k];
                    }
                }
            }
        }
        (rmax, rnez)
    }
}

/// WEZ of `model` for a shooter at `shooter_alt_gdm` against a target at
/// `target_alt_gdm`. `off_bore_deg`: target bearing off the shooter's nose;
/// `target_aspect_deg`: target heading relative to its line of sight to the
/// shooter (0 = hot); `angle_off_deg`: heading difference (legacy model only).
pub fn evaluate_model(
    model: MissileModel,
    shooter_alt_gdm: f64,
    target_alt_gdm: f64,
    off_bore_deg: f64,
    target_aspect_deg: f64,
    angle_off_deg: f64,
) -> WezRanges {
    match model {
        MissileModel::Legacy => evaluate(shooter_alt_gdm, off_bore_deg, angle_off_deg),
        MissileModel::GenericMraam => {
            let (rmax, rnez) = table().lookup(
                shooter_alt_gdm / SConv::FT2GDM,
                target_alt_gdm / SConv::FT2GDM,
                target_aspect_deg,
                off_bore_deg,
            );
            WezRanges {
                r_max: (rmax * SConv::NM2GDM).max(0.01),
                r_nez: (rnez.min(rmax) * SConv::NM2GDM).max(0.01),
            }
        }
    }
}

/// Threat / offensive factors used by FSM (range relative to envelopes).
pub fn offensive_factor(range_gdm: f64, own_r_max: f64, own_r_nez: f64) -> f64 {
    if range_gdm <= own_r_nez {
        2.0
    } else if range_gdm <= own_r_max {
        1.0 + (own_r_max - range_gdm) / (own_r_max - own_r_nez + 1e-6)
    } else {
        own_r_max / (range_gdm + 1e-6)
    }
}

pub fn threat_factor(range_gdm: f64, enemy_r_max: f64, enemy_r_nez: f64) -> f64 {
    offensive_factor(range_gdm, enemy_r_max, enemy_r_nez)
}

/// Flyout search used to build and verify the WEZ table (see `examples/gen_wez.rs`).
pub mod gen {
    use crate::config::{BehConfig, Behavior, Mission, RewardsConfig};
    use crate::fighter::Fighter;
    use crate::geometry::{clamp_hdg, heading_to};
    use crate::missile::{flyout, velocity_ms, TargetState};
    use crate::units::SConv;

    pub const SHOOTER_ALT: [f64; 5] = [5_000.0, 15_000.0, 25_000.0, 35_000.0, 45_000.0];
    pub const TARGET_ALT: [f64; 5] = [5_000.0, 15_000.0, 25_000.0, 35_000.0, 45_000.0];
    pub const ASPECT: [f64; 7] = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0];
    pub const OFF_BORE: [f64; 3] = [0.0, 30.0, 60.0];
    pub const MAX_NM: f64 = 120.0;

    pub fn aircraft(alt_ft: f64, pos: [f64; 3], hdg: f64) -> Fighter {
        let mut f = Fighter::new(
            1,
            1,
            [pos[0], alt_ft * SConv::FT2GDM, pos[2]],
            hdg,
            [0.0; 3],
            Behavior::External,
            Mission::Dca,
            BehConfig::default(),
            false,
            RewardsConfig::default(),
            &[],
            vec![],
        );
        f.speed = f.max_speed * Fighter::altitude_speed_factor(f.pos[1]);
        f.hdg_cmd = f.hdg;
        f.level_cmd = f.pos[1];
        f
    }

    /// One shot at `range_nm`. `sign` mirrors the geometry; `evade` = cold turn at launch.
    pub fn shot(
        shooter_ft: f64,
        target_ft: f64,
        aspect: f64,
        off_bore: f64,
        sign: f64,
        evade: bool,
        range_nm: f64,
    ) -> bool {
        let shooter = aircraft(shooter_ft, [0.0; 3], 0.0);
        let ob = (off_bore * sign).to_radians();
        let r = range_nm * SConv::NM2GDM;
        let tpos = [ob.sin() * r, 0.0, -ob.cos() * r];
        // Target heading so that its nose is `aspect` degrees off the line to the shooter.
        let to_shooter = heading_to(tpos, [0.0; 3]);
        let thdg = clamp_hdg(to_shooter + aspect * sign);
        let mut target = aircraft(target_ft, tpos, thdg);
        if evade {
            target.hdg_cmd = clamp_hdg(to_shooter + 180.0);
            target.g_cmd = target.max_g * 0.8;
        }
        let state = |f: &Fighter| TargetState {
            pos: f.pos,
            vel: velocity_ms(f.hdg, f.pitch, f.speed),
            alive: true,
        };
        let t0 = state(&target);
        let s_vel = velocity_ms(shooter.hdg, 0.0, shooter.speed);
        let (hit, _, _) = flyout(shooter.pos, 0.0, s_vel, t0, |dt, _| {
            target.physics_tick(dt);
            state(&target)
        });
        hit
    }

    /// Longest range (NM, 0.5 NM resolution) that hits.
    pub fn max_range(
        shooter_ft: f64,
        target_ft: f64,
        aspect: f64,
        off_bore: f64,
        sign: f64,
        evade: bool,
    ) -> f64 {
        let hits = |r: f64| shot(shooter_ft, target_ft, aspect, off_bore, sign, evade, r);
        // Coarse scan downward, then bisect the bracket.
        let mut hi = MAX_NM;
        let mut lo = None;
        let mut probes: Vec<f64> = (1..=19).rev().map(|k| k as f64 * 6.0).collect();
        probes.extend([4.0, 2.0, 1.0]);
        for r in probes {
            if hits(r) {
                lo = Some(r);
                break;
            }
            hi = r;
        }
        let Some(mut lo) = lo else { return 0.0 };
        while hi - lo > 0.5 {
            let mid = 0.5 * (lo + hi);
            if hits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alt_25k() -> f64 {
        25_000.0 * SConv::FT2GDM
    }

    fn nm(w: WezRanges) -> f64 {
        w.r_max * SConv::GDM2NM
    }

    #[test]
    fn embedded_table_matches_the_missile() {
        // Spot-check cells across the table against fresh flyouts: the missile
        // must hit just inside the tabulated Rmax and miss well outside it.
        let t = table();
        assert_eq!(
            t.model, "generic_mraam",
            "run examples/gen_wez.rs to build the table"
        );
        let cells = [
            (2, 2, 0, 0),
            (0, 0, 3, 0),
            (4, 4, 6, 1),
            (1, 3, 1, 0),
            (3, 1, 5, 1),
        ];
        for (s, tt, a, o) in cells {
            let (sa, ta, asp, ob) = (
                t.shooter_alt_ft[s],
                t.target_alt_ft[tt],
                t.target_aspect_deg[a],
                t.off_boresight_deg[o],
            );
            let rmax = t.rmax_nm[t.idx(s, tt, a, o)];
            assert!(rmax > 2.0, "cell {s},{tt},{a},{o} is empty");
            assert!(
                gen::shot(sa, ta, asp, ob, 1.0, false, rmax - 1.0),
                "should hit inside Rmax {rmax} at {sa}/{ta}/{asp}/{ob}"
            );
            assert!(
                !gen::shot(sa, ta, asp, ob, 1.0, false, rmax + 3.0),
                "should miss outside Rmax {rmax} at {sa}/{ta}/{asp}/{ob}"
            );
        }
    }

    #[test]
    fn generic_wez_grows_with_altitude_and_rnez_is_inside_rmax() {
        let ft = SConv::FT2GDM;
        let low = evaluate_model(
            MissileModel::GenericMraam,
            10_000.0 * ft,
            10_000.0 * ft,
            0.0,
            0.0,
            180.0,
        );
        let high = evaluate_model(
            MissileModel::GenericMraam,
            40_000.0 * ft,
            40_000.0 * ft,
            0.0,
            0.0,
            180.0,
        );
        assert!(high.r_max > 1.5 * low.r_max);
        let tail = evaluate_model(
            MissileModel::GenericMraam,
            25_000.0 * ft,
            25_000.0 * ft,
            0.0,
            180.0,
            0.0,
        );
        let head = evaluate_model(
            MissileModel::GenericMraam,
            25_000.0 * ft,
            25_000.0 * ft,
            0.0,
            0.0,
            180.0,
        );
        assert!(head.r_max > 3.0 * tail.r_max);
        assert!(head.r_nez < head.r_max && (tail.r_nez - tail.r_max).abs() < 0.1 * tail.r_max);
    }

    #[test]
    fn head_on_longer_than_beam() {
        let head = evaluate(alt_25k(), 0.0, 180.0);
        let beam = evaluate(alt_25k(), 0.0, 90.0);
        assert!(head.r_max > beam.r_max);
        assert!(head.r_nez < head.r_max);
    }

    #[test]
    fn head_beam_tail_recipe_order() {
        let alt = alt_25k();
        let head = evaluate(alt, 0.0, 180.0);
        let beam = evaluate(alt, 0.0, 90.0);
        let tail = evaluate(alt, 0.0, 0.0);
        assert!(
            head.r_max > beam.r_max && beam.r_max > tail.r_max,
            "head={:.1} beam={:.1} tail={:.1} NM",
            nm(head),
            nm(beam),
            nm(tail)
        );
        assert!(
            (nm(head) - 40.0).abs() < 8.0,
            "head-on Rmax {:.1} NM, expected ~40",
            nm(head)
        );
        assert!(
            (nm(beam) - 15.0).abs() < 8.0,
            "beam Rmax {:.1} NM, expected ~15",
            nm(beam)
        );
        assert!(
            (nm(tail) - 9.0).abs() < 8.0,
            "tail Rmax {:.1} NM, expected ~9",
            nm(tail)
        );
    }
}
