//! Redesigned WEZ model for B-ACE 2.0.
//!
//! Geometric engagement envelope (not the legacy Godot Expression polynomials).
//! \(R_{\max}\) grows with reciprocal headings (closing-speed proxy) and altitude.

use crate::units::SConv;

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
