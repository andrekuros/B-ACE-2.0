//! Prints the characteristic numbers of the generic missile (for docs and calibration).
//! cargo run --release -p bace-core --example missile_card
use bace_core::missile::{flyout, velocity_ms, TargetState};
use bace_core::SConv;

fn st(alt_ft: f64, z_nm: f64, hdg: f64, kt: f64) -> TargetState {
    TargetState {
        pos: [0.0, alt_ft * SConv::FT2GDM, z_nm * SConv::NM2GDM],
        vel: velocity_ms(hdg, 0.0, kt * SConv::KNOT2GDM_S),
        alive: true,
    }
}

fn main() {
    let straight = |dt: f64, t: &TargetState| TargetState {
        pos: [
            t.pos[0] + t.vel[0] * dt / 100.0,
            t.pos[1] + t.vel[1] * dt / 100.0,
            t.pos[2] + t.vel[2] * dt / 100.0,
        ],
        ..*t
    };
    println!("alt_ft  aspect  Rmax_NM  TOF_s");
    for alt in [5_000.0, 15_000.0, 25_000.0, 35_000.0, 45_000.0] {
        for (name, hdg) in [("head", 180.0), ("beam", 90.0), ("tail", 0.0)] {
            let mut best = (0.0, 0.0);
            let mut r = 1.0;
            while r < 110.0 {
                let s = st(alt, 0.0, 0.0, 600.0);
                let (hit, tof, _) = flyout(s.pos, 0.0, s.vel, st(alt, -r, hdg, 600.0), straight);
                if hit {
                    best = (r, tof);
                }
                r += 1.0;
            }
            println!("{alt:>6.0}  {name:>6}  {:>7.0}  {:>5.0}", best.0, best.1);
        }
    }
}
