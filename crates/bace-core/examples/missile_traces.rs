//! Missile fly-out traces as JSON (for figures).
//! cargo run --release -p bace-core --example missile_traces > traces.json
//!
//! Shooter at 600 kt heading north; target 30 NM ahead flying straight head-on
//! (and one case where it turns cold at launch). Samples every 0.5 s.
use bace_core::missile::{velocity_ms, Missile, TargetState};
use bace_core::{MissileModel, SConv};
use serde_json::json;

fn level(alt_ft: f64, z_nm: f64, hdg: f64, kt: f64) -> TargetState {
    TargetState {
        pos: [0.0, alt_ft * SConv::FT2GDM, z_nm * SConv::NM2GDM],
        vel: velocity_ms(hdg, 0.0, kt * SConv::KNOT2GDM_S),
        alive: true,
    }
}

fn trace(model: MissileModel, alt_ft: f64, range_nm: f64, target_hdg: f64) -> serde_json::Value {
    let s = level(alt_ft, 0.0, 0.0, 600.0);
    let mut tgt = level(alt_ft, -range_nm, target_hdg, 600.0);
    let mut m = Missile::launch_model(model, 1, 0, 1, 0, s.pos, 0.0, s.vel, &tgt);
    let dt = 0.05;
    let mut rows = Vec::new();
    let mut hit = false;
    let mut k = 0u32;
    while m.alive {
        if k % 10 == 0 {
            let speed_ms = if model == MissileModel::Legacy {
                m.speed * SConv::GDM2M
            } else {
                (m.vel[0].powi(2) + m.vel[1].powi(2) + m.vel[2].powi(2)).sqrt()
            };
            let dx = (tgt.pos[0] - m.pos[0]) * SConv::GDM2NM;
            let dz = (tgt.pos[2] - m.pos[2]) * SConv::GDM2NM;
            rows.push(json!({
                "t": m.time_alive,
                "mach": m.mach,
                "speed_ms": speed_ms,
                "alt_ft": m.pos[1] / SConv::FT2GDM,
                "downrange_nm": -m.pos[2] * SConv::GDM2NM,
                "range_to_target_nm": (dx * dx + dz * dz).sqrt(),
                "phase": m.phase(),
            }));
        }
        if m.tick(dt, &tgt) {
            hit = true;
            break;
        }
        tgt.pos = [
            tgt.pos[0] + tgt.vel[0] * dt / SConv::GDM2M,
            tgt.pos[1] + tgt.vel[1] * dt / SConv::GDM2M,
            tgt.pos[2] + tgt.vel[2] * dt / SConv::GDM2M,
        ];
        k += 1;
    }
    json!({
        "model": format!("{model:?}"),
        "launch_alt_ft": alt_ft,
        "range_nm": range_nm,
        "target": if target_hdg == 180.0 { "hot" } else { "cold" },
        "hit": hit,
        "tof_s": m.time_alive,
        "end": if hit { "hit".to_string() } else { m.miss_cause.clone().unwrap_or_default() },
        "samples": rows,
    })
}

fn main() {
    let mut cases = Vec::new();
    for alt in [10_000.0, 25_000.0, 40_000.0] {
        cases.push(trace(MissileModel::GenericMraam, alt, 30.0, 180.0));
    }
    cases.push(trace(MissileModel::GenericMraam, 25_000.0, 20.0, 0.0));
    cases.push(trace(MissileModel::GenericMraam, 25_000.0, 12.0, 0.0));
    cases.push(trace(MissileModel::Legacy, 25_000.0, 30.0, 180.0));
    println!("{}", serde_json::to_string(&cases).unwrap());
}
