//! Generate the WEZ table of the generic missile by flyout simulation.
//!
//!     cargo run --release -p bace-core --example gen_wez > crates/bace-core/data/wez_generic_mraam.json
//!
//! For every (shooter altitude, target altitude, target aspect, off-boresight)
//! cell the launch range is searched for two target behaviours, following the
//! open-literature WEZ definitions:
//! * Rmax: the target holds course, speed and altitude.
//! * Rnez: at launch the target turns cold (puts the shooter on its tail) at its
//!   break g and runs; the longest range that still hits.
//! Shooter and target fly at the simulator's aircraft speed for their altitude.
//! When a cell has two mirror-image geometries the shorter (conservative) range
//! is kept. Search logic: `bace_core::wez::gen`.
use bace_core::wez::gen::{max_range, ASPECT, OFF_BORE, SHOOTER_ALT, TARGET_ALT};
use bace_core::wez::WezTable;
use rayon::prelude::*;

fn main() {
    let mut cells = Vec::new();
    for &s in &SHOOTER_ALT {
        for &t in &TARGET_ALT {
            for &a in &ASPECT {
                for &o in &OFF_BORE {
                    cells.push((s, t, a, o));
                }
            }
        }
    }
    let results: Vec<(f64, f64)> = cells
        .par_iter()
        .map(|&(s, t, a, o)| {
            let mirrored = o > 0.0 && a > 0.0 && a < 180.0;
            let signs: &[f64] = if mirrored { &[1.0, -1.0] } else { &[1.0] };
            let rmax = signs
                .iter()
                .map(|&g| max_range(s, t, a, o, g, false))
                .fold(f64::MAX, f64::min);
            let rnez = signs
                .iter()
                .map(|&g| max_range(s, t, a, o, g, true))
                .fold(f64::MAX, f64::min);
            (rmax, rnez.min(rmax))
        })
        .collect();
    let table = WezTable {
        model: "generic_mraam".into(),
        shooter_alt_ft: SHOOTER_ALT.to_vec(),
        target_alt_ft: TARGET_ALT.to_vec(),
        target_aspect_deg: ASPECT.to_vec(),
        off_boresight_deg: OFF_BORE.to_vec(),
        rmax_nm: results.iter().map(|r| r.0).collect(),
        rnez_nm: results.iter().map(|r| r.1).collect(),
    };
    println!("{}", serde_json::to_string(&table).unwrap());
}
