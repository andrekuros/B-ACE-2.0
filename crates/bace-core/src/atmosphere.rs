//! U.S. Standard Atmosphere 1976, lower two layers (0–20 km).
//!
//! Troposphere: T = 288.15 K − 0.0065 K/m · h up to 11 km; isothermal 216.65 K
//! above it. Pressure from the hydrostatic equation, density from the ideal gas
//! law. Public reference: U.S. Standard Atmosphere, 1976 (NOAA/NASA/USAF).

const T0: f64 = 288.15; // K
const P0: f64 = 101_325.0; // Pa
const L: f64 = 0.0065; // K/m
const R: f64 = 287.052_87; // J/(kg·K), specific gas constant of dry air
const G0: f64 = 9.806_65; // m/s²
const GAMMA: f64 = 1.4;
const H_TROPO: f64 = 11_000.0; // m
const T_TROPO: f64 = T0 - L * H_TROPO; // 216.65 K
const R_EARTH: f64 = 6_356_766.0; // m, effective radius used by USSA-1976

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Air {
    /// Temperature, K.
    pub temperature: f64,
    /// Static pressure, Pa.
    pub pressure: f64,
    /// Density, kg/m³.
    pub density: f64,
    /// Speed of sound, m/s.
    pub sound: f64,
}

/// Standard atmosphere at geometric altitude `h_m` (metres), clamped to 0–20 km.
pub fn at(h_m: f64) -> Air {
    let z = h_m.clamp(0.0, 20_000.0);
    // Layers are defined in geopotential altitude.
    let h = R_EARTH * z / (R_EARTH + z);
    let (t, p) = if h <= H_TROPO {
        let t = T0 - L * h;
        (t, P0 * (t / T0).powf(G0 / (R * L)))
    } else {
        let p11 = P0 * (T_TROPO / T0).powf(G0 / (R * L));
        (T_TROPO, p11 * (-G0 * (h - H_TROPO) / (R * T_TROPO)).exp())
    };
    Air {
        temperature: t,
        pressure: p,
        density: p / (R * t),
        sound: (GAMMA * R * t).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_published_table_values() {
        // USSA-1976 tabulated values by geometric altitude.
        let cases = [
            (0.0, 1.225, 340.29),
            (5_000.0, 0.736_43, 320.55),
            (11_000.0, 0.364_80, 295.15),
            (15_000.0, 0.194_76, 295.07),
        ];
        for (h, rho, a) in cases {
            let air = at(h);
            assert!(
                (air.density - rho).abs() / rho < 2e-3,
                "rho at {h}: {}",
                air.density
            );
            assert!((air.sound - a).abs() < 0.2, "a at {h}: {}", air.sound);
        }
    }
}
