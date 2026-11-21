# Public references and data provenance

B-ACE is built to be shared without restriction. Every model element below comes from
open literature or from first-principles physics, and every number that describes a
weapon or an aircraft is **notional**: chosen to be of the order publicly reported for
its class, never taken from a specific system's data. The simulator contains no
classified, export-controlled, proprietary, or doctrinal material. Tactics are generic
concepts that are widely described in open books, papers, and articles.

If you extend the model, please keep to the same rule and add your source here.

## Atmosphere

| Element | Source | Use in B-ACE |
|---|---|---|
| Temperature, pressure, density, speed of sound vs altitude | U.S. Standard Atmosphere, 1976 (NOAA/NASA/USAF); summarized in [Wikipedia](https://en.wikipedia.org/wiki/U.S._Standard_Atmosphere) and [course notes](https://pages.hmc.edu/spjut/AdvRoc/PDF/StandardAtmosphere.pdf) | `atmosphere.rs`: troposphere lapse (−6.5 K/km) to 11 km, isothermal above, geometric→geopotential conversion. Unit-tested against tabulated values. |

## Missile (`MissileModel::GenericMraam`)

| Element | Source | Use in B-ACE |
|---|---|---|
| Point-mass air-to-air missile for combat simulation: thrust and mass loss, drag split into a zero-lift term and a manoeuvre-dependent term, maximum acceleration, autopilot time constant, PN guidance, kill radius | NASA TM-109057, *The Analysis of a Generic Air-to-Air Missile Simulation Model* (1994), [NTRS](https://ntrs.nasa.gov/api/citations/19940031931/downloads/19940031931.pdf) | Model structure. Same navigation constant (4) and time constant (0.25 s); lethal radius of the same order (12 m vs 10 m). |
| Proportional navigation; navigation constant typically 3–5 | NPTEL *Guidance of Missiles*, [lecture on PN](https://archive.nptel.ac.in/content/storage2/courses/101108054/module8/lecture22.pdf); Tiwari et al., *Defence Science Journal* 55(3), 2005, [PDF](https://publications.drdo.gov.in/ojs/index.php/dsj/article/download/1992/1029); Zarchan, *Tactical and Strategic Missile Guidance* (AIAA) | Pure PN, `a = N · Ω × V`, with lateral gravity compensation. |
| Drag build-up (zero-lift drag vs Mach with a transonic peak, reduced base drag with the motor burning, induced drag growing with normal force), boost–sustain propulsion | Fleeman, *Tactical Missile Design* / *Missile Design Guide* (AIAA); Fleeman, *Maximizing Missile Flight Performance* (Georgia Tech, public slides, [SlideShare](https://www.slideshare.net/slideshow/maximizing-missile-flight-performance-ppt/272561912)); NSWC TR 81-156, *Aerodynamic Design Manual for Tactical Weapons* (1981), [DTIC](https://apps.dtic.mil/sti/pdfs/ADA109180.pdf) | `MissileParams::cd0`, `power_on_drag_factor`, `k_induced`, `cn_max`; thrust from `Isp · g0 · ṁ`. |
| Class magnitudes for a medium-range AAM: launch mass ~150–160 kg, ~178 mm body, peak speed ~Mach 4, boost–sustain motor, inertial midcourse with datalink updates, active radar terminal phase | Public encyclopaedic description of the class, e.g. [Wikipedia: AIM-120 AMRAAM](https://en.wikipedia.org/wiki/AIM-120_AMRAAM) | Orders of magnitude only. The notional missile (150 kg, 178 mm, peaks near Mach 4) is not tuned to any system's published range. |
| Seeker activation ("pitbull", the A-pole) and datalink support | Open descriptions of active-radar missile employment, e.g. [SimHQ](https://www.simhq.com/_air/air_099c.html) and the article cited under Tactics | Datalink updates while the shooter tracks; inertial extrapolation otherwise; seeker acquires within 10 NM and a ±45° gimbal. |

All values are in `crates/bace-core/src/missile.rs` (`MissileParams`) and shown in the Lab (WEZ tab).

## Weapon engagement zone

| Element | Source | Use in B-ACE |
|---|---|---|
| Launch acceptability region / WEZ concept; `Rmax` as the longest range that hits a non-manoeuvring target; no-escape range against an evading target | Dantas et al., *Weapon Engagement Zone Maximum Launch Range Estimation Using a Deep Neural Network*, [arXiv:2111.04474](https://arxiv.org/abs/2111.04474); Veda Inc., *An Introduction to Launch Acceptability Regions* (1967), [DTIC AD0816286](https://apps.dtic.mil/sti/pdfs/AD0816286.pdf); [Aerospaceweb: LAR](https://aerospaceweb.org/question/weapons/q0250.shtml) | Definitions. `Rnez` uses an immediate cold turn at break g. |
| WEZ inputs: shooter and target altitude, target aspect, off-boresight | arXiv:2111.04474 (input variables of its WEZ model) | Table axes in `data/wez_generic_mraam.json`. |
| Envelope tables computed by missile fly-out (rather than hand formulas) | NASA TM-109057 (launch envelope vs. fly-out comparison); [arXiv:2610.32864](https://arxiv.org/abs/2610.32864) (fly-out-based WEZ) | `examples/gen_wez.rs` searches launch ranges with the same missile code the simulator runs; a test keeps the table in sync. |

## Aircraft

| Element | Source | Use in B-ACE |
|---|---|---|
| Point-mass energy: thrust excess recovers speed, induced drag grows with load factor squared, climb trades speed | Standard aircraft-performance texts (e.g. Anderson, *Aircraft Performance and Design*); energy-manoeuvrability discussion in Shaw (below) | `AircraftModel::Energy`: recovery time constant 25 s, bleed 0.19 m/s² per g², −g·sin γ in climbs. |
| Radar coverage | Generic fighter radar geometry | 50 NM, ±60° azimuth, +40/−20° elevation. Notional. |

## Tactics and scripted behaviour

| Element | Source | Use in B-ACE |
|---|---|---|
| BVR vocabulary and principles: maximum range vs no-escape zone, F-pole (range at missile arrival), A-pole (range when the missile goes active), crank (offset to keep the target on the radar gimbal limit while supporting a missile), notch, turning cold, defending at missile warning | Higby, *Promise and Reality: Beyond Visual Range (BVR) Air-to-Air Combat*, Air War College (2005); summarized in [MigFlug: BVR Air Combat Explained](https://migflug.com/afterburner/bvr-air-combat-explained-beyond-visual-range-f-pole-no-escape-zone/); Shaw, *Fighter Combat: Tactics and Maneuvering*, Naval Institute Press (1985), ISBN 978-0870210594 | FSM states (search, engage with crank, missile support, evade); missile-warning defence; F-pole and A-pole logged on every hit, miss and pitbull event. |
| Scripted agents parameterised by shot range, crank and break thresholds; diverse opponents found by evolutionary search; F-score and bootstrap confidence intervals | Kuroswiski, Wu & Passaro, *Enhancing BVR Air Combat Agent Development with Attention-Driven Reinforcement Learning*; Kuroswiski et al. (validation of scripted BVR agents) | `baseline1` FSM, the FSM search recipe, the Lab's Evaluate metrics. |
| Reference profiles (`configs/reference`) | Derived here from the principles above and checked by cross-play | Three non-dominated pilots that stop a non-manoeuvring striker, beat a random policy, and beat one another in a cycle. |

## What is deliberately not modelled

Electronic warfare, radar cross-section, Doppler notch effects on radars and seekers,
specific identification and rules of engagement, fuel, weapon-specific envelopes and any
real procedures. These are out of scope, and several are sensitive; keep them out.
