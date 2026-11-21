# What B-ACE 2.0 simulates

A model card: what is simulated, how agents see and act, and what is simplified. Every
element's open-literature source is listed in [PUBLIC_REFERENCES.md](PUBLIC_REFERENCES.md).
Weapon and aircraft numbers are notional. Units: positions in GDM (metres / 100), headings
in degrees with 0 = north (−z), altitudes reported in feet.

## Time

Physics runs at `phy_fps` = 20 Hz; missiles integrate internally at 100 Hz. One agent
decision repeats its action for `action_repeat` = 20 ticks, so one decision is one simulated
second. `max_cycles` counts decisions.

## Atmosphere

U.S. Standard Atmosphere 1976 to 20 km: density and speed of sound drive missile drag,
Mach number and manoeuvre limits.

## Aircraft (`env.aircraft_model`)

- `energy` (default): top speed 650 kt × (0.8 + 0.2·alt/25 kft). Speed recovers toward it
  (time constant 25 s), bleeds with the square of the turn load factor (0.19 m/s² per g²)
  and trades with climb (−g·sin γ). Floor 45% of top speed. Slower aircraft turn tighter.
- `constant_speed`: B-ACE 2.0 behaviour; manoeuvring costs nothing.
- Turn rate limited by commanded g (max 9 g, less above 25 kft). Pitch tracks the commanded
  flight level between 1,000 and 50,000 ft. Six missiles each.

## Radar, datalink, warning

- Perfect detection inside 50 NM horizontal range, ±60° azimuth, +40/−20° elevation.
- `share_tracks`: a team pools detections (an enemy seen by any ally is a track for all).
- Missile warning (`env.missile_warning`, default on): an aircraft knows the bearing of an
  enemy missile whose seeker is active against it (what a radar warning receiver reports).
  Scripted pilots defend on it.

## Missiles (`env.missile_model`)

### `generic_mraam` (default): notional generic medium-range missile

| | |
|---|---|
| Mass | 150 kg at launch, 30 kg boost + 20 kg sustain propellant |
| Motor | 4 s boost, 8 s sustain, Isp 245 s (≈18 kN then ≈6 kN) |
| Body | 178 mm diameter |
| Drag | zero-lift `C_D0(M)`: 0.30 subsonic, peak 0.70 at Mach 1.1, then ∝ M^−0.65; ×0.8 while burning; induced `k·C_N²`, k = 0.12 |
| Manoeuvre | lateral acceleration ≤ min(25·q·S/m, 40 g): thin air at altitude limits it |
| Guidance | pure proportional navigation, N = 4, first-order lag 0.25 s, gravity compensation, loft up to 8 km above the target estimate in midcourse |
| Midcourse | datalink updates while supported, otherwise inertial extrapolation of the last estimate |
| Terminal | active seeker acquires within 10 NM and ±45° (the A-pole), tracks while the target stays in its gimbal |
| Fuze | closest approach within 12 m over each 10 ms step |
| Ends | `energy` (below 300 m/s or the target's speed after burnout), `overshoot`, `ground`, `timeout` (100 s), `target_gone` |

Peak speed is about Mach 4. Head-on Rmax grows from about 24 NM at 5,000 ft to 68 NM at
45,000 ft; tail-chase Rmax is a quarter (low) to two-fifths (high) of head-on at the same
altitude. The Lab's WEZ tab prints the full range card.

### `legacy`

The B-ACE 2.0 kinematic missile (constant 900 kt, horizontal pure pursuit, 120 s life,
50 m horizontal hit test), kept to reproduce earlier results.

### Datalink support (`env.missile_support`)

- `track` (default): updates only while the shooter is alive and tracks the target (own
  radar or shared datalink); regained when the track returns. Without support the generic
  missile flies to the last predicted point and can still acquire with its seeker, so
  support matters against a target that manoeuvres after launch, not against one that
  flies straight. The legacy missile without support flies straight and cannot go active.
- `launch`: perfect updates from launch until the seeker is active.

## Weapon engagement zone

Agents observe `Rmax` and the no-escape range `Rnez` for every track, in both directions
(our envelope against it, its envelope against us).

- Generic missile: looked up in `crates/bace-core/data/wez_generic_mraam.json`, built by
  flying the missile (`cargo run --release -p bace-core --example gen_wez`). Axes: shooter
  altitude and target altitude (5–45 kft), target aspect (0° hot to 180° cold), off-boresight
  (0–60°). `Rmax`: target holds course and speed. `Rnez`: target turns cold at launch and
  runs. Interpolated multilinearly; mirror geometries keep the shorter range. A test checks
  the table against fresh fly-outs, and the Lab's WEZ tab and Diagnostics measure it against
  launched missiles.
- Legacy missile: the B-ACE 2.0 analytic envelope.

Launches need a radar track, so shots beyond 50 NM are not possible even where `Rmax` is
larger.

## Scripted pilots (`baseline1`)

| Parameter | Meaning |
|-----------|---------|
| `d_shot` | Shoot when range ≤ `d_shot` × Rmax and the target is within 30° of the nose. The turn-in starts at 1.15 × `d_shot`. |
| `l_crank` | Crank (hold the target 40° off the nose) when the target's threat factor exceeds it. |
| `l_break` | Break (turn away, held at least 8 s) when the threat factor reaches it, +0.5 while supporting a missile. |

Threat and offensive factors are `Rmax/range` outside the envelope, 1 to 2 between `Rmax`
and `Rnez`, and 2 inside `Rnez`. A missile warning always triggers a break (missile on the
tail). The crank/support side is chosen once per engagement. `beh_noise` jitters the three
thresholds every decision; `0` keeps the FSM deterministic.

Pilot sets:
- **Reference** (`configs/reference`, `make_env(opponent="ref_balanced")` etc.): aggressive
  (0.40, 1.5, 2.0), balanced (0.60, 1.5, 2.0), conservative (0.75, 1.2, 1.9). Each stops a
  non-manoeuvring striker and beats a random policy; in 1v1 they beat one another in a cycle.
- **Searched** (`configs/baselines`): profiles from the FSM search recipe.
- **FSM-B1**: the paper's thresholds (1.04, 1.06, 1.05), tuned to the earlier envelope. With
  the flyout-derived WEZ they shoot just outside Rmax and break early, so they are very
  passive.
- `duck` flies straight to its target; `fire_once` launches one missile at the first track.

## Shot log

`outcome()["shots"]` lists every missile of both teams: launch time and range, the shooter's
`Rmax`/`Rnez` against the target at launch, shooter and target altitude, target aspect,
shooter speed, A-pole, F-pole, time of flight and result (`hit`, a miss cause, or
`in_flight` if the episode ended first).

## Observations, actions, rewards

The Simulate tab's observation inspector lists every field with its meaning. In short: own
state (9), then 13 values per enemy and 6 per ally. Undetected enemies get fixed placeholders
(`dist = −1`, `dist_target = 0.5`, everything else 0). Actions are
`[d_heading, altitude_level, g, fire]` in [−1, 1] (`altitude_level` is an absolute flight
level: −1 → the 1,000 ft floor, 0 → 25,000 ft, 1 → 50,000 ft). Reward terms follow Table 3
of the paper and are configurable under `env.rewards`.

## Reproducing B-ACE 2.0 results

```json
{"env": {"missile_model": "legacy", "missile_support": "launch", "aircraft_model": "constant_speed"}}
```

restores 2.0 physics. Scripted pilots still use the corrected FSM and the missile-warning
defence (see CHANGELOG).

## Known limitations

1. Perfect sensors: geometric detection without noise, clutter, radar cross-section or
   electronic warfare; no Doppler notch effect on radars or seekers.
2. Point-mass missile and aircraft: no attitude dynamics, no fuel.
3. Scripted pilots manoeuvre in the horizontal plane; altitude stays at the spawn level
   unless an RL agent commands it.
4. The WEZ has no minimum range; very short shots at high off-boresight or high altitude
   can miss even though they are inside `Rmax`.
5. Scripted episodes are identical across seeds unless `beh_noise` or `rnd_offset_range` is
   set.
