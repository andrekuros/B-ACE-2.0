// Names and meanings of the flat observation (crates/bace-core/src/obs.rs).
export const OWN_FIELDS = [
  ["x", "East position / 3000 GDM"],
  ["z", "South position / 3000 GDM"],
  ["altitude", "Altitude / 150 GDM (≈ 49 kft)"],
  ["dist_target", "Distance to own mission target / 3000"],
  ["aspect_to_target", "Bearing to target relative to nose / 180°"],
  ["heading", "Heading / 180°"],
  ["speed", "Speed / max speed"],
  ["missiles", "Missiles left / 6"],
  ["supporting_missile", "1 while guiding a missile by datalink"],
];
export const ENEMY_FIELDS = [
  ["alt_diff", "Own minus enemy altitude / 150"],
  ["aspect", "Bearing to enemy relative to nose / 180°"],
  ["angle_off", "Enemy heading minus own heading / 180°"],
  ["dist", "Range / 3000 (−1 if not detected)"],
  ["dist_target", "Enemy distance to its target / 3000 (0.5 if not detected)"],
  ["own_r_max", "Our max launch range vs this enemy / 926"],
  ["own_r_nez", "Our no-escape range / 926"],
  ["enemy_r_max", "Enemy max launch range vs us / 926"],
  ["enemy_r_nez", "Enemy no-escape range vs us / 926"],
  ["threat_factor", "How deep we are in its envelope, minus 1"],
  ["offensive_factor", "How deep it is in our envelope, minus 1"],
  ["is_missile_support", "1 if our in-flight missile targets it"],
  ["detected", "1 if tracked by radar or datalink"],
];
export const ALLY_FIELDS = [
  ["alt_diff", "Own minus ally altitude / 150"],
  ["aspect", "Bearing to ally relative to nose / 180°"],
  ["angle_off", "Ally heading minus own heading / 180°"],
  ["dist", "Range to ally / 3000"],
  ["dist_target", "Ally distance to its target / 3000"],
  ["detected", "1 if ally state is shared by datalink"],
];

/** Rows: { group, name, desc, flat, value, placeholder } in flat-vector order. */
export function obsRows(obs) {
  const rows = [];
  let k = 0;
  for (const [n, d] of OWN_FIELDS) rows.push({ group: "Own aircraft", name: n, desc: d, flat: k++, value: obs.own[n] });
  obs.enemies.forEach((e, i) => {
    const ph = e.detected < 0.5;
    for (const [n, d] of ENEMY_FIELDS) rows.push({ group: `Enemy ${i + 1} (id ${e.id})`, name: n, desc: d, flat: k++, value: e[n], placeholder: ph && n !== "detected" });
  });
  obs.allies.forEach((a, i) => {
    const ph = a.detected < 0.5;
    for (const [n, d] of ALLY_FIELDS) rows.push({ group: `Ally ${i + 1} (id ${a.id})`, name: n, desc: d, flat: k++, value: a[n], placeholder: ph && n !== "detected" });
  });
  return rows;
}
