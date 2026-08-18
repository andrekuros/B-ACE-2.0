const FSM_LABEL = {
  Search: "SEARCH",
  Engage: "ENGAGE",
  MissileSupport: "SUPPORT",
  Evade: "EVADE",
};

export function createLegend(host) {
  const el = document.createElement("div");
  el.className = "legend";
  el.innerHTML = `
    <div><span class="swatch blue"></span> Blue</div>
    <div><span class="swatch red"></span> Red</div>
    <div><span class="swatch gold"></span> HPT / WEZ</div>
    <div><span class="swatch violet"></span> Uplink missile</div>
    <div><span class="swatch amber"></span> Pitbull</div>
    <div class="hint">Radar volume · tracks · altitude stems</div>
  `;
  host.appendChild(el);
  return el;
}

export function createLabel(text) {
  const el = document.createElement("div");
  el.className = "ac-label";
  el.textContent = text;
  return el;
}

export function formatFighterLabel(f) {
  const name = f.agent_name || `AC-${f.id}`;
  const fsm = FSM_LABEL[f.fsm] || f.fsm || "";
  const msl = Number.isFinite(f.missiles) ? f.missiles : "?";
  return `${name}  ${fsm}  M${msl}`;
}

export function formatMeta(snapshot, extras = "") {
  if (!snapshot) return extras;
  const blue = (snapshot.fighters || []).filter((f) => f.team === 0);
  const red = (snapshot.fighters || []).filter((f) => f.team === 1);
  const ba = blue.filter((f) => f.alive).length;
  const ra = red.filter((f) => f.alive).length;
  const msl = (snapshot.missiles || []).length;
  const tracks = blue.reduce((n, f) => n + (f.tracks || []).filter((t) => t.detected).length, 0);
  return [
    `step ${snapshot.action_step} · ${snapshot.end}`,
    `blue ${ba}/${blue.length}  red ${ra}/${red.length}  msl ${msl}  tracks ${tracks}`,
    extras,
  ]
    .filter(Boolean)
    .join("\n");
}
