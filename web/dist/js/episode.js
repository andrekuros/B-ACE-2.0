// Episode frames: { snapshot, agents: {name: {obs, flat_obs, reward, reward_breakdown}}, events: [] }.
import { acName, NM } from "./util.js";

export const EPISODE_FORMAT = "bace-episode/1";

export function serializeEpisode(frames, config, extra = {}) {
  return JSON.stringify({
    format: EPISODE_FORMAT,
    created_at: new Date().toISOString(),
    config,
    ...extra,
    frames,
  });
}

/** Older recordings have no simulated time; derive it from the decision index. */
function fillTime(frames, cfg) {
  const dt = (cfg?.env?.action_repeat || 20) / (cfg?.env?.phy_fps || 20);
  frames.forEach((f, i) => {
    if (f.snapshot.time_s == null || (i > 0 && !f.snapshot.time_s)) f.snapshot.time_s = (f.snapshot.action_step ?? i) * dt;
  });
  return frames;
}

/** Accept Lab files, Python recorder files, and server recordings (meta + steps). */
export function parseEpisode(json) {
  const d = typeof json === "string" ? JSON.parse(json) : json;
  if (Array.isArray(d.frames)) {
    const frames = d.frames.map((f) => ({ snapshot: f.snapshot, agents: f.agents || {}, events: f.events || [] }));
    fillTime(frames, d.config);
    if (!frames.some((f) => f.events.length)) deriveEvents(frames);
    return { frames, config: d.config || d.meta?.config || null, label: d.label || d.created_at || "episode" };
  }
  if (Array.isArray(d.steps)) {
    const frames = d.steps.map((s) => {
      const agents = {};
      for (const [k, obs] of Object.entries(s.obs || {})) {
        agents[k] = { obs, reward: s.rewards?.[k] ?? 0, reward_breakdown: s.reward_breakdowns?.[k] };
      }
      return { snapshot: s.snapshot, agents, events: s.events || [] };
    });
    fillTime(frames, d.meta?.config);
    if (!frames.some((f) => f.events.length)) deriveEvents(frames);
    return { frames, config: d.meta?.config || null, label: d.meta?.run_id || "recording" };
  }
  throw new Error("Not a B-ACE episode: expected a `frames` or `steps` array.");
}

/** Reconstruct launch/hit/miss events for recordings made before events existed. */
export function deriveEvents(frames) {
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1].snapshot, cur = frames[i].snapshot;
    const t = cur.time_s ?? i;
    const ids = new Set(prev.missiles.map((m) => m.id));
    const ev = [];
    for (const m of cur.missiles) {
      if (!ids.has(m.id)) ev.push({ t, step: cur.action_step, kind: "launch", actor: m.shooter_id || 0, target: m.target_id, missile: m.id, value: 0, detail: "" });
    }
    const now = new Set(cur.missiles.map((m) => m.id));
    for (const m of prev.missiles) {
      if (now.has(m.id)) continue;
      const tgt = cur.fighters.find((f) => f.id === m.target_id);
      const wasAlive = prev.fighters.find((f) => f.id === m.target_id)?.alive;
      const hit = wasAlive && tgt && !tgt.alive;
      ev.push({ t, step: cur.action_step, kind: hit ? "hit" : "miss", actor: m.shooter_id || 0, target: m.target_id, missile: m.id, value: 0, detail: hit ? "" : "unknown" });
    }
    if (cur.end !== "ongoing" && prev.end === "ongoing") ev.push({ t, step: cur.action_step, kind: "end", actor: 0, value: 0, detail: cur.end });
    frames[i].events = ev;
  }
}

const END_TEXT = {
  red_killed: "red force destroyed", RedKilled: "red force destroyed",
  blue_killed: "blue force destroyed", BlueKilled: "blue force destroyed",
  mutual_kill: "both forces destroyed", MutualKill: "both forces destroyed",
  max_cycles: "time limit reached", MaxCycles: "time limit reached",
  red_mission: "red striker reached its target", RedMission: "red striker reached its target",
};
export const endText = (e) => END_TEXT[e] || e;

const MISS_TEXT = {
  energy: "out of energy", overshoot: "overshot", ground: "hit the ground", timeout: "flight time ran out",
  target_gone: "target already down", lost_support: "lost datalink",
};

export function eventText(e, snap) {
  const a = acName(e.actor, snap);
  const t = e.target != null ? acName(e.target, snap) : "";
  switch (e.kind) {
    case "launch": return `${a} fired at ${t}${e.value ? `, ${e.value.toFixed(1)} NM` : ""}`;
    case "pitbull": return `${a}’s missile went active, ${e.value.toFixed(0)} s after launch${e.pole_nm ? `, A-pole ${e.pole_nm.toFixed(1)} NM` : ""}`;
    case "hit": return `${a} hit ${t}${e.value ? `, ${e.value.toFixed(0)} s flight` : ""}${e.pole_nm ? `, F-pole ${e.pole_nm.toFixed(1)} NM` : ""}`;
    case "miss": return `${a}’s missile missed ${t}${e.detail ? ` (${MISS_TEXT[e.detail] || e.detail.replace("_", " ")})` : ""}`;
    case "track_gained": return `${a} tracks ${t}`;
    case "track_lost": return `${a} lost ${t}`;
    case "end": return `Episode over: ${endText(e.detail)}`;
    default: return e.kind;
  }
}

/** Per-frame engagement series for one aircraft (ranges in NM). */
export function engagementSeries(frames, id) {
  const out = { t: [], range: [], ownMax: [], ownNez: [], enMax: [], enNez: [] };
  for (const fr of frames) {
    const f = fr.snapshot.fighters.find((x) => x.id === id);
    const tt = fr.snapshot.time_s ?? fr.snapshot.action_step;
    out.t.push(tt);
    const tr = f && f.alive ? (f.tracks || []).find((x) => x.id === f.hpt_id && x.detected) : null;
    out.range.push(tr ? tr.dist / NM : null);
    out.ownMax.push(tr ? tr.own_r_max / NM : null);
    out.ownNez.push(tr && tr.own_r_nez ? tr.own_r_nez / NM : null);
    out.enMax.push(tr && tr.enemy_r_max ? tr.enemy_r_max / NM : null);
    out.enNez.push(tr && tr.enemy_r_nez ? tr.enemy_r_nez / NM : null);
  }
  return out;
}

export const REWARD_KEYS = [
  ["mission", "Mission shaping", "ok"],
  ["keep_track", "Keep track", "#5f8fbf"],
  ["detect_loss", "Lost track", "#9a7bbf"],
  ["missile_fire", "Missile fired", "violet"],
  ["missile_no_fire", "Fire refused", "#9aa5a9"],
  ["missile_miss", "Missile missed", "ochre"],
  ["hit_enemy", "Hit enemy", "blue"],
  ["hit_own", "Was hit", "red"],
  ["terminal", "Terminal", "ink2"],
];
