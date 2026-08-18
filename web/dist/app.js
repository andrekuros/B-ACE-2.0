import { CombatScene } from "./scene.js";
import { DEFAULT_VIEW_CONFIG } from "./models.js";
import { formatMeta } from "./hud.js";

const API = "/api";
const CFG_KEY = "bace-view-config";

const state = {
  view: "live",
  jobId: null,
  ws: null,
  episode: null,
  replayIdx: 0,
  playing: false,
  envIndex: 0,
  snapshots: [],
  exp: null,
  expSort: { key: "seed", dir: 1 },
};

function $(id) {
  return document.getElementById(id);
}

function loadCfg() {
  try {
    return { ...DEFAULT_VIEW_CONFIG, ...JSON.parse(localStorage.getItem(CFG_KEY) || "{}") };
  } catch {
    return { ...DEFAULT_VIEW_CONFIG };
  }
}

function saveCfg(cfg) {
  localStorage.setItem(CFG_KEY, JSON.stringify(cfg));
}

const liveScene = new CombatScene($("live-canvas"), {
  hudRoot: $("live-viewport"),
  legendHost: $("live-viewport"),
  split: true,
});
const replayScene = new CombatScene($("replay-canvas"), {
  hudRoot: $("replay-viewport"),
  legendHost: $("replay-viewport"),
  split: false,
});
replayScene.setConfig({ layout: "focus" });

liveScene.onFocus = (index) => {
  state.envIndex = index;
  $("env-picker").value = String(index);
  setLayout("focus");
};

async function api(path, opts) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function switchView(name) {
  state.view = name;
  document.querySelectorAll("nav button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === name);
  });
  ["live", "replay", "experiment"].forEach((v) => {
    $(`view-${v}`).classList.toggle("hidden", v !== name);
  });
  requestAnimationFrame(() => {
    liveScene.resize();
    replayScene.resize();
  });
}

document.querySelectorAll("nav button").forEach((b) => {
  b.addEventListener("click", () => switchView(b.dataset.view));
});

function applyViewConfig(cfg) {
  liveScene.setConfig(cfg);
  $("layout-seg").querySelectorAll("button").forEach((b) => {
    b.classList.toggle("active", b.dataset.layout === cfg.layout);
  });
  document.querySelectorAll('.cam-modes[data-scene="live"] button').forEach((b) => {
    b.classList.toggle("active", b.dataset.cam === cfg.camera);
  });
  $("cfg-radar").checked = cfg.radar;
  $("cfg-tracks").checked = cfg.tracks;
  $("cfg-wez").checked = cfg.wez;
  $("cfg-labels").checked = cfg.labels;
  $("cfg-stems").checked = cfg.stems;
  $("cfg-ac-scale").value = cfg.aircraftScale;
  $("cfg-msl-scale").value = cfg.missileScale;
  $("cfg-alt-scale").value = cfg.altScale;
  $("back-split").classList.toggle("hidden", cfg.layout !== "focus");
  saveCfg(cfg);
  showLiveSnapshot();
}

function currentCfg() {
  return { ...liveScene.config };
}

function setLayout(layout) {
  const cfg = currentCfg();
  cfg.layout = layout;
  if (layout === "split" && cfg.camera === "chase") cfg.camera = "orbit";
  applyViewConfig(cfg);
}

$("layout-seg").querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("click", () => setLayout(btn.dataset.layout));
});
$("back-split").addEventListener("click", () => setLayout("split"));

document.querySelectorAll(".cam-modes").forEach((group) => {
  group.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const scene = group.dataset.scene === "replay" ? replayScene : liveScene;
      if (scene === liveScene && liveScene.config.layout === "split" && btn.dataset.cam === "chase") {
        setLayout("focus");
      }
      scene.setCameraMode(btn.dataset.cam);
      group.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
      if (scene === liveScene) saveCfg(liveScene.config);
    });
  });
});

["radar", "tracks", "wez", "labels", "stems"].forEach((key) => {
  $(`cfg-${key}`).addEventListener("change", (e) => {
    liveScene.setConfig({ [key]: e.target.checked });
    saveCfg(liveScene.config);
  });
});

$("cfg-ac-scale").addEventListener("input", (e) => {
  liveScene.setConfig({ aircraftScale: Number(e.target.value) });
  saveCfg(liveScene.config);
});
$("cfg-msl-scale").addEventListener("input", (e) => {
  liveScene.setConfig({ missileScale: Number(e.target.value) });
  saveCfg(liveScene.config);
});
$("cfg-alt-scale").addEventListener("input", (e) => {
  liveScene.setConfig({ altScale: Number(e.target.value) });
  saveCfg(liveScene.config);
});

$("env-picker").addEventListener("change", () => {
  state.envIndex = Number($("env-picker").value) || 0;
  liveScene.setFocusIndex(state.envIndex);
  if (liveScene.config.layout === "split") setLayout("focus");
  showLiveSnapshot();
});

function fillEnvPicker(n) {
  const sel = $("env-picker");
  if (sel.options.length === n) {
    state.envIndex = Math.min(Number(sel.value) || 0, Math.max(0, n - 1));
    return;
  }
  const prev = sel.value;
  sel.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `Env ${i + 1}`;
    sel.appendChild(opt);
  }
  const idx = Math.min(Number(prev) || 0, Math.max(0, n - 1));
  sel.value = String(idx);
  state.envIndex = idx;
}

function showLiveSnapshot() {
  const snaps = state.snapshots;
  if (!snaps.length) return;
  const i = Math.min(state.envIndex, snaps.length - 1);
  liveScene.setFocusIndex(i);
  liveScene.applyAll(snaps, i);
  const snap = snaps[i];
  $("live-meta").textContent = formatMeta(
    snap,
    liveScene.config.layout === "split"
      ? `split ${Math.min(4, snaps.length)}/${snaps.length}`
      : `focus env ${i + 1}/${snaps.length}`
  );
}

async function refreshHealth() {
  try {
    const h = await api("/health");
    $("health").textContent = `ok · v${h.version}`;
  } catch {
    $("health").textContent = "server offline";
  }
}

async function refreshJobs() {
  const jobs = await api("/jobs");
  const ul = $("job-list");
  ul.innerHTML = "";
  jobs.forEach((j) => {
    const li = document.createElement("li");
    li.textContent = `${j.id.slice(0, 8)} · ${j.num_envs} envs · ${j.status}`;
    li.onclick = () => connectJob(j.id);
    if (j.id === state.jobId) li.classList.add("active");
    ul.appendChild(li);
  });
}

function connectJob(id) {
  state.jobId = id;
  if (state.ws) state.ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/api/jobs/${id}/ws`);
  state.ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    state.snapshots = msg.snapshots || [];
    fillEnvPicker(state.snapshots.length);
    showLiveSnapshot();
  };
  refreshJobs();
}

$("start-job").onclick = async () => {
  const job = await api("/jobs", {
    method: "POST",
    body: JSON.stringify({
      num_envs: Number($("num-envs").value),
      max_cycles: Number($("max-cycles").value),
      record: true,
      blue_behavior: "baseline1",
      red_behavior: "duck",
    }),
  });
  await refreshJobs();
  connectJob(job.id);
};

async function refreshRuns() {
  const runs = await api("/runs");
  const ul = $("run-list");
  ul.innerHTML = "";
  runs.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = `${r.run_id.slice(0, 8)} · ${r.end} · ${r.total_steps} steps`;
    li.onclick = () => loadRun(r.run_id);
    ul.appendChild(li);
  });
}

async function loadRun(id) {
  state.episode = await api(`/runs/${id}`);
  state.replayIdx = 0;
  $("scrubber").max = Math.max(0, (state.episode.steps || []).length - 1);
  $("scrubber").value = 0;
  renderReplay();
}

function renderReplay() {
  const ep = state.episode;
  if (!ep || !ep.steps.length) return;
  const step = ep.steps[state.replayIdx];
  replayScene.apply(step.snapshot);
  const agent = Object.keys(step.obs || {})[0];
  const bd = step.reward_breakdowns?.[agent];
  $("replay-inspect").textContent =
    formatMeta(step.snapshot, `replay ${state.replayIdx + 1}/${ep.steps.length}`) +
    `\nagent=${agent}\nreward=${step.rewards?.[agent]}\n` +
    `breakdown=${JSON.stringify(bd, null, 2)}`;
}

$("refresh-runs").onclick = refreshRuns;
$("scrubber").oninput = (e) => {
  state.replayIdx = Number(e.target.value);
  renderReplay();
};
$("play-replay").onclick = () => {
  state.playing = !state.playing;
  $("play-replay").textContent = state.playing ? "Pause" : "Play";
};

setInterval(() => {
  if (!state.playing || !state.episode) return;
  const max = (state.episode.steps || []).length - 1;
  if (state.replayIdx >= max) {
    state.playing = false;
    $("play-replay").textContent = "Play";
    return;
  }
  const speed = Number($("replay-speed").value) || 1;
  state.replayIdx = Math.min(max, state.replayIdx + Math.max(1, Math.round(speed)));
  $("scrubber").value = state.replayIdx;
  renderReplay();
}, 200);

const END_COLOR = {
  RedKilled: "#3ecf8e",
  BlueKilled: "#ff6b5a",
  MutualKill: "#c4b5fd",
  MaxCycles: "#ffd166",
  RedMission: "#4aa3ff",
};

function renderExperiment(out) {
  state.exp = out;
  const n = out.cases || 0;
  const cards = [
    ["cases", n],
    ["win rate", `${((out.win_rate || 0) * 100).toFixed(0)}%`],
    ["red killed", out.red_killed ?? 0],
    ["blue killed", out.blue_killed ?? 0],
    ["mutual", out.mutual_kill ?? 0],
    ["timeout", out.timeouts ?? 0],
    ["mean steps", (out.mean_steps || 0).toFixed(0)],
  ];
  $("exp-summary").innerHTML = cards
    .map(([k, v]) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");

  const counts = {
    red_killed: out.red_killed || 0,
    blue_killed: out.blue_killed || 0,
    mutual_kill: out.mutual_kill || 0,
      max_cycles: out.timeouts || 0,
  };
  $("exp-bars").innerHTML = Object.entries(counts)
    .map(([k, v]) => {
      const pct = n ? (100 * v) / n : 0;
      return `<div class="bar-row"><span>${k}</span><div class="bar-track"><div class="bar-fill ${k}" style="width:${pct}%"></div></div><span>${v}</span></div>`;
    })
    .join("");

  const rows = out.results || [];
  const xs = rows.map((r) => Number(r.d_shot) || 0);
  const ys = rows.map((r) => Number(r.steps) || 0);
  const minX = Math.min(...xs, 0.7);
  const maxX = Math.max(...xs, 0.85);
  const minY = 0;
  const maxY = Math.max(...ys, 1);
  const svg = $("exp-scatter");
  const dots = rows
    .map((r) => {
      const x = 24 + ((Number(r.d_shot) - minX) / (maxX - minX || 1)) * 280;
      const y = 160 - ((Number(r.steps) - minY) / (maxY - minY || 1)) * 140;
      const c = END_COLOR[r.end] || "#8aa0b5";
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="${c}" opacity="0.9"><title>seed ${r.seed} d_shot=${r.d_shot} ${r.end}</title></circle>`;
    })
    .join("");
  svg.innerHTML = `
    <rect x="0" y="0" width="320" height="180" fill="transparent"/>
    <text x="12" y="16" fill="#8aa0b5" font-size="10" font-family="IBM Plex Mono">steps</text>
    <text x="250" y="174" fill="#8aa0b5" font-size="10" font-family="IBM Plex Mono">d_shot</text>
    ${dots}
  `;
  renderExpTable();
}

function renderExpTable() {
  const rows = [...(state.exp?.results || [])];
  const { key, dir } = state.expSort;
  rows.sort((a, b) => {
    const va = a[key];
    const vb = b[key];
    if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
    return String(va).localeCompare(String(vb)) * dir;
  });
  const tb = $("exp-table").querySelector("tbody");
  tb.innerHTML = rows
    .map(
      (r) =>
        `<tr><td>${r.seed}</td><td>${Number(r.d_shot).toFixed(2)}</td><td>${r.end}</td><td>${r.steps}</td><td>${r.blue_alive}</td><td>${r.red_alive}</td></tr>`
    )
    .join("");
}

$("exp-table").querySelectorAll("th").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (state.expSort.key === key) state.expSort.dir *= -1;
    else state.expSort = { key, dir: 1 };
    renderExpTable();
  });
});

$("run-exp").onclick = async () => {
  $("exp-summary").innerHTML = `<div class="card"><div class="k">status</div><div class="v">running…</div></div>`;
  const out = await api("/experiment", {
    method: "POST",
    body: JSON.stringify({
      cases: Number($("exp-cases").value),
      max_cycles: Number($("exp-cycles").value),
      blue_behavior: $("exp-blue").value,
      red_behavior: $("exp-red").value,
    }),
  });
  renderExperiment(out);
};

applyViewConfig(loadCfg());
refreshHealth();
refreshJobs();
refreshRuns();
setInterval(refreshHealth, 5000);
