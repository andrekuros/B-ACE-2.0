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
      num_agents: Number($("num-agents").value),
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

function fillExpTable(recipe, out) {
  const thead = $("exp-table").querySelector("thead");
  const tbody = $("exp-table").querySelector("tbody");
  tbody.innerHTML = "";
  if (recipe === "wez") {
    thead.innerHTML = "<tr><th>range</th><th>alt</th><th>aspect</th><th>n</th><th>hits</th><th>hit rate</th><th>RMax</th></tr>";
    (out.cells || []).forEach((c) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${c.range_nm}</td><td>${c.altitude_ft}</td><td>${c.aspect}</td><td>${c.n}</td><td>${c.hits}</td><td>${(c.hit_rate || 0).toFixed(2)}</td><td>${(c.analytic_rmax_nm || 0).toFixed(1)}</td>`;
      tbody.appendChild(tr);
    });
  } else {
    thead.innerHTML = "<tr><th>d_shot</th><th>l_crank</th><th>l_break</th><th>fitness</th><th>kills</th><th>deaths</th><th>mission</th></tr>";
    (out.last_generation || []).forEach((r) => {
      const g = r.genome || {};
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${(g.d_shot || 0).toFixed(3)}</td><td>${(g.l_crank || 0).toFixed(3)}</td><td>${(g.l_break || 0).toFixed(3)}</td><td>${(r.fitness || 0).toFixed(3)}</td><td>${(r.mean_kills || 0).toFixed(2)}</td><td>${(r.mean_deaths || 0).toFixed(2)}</td><td>${(r.mission_rate || 0).toFixed(2)}</td>`;
      tbody.appendChild(tr);
    });
  }
}

$("run-exp").onclick = async () => {
  $("exp-out").textContent = "running…";
  $("exp-summary").textContent = "";
  const recipe = $("exp-recipe").value;
  const body = {
    recipe,
    max_cycles: Number($("exp-cycles").value),
  };
  if (recipe === "fsm") {
    body.pop = Number($("exp-cases").value);
    body.generations = Number($("exp-gens").value);
  } else {
    body.repeats = Number($("exp-cases").value);
  }
  const out = await api("/experiment", {
    method: "POST",
    body: JSON.stringify(body),
  });
  $("exp-summary").textContent = out.summary || "";
  fillExpTable(recipe, out);
  $("exp-out").textContent = JSON.stringify(
    { recipe: out.recipe, summary: out.summary, params: out.params, elites: out.elites },
    null,
    2
  );
};

$("exp-recipe").onchange = () => {
  const fsm = $("exp-recipe").value === "fsm";
  $("exp-gen-row").classList.toggle("hidden", !fsm);
  $("exp-cases-label").childNodes[0].textContent = fsm ? "Pop " : "Repeats / pop ";
};

applyViewConfig(loadCfg());
refreshHealth();
refreshJobs();
refreshRuns();
setInterval(refreshHealth, 5000);
