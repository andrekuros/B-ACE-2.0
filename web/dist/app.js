import { CombatScene } from "./scene.js";
import { formatMeta } from "./hud.js";

const API = "/api";

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

const liveScene = new CombatScene($("live-canvas"), {
  hudRoot: $("live-viewport"),
  legendHost: $("live-viewport"),
});
const replayScene = new CombatScene($("replay-canvas"), {
  hudRoot: $("replay-viewport"),
  legendHost: $("replay-viewport"),
});

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

document.querySelectorAll(".cam-modes").forEach((group) => {
  group.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      group.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
      const scene = group.dataset.scene === "replay" ? replayScene : liveScene;
      scene.setCameraMode(btn.dataset.cam);
    });
  });
});

$("env-picker").addEventListener("change", () => {
  state.envIndex = Number($("env-picker").value) || 0;
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
  const snap = snaps[i];
  liveScene.apply(snap);
  $("live-meta").textContent = formatMeta(snap, `env ${i + 1}/${snaps.length}`);
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

$("run-exp").onclick = async () => {
  $("exp-out").textContent = "running…";
  const out = await api("/experiment", {
    method: "POST",
    body: JSON.stringify({
      cases: Number($("exp-cases").value),
      max_cycles: Number($("exp-cycles").value),
    }),
  });
  $("exp-out").textContent = JSON.stringify(out, null, 2);
};

refreshHealth();
refreshJobs();
refreshRuns();
setInterval(refreshHealth, 5000);
