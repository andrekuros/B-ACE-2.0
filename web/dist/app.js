const API = "/api";

const state = {
  view: "live",
  jobId: null,
  ws: null,
  episode: null,
  replayIdx: 0,
  playing: false,
  trails: new Map(),
};

function $(id) { return document.getElementById(id); }

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
}

document.querySelectorAll("nav button").forEach((b) => {
  b.addEventListener("click", () => switchView(b.dataset.view));
});

async function refreshHealth() {
  try {
    const h = await api("/health");
    $("health").textContent = `ok · v${h.version}`;
  } catch {
    $("health").textContent = "server offline";
  }
}

function worldToCanvas(pos, w, h) {
  // pos in GDM; map ±80 NM (~±1480 GDM) into a tile
  const scale = w / (160 * 18.52);
  return [w / 2 + pos[0] * scale, h / 2 + pos[2] * scale];
}

function drawSnapshots(canvas, snapshots) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const n = Math.max(1, (snapshots || []).length);
  const cols = n <= 1 ? 1 : 2;
  const rows = Math.ceil(n / cols);
  const tw = canvas.width / cols;
  const th = canvas.height / rows;

  (snapshots || []).forEach((snap, envIdx) => {
    const col = envIdx % cols;
    const row = Math.floor(envIdx / cols);
    const ox = col * tw;
    const oy = row * th;
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, tw, th);
    ctx.clip();
    ctx.strokeStyle = "rgba(138,160,181,0.25)";
    ctx.beginPath();
    ctx.moveTo(ox + tw / 2, oy);
    ctx.lineTo(ox + tw / 2, oy + th);
    ctx.moveTo(ox, oy + th / 2);
    ctx.lineTo(ox + tw, oy + th / 2);
    ctx.stroke();
    if (n > 1) {
      ctx.fillStyle = "rgba(138,160,181,0.7)";
      ctx.font = "11px IBM Plex Mono, monospace";
      ctx.fillText(`env ${envIdx}`, ox + 8, oy + 14);
    }
    (snap.fighters || []).forEach((f) => {
      const key = `${envIdx}-${f.id}`;
      let trail = state.trails.get(key);
      if (!trail) {
        trail = [];
        state.trails.set(key, trail);
      }
      const [lx, ly] = worldToCanvas(f.pos, tw, th);
      trail.push([ox + lx, oy + ly]);
      if (trail.length > 80) trail.shift();
      ctx.strokeStyle = f.team === 0 ? "rgba(74,163,255,0.45)" : "rgba(255,107,90,0.45)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      trail.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
      ctx.stroke();
      const r = (snap.fighters || []).length > 4 ? 3.5 : 5;
      ctx.fillStyle = f.team === 0 ? "#4aa3ff" : "#ff6b5a";
      ctx.globalAlpha = f.alive ? 1 : 0.25;
      ctx.beginPath();
      ctx.arc(ox + lx, oy + ly, r, 0, Math.PI * 2);
      ctx.fill();
      const rad = (f.hdg * Math.PI) / 180;
      ctx.strokeStyle = ctx.fillStyle;
      ctx.beginPath();
      ctx.moveTo(ox + lx, oy + ly);
      ctx.lineTo(ox + lx + Math.sin(rad) * 12, oy + ly - Math.cos(rad) * 12);
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    (snap.missiles || []).forEach((m) => {
      const [x, y] = worldToCanvas(m.pos, tw, th);
      ctx.fillStyle = m.pitbull ? "#ffd166" : "#c4b5fd";
      ctx.fillRect(ox + x - 2, oy + y - 2, 4, 4);
    });
    ctx.restore();
    ctx.strokeStyle = "rgba(138,160,181,0.35)";
    ctx.strokeRect(ox + 0.5, oy + 0.5, tw - 1, th - 1);
  });
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
  state.trails = new Map();
  if (state.ws) state.ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/api/jobs/${id}/ws`);
  state.ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    drawSnapshots($("live-canvas"), msg.snapshots);
    $("live-meta").textContent = `job ${msg.id.slice(0, 8)} · ${msg.status}\n` +
      (msg.snapshots || []).map((s, i) => `env${i}: step=${s.action_step} end=${s.end}`).join("\n");
  };
  refreshJobs();
}

$("start-job").onclick = async () => {
  const job = await api("/jobs", {
    method: "POST",
    body: JSON.stringify({
      num_envs: Number($("num-envs").value),
      num_agents: Number($("num-agents") ? $("num-agents").value : 1),
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
  state.trails = new Map();
  $("scrubber").max = Math.max(0, (state.episode.steps || []).length - 1);
  $("scrubber").value = 0;
  renderReplay();
}

function renderReplay() {
  const ep = state.episode;
  if (!ep || !ep.steps.length) return;
  const step = ep.steps[state.replayIdx];
  drawSnapshots($("replay-canvas"), [step.snapshot]);
  const agent = Object.keys(step.obs || {})[0];
  const bd = step.reward_breakdowns?.[agent];
  $("replay-inspect").textContent =
    `step ${step.action_step} / ${ep.meta.total_steps}\n` +
    `end=${ep.meta.end}\n` +
    `agent=${agent}\n` +
    `reward=${step.rewards?.[agent]}\n` +
    `breakdown=${JSON.stringify(bd, null, 2)}\n` +
    `own=${JSON.stringify(step.obs?.[agent]?.own, null, 2)}`;
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
  state.replayIdx += 1;
  $("scrubber").value = state.replayIdx;
  renderReplay();
}, 200);

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

$("exp-recipe").onchange = () => {
  const fsm = $("exp-recipe").value === "fsm";
  $("exp-gen-row").classList.toggle("hidden", !fsm);
  $("exp-cases-label").childNodes[0].textContent = fsm ? "Pop " : "Repeats / pop ";
};

refreshHealth();
refreshJobs();
refreshRuns();
setInterval(refreshHealth, 5000);
