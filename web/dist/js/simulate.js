// Simulate tab: scenario form, stepping, playback, telemetry, inspector.
import { $, COLORS, NM, alpha, toFt, fmt, fmtTime, prng, clamp, toast, download, teamColor, acName, fitCanvas } from "./util.js";
import { PRESETS, buildScenario, pythonSnippet } from "./scenario.js";
import { TacticalChart } from "./tactical.js";
import { lineChart } from "./charts.js";
import { engagementSeries, eventText, endText, REWARD_KEYS, serializeEpisode } from "./episode.js";
import { obsRows } from "./obsdoc.js";

const SIDE_KEYS = ["pilot", "num_agents", "mission", "share_tracks", "init_x", "init_z", "init_y", "init_hdg", "rnd_xz", "beh_noise"];
const STATE_KEY = "bace-lab-scenario-v2";
const FSM_NAME = { Search: "Searching", Engage: "Engaging", MissileSupport: "Supporting missile", Evade: "Evading" };

export class SimulateTab {
  constructor(engine, book) {
    this.engine = engine;
    this.book = book;
    this.chart = new TacticalChart($("tactical"));
    this.chart.request = () => (this.dirty = true);
    this.chart.onSelect = (id) => { this.selected = id; this.telemetryDirty = true; };
    this.frames = [];
    this.cursor = 0;
    this.playing = false;
    this.speed = 20;
    this.acc = 0;
    this.sim = null;
    this.ended = false;
    this.mode = "live";
    this.keys = new Set();
    this.manual = { pendingFire: false, alt: null };
    this.state = this._loadState();
    this._buildForm();
    this._bindTransport();
    this._bindKeys();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // ---------------- scenario form ----------------
  _loadState() {
    try {
      const s = JSON.parse(localStorage.getItem(STATE_KEY) || "null");
      if (s?.blue && s?.red && s?.env) return s;
    } catch { /* fall through */ }
    return this._fromPreset(PRESETS[0]);
  }
  _fromPreset(p) {
    const rewards = structuredClone(this.engine.info.default_config.env.rewards);
    return { preset: p.id, blue: { ...p.blue }, red: { ...p.red }, env: { ...p.env, rewards } };
  }
  _saveState() {
    localStorage.setItem(STATE_KEY, JSON.stringify(this.state));
  }

  _buildForm() {
    const ps = $("preset");
    ps.innerHTML = PRESETS.map((p) => `<option value="${p.id}">${p.label}</option>`).join("") + `<option value="custom">Custom</option>`;
    ps.value = PRESETS.some((p) => p.id === this.state.preset) ? this.state.preset : "custom";
    ps.onchange = () => {
      const p = PRESETS.find((x) => x.id === ps.value);
      if (!p) return;
      this.state = this._fromPreset(p);
      this._fillForm();
      this.start();
    };
    this.refreshPilots();
    const rf = $("reward-fields");
    rf.innerHTML = Object.keys(this.engine.info.default_config.env.rewards)
      .map((k) => `<label class="field">${k.replace(/_factor$/, "").replaceAll("_", " ")}<input type="number" step="0.001" data-reward="${k}" /></label>`)
      .join("");
    document.querySelectorAll(".scenario input, .scenario select").forEach((el) => {
      if (el.id === "preset") return;
      el.addEventListener("change", () => { this._readForm(); $("preset").value = "custom"; this.state.preset = "custom"; this._saveState(); });
    });
    this._fillForm();
    $("apply-scenario").onclick = () => { this._readForm(); this.start(); };
    $("new-seed").onclick = () => { this._readForm(); this.state.env.seed = Number(this.state.env.seed) + 1; $("env-seed").value = this.state.env.seed; this.start(); };
    $("copy-json").onclick = () => this._copy(JSON.stringify(this.config(), null, 2), "Scenario JSON copied");
    $("copy-python").onclick = () => this._copy(pythonSnippet(this.config(), this.state, this.book), "Python snippet copied");
    $("import-json").onclick = () => this._import();
  }

  refreshPilots() {
    document.querySelectorAll('.side select[data-k="pilot"]').forEach((sel) => {
      const side = sel.closest(".side").dataset.side;
      sel.innerHTML = this.book.all().map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
      sel.value = this.state[side].pilot;
    });
  }

  _fillForm() {
    $("preset").value = PRESETS.some((p) => p.id === this.state.preset) ? this.state.preset : "custom";
    $("preset-note").textContent = PRESETS.find((p) => p.id === this.state.preset)?.note || "Your own scenario. Changes are kept in this browser.";
    for (const side of ["blue", "red"]) {
      const root = document.querySelector(`.side[data-side="${side}"]`);
      for (const k of SIDE_KEYS) {
        const el = root.querySelector(`[data-k="${k}"]`);
        if (el) el.value = String(this.state[side][k]);
      }
    }
    $("env-seed").value = this.state.env.seed;
    $("env-max").value = this.state.env.max_cycles;
    $("env-support").value = this.state.env.missile_support || "track";
    $("env-model").value = this.state.env.missile_model || "generic_mraam";
    document.querySelectorAll("[data-reward]").forEach((el) => (el.value = this.state.env.rewards?.[el.dataset.reward] ?? ""));
  }

  _readForm() {
    for (const side of ["blue", "red"]) {
      const root = document.querySelector(`.side[data-side="${side}"]`);
      for (const k of SIDE_KEYS) {
        const el = root.querySelector(`[data-k="${k}"]`);
        if (!el) continue;
        this.state[side][k] = el.type === "number" ? Number(el.value) : el.value;
      }
      this.state[side].num_agents = clamp(Number(this.state[side].num_agents) || 1, 1, 4);
    }
    this.state.env.seed = Number($("env-seed").value) || 0;
    this.state.env.max_cycles = Math.max(10, Number($("env-max").value) || 600);
    this.state.env.missile_support = $("env-support").value;
    this.state.env.missile_model = $("env-model").value;
    this.state.env.rewards = {};
    document.querySelectorAll("[data-reward]").forEach((el) => (this.state.env.rewards[el.dataset.reward] = Number(el.value)));
    $("preset-note").textContent = PRESETS.find((p) => p.id === this.state.preset)?.note || "Your own scenario. Changes are kept in this browser.";
  }

  config() {
    return buildScenario(this.state, this.book);
  }

  async _copy(text, msg) {
    try { await navigator.clipboard.writeText(text); toast(msg); } catch { download("bace_scenario.txt", text, "text/plain"); }
  }

  _import() {
    const dlg = $("import-dialog");
    $("import-text").value = JSON.stringify(this.config(), null, 2);
    dlg.showModal();
    dlg.onclose = () => {
      if (dlg.returnValue !== "ok") return;
      try {
        const cfg = JSON.parse($("import-text").value);
        this.startConfig(cfg, "Imported scenario");
      } catch (e) { toast(`Could not import: ${e.message}`); }
    };
  }

  /** Map a policy to every learner name from the pilot chosen for its side. */
  _policies(frame) {
    const pol = {};
    for (const name of Object.keys(frame.agents || {})) {
      const side = name.startsWith("red_") ? "red" : "blue";
      const p = this.book.get(this.state[side].pilot);
      pol[name] = p.rl ? p.policy : "idle";
      if (pol[name] === "manual" && name !== "agent_0") pol[name] = "idle";
    }
    return pol;
  }

  // ---------------- episode control ----------------
  start() {
    this._saveState();
    this.startConfig(this.config());
  }

  startConfig(cfg, label = null) {
    try {
      this.sim?.free();
      this.sim = this.engine.createSim(cfg);
      this.cfg = cfg;
      const f0 = this.sim.reset(Number(cfg.env?.seed ?? -1));
      this.mode = "live";
      this.frames = [f0];
      this.policies = label ? Object.fromEntries(Object.keys(f0.agents).map((k) => [k, "random"])) : this._policies(f0);
      this.rng = prng((Number(cfg.env?.seed) || 1) * 7919);
      this.cursor = 0;
      this.ended = f0.end !== "ongoing";
      this.acc = 0;
      this.manual = { pendingFire: false, alt: null };
      $("manual-help").hidden = !Object.values(this.policies).includes("manual");
      const blue = f0.snapshot.fighters.find((f) => f.team === 0);
      this.selected = blue?.id ?? null;
      this.chart.selected = this.selected;
      this.chart.userMoved = false;
      this.chart._fitted = false;
      this.setPlaying(true);
      this.telemetryDirty = true;
      if (label) toast(label);
    } catch (e) {
      toast(`Scenario rejected: ${e.message}`);
    }
  }

  /** Show a recorded episode (no stepping). */
  loadEpisode(ep) {
    this.sim?.free();
    this.sim = null;
    this.mode = "replay";
    this.cfg = ep.config;
    this.frames = ep.frames;
    this.cursor = 0;
    this.ended = true;
    this.selected = ep.frames[0]?.snapshot.fighters.find((f) => f.team === 0)?.id ?? null;
    this.chart.selected = this.selected;
    this.chart.userMoved = false;
    this.chart._fitted = false;
    $("manual-help").hidden = true;
    $("preset-note").textContent = `Showing a replay (${ep.frames.length} decisions). The form below is not this episode’s setup; press Start episode to go back to a live run.`;
    this.setPlaying(true);
    this.telemetryDirty = true;
  }

  _actions(frame) {
    const acts = {};
    for (const [name, pol] of Object.entries(this.policies)) {
      if (pol === "random") {
        const r = this.rng;
        acts[name] = [r() * 2 - 1, r() * 2 - 1, r() * 2 - 1, r() < 0.05 ? 1 : -1];
      } else if (pol === "manual") {
        const me = frame.snapshot.fighters.find((f) => f.agent_name === name);
        if (this.manual.alt == null && me) this.manual.alt = toFt(me.pos[1]);
        let turn = 0;
        if (this.keys.has("ArrowLeft")) turn -= 0.25;
        if (this.keys.has("ArrowRight")) turn += 0.25;
        if (this.keys.has("ArrowUp")) this.manual.alt = Math.min(50000, this.manual.alt + 2000);
        if (this.keys.has("ArrowDown")) this.manual.alt = Math.max(1000, this.manual.alt - 2000);
        const fire = this.manual.pendingFire ? 1 : -1;
        this.manual.pendingFire = false;
        acts[name] = [turn, (this.manual.alt - 25000) / 25000, turn ? 0.6 : -1, fire];
      } else {
        acts[name] = [0, (toFt(frame.snapshot.fighters.find((f) => f.agent_name === name)?.pos[1] ?? 25000 * 0.003048) - 25000) / 25000, -1, -1];
      }
    }
    return acts;
  }

  _stepSim() {
    if (!this.sim || this.ended) return false;
    const last = this.frames[this.frames.length - 1];
    const f = this.sim.step(this._actions(last));
    this.frames.push(f);
    if (f.end !== "ongoing") this.ended = true;
    return true;
  }

  /** Keep one frame ahead of the cursor so playback can interpolate. */
  _ensureAhead() {
    while (this.cursor + 1 >= this.frames.length && this._stepSim()) { /* step */ }
  }

  advance(n = 1) {
    for (let k = 0; k < n; k++) {
      this._ensureAhead();
      if (this.cursor + 1 < this.frames.length) this.cursor++;
      else { this.setPlaying(false); break; }
    }
    this.telemetryDirty = true;
  }

  setPlaying(p) {
    this.playing = p;
    $("play").textContent = p ? "Pause" : this.cursor + 1 >= this.frames.length && this.ended ? "Replay" : "Play";
  }

  // ---------------- input ----------------
  _bindTransport() {
    $("play").onclick = () => {
      if (!this.playing && this.ended && this.cursor + 1 >= this.frames.length) this.cursor = 0;
      this.setPlaying(!this.playing);
    };
    $("step").onclick = () => { this.setPlaying(false); this.advance(1); };
    $("speed").querySelectorAll("button").forEach((b) => {
      b.onclick = () => {
        this.speed = Number(b.dataset.speed);
        $("speed").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      };
    });
    $("scrubber").oninput = (e) => {
      this.cursor = clamp(Number(e.target.value), 0, this.frames.length - 1);
      this.acc = 0;
      this.telemetryDirty = true;
    };
    $("save-episode").onclick = () => {
      if (!this.frames.length) return;
      download(`bace_episode_seed${this.cfg?.env?.seed ?? 0}.json`, serializeEpisode(this.frames, this.cfg, { source: "B-ACE Lab" }));
    };
    $("fit-view").onclick = () => { this.chart.userMoved = false; this.chart.fitTo(this.frames[0]); };
    $("layers").querySelectorAll("[data-layer]").forEach((el) => {
      el.onchange = () => (this.chart.layers[el.dataset.layer] = el.checked);
    });
    $("toggle-3d").onclick = () => this._toggle3d();
    $("events").onclick = (e) => {
      const li = e.target.closest("li[data-i]");
      if (!li) return;
      this.cursor = Number(li.dataset.i);
      this.setPlaying(false);
      this.telemetryDirty = true;
    };
    window.addEventListener("resize", () => (this.telemetryDirty = true));
  }

  _bindKeys() {
    const manualKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " "];
    window.addEventListener("keydown", (e) => {
      if (!this.active || e.target.closest("input, select, textarea, dialog")) return;
      if (Object.values(this.policies || {}).includes("manual") && manualKeys.includes(e.key)) {
        e.preventDefault();
        if (e.key === " ") this.manual.pendingFire = true;
        else this.keys.add(e.key);
      } else if (e.key === " ") {
        e.preventDefault();
        $("play").click();
      } else if (e.key === ".") { $("step").click(); }
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.key));
  }

  async _toggle3d() {
    const host = $("three-host");
    if (!host.hidden) { host.hidden = true; $("toggle-3d").textContent = "3D"; return; }
    try {
      if (!this.scene3d) {
        const { CombatScene } = await import("../scene.js");
        host.hidden = false;
        this.scene3d = new CombatScene($("three-canvas"), { hudRoot: host, legendHost: null, split: false });
        this.scene3d.setConfig({ layout: "focus" });
      }
      host.hidden = false;
      $("toggle-3d").textContent = "2D";
      this.scene3d.resize();
    } catch (e) {
      host.hidden = true;
      toast(`3D view needs three.js from the CDN: ${e.message}`);
    }
  }

  // ---------------- render loop ----------------
  _loop(ts) {
    const dt = this._lastTs ? Math.min(0.1, (ts - this._lastTs) / 1000) : 0;
    this._lastTs = ts;
    if (this.active && this.frames.length) {
      if (this.playing) {
        this.acc += dt * this.speed;
        const n = Math.floor(this.acc);
        if (n > 0) { this.acc -= n; this.advance(Math.min(n, 400)); }
        if (this.playing) this._ensureAhead();
      }
      const frac = this.playing && this.speed <= 20 && this.cursor + 1 < this.frames.length ? this.acc : 0;
      if (!$("three-host").hidden && this.scene3d) this.scene3d.apply(this.frames[this.cursor].snapshot);
      else this.chart.draw(this.frames, this.cursor, frac);
      this._clock();
      if (this.telemetryDirty && ts - (this._lastTel || 0) > (this.playing ? 120 : 0)) {
        this._lastTel = ts;
        this.telemetryDirty = false;
        this.renderTelemetry();
      }
    }
    requestAnimationFrame(this._loop);
  }

  _clock() {
    const f = this.frames[this.cursor];
    const s = f.snapshot;
    const blue = s.fighters.filter((x) => x.team === 0), red = s.fighters.filter((x) => x.team === 1);
    const alive = (xs) => xs.filter((x) => x.alive).length;
    const state = this.mode === "replay" ? "Replay" : s.end === "ongoing" ? "Live" : "Finished";
    $("clock").innerHTML = `<b>${fmtTime(s.time_s ?? this.cursor)}</b>${state}, decision ${s.action_step}<br>` +
      `<span style="color:${COLORS.blue}">Blue ${alive(blue)}/${blue.length}</span>&ensp;<span style="color:${COLORS.red}">Red ${alive(red)}/${red.length}</span>` +
      (s.end !== "ongoing" ? `<br>${endText(s.end)}` : "");
    const sc = $("scrubber");
    sc.max = String(this.frames.length - 1);
    sc.value = String(this.cursor);
  }

  // ---------------- telemetry ----------------
  renderTelemetry() {
    const frames = this.frames;
    if (!frames.length) return;
    const frame = frames[this.cursor];
    const snap = frame.snapshot;
    const f = snap.fighters.find((x) => x.id === this.selected);
    const tNow = snap.time_s ?? this.cursor;
    const xFmt = (t) => fmtTime(t);
    this._ticks();
    this._altStrip(tNow);
    this._events();
    if (!f) { $("sel-title").textContent = "Select an aircraft"; return; }

    const kind = f.behavior === "external" ? "RL agent (external)" : f.behavior ? `Scripted, ${f.behavior.replace("_", " ")}` : f.agent_name ? "RL agent" : "Scripted";
    $("sel-title").innerHTML = `<span class="swatch ${f.team === 0 ? "blue" : "red"}"></span>${acName(f.id, snap)}`;
    $("sel-sub").textContent = f.mission ? `${kind}. Mission: ${f.mission === "striker" ? "strike target" : "defend"}.` : `${kind}. Recorded before 2.1: reward and mission details are not in this file.`;
    const hpt = f.tracks?.find((t) => t.id === f.hpt_id && t.detected);
    const kv = [
      ["State", !f.alive ? "Shot down" : f.behavior === "external" ? "Learning agent" : f.behavior === "baseline1" ? FSM_NAME[f.fsm] || f.fsm : f.behavior === "duck" ? "Flying to target" : FSM_NAME[f.fsm] || f.fsm],
      ["Altitude", `${Math.round(toFt(f.pos[1])).toLocaleString()} ft`],
      ["Speed", `${Math.round(f.speed * 194.4)} kt`],
      ["Heading", `${Math.round((f.hdg + 360) % 360)}°`],
      ["Missiles", `${f.missiles}${f.supporting_missile ? ", guiding one" : ""}`],
      ["Target", hpt ? `${acName(hpt.id, snap)} at ${fmt(hpt.dist / NM, 1)} NM` : "none"],
      ["Reward", fmt(f.reward ? Object.values(f.reward).reduce((a, b) => a + b, 0) : NaN, 3)],
      ["Tracks", `${(f.tracks || []).filter((t) => t.detected).length} of ${(f.tracks || []).length}`],
    ];
    $("sel-kv").innerHTML = kv.map(([k, v]) => `<span class="k">${k}</span><span>${v}</span>`).join("");

    // Engagement chart
    const es = engagementSeries(frames, f.id);
    const pts = (arr) => arr.map((v, i) => [es.t[i], v]);
    const band = (lo, hi) => {
      const L = [], H = [];
      lo.forEach((v, i) => { if (v != null && hi[i] != null) { L.push([es.t[i], v]); H.push([es.t[i], hi[i]]); } });
      return { lo: L, hi: H };
    };
    const markers = [];
    for (const fr of frames) for (const e of fr.events || []) {
      if (e.kind === "launch" && e.actor === f.id) markers.push({ x: e.t, color: COLORS.violet });
      if (e.kind === "launch" && e.target === f.id) markers.push({ x: e.t, color: COLORS.red, dash: [1, 3] });
      if (e.kind === "hit" && (e.actor === f.id || e.target === f.id)) markers.push({ x: e.t, color: COLORS.ochre, width: 2, dash: [] });
    }
    lineChart($("eng-chart"), {
      height: 170, xFmt, yLabel: "NM", yMin: 0, cursorX: tNow, markers, empty: "No target tracked yet",
      bands: [{ ...band(es.ownNez, es.ownMax), color: alpha(teamColor(f.team), 0.16) }],
      series: [
        { points: pts(es.enMax), color: f.team === 0 ? COLORS.red : COLORS.blue, width: 1.2, dash: [5, 3], label: "Their Rmax" },
        { points: pts(es.enNez), color: f.team === 0 ? COLORS.red : COLORS.blue, width: 1.2, label: "Their NEZ" },
        { points: pts(es.range), color: COLORS.ink, width: 2.2, label: "Range" },
      ],
      legend: true,
    });

    // Reward chart
    const series = [];
    const total = [];
    const comps = Object.fromEntries(REWARD_KEYS.map(([k]) => [k, []]));
    for (const fr of frames) {
      const g = fr.snapshot.fighters.find((x) => x.id === f.id);
      const t = fr.snapshot.time_s ?? 0;
      if (!g?.reward) continue;
      let sum = 0;
      for (const [k] of REWARD_KEYS) { sum += g.reward[k] || 0; comps[k].push([t, g.reward[k] || 0]); }
      total.push([t, sum]);
    }
    series.push({ points: total, color: COLORS.ink, width: 2.2, label: "Total" });
    for (const [k, label, color] of REWARD_KEYS) {
      if (comps[k].some((p) => Math.abs(p[1]) > 1e-9)) series.push({ points: comps[k], color: COLORS[color] || color, width: 1.2, label });
    }
    lineChart($("rew-chart"), { height: 150, xFmt, series, cursorX: tNow, legend: true });

    this._obs(frame, f);
  }

  _obs(frame, f) {
    const name = f.agent_name;
    const a = name ? frame.agents?.[name] : null;
    if (!a?.obs) {
      $("obs-size").textContent = "";
      $("obs-table").innerHTML = `<p class="hint">${f.behavior === "external" ? "No observation in this frame." : "Scripted aircraft don’t receive observations. Select an RL slot to inspect its input vector."}</p>`;
      return;
    }
    const prev = this.frames[this.cursor - 1]?.agents?.[name]?.obs;
    $("obs-size").title = "Greyed rows are placeholders for aircraft not currently tracked. Hover a row for its meaning.";
    const rows = obsRows(a.obs);
    const prevRows = prev ? obsRows(prev) : null;
    $("obs-size").textContent = `${rows.length} values`;
    let html = "<table>";
    let group = null;
    rows.forEach((r, i) => {
      if (r.group !== group) { group = r.group; html += `<tr><th colspan="3">${group}</th></tr>`; }
      const bad = !Number.isFinite(r.value) || Math.abs(r.value) > 5;
      // Highlight only detection flips, the moments that change what the agent knows.
      const flip = r.name === "detected" && prevRows && prevRows[i] && prevRows[i].value !== r.value;
      const cls = [bad ? "flag" : "", r.placeholder ? "ph" : "", flip ? "chg" : ""].join(" ");
      html += `<tr class="${cls}" title="${r.desc}"><td class="i">${r.flat}</td><td>${r.name}</td><td class="v">${Number.isFinite(r.value) ? r.value.toFixed(3) : r.value}</td></tr>`;
    });
    $("obs-table").innerHTML = html + "</table>";
  }

  _events() {
    const items = [];
    const snap = this.frames[this.cursor].snapshot;
    this.frames.forEach((fr, i) => {
      for (const e of fr.events || []) {
        if (e.kind === "track_gained" || e.kind === "track_lost") {
          if (e.actor !== this.selected) continue;
        }
        items.push({ e, i });
      }
    });
    const key = `${items.length}:${this.cursor}:${this.selected}`;
    if (key === this._evKey) return;
    this._evKey = key;
    $("events").innerHTML = items.length
      ? items.map(({ e, i }) => `<li data-i="${i}" class="${i > this.cursor ? "future" : ""}"><time>${fmtTime(e.t)}</time><span class="${e.kind}">${eventText(e, snap)}</span></li>`).join("")
      : `<li class="future"><span>Nothing yet. Launches, hits and misses appear here; track changes for the selected aircraft too.</span></li>`;
  }

  _ticks() {
    const c = $("event-ticks");
    const { ctx, w, h } = fitCanvas(c, 8);
    ctx.clearRect(0, 0, w, h);
    const n = Math.max(1, this.frames.length - 1);
    this.frames.forEach((fr, i) => {
      for (const e of fr.events || []) {
        const col = e.kind === "launch" ? COLORS.violet : e.kind === "hit" ? COLORS.red : e.kind === "end" ? COLORS.ink : null;
        if (!col) continue;
        ctx.fillStyle = col;
        ctx.fillRect(8 + (i / n) * (w - 16) - 1, 0, 2, 8);
      }
    });
  }

  _altStrip(tNow) {
    const series = [];
    const byId = new Map();
    for (const fr of this.frames) {
      const t = fr.snapshot.time_s ?? 0;
      for (const f of fr.snapshot.fighters) {
        if (!byId.has(f.id)) byId.set(f.id, { pts: [], team: f.team, dead: false });
        const r = byId.get(f.id);
        if (r.dead) continue;
        r.pts.push([t, toFt(f.pos[1]) / 1000]);
        if (!f.alive) r.dead = true;
      }
    }
    for (const [id, r] of byId) series.push({ points: r.pts, color: teamColor(r.team), width: id === this.selected ? 2.2 : 1 });
    const c = $("alt-strip");
    lineChart(c, { height: c.parentElement.clientHeight, series, cursorX: tNow, yLabel: "kft", xFmt: fmtTime, yMin: 0 });
  }
}
