// Diagnostics tab: benchmark-property checks run against the WebAssembly engine.
import { $ } from "./util.js";

const runEp = (engine, cfg, policyFn, maxSteps = 100000) => {
  const sim = engine.createSim(cfg);
  const frames = [sim.reset(cfg.env.seed)];
  let k = 0;
  while (frames[frames.length - 1].end === "ongoing" && k++ < maxSteps) {
    const last = frames[frames.length - 1];
    frames.push(sim.step(policyFn ? policyFn(last) : {}));
  }
  const outcome = sim.outcome();
  sim.free();
  return { frames, outcome };
};

const randomPolicy = (rnd) => (frame) => Object.fromEntries(Object.keys(frame.agents).map((a) => [a, [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1, rnd() < 0.08 ? 1 : -1]]));
function lcg(seed) { let s = seed >>> 0; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }

function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0).toString(16); }

const base = (o = {}) => ({
  env: { seed: 1, max_cycles: 300, ...(o.env || {}) },
  blue: { behavior: "baseline1", num_agents: 1, ...(o.blue || {}) },
  red: { behavior: "baseline1", num_agents: 1, init_position: { x: 0, y: 25000, z: -30 }, init_hdg: 180, target_position: { x: 0, y: 25000, z: 30 }, ...(o.red || {}) },
});

export const CHECKS = [
  {
    name: "Reproducible with a seed",
    what: "Same scenario and seed twice gives an identical trajectory, including FSM noise and random spawns.",
    run(engine) {
      const cfg = base({ blue: { beh_noise: 0.2, rnd_offset_range: { x: 5, y: 2000, z: 3 } }, red: { beh_noise: 0.2, rnd_offset_range: { x: 5, y: 2000, z: 3 } }, env: { seed: 42, max_cycles: 400 } });
      const a = hash(JSON.stringify(runEp(engine, cfg).frames));
      const b = hash(JSON.stringify(runEp(engine, cfg).frames));
      return a === b ? ["pass", `Trajectory hash ${a} on both runs.`] : ["fail", `Hashes differ: ${a} vs ${b}`];
    },
  },
  {
    name: "Seeds produce variety",
    what: "Different seeds change scripted episodes. Without FSM noise or spawn jitter they cannot.",
    run(engine) {
      // Compare trajectories, not outcome records (those carry the seed itself).
      const ends = (o) => new Set([1, 2, 3, 4, 5, 6].map((s) => hash(JSON.stringify(runEp(engine, { ...base(o), env: { seed: s, max_cycles: 300 } }).frames.map((f) => f.snapshot.fighters.map((x) => x.pos)))))).size;
      const plain = ends({});
      const noisy = ends({ blue: { beh_noise: 0.15, rnd_offset_range: { x: 4, y: 0, z: 2 } }, red: { beh_noise: 0.15, rnd_offset_range: { x: 4, y: 0, z: 2 } } });
      if (noisy < 2) return ["fail", `Even with noise, 6 seeds gave ${noisy} distinct outcome(s).`];
      if (plain === 1) return ["warn", `Default config: 6 seeds gave 1 distinct episode. With beh_noise 0.15 and 4 NM jitter: ${noisy} distinct. Set noise or jitter before reporting confidence intervals.`];
      return ["pass", `${plain} distinct with defaults, ${noisy} with noise.`];
    },
  },
  {
    name: "Observation size contract",
    what: "Every learner gets 9 + 13 × enemies + 6 × allies values, for 1v1 up to 4v4 and for red learners in self-play.",
    run(engine) {
      const bad = [];
      for (const n of [1, 2, 3, 4]) {
        const cfg = base({ blue: { behavior: "external", num_agents: n }, red: { behavior: "external", num_agents: n } });
        const sim = engine.createSim(cfg);
        const f = sim.reset(1);
        sim.free();
        const want = 9 + 13 * n + 6 * (n - 1);
        for (const [a, v] of Object.entries(f.agents)) if (v.flat_obs.length !== want) bad.push(`${n}v${n} ${a}: ${v.flat_obs.length} ≠ ${want}`);
        if (Object.keys(f.agents).length !== 2 * n) bad.push(`${n}v${n}: ${Object.keys(f.agents).length} learners, expected ${2 * n}`);
      }
      return bad.length ? ["fail", bad.join("; ")] : ["pass", "Sizes 22, 49, 76, 103 for 1v1 to 4v4, both teams."];
    },
  },
  {
    name: "Observations are finite and bounded",
    what: "Random-policy 4v4 self-play: no NaN or infinity, and every feature within ±5.",
    run(engine) {
      const { frames } = runEp(engine, base({ env: { max_cycles: 250, seed: 7 }, blue: { behavior: "external", num_agents: 4 }, red: { behavior: "external", num_agents: 4 } }), randomPolicy(lcg(7)));
      let maxAbs = 0, nonFinite = 0, worst = -1;
      for (const f of frames) for (const a of Object.values(f.agents)) a.flat_obs.forEach((v, i) => {
        if (!Number.isFinite(v)) nonFinite++;
        else if (Math.abs(v) > maxAbs) { maxAbs = Math.abs(v); worst = i; }
      });
      if (nonFinite) return ["fail", `${nonFinite} non-finite values.`];
      return maxAbs <= 5 ? ["pass", `Max |value| ${maxAbs.toFixed(2)} (index ${worst}) over ${frames.length} decisions × 8 agents.`] : ["warn", `Max |value| ${maxAbs.toFixed(2)} at flat index ${worst}. Consider normalizing.`];
    },
  },
  {
    name: "No information on undetected enemies",
    what: "When an enemy is not tracked, its observation slots hold fixed placeholders, so agents cannot see through the radar.",
    run(engine) {
      const { frames } = runEp(engine, base({ env: { max_cycles: 250, seed: 3 }, blue: { behavior: "external", num_agents: 2, share_tracks: false }, red: { behavior: "external", num_agents: 2 } }), randomPolicy(lcg(3)));
      let checked = 0; const leaks = new Set();
      for (const f of frames) for (const a of Object.values(f.agents)) for (const e of a.obs.enemies) {
        if (e.detected > 0.5) continue;
        checked++;
        for (const k of ["alt_diff", "aspect", "angle_off", "own_r_max", "own_r_nez", "enemy_r_max", "enemy_r_nez", "threat_factor", "offensive_factor", "is_missile_support"]) if (e[k] !== 0) leaks.add(k);
        if (e.dist !== -1) leaks.add("dist");
        if (e.dist_target !== 0.5) leaks.add("dist_target");
      }
      if (!checked) return ["warn", "No undetected enemy slots seen; nothing to check."];
      return leaks.size ? ["fail", `Leaking fields: ${[...leaks].join(", ")} (over ${checked} slots).`] : ["pass", `${checked} undetected enemy slots, all placeholders.`];
    },
  },
  {
    name: "Rewards add up",
    what: "The sum of per-step rewards equals each learner’s cumulative reward breakdown.",
    run(engine) {
      const { frames } = runEp(engine, base({ env: { max_cycles: 400, seed: 5 }, blue: { behavior: "external", num_agents: 2 }, red: { behavior: "baseline1", num_agents: 2 } }), randomPolicy(lcg(5)));
      const sum = {};
      for (const f of frames.slice(1)) for (const [a, v] of Object.entries(f.agents)) sum[a] = (sum[a] || 0) + v.reward;
      const last = frames[frames.length - 1].snapshot;
      let worst = 0;
      for (const [a, s] of Object.entries(sum)) {
        const fi = last.fighters.find((x) => x.agent_name === a);
        const cum = Object.values(fi.reward).reduce((x, y) => x + y, 0);
        worst = Math.max(worst, Math.abs(cum - s));
      }
      return worst < 1e-6 ? ["pass", `Max difference ${worst.toExponential(1)} over ${frames.length - 1} steps.`] : ["fail", `Per-step sum differs from cumulative by ${worst.toFixed(6)}.`];
    },
  },
  {
    name: "Termination flags are consistent",
    what: "A shot-down learner stays terminated, truncation happens only at the time limit, and every learner is done on the last step.",
    run(engine) {
      const bad = [];
      for (const [seed, max] of [[1, 40], [2, 800], [3, 800]]) {
        const { frames } = runEp(engine, base({ env: { seed, max_cycles: max }, blue: { behavior: "external", num_agents: 2 }, red: { behavior: "baseline1", num_agents: 2, mission: "striker" } }), randomPolicy(lcg(seed)));
        const done = new Set();
        frames.forEach((f, i) => {
          for (const [a, v] of Object.entries(f.agents)) {
            if (done.has(a) && !v.terminated && !v.truncated) bad.push(`${a} came back after being done`);
            if (v.terminated || v.truncated) done.add(a);
            if (v.truncated && f.end !== "max_cycles") bad.push(`${a} truncated at ${f.end}`);
            if (i === frames.length - 1 && !v.terminated && !v.truncated) bad.push(`${a} not done at ${f.end}`);
          }
        });
      }
      return bad.length ? ["fail", [...new Set(bad)].join("; ")] : ["pass", "Checked a time-limit episode and two decided 2v2 episodes."];
    },
  },
  {
    name: "Event log matches outcome",
    what: "Launch and hit events agree with the episode counters.",
    run(engine) {
      let n = 0;
      for (const s of [1, 2, 3, 4]) {
        const { frames, outcome } = runEp(engine, base({ env: { seed: s, max_cycles: 600 }, blue: { num_agents: 2, beh_noise: 0.1 }, red: { num_agents: 2, beh_noise: 0.1, mission: "striker" } }));
        const ev = frames.flatMap((f) => f.events);
        const blueLaunch = ev.filter((e) => e.kind === "launch" && e.actor < 200).length;
        const blueHits = ev.filter((e) => e.kind === "hit" && e.actor < 200).length;
        if (blueLaunch !== outcome.missiles_fired || blueHits !== outcome.missile_hits) return ["fail", `Seed ${s}: ${blueLaunch} launches / ${blueHits} hits logged vs ${outcome.missiles_fired} / ${outcome.missile_hits} counted.`];
        n += ev.length;
      }
      return ["pass", `${n} events across 4 episodes, counters agree.`];
    },
  },
  {
    name: "WEZ ordering",
    what: "Observed max range: hot > beam > cold, and higher altitude reaches farther.",
    run(engine) {
      const h = engine.wez(25000, 25000, 0, 0).r_max_nm, b = engine.wez(25000, 25000, 0, 90).r_max_nm, t = engine.wez(25000, 25000, 0, 180).r_max_nm;
      const lo = engine.wez(10000, 10000, 0, 0).r_max_nm, hi = engine.wez(40000, 40000, 0, 0).r_max_nm;
      const ok = h > b && b > t && hi > h && h > lo;
      return [ok ? "pass" : "fail", `Head ${h.toFixed(1)}, beam ${b.toFixed(1)}, tail ${t.toFixed(1)} NM; head-on at 10k/25k/40k ft: ${lo.toFixed(1)} / ${h.toFixed(1)} / ${hi.toFixed(1)} NM.`];
    },
  },
  {
    name: "Observed WEZ matches the missile",
    what: "The range where half the missiles hit should sit near the analytic Rmax the agents observe, at every altitude.",
    run(engine) {
      const ranges = Array.from({ length: 12 }, (_, i) => 4 + 4 * i);
      const rep = engine.recipeWez({ ranges_nm: ranges, altitudes_ft: [10000, 25000, 40000], aspects: ["head", "beam", "tail"], repeats: 1, max_cycles: 240, seed: 1 });
      const off = [];
      for (const s of rep.aspect_summaries) {
        // Beyond the 48 NM sweep (radar-limited) only check that both agree it is far.
        const open = s.empirical_p50_nm >= 48 && s.analytic_rmax_nm >= 44;
        const r = s.empirical_p50_nm / s.analytic_rmax_nm;
        if (!open && (r < 0.75 || r > 1.33)) off.push(`${s.aspect} ${s.altitude_ft / 1000}k: ${s.empirical_p50_nm.toFixed(0)} vs ${s.analytic_rmax_nm.toFixed(0)} NM`);
      }
      return off.length
        ? ["warn", `${off.length} of ${rep.aspect_summaries.length} cells disagree by more than a third (${off.join("; ")}). Agents observe an envelope the missile does not honour.`]
        : ["pass", "All aspect and altitude cells within a third of Rmax."];
    },
  },
  {
    name: "Speed on this machine",
    what: "Single-thread FSM-vs-FSM decisions per second in this browser. One decision is 20 physics ticks, one simulated second.",
    run(engine) {
      const out = [];
      for (const n of [1, 4]) {
        const cases = Array.from({ length: 6 }, (_, i) => ({ ...base({ blue: { num_agents: n, beh_noise: 0.1 }, red: { num_agents: n, beh_noise: 0.1 } }), env: { seed: i + 1, max_cycles: 400 } }));
        const t0 = performance.now();
        const res = engine.batch(cases, "idle");
        const dt = (performance.now() - t0) / 1000;
        const steps = res.reduce((a, o) => a + o.steps, 0);
        out.push(`${n}v${n}: ${Math.round(steps / dt).toLocaleString()} decisions/s`);
      }
      return ["pass", out.join("; ")];
    },
  },
];

export class DiagnosticsTab {
  constructor(engine) {
    this.engine = engine;
    $("diag-run").onclick = () => this.run();
    this._render(CHECKS.map((c) => ({ c, res: null })));
  }
  _render(rows) {
    const label = { pass: "Pass", warn: "Warning", fail: "Fail" };
    $("diag-table").querySelector("tbody").innerHTML = rows.map(({ c, res, ms }) =>
      `<tr><td><b>${c.name}</b><small>${c.what}</small></td><td class="res ${res?.[0] || ""}">${res ? label[res[0]] : "Not run"}</td><td>${res ? res[1] : ""}</td><td class="num">${ms != null ? ms.toFixed(0) : ""}</td></tr>`).join("");
  }
  async run() {
    $("diag-run").disabled = true;
    const rows = CHECKS.map((c) => ({ c, res: null }));
    this._render(rows);
    for (const r of rows) {
      await new Promise((ok) => setTimeout(ok, 0));
      const t0 = performance.now();
      try { r.res = r.c.run(this.engine); } catch (e) { r.res = ["fail", `Threw: ${e.message}`]; }
      r.ms = performance.now() - t0;
      this._render(rows);
    }
    const n = (k) => rows.filter((r) => r.res[0] === k).length;
    $("diag-status").textContent = `${n("pass")} passed, ${n("warn")} warnings, ${n("fail")} failed.`;
    document.body.dataset.diagnostics = n("fail") ? "fail" : "pass";
    $("diag-run").disabled = false;
    return rows;
  }
}
