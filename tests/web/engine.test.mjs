// Node tests for the WebAssembly engine (no browser). Run: node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadEngine } from "../../web/dist/engine/bace.js";

const wasm = readFileSync(fileURLToPath(new URL("../../web/dist/engine/bace.wasm", import.meta.url)));
const engine = await loadEngine(wasm);

test("info exposes defaults and FSM profiles", () => {
  assert.equal(engine.info.default_config.env.phy_fps, 20);
  for (const k of ["aggressive", "balanced", "cautious"]) assert.equal(engine.info.baselines[k].behavior, "baseline1");
});

test("an engagement emits launch, hit and end events in time order", () => {
  const balanced = engine.info.reference.balanced.beh_config;
  const sim = engine.createSim({ env: { max_cycles: 500 }, blue: { behavior: "baseline1", beh_config: balanced }, red: { behavior: "duck", mission: "striker" } });
  let f = sim.reset(1);
  const kinds = [];
  let t = 0;
  while (f.end === "ongoing") {
    f = sim.step({});
    for (const e of f.events) { assert.ok(e.t >= t); t = e.t; kinds.push(e.kind); }
  }
  assert.ok(kinds.includes("launch") && kinds.includes("pitbull") && kinds.includes("hit"));
  assert.equal(kinds.at(-1), "end");
  assert.equal(sim.outcome().blue_kills, 1);
  sim.free();
});

test("partial batch cases keep red on its own side", () => {
  const [o] = engine.batch([{ env: { max_cycles: 2 }, red: { behavior: "duck" } }]);
  assert.equal(o.steps, 2);
  const sim = engine.createSim({ red: { behavior: "duck" } });
  const red = sim.reset(1).snapshot.fighters.find((x) => x.team === 1);
  assert.ok(red.pos[2] < 0, "red spawns north of the bullseye");
  sim.free();
});

test("generic missile reports Mach and the A-pole / F-pole", () => {
  // 30 NM head-on: inside the envelope, beyond the 10 NM seeker range.
  const sim = engine.createSim({
    env: { max_cycles: 500 },
    blue: { behavior: "fire_once", init_position: { x: 0, y: 25000, z: 15 } },
    red: { behavior: "duck", mission: "striker", init_position: { x: 0, y: 25000, z: -15 } },
  });
  let f = sim.reset(2);
  let peak = 0;
  const ev = [];
  while (f.end === "ongoing") {
    f = sim.step({});
    for (const m of f.snapshot.missiles) peak = Math.max(peak, m.mach);
    ev.push(...f.events);
  }
  assert.ok(peak > 3 && peak < 4.6, `peak Mach ${peak}`);
  const hit = ev.find((e) => e.kind === "hit");
  const act = ev.find((e) => e.kind === "pitbull");
  assert.ok(act.pole_nm > hit.pole_nm && hit.pole_nm > 0, "A-pole is farther than F-pole");
  sim.free();
});

test("batch is deterministic per seed", () => {
  const cases = [1, 2, 3].map((s) => ({ env: { seed: s, max_cycles: 300 }, blue: { behavior: "external" }, red: { behavior: "baseline1", beh_noise: 0.1 } }));
  assert.deepEqual(engine.batch(cases, "random"), engine.batch(cases, "random"));
});

test("errors come back as exceptions, not crashes", () => {
  assert.throws(() => engine.createSim({ env: { max_cycles: "many" } }), /bad scenario/);
});
