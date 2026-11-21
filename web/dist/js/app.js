// B-ACE Lab entry point.
import { $, toast, applyTheme, themePreference } from "./util.js";
import { loadEngine } from "../engine/bace.js";
import { PilotBook } from "./scenario.js";
import { SimulateTab } from "./simulate.js";
import { EvaluateTab } from "./evaluate.js";
import { WezTab } from "./wez.js";
import { FsmTab } from "./fsm.js";
import { DiagnosticsTab } from "./diagnostics.js";
import { ReplayTab } from "./replay.js";
import { WorkerPool } from "./pool.js";

const TABS = ["simulate", "evaluate", "wez", "fsm", "diagnostics", "replay"];

function bindTheme(onChange) {
  const mark = (pref) => document.querySelectorAll("[data-theme-pref]").forEach((b) => b.classList.toggle("on", b.dataset.themePref === pref));
  document.querySelectorAll("[data-theme-pref]").forEach((b) => {
    b.onclick = () => { applyTheme(b.dataset.themePref); mark(b.dataset.themePref); onChange(); };
  });
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    if (themePreference() === "auto") { applyTheme("auto"); onChange(); }
  });
  applyTheme(themePreference());
  mark(themePreference());
}

async function main() {
  let engine;
  const redraw = [];
  bindTheme(() => redraw.forEach((f) => f()));
  try {
    engine = await loadEngine();
  } catch (e) {
    $("engine-status").textContent = "Engine failed to load";
    $("engine-status").classList.add("err");
    document.body.insertAdjacentHTML("beforeend", `<p class="empty" style="padding:2rem">${e.message}</p>`);
    throw e;
  }
  $("engine-status").textContent = `Engine ${engine.info.version}, WebAssembly`;
  const book = new PilotBook(engine.info.baselines, engine.info.reference);
  const pool = new WorkerPool();
  const sim = new SimulateTab(engine, book);
  const tabs = {
    simulate: sim,
    evaluate: new EvaluateTab(book, pool, (cfg) => { go("simulate"); sim.startConfig(cfg, "Episode 1 of the selected match"); }),
    wez: new WezTab(engine, pool),
    fsm: new FsmTab(book, pool, () => { sim.refreshPilots(); tabs.evaluate.refreshPilots(); }),
    diagnostics: new DiagnosticsTab(engine),
    replay: new ReplayTab((ep) => { go("simulate"); sim.loadEpisode(ep); toast(`Loaded ${ep.frames.length} decisions`); }),
  };

  function go(name) {
    if (!TABS.includes(name)) name = "simulate";
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    for (const t of TABS) {
      $(`tab-${t}`).hidden = t !== name;
      tabs[t].active = t === name;
      document.querySelector(`[data-tab="${t}"]`).setAttribute("aria-selected", String(t === name));
    }
    tabs[name].show?.();
    sim.telemetryDirty = true;
  }
  redraw.push(() => {
    sim.telemetryDirty = true;
    const cur = location.hash.slice(1) || "simulate";
    tabs[cur]?.show?.();
    if (tabs.evaluate.results) tabs.evaluate.render();
  });
  window.addEventListener("hashchange", () => go(location.hash.slice(1)));
  go(location.hash.slice(1) || "simulate");
  sim.start();
  window.bace = { engine, sim, tabs, go };
  document.body.dataset.ready = "1";
}

main();
