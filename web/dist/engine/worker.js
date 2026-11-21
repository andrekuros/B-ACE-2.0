// Runs batches / recipes off the main thread. One engine instance per worker.
import { loadEngine } from "./bace.js";

const ready = loadEngine();

self.onmessage = async (ev) => {
  const { id, kind, payload } = ev.data;
  try {
    const engine = await ready;
    let result;
    if (kind === "batch") result = engine.batch(payload.cases, payload.policy);
    else if (kind === "wez") result = engine.recipeWez(payload);
    else if (kind === "fsm") result = engine.recipeFsm(payload);
    else throw new Error(`unknown job ${kind}`);
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message || err) });
  }
};
