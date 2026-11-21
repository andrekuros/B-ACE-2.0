// Thin loader for bace.wasm (crates/bace-wasm). No generated glue: JSON in, JSON out.
const enc = new TextEncoder();
const dec = new TextDecoder();

export class Engine {
  constructor(exports) {
    this.x = exports;
    this.info = this._json(this.x.lab_info());
  }

  _read() {
    const ptr = this.x.result_ptr();
    const len = this.x.result_len();
    return dec.decode(new Uint8Array(this.x.memory.buffer, ptr, len));
  }

  _json(code) {
    const v = JSON.parse(this._read());
    if (code < 0 || (v && v.error)) throw new Error(v?.error || "engine error");
    return v;
  }

  _withStr(str, fn) {
    const bytes = enc.encode(str ?? "");
    const len = bytes.length;
    const ptr = this.x.alloc(len);
    new Uint8Array(this.x.memory.buffer, ptr, len).set(bytes);
    try {
      return fn(ptr, len);
    } finally {
      this.x.dealloc(ptr, len);
    }
  }

  createSim(scenario) {
    const h = this._withStr(JSON.stringify(scenario || {}), (p, l) => this.x.sim_create(p, l));
    if (h < 0) this._json(h);
    return new Sim(this, h);
  }

  batch(cases, policy = "idle") {
    return this._withStr(JSON.stringify({ cases, policy }), (p, l) => this._json(this.x.batch_run(p, l)));
  }

  /** WEZ the agents observe. targetAspect: 0 = target hot, 180 = cold. */
  wez(shooterFt, targetFt, offBoreDeg, targetAspectDeg, legacy = false) {
    return this._json(this.x.wez_evaluate(legacy ? 1 : 0, shooterFt, targetFt, offBoreDeg, targetAspectDeg));
  }

  recipeWez(params) {
    return this._withStr(JSON.stringify(params || {}), (p, l) => this._json(this.x.recipe_wez(p, l)));
  }

  recipeFsm(params) {
    return this._withStr(JSON.stringify(params || {}), (p, l) => this._json(this.x.recipe_fsm(p, l)));
  }
}

export class Sim {
  constructor(engine, handle) {
    this.e = engine;
    this.h = handle;
  }
  reset(seed = -1) {
    return this.e._json(this.e.x.sim_reset(this.h, seed));
  }
  step(actions = {}) {
    return this.e._withStr(JSON.stringify(actions), (p, l) => this.e._json(this.e.x.sim_step(this.h, p, l)));
  }
  outcome() {
    return this.e._json(this.e.x.sim_outcome(this.h));
  }
  free() {
    this.e.x.sim_free(this.h);
    this.h = -1;
  }
}

/** Load from a URL (browser/worker) or from raw bytes (Node tests). */
export async function loadEngine(source = new URL("./bace.wasm", import.meta.url)) {
  let instance;
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    ({ instance } = await WebAssembly.instantiate(source, {}));
  } else {
    const res = await fetch(source);
    if (!res.ok) throw new Error(`bace.wasm not found (${res.status}). Build it with scripts/build_web.sh`);
    const bytes = await res.arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes, {}));
  }
  return new Engine(instance.exports);
}
