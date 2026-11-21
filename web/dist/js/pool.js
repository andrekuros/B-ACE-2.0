// A small pool of engine workers for batches and recipes.
export class WorkerPool {
  constructor(size = Math.max(1, Math.min(6, (navigator.hardwareConcurrency || 4) - 1))) {
    this.size = size;
    this.workers = [];
    this.queue = [];
    this.pending = new Map();
    this.idle = [];
    this.nextId = 1;
  }
  _spawn() {
    const w = new Worker(new URL("../engine/worker.js", import.meta.url), { type: "module" });
    w.onmessage = (ev) => {
      const { id, ok, result, error } = ev.data;
      const p = this.pending.get(id);
      this.pending.delete(id);
      this.idle.push(w);
      this._pump();
      if (p) (ok ? p.resolve(result) : p.reject(new Error(error)));
    };
    w.onerror = (ev) => {
      for (const [id, p] of this.pending) if (p.worker === w) { p.reject(new Error(ev.message || "worker failed")); this.pending.delete(id); }
    };
    this.workers.push(w);
    this.idle.push(w);
  }
  run(kind, payload) {
    if (!this.workers.length) for (let i = 0; i < this.size; i++) this._spawn();
    return new Promise((resolve, reject) => {
      this.queue.push({ kind, payload, resolve, reject });
      this._pump();
    });
  }
  _pump() {
    while (this.idle.length && this.queue.length) {
      const w = this.idle.pop();
      const job = this.queue.shift();
      const id = this.nextId++;
      this.pending.set(id, { ...job, worker: w });
      w.postMessage({ id, kind: job.kind, payload: job.payload });
    }
  }
}
