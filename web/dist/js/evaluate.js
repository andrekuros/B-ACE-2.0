// Evaluate tab: cross-play matrix of blue pilots vs red pilots.
import { $, COLORS, alpha, fmt, toast, download } from "./util.js";
import { METRICS, summarize } from "./metrics.js";

export class EvaluateTab {
  constructor(book, pool, onWatch) {
    this.book = book;
    this.pool = pool;
    this.onWatch = onWatch;
    this.metric = "fscore";
    this.results = null;
    this.refreshPilots();
    $("eval-run").onclick = () => this.run();
    $("eval-export").onclick = () => this.exportCsv();
    $("eval-metric").querySelectorAll("button").forEach((b) => {
      b.onclick = () => {
        this.metric = b.dataset.m;
        $("eval-metric").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        this.render();
      };
    });
  }

  refreshPilots() {
    const pilots = this.book.all({ live: false });
    const defBlue = new Set(["ref_aggressive", "ref_balanced", "ref_conservative", "rl_random"]);
    const defRed = new Set(["duck", "ref_aggressive", "ref_balanced", "ref_conservative", "fsm_b1"]);
    for (const [id, def] of [["eval-blue", defBlue], ["eval-red", defRed]]) {
      const prev = new Set([...document.querySelectorAll(`#${id} input:checked`)].map((x) => x.value));
      $(id).innerHTML = pilots.map((p) => {
        const on = prev.size ? prev.has(p.id) : def.has(p.id);
        return `<label><input type="checkbox" value="${p.id}" ${on ? "checked" : ""}/> ${p.label}</label>`;
      }).join("");
    }
  }

  _cases(blueId, redId, opts) {
    const cases = [];
    for (let i = 0; i < opts.eps; i++) {
      const side = (id, z, hdg) => ({
        ...this.book.team(id),
        num_agents: opts.n,
        init_position: { x: 0, y: 25000, z },
        init_hdg: hdg,
        target_position: { x: 0, y: 25000, z: -z },
        rnd_offset_range: { x: opts.jitter, y: 0, z: opts.jitter / 2 },
        beh_noise: opts.noise,
        mission: "dca",
      });
      const blue = side(blueId, 30, 0);
      const red = side(redId, -30, 180);
      red.mission = opts.mission;
      cases.push({ env: { seed: opts.seed + i, max_cycles: opts.max, missile_support: opts.support, missile_model: opts.model }, blue, red });
    }
    return cases;
  }

  async run() {
    const blues = [...document.querySelectorAll("#eval-blue input:checked")].map((x) => x.value);
    const reds = [...document.querySelectorAll("#eval-red input:checked")].map((x) => x.value);
    if (!blues.length || !reds.length) { toast("Pick at least one pilot on each side"); return; }
    const opts = {
      n: Number($("eval-n").value) || 1,
      eps: Math.max(2, Number($("eval-eps").value) || 30),
      max: Number($("eval-max").value) || 600,
      mission: $("eval-mission").value,
      noise: Number($("eval-noise").value) || 0,
      jitter: Number($("eval-jitter").value) || 0,
      support: $("eval-support").value,
      model: $("eval-model").value,
      seed: 1000,
    };
    const prog = $("eval-progress");
    prog.hidden = false; prog.value = 0; prog.max = blues.length * reds.length;
    $("eval-run").disabled = true;
    const t0 = performance.now();
    $("eval-status").textContent = `Running ${blues.length * reds.length} matches × ${opts.eps} episodes on ${this.pool.size} workers…`;
    const cells = {};
    const jobs = [];
    for (const b of blues) for (const r of reds) {
      const policy = [b, r].some((id) => this.book.get(id).policy === "random") ? "random" : "idle";
      const cases = this._cases(b, r, opts);
      jobs.push(this.pool.run("batch", { cases, policy }).then((outs) => {
        cells[`${b}|${r}`] = { blue: b, red: r, outcomes: outs, stats: summarize(outs), cases };
        prog.value += 1;
      }));
    }
    try {
      await Promise.all(jobs);
      const secs = (performance.now() - t0) / 1000;
      const total = blues.length * reds.length * opts.eps;
      $("eval-status").textContent = `${total} episodes in ${secs.toFixed(1)} s. Click a cell for details and a sample episode.`;
      this.results = { blues, reds, cells, opts };
      $("eval-export").disabled = false;
      this.selected = null;
      this.render();
    } catch (e) {
      $("eval-status").textContent = `Evaluation failed: ${e.message}`;
    } finally {
      $("eval-run").disabled = false;
      prog.hidden = true;
    }
  }

  _color(metric, v, range) {
    if (!Number.isFinite(v)) return COLORS.hair2;
    const diverging = metric === "fscore" || metric === "eprew";
    if (diverging) {
      const t = Math.max(-1, Math.min(1, v / (range || 1)));
      return t >= 0 ? alpha(COLORS.blue, 0.08 + 0.62 * t) : alpha(COLORS.red, 0.08 + 0.62 * -t);
    }
    const t = Math.max(0, Math.min(1, v / (range || 1)));
    return alpha(metric === "rscore" ? COLORS.red : COLORS.blue, 0.06 + 0.64 * t);
  }

  render() {
    const R = this.results;
    if (!R) return;
    const m = this.metric;
    const vals = Object.values(R.cells).map((c) => c.stats[m].mean).filter(Number.isFinite);
    const range = m === "macc" || m === "beff" ? 1 : Math.max(1e-6, ...vals.map(Math.abs));
    const name = (id) => this.book.get(id).label;
    let html = `<table><tr><th class="corner">Blue ↓ vs red →</th>${R.reds.map((r) => `<th>${name(r)}</th>`).join("")}</tr>`;
    for (const b of R.blues) {
      html += `<tr><th>${name(b)}</th>`;
      for (const r of R.reds) {
        const c = R.cells[`${b}|${r}`];
        const s = c.stats[m];
        const dark = Math.abs(s.mean) / range > 0.55;
        html += `<td data-k="${b}|${r}" class="${this.selected === `${b}|${r}` ? "sel" : ""}" style="background:${this._color(m, s.mean, range)};color:${dark ? COLORS.paper : "inherit"}">` +
          `<b>${fmt(s.mean, METRICS[m].fmt)}</b><small>${fmt(s.lo, 2)} to ${fmt(s.hi, 2)}</small></td>`;
      }
      html += "</tr>";
    }
    $("eval-matrix").innerHTML = html + "</table>";
    $("eval-matrix").querySelectorAll("td[data-k]").forEach((td) => (td.onclick = () => { this.selected = td.dataset.k; this.render(); }));
    this._detail();
  }

  _detail() {
    const c = this.results?.cells[this.selected];
    if (!c) { $("eval-detail").innerHTML = ""; return; }
    const name = (id) => this.book.get(id).label;
    const rows = Object.entries(METRICS).map(([k, d]) => {
      const s = c.stats[k];
      return `<tr><td>${d.label}</td><td class="num">${fmt(s.mean, 3)}</td><td class="num">${fmt(s.lo, 3)}</td><td class="num">${fmt(s.hi, 3)}</td></tr>`;
    }).join("");
    const ends = {};
    c.outcomes.forEach((o) => (ends[o.end] = (ends[o.end] || 0) + 1));
    $("eval-detail").innerHTML = `<div class="detail"><h3>${name(c.blue)} vs ${name(c.red)}, ${c.stats.n} episodes</h3>
      <table><tr><th>Metric</th><th class="num">Mean</th><th class="num">95% CI low</th><th class="num">95% CI high</th></tr>${rows}</table>
      <p class="hint" style="margin-top:.6rem">Endings: ${Object.entries(ends).map(([k, v]) => `${k.replaceAll("_", " ")} ${v}`).join(", ")}.</p>
      <div class="row"><button id="eval-watch" class="primary">Watch episode 1 in Simulate</button><button id="eval-json">Download outcomes</button></div></div>`;
    $("eval-watch").onclick = () => this.onWatch(c.cases[0], c.blue, c.red);
    $("eval-json").onclick = () => download(`eval_${c.blue}_vs_${c.red}.json`, JSON.stringify({ blue: c.blue, red: c.red, stats: c.stats, outcomes: c.outcomes }, null, 2));
  }

  exportCsv() {
    const R = this.results;
    if (!R) return;
    const keys = Object.keys(METRICS);
    const head = ["blue", "red", "episodes", ...keys.flatMap((k) => [k, `${k}_lo`, `${k}_hi`])];
    const lines = [head.join(",")];
    for (const c of Object.values(R.cells)) {
      lines.push([c.blue, c.red, c.stats.n, ...keys.flatMap((k) => [c.stats[k].mean, c.stats[k].lo, c.stats[k].hi].map((v) => (Number.isFinite(v) ? v.toFixed(4) : "")))].join(","));
    }
    download("bace_crossplay.csv", lines.join("\n"), "text/csv");
  }
}
