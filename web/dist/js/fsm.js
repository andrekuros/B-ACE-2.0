// FSM search tab: run the GA recipe and promote elites to pilots.
import { $, COLORS, fmt, toast } from "./util.js";
import { lineChart } from "./charts.js";

export class FsmTab {
  constructor(book, pool, onPilotsChanged) {
    this.book = book;
    this.pool = pool;
    this.onPilotsChanged = onPilotsChanged;
    $("fsm-run").onclick = () => this.run();
  }
  show() { this.chart(); }

  async run() {
    const p = {
      pop: Number($("fsm-pop").value), generations: Number($("fsm-gen").value), episodes: Number($("fsm-eps").value),
      max_cycles: Number($("fsm-max").value), num_agents: Number($("fsm-n").value), seed: Number($("fsm-seed").value),
      eval_agents: 0, pool_interval: 3,
      beh_noise: Number($("fsm-noise").value) || 0, spawn_jitter_nm: Number($("fsm-jitter").value) || 0,
    };
    $("fsm-run").disabled = true;
    $("fsm-status").textContent = `Evolving ${p.pop} genomes for ${p.generations} generations. This runs on one worker; larger searches can take a minute.`;
    const t0 = performance.now();
    try {
      this.report = await this.pool.run("fsm", p);
      $("fsm-status").textContent = `${this.report.summary || "Done"} (${((performance.now() - t0) / 1000).toFixed(1)} s)`;
      this.chart();
      this.table();
    } catch (e) {
      $("fsm-status").textContent = `Search failed: ${e.message}`;
    } finally {
      $("fsm-run").disabled = false;
    }
  }

  chart() {
    const h = this.report?.history || [];
    lineChart($("fsm-chart"), {
      height: 260, legend: true, empty: "Run a search to see fitness by generation.", xInt: true, xLabel: "generation",
      series: h.length ? [
        { points: h.map((g) => [g.generation, g.best_fitness]), color: COLORS.ink, width: 2.2, dots: true, label: "Best vs fixed reference" },
        { points: h.map((g) => [g.generation, g.train_fitness]), color: COLORS.blue, dots: true, label: "Against training pool" },
        { points: h.map((g) => [g.generation, g.frozen_fitness]), color: COLORS.ochre, dash: [5, 3], label: "Against frozen archive" },
      ] : [],
    });
  }

  table() {
    const el = this.report?.elites || [];
    if (!el.length) { $("fsm-elites").innerHTML = `<p class="empty">No elites in this run.</p>`; return; }
    $("fsm-elites").innerHTML = `<table><tr><th>Elite</th><th class="num">Shot dist.</th><th class="num">Crank</th><th class="num">Break</th><th class="num">Fitness</th><th class="num">Kills</th><th class="num">Losses</th><th class="num">Mission</th><th></th></tr>` +
      el.map((e, i) => `<tr><td>${e.label}</td><td class="num">${fmt(e.genome.d_shot, 3)}</td><td class="num">${fmt(e.genome.l_crank, 3)}</td><td class="num">${fmt(e.genome.l_break, 3)}</td><td class="num">${fmt(e.fitness, 3)}</td><td class="num">${fmt(e.mean_kills, 2)}</td><td class="num">${fmt(e.mean_deaths, 2)}</td><td class="num">${fmt(e.mission_rate, 2)}</td><td><button class="small" data-i="${i}">Add as pilot</button></td></tr>`).join("") + "</table>";
    $("fsm-elites").querySelectorAll("button[data-i]").forEach((b) => {
      b.onclick = () => {
        const e = el[Number(b.dataset.i)];
        this.book.addCustom(`FSM ${e.label} (searched)`, e.genome);
        this.onPilotsChanged();
        b.disabled = true;
        toast(`Added “FSM ${e.label} (searched)” to the pilot lists`);
      };
    });
  }
}
