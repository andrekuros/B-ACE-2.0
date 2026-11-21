// WEZ tab: analytic envelope explorer + empirical validation recipe.
import { $, COLORS, alpha, fitCanvas, fmt } from "./util.js";
import { lineChart } from "./charts.js";

const ASP_COLOR = { head: COLORS.blue, beam: COLORS.violet, tail: COLORS.ochre };
const ALT_DASH = { 0: [], 1: [6, 3], 2: [2, 3] };
const ASP_TEXT = { head: "Head-on", beam: "Beam", tail: "Tail chase" };

export class WezTab {
  constructor(engine, pool) {
    this.engine = engine;
    this.pool = pool;
    for (const id of ["wez-alt", "wez-talt", "wez-model"]) $(id).oninput = () => this.polar();
    $("wez-run").onclick = () => this.run();
    window.addEventListener("resize", () => { if (this.active) { this.polar(); this.chart(); } });
    this.card();
    this.params();
  }
  show() { this.polar(); this.chart(); }

  card() {
    const t = this.engine.info.wez_table;
    const asp = [0, 90, 180];
    const name = { 0: "Hot (head-on)", 90: "Beam", 180: "Cold (tail)" };
    const cell = (alt, a) => this.engine.wez(alt, alt, 0, a);
    let html = `<table><tr><th>Altitude ft</th>${asp.map((a) => `<th class="num">${name[a]} R<sub>max</sub></th><th class="num">NEZ</th>`).join("")}</tr>`;
    for (const alt of t.shooter_alt_ft) {
      html += `<tr><td>${alt.toLocaleString()}</td>${asp.map((a) => { const w = cell(alt, a); return `<td class="num">${fmt(w.r_max_nm, 1)}</td><td class="num">${fmt(w.r_nez_nm, 1)}</td>`; }).join("")}</tr>`;
    }
    $("wez-card").innerHTML = html + "</table>";
  }

  params() {
    const p = this.engine.info.missile_params;
    const rows = [
      ["Launch mass", `${p.mass_launch} kg`], ["Propellant, boost / sustain", `${p.prop_boost} / ${p.prop_sustain} kg`],
      ["Burn, boost / sustain", `${p.t_boost} / ${p.t_sustain} s`], ["Specific impulse", `${p.isp} s`],
      ["Diameter", `${p.diameter} m`], ["Max normal-force coefficient", p.cn_max], ["Structural limit", `${p.g_max} g`],
      ["Proportional navigation constant", p.nav_const], ["Autopilot time constant", `${p.tau} s`],
      ["Seeker acquisition range / gimbal", `${(p.seeker_range_m / 1852).toFixed(0)} NM / ±${p.seeker_fov_deg}°`],
      ["Lethal radius", `${p.lethal_radius_m} m`], ["Max time of flight", `${p.max_tof} s`], ["Spent below", `${p.min_speed} m/s after burnout`],
    ];
    $("wez-params").innerHTML = `<table>${rows.map(([k, v]) => `<tr><td>${k}</td><td class="num">${v}</td></tr>`).join("")}</table>
      <p class="hint">Notional values of the order publicly reported for the medium-range class. They describe no real weapon. Every element’s open source is listed in <a href="https://github.com/andrekuros/B-ACE-2.0/blob/main/docs/PUBLIC_REFERENCES.md" target="_blank" rel="noopener">PUBLIC_REFERENCES.md</a>.</p>`;
  }

  polar() {
    const alt = Number($("wez-alt").value);
    const talt = Number($("wez-talt").value);
    const legacy = $("wez-model").value === "legacy";
    $("wez-alt-out").textContent = `${alt.toLocaleString()} ft`;
    $("wez-talt-out").textContent = `${talt.toLocaleString()} ft`;
    const c = $("wez-polar");
    const { ctx, w } = fitCanvas(c, c.clientWidth);
    const h = w;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, R = w / 2 - 26;
    const maxNm = 70;
    const k = R / maxNm;
    ctx.strokeStyle = COLORS.hair; ctx.fillStyle = COLORS.ink3; ctx.lineWidth = 1;
    ctx.font = "500 11px 'Barlow Semi Condensed', sans-serif"; ctx.textAlign = "left";
    for (const r of [10, 20, 30, 40, 50, 60, 70]) {
      ctx.beginPath(); ctx.arc(cx, cy, r * k, 0, Math.PI * 2); ctx.stroke();
      if (r % 20 === 0) ctx.fillText(`${r}`, cx + 3, cy - r * k + 10);
    }
    ctx.textAlign = "center";
    ctx.fillStyle = COLORS.ink2;
    ctx.fillText("Target hot", cx, 12);
    ctx.fillText("Target cold", cx, h - 4);
    ctx.save(); ctx.translate(w - 8, cy); ctx.rotate(Math.PI / 2); ctx.fillText("Beam", 0, 0); ctx.restore();
    ctx.save(); ctx.translate(8, cy); ctx.rotate(-Math.PI / 2); ctx.fillText("Beam", 0, 0); ctx.restore();
    const curve = (key) => {
      const pts = [];
      for (let d = 0; d <= 360; d += 3) {
        const asp = d <= 180 ? d : 360 - d; // top = hot
        const v = this.engine.wez(alt, talt, 0, asp, legacy)[key];
        const a = (d * Math.PI) / 180;
        pts.push([cx + Math.sin(a) * v * k, cy - Math.cos(a) * v * k]);
      }
      return pts;
    };
    const draw = (pts, fill, stroke, dash) => {
      ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.setLineDash(dash || []); ctx.strokeStyle = stroke; ctx.lineWidth = 1.8; ctx.stroke(); ctx.setLineDash([]);
    };
    draw(curve("r_max_nm"), alpha(COLORS.blue, 0.08), COLORS.blue, [6, 4]);
    draw(curve("r_nez_nm"), alpha(COLORS.blue, 0.2), COLORS.blue);
    ctx.fillStyle = COLORS.ink;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    const head = this.engine.wez(alt, talt, 0, 0, legacy), beam = this.engine.wez(alt, talt, 0, 90, legacy), tail = this.engine.wez(alt, talt, 0, 180, legacy);
    ctx.textAlign = "left"; ctx.fillStyle = COLORS.ink2;
    ctx.fillText(`Rmax hot / beam / cold ${fmt(head.r_max_nm, 0)} / ${fmt(beam.r_max_nm, 0)} / ${fmt(tail.r_max_nm, 0)} NM`, 6, h - 34);
    ctx.fillText(`NEZ ${fmt(head.r_nez_nm, 0)} / ${fmt(beam.r_nez_nm, 0)} / ${fmt(tail.r_nez_nm, 0)} NM`, 6, h - 20);
  }

  async run() {
    const nums = (s) => s.split(/[,\s]+/).map(Number).filter((v) => Number.isFinite(v) && v > 0);
    const ranges = nums($("wez-ranges").value);
    const alts = nums($("wez-alts").value);
    const aspects = [...document.querySelectorAll('input[name="wez-asp"]:checked')].map((x) => x.value);
    if (!ranges.length || !alts.length || !aspects.length) { $("wez-status").textContent = "Give at least one range, altitude and aspect."; return; }
    $("wez-run").disabled = true;
    $("wez-status").textContent = `Running ${ranges.length * alts.length * aspects.length} launches…`;
    const t0 = performance.now();
    try {
      const model = $("wez-model").value === "legacy" ? "legacy" : "generic_mraam";
      const reps = await Promise.all(alts.map((a) => this.pool.run("wez", { ranges_nm: ranges, altitudes_ft: [a], aspects, repeats: 1, max_cycles: 240, seed: 1, missile_model: model })));
      this.report = { cells: reps.flatMap((r) => r.cells), summaries: reps.flatMap((r) => r.aspect_summaries), alts, aspects };
      $("wez-status").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
      this.chart();
    } catch (e) {
      $("wez-status").textContent = `WEZ recipe failed: ${e.message}`;
    } finally {
      $("wez-run").disabled = false;
    }
  }

  chart() {
    const R = this.report;
    if (!R) { lineChart($("wez-chart"), { height: 360, series: [], empty: "Run the recipe to measure hit rate against launch range." }); return; }
    const series = [], markers = [];
    R.alts.forEach((alt, ai) => {
      for (const asp of R.aspects) {
        const cells = R.cells.filter((c) => c.altitude_ft === alt && c.aspect === asp).sort((a, b) => a.range_nm - b.range_nm);
        if (!cells.length) continue;
        series.push({ points: cells.map((c) => [c.range_nm, c.hit_rate]), color: ASP_COLOR[asp], dash: ALT_DASH[ai % 3], dots: true, label: `${ASP_TEXT[asp]} ${alt / 1000}k` });
        markers.push({ x: cells[0].analytic_rmax_nm, color: ASP_COLOR[asp], dash: [4, 4], width: 1.4 });
      }
    });
    lineChart($("wez-chart"), { height: 360, series, markers, yMin: 0, yMax: 1.05, yLabel: "hit rate", legend: true, xFmt: (v) => `${v} NM` });
    const rows = R.summaries.map((s) => {
      const cells = R.cells.filter((c) => c.altitude_ft === s.altitude_ft && c.aspect === s.aspect);
      const far = cells.reduce((a, c) => (c.range_nm > a.range_nm ? c : a), cells[0]);
      const open = far && far.hit_rate >= 0.5;
      const ratio = s.empirical_p50_nm / s.analytic_rmax_nm;
      const p50 = open ? `beyond ${fmt(far.range_nm, 0)}` : fmt(s.empirical_p50_nm, 1);
      return `<tr><td>${ASP_TEXT[s.aspect] || s.aspect}</td><td class="num">${s.altitude_ft.toLocaleString()}</td><td class="num">${p50}</td><td class="num">${fmt(s.analytic_rmax_nm, 1)}</td><td class="num">${open ? `over ${fmt(ratio, 2)}` : fmt(ratio, 2)}</td></tr>`;
    }).join("");
    $("wez-summary").innerHTML = `<table><tr><th>Aspect</th><th class="num">Altitude ft</th><th class="num">50% hit range NM</th><th class="num">Observed Rmax NM</th><th class="num">Ratio</th></tr>${rows}</table>
      <p class="hint">Ratio near 1 means the envelope the agents observe matches what the missile can actually do. Well below 1, agents are told they are in range when they are not.</p>`;
  }
}
