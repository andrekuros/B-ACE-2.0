// Minimal canvas line charts (no dependencies).
import { COLORS, fitCanvas } from "./util.js";

function niceTicks(min, max, n = 4) {
  const span = max - min || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) || mag * 10;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

/**
 * opts: { series: [{ points:[[x,y]], color, width, dash, label }], bands: [{ lo:[[x,y]], hi:[[x,y]], color }],
 *         markers: [{ x, color, label }], cursorX, xLabel, yLabel, yMin, yMax, xMin, xMax, height, legend: bool }
 */
export function lineChart(canvas, opts) {
  const { ctx, w, h } = fitCanvas(canvas, opts.height);
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px 'Barlow Semi Condensed', Barlow, sans-serif";
  // Lay out legend entries first so the plot starts below however many rows they need.
  const legendItems = [];
  if (opts.legend) {
    let lx = 36, row = 0;
    for (const s of (opts.series || []).filter((s) => s.label)) {
      const wItem = 18 + ctx.measureText(s.label).width;
      if (lx + wItem > w - 4 && lx > 36) { row++; lx = 36; }
      legendItems.push({ s, x: lx, y: row * 14 });
      lx += wItem + 4;
    }
  }
  const legendH = legendItems.length ? (legendItems[legendItems.length - 1].y + 16) : 0;
  const pad = { l: 36, r: 8, t: opts.legend ? legendH + 4 : 8, b: 20 };
  const all = [...(opts.series || []).flatMap((s) => s.points), ...(opts.bands || []).flatMap((b) => [...b.lo, ...b.hi])];
  if (!all.length) {
    ctx.fillStyle = COLORS.ink3;
    ctx.font = "13px Barlow, sans-serif";
    ctx.fillText(opts.empty || "No data yet", pad.l, h / 2);
    return;
  }
  let xMin = opts.xMin ?? Math.min(...all.map((p) => p[0]));
  let xMax = opts.xMax ?? Math.max(...all.map((p) => p[0]));
  let yMin = opts.yMin ?? Math.min(...all.map((p) => p[1]));
  let yMax = opts.yMax ?? Math.max(...all.map((p) => p[1]));
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) { yMax += 1; yMin -= opts.yMin != null ? 0 : 1; }
  const X = (x) => pad.l + ((x - xMin) / (xMax - xMin)) * (w - pad.l - pad.r);
  const Y = (y) => h - pad.b - ((y - yMin) / (yMax - yMin)) * (h - pad.t - pad.b);

  ctx.font = "11px 'Barlow Semi Condensed', Barlow, sans-serif";
  ctx.strokeStyle = COLORS.hair2;
  ctx.fillStyle = COLORS.ink3;
  ctx.lineWidth = 1;
  for (const t of niceTicks(yMin, yMax, 4)) {
    const y = Math.round(Y(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(Math.abs(t) >= 100 ? t.toFixed(0) : +t.toFixed(2), pad.l - 4, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of niceTicks(xMin, xMax, 6)) {
    if (opts.xInt && !Number.isInteger(t)) continue;
    ctx.fillText(opts.xFmt ? opts.xFmt(t) : t, X(t), h - pad.b + 4);
  }
  if (opts.yLabel) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(opts.yLabel, 2, 0); }

  ctx.save();
  ctx.beginPath(); ctx.rect(pad.l, pad.t - 2, w - pad.l - pad.r, h - pad.t - pad.b + 4); ctx.clip();
  for (const b of opts.bands || []) {
    if (!b.lo.length) continue;
    ctx.fillStyle = b.color;
    ctx.beginPath();
    b.lo.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
    for (let i = b.hi.length - 1; i >= 0; i--) ctx.lineTo(X(b.hi[i][0]), Y(b.hi[i][1]));
    ctx.closePath(); ctx.fill();
  }
  for (const s of opts.series || []) {
    if (!s.points.length) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    for (const p of s.points) {
      if (p[1] == null || !Number.isFinite(p[1])) { pen = false; continue; }
      if (pen) ctx.lineTo(X(p[0]), Y(p[1])); else { ctx.moveTo(X(p[0]), Y(p[1])); pen = true; }
    }
    ctx.stroke();
    if (s.dots) {
      ctx.fillStyle = s.color;
      for (const p of s.points) if (Number.isFinite(p[1])) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.6, 0, 7); ctx.fill(); }
    }
  }
  ctx.setLineDash([]);
  for (const m of opts.markers || []) {
    const x = Math.round(X(m.x)) + 0.5;
    ctx.strokeStyle = m.color; ctx.lineWidth = m.width || 1; ctx.setLineDash(m.dash || [2, 2]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (opts.cursorX != null) {
    const x = Math.round(X(opts.cursorX)) + 0.5;
    ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
  }
  ctx.restore();

  ctx.textAlign = "left"; ctx.textBaseline = "top";
  for (const it of legendItems) {
    ctx.fillStyle = it.s.color; ctx.fillRect(it.x, it.y + 5, 10, 3);
    ctx.fillStyle = COLORS.ink2; ctx.fillText(it.s.label, it.x + 13, it.y);
  }
  return { X, Y };
}
