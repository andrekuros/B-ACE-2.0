// Top-down tactical chart in an aeronautical-chart idiom.
// World: x east, z south (heading 0 flies toward -z, i.e. up the screen).
import { NM, COLORS, alpha, teamColor, toFt, fitCanvas, acName } from "./util.js";

const FSM_TEXT = { Search: "search", Engage: "engage", MissileSupport: "support", Evade: "evade" };
const TRAIL = 900;

export class TacticalChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.view = { x: 0, z: 0, s: 6 }; // s = px per GDM... set by fit
    this.layers = { radar: true, wez: true, tracks: true, trails: true, labels: true };
    this.selected = null;
    this.onSelect = null;
    this._bind();
  }

  _bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, vx: this.view.x, vz: this.view.z, moved: false };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; c.classList.add("dragging"); }
      this.view.x = drag.vx - dx / this.view.s;
      this.view.z = drag.vz - dy / this.view.s;
      this.userMoved = true;
      this.request?.();
    });
    c.addEventListener("pointerup", (e) => {
      c.classList.remove("dragging");
      if (drag && !drag.moved) this._click(e);
      drag = null;
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const before = this.toWorld(mx, my);
      this.view.s *= Math.exp(-e.deltaY * 0.0015);
      this.view.s = Math.max(0.05, Math.min(40, this.view.s));
      const after = this.toWorld(mx, my);
      this.view.x += before.x - after.x;
      this.view.z += before.z - after.z;
      this.userMoved = true;
      this.request?.();
    }, { passive: false });
    c.addEventListener("dblclick", () => { this.userMoved = false; this.fitTo(this._lastFrame); this.request?.(); });
  }

  toScreen(x, z) {
    return [this.w / 2 + (x - this.view.x) * this.view.s, this.h / 2 + (z - this.view.z) * this.view.s];
  }
  toWorld(px, py) {
    return { x: this.view.x + (px - this.w / 2) / this.view.s, z: this.view.z + (py - this.h / 2) / this.view.s };
  }

  _click(e) {
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 22;
    for (const f of this._drawn || []) {
      const d = Math.hypot(f.sx - mx, f.sy - my);
      if (d < bd) { bd = d; best = f.id; }
    }
    if (best != null) { this.selected = best; this.onSelect?.(best); this.request?.(); }
  }

  /** Fit the view to aircraft and their objectives. */
  fitTo(frame) {
    if (!frame) return;
    const { w, h } = fitCanvas(this.canvas);
    const pts = [];
    for (const f of frame.snapshot.fighters) {
      pts.push([f.pos[0], f.pos[2]]);
      if (f.target_pos) pts.push([f.target_pos[0], f.target_pos[2]]);
    }
    if (!pts.length) return;
    const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
    const minx = Math.min(...xs), maxx = Math.max(...xs), minz = Math.min(...zs), maxz = Math.max(...zs);
    this.view.x = (minx + maxx) / 2;
    this.view.z = (minz + maxz) / 2;
    const spanX = Math.max(maxx - minx, 20 * NM), spanZ = Math.max(maxz - minz, 20 * NM);
    this.view.s = Math.min((w - 120) / spanX, (h - 140) / spanZ);
  }

  /**
   * frames: episode frames; i: current index; frac: 0..1 toward i+1 (smooth playback).
   */
  draw(frames, i, frac = 0) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    this.w = w; this.h = h;
    const frame = frames[i];
    this._lastFrame = frames[0];
    ctx.fillStyle = COLORS.paper;
    ctx.fillRect(0, 0, w, h);
    if (!frame) return;
    if (!this.userMoved && !this._fitted) { this.fitTo(frames[0]); this._fitted = true; }

    this._grid(ctx);
    this._bullseye(ctx);
    const next = frac > 0 ? frames[i + 1] : null;
    const snap = frame.snapshot;
    const lerpPos = (id, pos, list) => {
      if (!next) return pos;
      const n = next.snapshot[list].find((o) => o.id === id);
      if (!n) return pos;
      return [pos[0] + (n.pos[0] - pos[0]) * frac, pos[1] + (n.pos[1] - pos[1]) * frac, pos[2] + (n.pos[2] - pos[2]) * frac];
    };
    const pos = new Map(snap.fighters.map((f) => [f.id, f.alive ? lerpPos(f.id, f.pos, "fighters") : f.pos]));

    this._objectives(ctx, snap);
    if (this.layers.trails) this._trails(ctx, frames, i, pos);
    this._missiles(ctx, frames, i, frac);
    if (this.layers.radar) for (const f of snap.fighters) if (f.alive && f.id === this.selected) this._radar(ctx, f, pos.get(f.id), true);
    if (this.layers.tracks) this._tracks(ctx, snap, pos);
    if (this.layers.wez) {
      const sel = snap.fighters.find((f) => f.id === this.selected && f.alive);
      if (sel) this._wez(ctx, sel, pos);
    }
    this._bursts(ctx, frames, i);
    this._drawn = [];
    this._labels = [];
    // Draw live aircraft last so they sit above wreckage.
    const order = [...snap.fighters].sort((a, b) => Number(a.alive) - Number(b.alive));
    for (const f of order) this._aircraft(ctx, f, pos.get(f.id), snap);
    this._scale(ctx);
  }

  _grid(ctx) {
    const step = 10 * NM;
    const a = this.toWorld(0, 0), b = this.toWorld(this.w, this.h);
    if (this.view.s * step < 14) return;
    ctx.strokeStyle = alpha(COLORS.ink3, 0.16);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = Math.floor(a.x / step) * step; x <= b.x; x += step) {
      const [sx] = this.toScreen(x, 0); ctx.moveTo(Math.round(sx) + 0.5, 0); ctx.lineTo(Math.round(sx) + 0.5, this.h);
    }
    for (let z = Math.floor(a.z / step) * step; z <= b.z; z += step) {
      const [, sy] = this.toScreen(0, z); ctx.moveTo(0, Math.round(sy) + 0.5); ctx.lineTo(this.w, Math.round(sy) + 0.5);
    }
    ctx.stroke();
  }

  _bullseye(ctx) {
    const [cx, cy] = this.toScreen(0, 0);
    const s = this.view.s;
    ctx.save();
    ctx.strokeStyle = alpha(COLORS.blue, 0.38);
    ctx.fillStyle = alpha(COLORS.blue, 0.65);
    ctx.lineWidth = 1;
    ctx.font = "500 11px 'Barlow Semi Condensed', sans-serif";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (const r of [10, 20, 30, 40, 50, 60]) {
      ctx.setLineDash(r % 20 === 0 ? [] : [2, 4]);
      ctx.beginPath(); ctx.arc(cx, cy, r * NM * s, 0, Math.PI * 2); ctx.stroke();
      if (r % 20 === 0) ctx.fillText(`${r} NM`, cx + 3, cy - r * NM * s + 7);
    }
    ctx.setLineDash([]);
    const R = Math.max(26, Math.min(70, 6 * NM * s));
    for (let d = 0; d < 360; d += 10) {
      const a = (d * Math.PI) / 180;
      const l = d % 90 === 0 ? 10 : d % 30 === 0 ? 6 : 3;
      ctx.beginPath();
      ctx.moveTo(cx + Math.sin(a) * R, cy - Math.cos(a) * R);
      ctx.lineTo(cx + Math.sin(a) * (R - l), cy - Math.cos(a) * (R - l));
      ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - R - 10); ctx.lineTo(cx - 4, cy - R); ctx.lineTo(cx + 4, cy - R); ctx.closePath(); ctx.fill();
    ctx.textAlign = "center";
    ctx.fillText("N", cx, cy - R - 18);
    ctx.fillText("Bullseye", cx, cy + R + 10);
    ctx.restore();
  }

  _objectives(ctx, snap) {
    const seen = new Set();
    for (const f of snap.fighters) {
      if (!f.target_pos) continue;
      const key = `${f.team}:${f.target_pos[0].toFixed(0)}:${f.target_pos[2].toFixed(0)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const [x, y] = this.toScreen(f.target_pos[0], f.target_pos[2]);
      const col = teamColor(f.team);
      ctx.save();
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.2;
      ctx.font = "500 11px 'Barlow Semi Condensed', sans-serif";
      ctx.textAlign = "center";
      if (f.mission === "striker") {
        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x - 14, y); ctx.lineTo(x + 14, y); ctx.moveTo(x, y - 14); ctx.lineTo(x, y + 14); ctx.stroke();
        ctx.fillText(`${f.team === 0 ? "Blue" : "Red"} strike target`, x, y + 25);
      } else {
        ctx.globalAlpha = 0.55;
        ctx.strokeRect(x - 4, y - 4, 8, 8);
        ctx.fillText(`${f.team === 0 ? "Blue" : "Red"} waypoint`, x, y + 17);
      }
      ctx.restore();
    }
  }

  _trails(ctx, frames, i, pos) {
    const start = Math.max(0, i - TRAIL);
    for (const f of frames[i].snapshot.fighters) {
      ctx.strokeStyle = alpha(teamColor(f.team), 0.45);
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      let first = true;
      for (let k = start; k <= i; k++) {
        const g = frames[k].snapshot.fighters.find((x) => x.id === f.id);
        if (!g) continue;
        const [x, y] = this.toScreen(g.pos[0], g.pos[2]);
        if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
        if (!g.alive) break;
      }
      if (f.alive) { const p = pos.get(f.id); ctx.lineTo(...this.toScreen(p[0], p[2])); }
      ctx.stroke();
    }
  }

  _missileColor(m) {
    if (m.pitbull) return COLORS.ochre;
    if (m.has_support) return COLORS.violet;
    return COLORS.ink3;
  }

  _missiles(ctx, frames, i, frac) {
    const snap = frames[i].snapshot;
    const next = frac > 0 ? frames[i + 1] : null;
    const live = new Set(snap.missiles.map((m) => m.id));
    // Trails for missiles alive now and for recently finished ones (fade).
    const recent = new Map();
    for (let k = Math.max(0, i - 160); k <= i; k++) {
      for (const m of frames[k].snapshot.missiles) {
        if (!recent.has(m.id)) recent.set(m.id, { pts: [], last: k, team: m.team });
        const r = recent.get(m.id);
        r.pts.push([m.pos[0], m.pos[2], m]);
        r.last = k;
      }
    }
    for (const [id, r] of recent) {
      const alive = live.has(id);
      const age = i - r.last;
      if (!alive && age > 40) continue;
      ctx.globalAlpha = alive ? 1 : Math.max(0.15, 1 - age / 40);
      ctx.lineWidth = 1.4;
      ctx.setLineDash(r.team === 0 ? [] : [5, 3]);
      for (let k = 1; k < r.pts.length; k++) {
        ctx.strokeStyle = this._missileColor(r.pts[k][2]);
        ctx.beginPath();
        ctx.moveTo(...this.toScreen(r.pts[k - 1][0], r.pts[k - 1][1]));
        ctx.lineTo(...this.toScreen(r.pts[k][0], r.pts[k][1]));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      if (alive) {
        const m = snap.missiles.find((x) => x.id === id);
        let p = m.pos;
        const n = next?.snapshot.missiles.find((x) => x.id === id);
        if (n) p = [p[0] + (n.pos[0] - p[0]) * frac, 0, p[2] + (n.pos[2] - p[2]) * frac];
        const [x, y] = this.toScreen(p[0], p[2]);
        const a = (m.hdg * Math.PI) / 180;
        ctx.fillStyle = this._missileColor(m);
        ctx.beginPath();
        ctx.moveTo(x + Math.sin(a) * 7, y - Math.cos(a) * 7);
        ctx.lineTo(x + Math.sin(a + 2.6) * 4, y - Math.cos(a + 2.6) * 4);
        ctx.lineTo(x + Math.sin(a - 2.6) * 4, y - Math.cos(a - 2.6) * 4);
        ctx.closePath(); ctx.fill();
        if (this.layers.labels && m.mach) {
          ctx.font = "500 10px 'Barlow Semi Condensed', sans-serif";
          ctx.textAlign = "left";
          ctx.fillText(`M${m.mach.toFixed(1)}`, x + 7, y + 3);
        }
      }
      ctx.globalAlpha = 1;
    }
  }

  _radar(ctx, f, p, selected) {
    const [x, y] = this.toScreen(p[0], p[2]);
    const r = f.radar_range * this.view.s;
    const a0 = ((f.hdg - f.radar_hfov - 90) * Math.PI) / 180;
    const a1 = ((f.hdg + f.radar_hfov - 90) * Math.PI) / 180;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, r, a0, a1);
    ctx.closePath();
    ctx.fillStyle = alpha(teamColor(f.team), selected ? 0.1 : 0.035);
    ctx.fill();
    if (selected) {
      ctx.strokeStyle = alpha(teamColor(f.team), 0.45);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  _tracks(ctx, snap, pos) {
    ctx.lineWidth = 1;
    ctx.setLineDash([1, 3]);
    for (const f of snap.fighters) {
      if (!f.alive) continue;
      const p = pos.get(f.id);
      for (const t of f.tracks || []) {
        if (!t.detected) continue;
        const q = pos.get(t.id);
        if (!q) continue;
        const strong = f.id === this.selected || t.id === f.hpt_id;
        ctx.strokeStyle = alpha(teamColor(f.team), strong ? 0.7 : 0.28);
        ctx.beginPath();
        ctx.moveTo(...this.toScreen(p[0], p[2]));
        ctx.lineTo(...this.toScreen(q[0], q[2]));
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  _arc(ctx, cx, cy, r, bearingDeg, spanDeg) {
    const a = ((bearingDeg - 90) * Math.PI) / 180;
    const s = (spanDeg * Math.PI) / 180;
    ctx.beginPath();
    ctx.arc(cx, cy, r, a - s, a + s);
    ctx.stroke();
  }

  _wez(ctx, f, pos) {
    const t = (f.tracks || []).find((x) => x.id === f.hpt_id && x.detected);
    if (!t) return;
    const p = pos.get(f.id), q = pos.get(t.id);
    if (!q) return;
    const [x, y] = this.toScreen(p[0], p[2]);
    const [ex, ey] = this.toScreen(q[0], q[2]);
    const brg = (Math.atan2(q[0] - p[0], -(q[2] - p[2])) * 180) / Math.PI;
    const s = this.view.s;
    const lbl = (txt, r, b, col) => {
      const a = ((b - 90) * Math.PI) / 180;
      ctx.fillStyle = col;
      ctx.fillText(txt, x + Math.cos(a) * r, y + Math.sin(a) * r - 6);
    };
    ctx.save();
    ctx.font = "500 11px 'Barlow Semi Condensed', sans-serif";
    ctx.textAlign = "center";
    const own = teamColor(f.team);
    ctx.strokeStyle = own; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    this._arc(ctx, x, y, t.own_r_max * s, brg, 14);
    ctx.setLineDash([]);
    if (t.own_r_nez) this._arc(ctx, x, y, t.own_r_nez * s, brg, 14);
    lbl("Rmax", t.own_r_max * s + 12, brg - 20, own);
    if (t.own_r_nez) lbl("NEZ", t.own_r_nez * s + 12, brg - 20, own);
    // Threat: enemy envelope against us, drawn from the enemy toward us.
    if (t.enemy_r_max) {
      const thr = f.team === 0 ? COLORS.red : COLORS.blue;
      ctx.strokeStyle = thr; ctx.lineWidth = 1.4;
      const back = brg + 180;
      ctx.setLineDash([6, 4]);
      this._arc(ctx, ex, ey, t.enemy_r_max * s, back, 10);
      ctx.setLineDash([]);
      if (t.enemy_r_nez) this._arc(ctx, ex, ey, t.enemy_r_nez * s, back, 10);
    }
    ctx.restore();
  }

  _bursts(ctx, frames, i) {
    const tNow = frames[i].snapshot.time_s ?? i;
    for (let k = Math.max(0, i - 60); k <= i; k++) {
      for (const e of frames[k].events || []) {
        if (e.kind !== "hit") continue;
        const v = frames[k].snapshot.fighters.find((f) => f.id === e.target);
        if (!v) continue;
        const [x, y] = this.toScreen(v.pos[0], v.pos[2]);
        const age = tNow - e.t;
        const r = 10 + Math.min(age, 6) * 3;
        ctx.strokeStyle = COLORS.ochre;
        ctx.globalAlpha = Math.max(0.25, 1 - age / 60);
        ctx.lineWidth = 1.6;
        for (let a = 0; a < 8; a++) {
          const ang = (a * Math.PI) / 4;
          ctx.beginPath();
          ctx.moveTo(x + Math.cos(ang) * 6, y + Math.sin(ang) * 6);
          ctx.lineTo(x + Math.cos(ang) * r, y + Math.sin(ang) * r);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
    }
  }

  _aircraft(ctx, f, p, snap) {
    const [x, y] = this.toScreen(p[0], p[2]);
    this._drawn.push({ id: f.id, sx: x, sy: y });
    const col = f.alive ? teamColor(f.team) : COLORS.ink3;
    const a = (f.hdg * Math.PI) / 180;
    ctx.save();
    if (f.id === this.selected) {
      ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
    if (f.alive) {
      // Velocity leader: 30 seconds of flight.
      const lead = Math.max(14, f.speed * 30 * this.view.s);
      ctx.beginPath(); ctx.moveTo(x + Math.sin(a) * 7, y - Math.cos(a) * 7);
      ctx.lineTo(x + Math.sin(a) * (7 + lead), y - Math.cos(a) * (7 + lead)); ctx.stroke();
    }
    ctx.fillStyle = COLORS.panel;
    ctx.beginPath();
    if (f.team === 0) ctx.arc(x, y, 7, 0, Math.PI * 2);
    else { ctx.moveTo(x, y - 8.5); ctx.lineTo(x + 8.5, y); ctx.lineTo(x, y + 8.5); ctx.lineTo(x - 8.5, y); ctx.closePath(); }
    ctx.fill(); ctx.stroke();
    if (!f.alive) {
      ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke();
    } else if (f.supporting_missile) {
      ctx.fillStyle = COLORS.violet;
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill();
    }
    if (this.layers.labels) {
      ctx.font = "600 12px 'Barlow Semi Condensed', sans-serif";
      ctx.fillStyle = f.alive ? COLORS.ink : COLORS.ink3;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      const name = acName(f.id, snap);
      // Greedy de-collision: nudge the label block down until it is clear.
      const bw = 118, bh = 26;
      let ly = y - 3;
      for (let k = 0; k < 6; k++) {
        const hit = this._labels.some((r) => x + 12 < r.x + r.w && x + 12 + bw > r.x && ly - 11 < r.y + r.h && ly - 11 + bh > r.y);
        if (!hit) break;
        ly += bh;
      }
      this._labels.push({ x: x + 12, y: ly - 11, w: bw, h: bh });
      if (ly !== y - 3) {
        ctx.strokeStyle = COLORS.hair; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x + 6, y + 3); ctx.lineTo(x + 11, ly - 4); ctx.stroke();
      }
      ctx.fillText(name, x + 12, ly);
      ctx.font = "500 11px 'Barlow Semi Condensed', sans-serif";
      ctx.fillStyle = COLORS.ink2;
      const fl = `FL${Math.round(toFt(p[1]) / 100).toString().padStart(3, "0")}`;
      const state = !f.alive ? "down" : f.behavior === "external" ? "RL" : f.behavior === "baseline1" ? FSM_TEXT[f.fsm] || "" : (f.behavior || "").replace("_", " ");
      ctx.fillText(`${fl}  ${f.missiles} msl  ${state}`, x + 12, ly + 13);
    }
    ctx.restore();
  }

  _scale(ctx) {
    const targetPx = 110;
    const nm = [1, 2, 5, 10, 20, 50].find((n) => n * NM * this.view.s >= targetPx * 0.6) || 50;
    const px = nm * NM * this.view.s;
    const x = this.w - px - 20, y = this.h - 44;
    ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x, y); ctx.lineTo(x + px, y); ctx.lineTo(x + px, y - 4); ctx.stroke();
    ctx.fillStyle = COLORS.ink; ctx.font = "500 11px 'Barlow Semi Condensed', sans-serif"; ctx.textAlign = "center";
    ctx.fillText(`${nm} NM`, x + px / 2, y - 6);
  }
}
