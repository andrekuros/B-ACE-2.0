// Paper metrics (Kuroswiski, Wu & Passaro) with percentile bootstrap CIs.
import { prng } from "./util.js";

export const METRICS = {
  eprew: { label: "Return", fmt: 2, per: (o) => o.episode_return },
  macc: { label: "Mission", fmt: 2, per: (o) => (o.mission_success ? 1 : 0) },
  bscore: { label: "B-score", fmt: 2, per: (o) => o.blue_kills },
  rscore: { label: "R-score", fmt: 2, per: (o) => o.blue_deaths },
  fscore: { label: "F-score", fmt: 2, per: (o) => o.blue_kills - o.blue_deaths },
  beff: { label: "Missile eff.", fmt: 2, pooled: (os) => { const f = os.reduce((a, o) => a + o.missiles_fired, 0); return f ? os.reduce((a, o) => a + o.missile_hits, 0) / f : NaN; } },
};

function stat(def, os) {
  if (def.pooled) return def.pooled(os);
  return os.reduce((a, o) => a + def.per(o), 0) / Math.max(1, os.length);
}

/** Mean and 95% percentile-bootstrap CI over episodes. Deterministic (seeded). */
export function summarize(outcomes, B = 1000, seed = 12345) {
  const rnd = prng(seed);
  const n = outcomes.length;
  const res = {};
  for (const [k, def] of Object.entries(METRICS)) {
    const mean = stat(def, outcomes);
    const boots = [];
    for (let b = 0; b < B; b++) {
      const s = new Array(n);
      for (let i = 0; i < n; i++) s[i] = outcomes[(rnd() * n) | 0];
      const v = stat(def, s);
      if (Number.isFinite(v)) boots.push(v);
    }
    boots.sort((a, b) => a - b);
    const q = (p) => (boots.length ? boots[Math.min(boots.length - 1, Math.floor(p * boots.length))] : NaN);
    res[k] = { mean, lo: q(0.025), hi: q(0.975) };
  }
  res.n = n;
  return res;
}
