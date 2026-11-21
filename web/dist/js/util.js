// Units and small helpers shared across the Lab.
export const NM = 18.52;          // sim units (GDM = metres / 100) per nautical mile
export const FT = 0.003048;       // GDM per foot
export const toNM = (gdm) => gdm / NM;
export const toFt = (gdm) => gdm / FT;

const LIGHT = {
  paper: "#eaf0ec", panel: "#f6f8f7", ink: "#1b2a33", ink2: "#4a5b63", ink3: "#7b8b90",
  hair: "#b7c5c1", hair2: "#d4ddda",
  blue: "#1f5fa8", red: "#b3263e", ochre: "#c08a12", violet: "#6a4ab0", ok: "#2f7d4f",
};
// Night chart: deep slate ground, lifted inks for contrast.
const DARK = {
  paper: "#13202a", panel: "#18262f", ink: "#dbe5e9", ink2: "#a8b8bf", ink3: "#7d9099",
  hair: "#3a4f59", hair2: "#26363f",
  blue: "#6ea8ec", red: "#ec6f84", ochre: "#e2b04c", violet: "#ab8cf0", ok: "#64c28a",
};
export const COLORS = { ...LIGHT };

/** hex colour + alpha -> rgba() string. */
export function alpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

const THEME_KEY = "bace-lab-theme";
export function themePreference() {
  try { return localStorage.getItem(THEME_KEY) || "auto"; } catch { return "auto"; }
}
/** Apply "light", "dark" or "auto" (follow the system). Returns the resolved theme. */
export function applyTheme(pref = themePreference()) {
  const dark = pref === "dark" || (pref === "auto" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  Object.assign(COLORS, dark ? DARK : LIGHT);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  try { localStorage.setItem(THEME_KEY, pref); } catch { /* private mode */ }
  return dark ? "dark" : "light";
}

export const teamColor = (team) => (team === 0 ? COLORS.blue : COLORS.red);

/** Display name for an aircraft id (101.. blue, 201.. red). */
export function acName(id, snapshot) {
  const f = snapshot?.fighters?.find((x) => x.id === id);
  if (f?.agent_name) return f.agent_name;
  const team = id >= 200 ? "Red" : "Blue";
  return `${team} ${id % 100}`;
}

/** Deterministic PRNG (mulberry32) for reproducible browser-side policies. */
export function prng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const fmtTime = (s) => {
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${ss}`;
};
export const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");
export const $ = (id) => document.getElementById(id);
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export function toast(msg) {
  const t = $("toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._h);
  toast._h = setTimeout(() => t.classList.remove("show"), 2200);
}

export function download(name, text, type = "application/json") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** Size a canvas to its CSS box at device pixel ratio; returns a 2D context in CSS pixels. */
export function fitCanvas(canvas, cssHeight) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 300;
  const h = cssHeight ?? (canvas.clientHeight || Number(canvas.getAttribute("height")) || 150);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    if (cssHeight != null) canvas.style.height = `${h}px`;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
