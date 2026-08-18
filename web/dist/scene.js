import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import {
  createFighter,
  createMissile,
  createRadarVolume,
  createAltitudeStem,
  createWezRing,
  DEFAULT_VIEW_CONFIG,
} from "./models.js";
import { createLabel, createLegend, formatFighterLabel } from "./hud.js";

const NM = 18.52;
const TRAIL_LEN = 28;
const SPLIT_CAP = 4;

function headingQuat(hdgDeg, pitchDeg = 0) {
  const q = new THREE.Quaternion();
  q.setFromEuler(
    new THREE.Euler(-THREE.MathUtils.degToRad(pitchDeg || 0), Math.PI - THREE.MathUtils.degToRad(hdgDeg || 0), 0, "YXZ")
  );
  return q;
}

function teamColor(team, alive) {
  const c = team === 0 ? 0x4aa3ff : 0xff6b5a;
  return alive ? c : 0x667788;
}

function gridFor(n) {
  if (n <= 1) return { cols: 1, rows: 1 };
  if (n === 2) return { cols: 2, rows: 1 };
  return { cols: 2, rows: 2 };
}

export class CombatScene {
  constructor(canvas, { hudRoot, legendHost, split = false } = {}) {
    this.canvas = canvas;
    this.hudRoot = hudRoot || canvas.parentElement;
    this.splitEnabled = split;
    this.config = { ...DEFAULT_VIEW_CONFIG, layout: split ? "split" : "focus" };
    this.cameraMode = this.config.camera;
    this.followId = null;
    this.focusIndex = 0;
    this.snapshots = [];
    this.clock = new THREE.Clock();
    this.fighters = new Map();
    this.missiles = new Map();
    this.trails = new Map();
    this.onFocus = null;
    this._raf = 0;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b141c);
    this.scene.fog = new THREE.Fog(0x0b141c, 420, 3200);

    const w = canvas.clientWidth || 900;
    const h = canvas.clientHeight || 560;
    this.camera = new THREE.PerspectiveCamera(48, w / h, 0.5, 8000);
    this.camera.position.set(220, 280, 520);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.setScissorTest(false);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(w, h);
    this.labelRenderer.domElement.className = "label-layer";
    this.hudRoot.appendChild(this.labelRenderer.domElement);

    this.paneLayer = document.createElement("div");
    this.paneLayer.className = "pane-layer";
    this.hudRoot.appendChild(this.paneLayer);

    this.controls = new OrbitControls(this.camera, this.labelRenderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 2400;
    this.controls.target.set(0, 80, 0);

    this._lights();
    this._theater();
    if (legendHost) createLegend(legendHost);

    this.labelRenderer.domElement.addEventListener("click", (ev) => this._onClick(ev));

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(this.hudRoot);
    this.resize();
    this._loop = this._loop.bind(this);
    this._raf = requestAnimationFrame(this._loop);
  }

  simToWorld(pos) {
    return new THREE.Vector3(pos[0], pos[1] * this.config.altScale, pos[2]);
  }

  setConfig(patch) {
    Object.assign(this.config, patch);
    if (patch.camera) this.setCameraMode(patch.camera);
    if (patch.layout === "split" && this.cameraMode === "chase") {
      this.setCameraMode("orbit");
    }
    this._syncLabels();
  }

  setCameraMode(mode) {
    if (this.config.layout === "split" && mode === "chase") mode = "orbit";
    this.cameraMode = mode;
    this.config.camera = mode;
    this.controls.enabled = mode === "orbit";
    if (mode === "tactical") {
      this.camera.position.set(0, 980, 40);
      this.controls.target.set(0, 40, 0);
      this.camera.lookAt(this.controls.target);
    }
    if (mode === "orbit" && this.camera.position.y > 800) {
      this.camera.position.set(220, 280, 520);
      this.controls.target.set(0, 80, 0);
    }
  }

  setFollowId(id) {
    this.followId = id;
  }

  setFocusIndex(i) {
    this.focusIndex = Math.max(0, i | 0);
  }

  applyAll(snapshots, focusIndex = 0) {
    this.snapshots = snapshots || [];
    this.focusIndex = Math.min(focusIndex, Math.max(0, this.snapshots.length - 1));
  }

  apply(snapshot, envKey = 0) {
    if (!snapshot) return;
    const byId = new Map((snapshot.fighters || []).map((f) => [f.id, f]));
    const seenF = new Set();
    for (const f of snapshot.fighters || []) {
      seenF.add(f.id);
      let rec = this.fighters.get(f.id);
      if (!rec) {
        rec = this._spawnFighter(f);
        this.fighters.set(f.id, rec);
      }
      this._updateFighter(rec, f, byId);
    }
    for (const id of [...this.fighters.keys()]) {
      if (!seenF.has(id)) this._removeFighter(id);
    }

    const seenM = new Set();
    for (const m of snapshot.missiles || []) {
      seenM.add(m.id);
      let rec = this.missiles.get(m.id);
      if (!rec) {
        rec = this._spawnMissile(m);
        this.missiles.set(m.id, rec);
      }
      this._updateMissile(rec, m, byId, envKey);
    }
    for (const id of [...this.missiles.keys()]) {
      if (!seenM.has(id)) this._removeMissile(id);
    }

    for (const [key, line] of this.trails) {
      if (!key.startsWith(`${envKey}:`)) line.visible = false;
    }

    if (!this.followId) {
      const blue = (snapshot.fighters || []).find((f) => f.team === 0 && f.alive);
      this.followId = blue?.id ?? snapshot.fighters?.[0]?.id ?? null;
    }
  }

  _spawnFighter(f) {
    const mesh = createFighter(f.team);
    const radar = createRadarVolume();
    const stem = createAltitudeStem();
    const wez = createWezRing();
    const tracks = new THREE.Group();
    tracks.name = "tracks";
    const labelEl = createLabel(formatFighterLabel(f));
    const label = new CSS2DObject(labelEl);
    label.position.set(0, 5.4, 0);
    mesh.add(radar, wez, label);
    this.scene.add(mesh, stem, tracks);
    return { mesh, radar, stem, wez, tracks, labelEl, label };
  }

  _updateFighter(rec, f, byId) {
    const p = this.simToWorld(f.pos);
    rec.mesh.position.copy(p);
    rec.mesh.quaternion.copy(headingQuat(f.hdg, f.pitch));
    rec.mesh.scale.setScalar(this.config.aircraftScale);
    rec.mesh.visible = true;
    rec.mesh.traverse((ch) => {
      if (ch.isMesh && ch.material && "opacity" in ch.material && ch.name !== "radar") {
        ch.material.transparent = !f.alive;
        if (ch.name !== "exhaust") ch.material.opacity = f.alive ? 1 : 0.28;
      }
    });

    const exhaust = rec.mesh.getObjectByName("exhaust");
    if (exhaust?.material) {
      exhaust.material.emissiveIntensity = f.alive ? 1.2 + 0.6 * Math.sin(this.clock.elapsedTime * 8) : 0.05;
    }
    const glow = rec.mesh.getObjectByName("exhaustLight");
    if (glow) glow.intensity = f.alive ? 1.6 : 0;

    rec.labelEl.textContent = formatFighterLabel(f);
    rec.labelEl.classList.toggle("dead", !f.alive);
    rec.labelEl.classList.toggle("blue", f.team === 0);
    rec.labelEl.classList.toggle("red", f.team === 1);
    rec.label.visible = !!this.config.labels && this.config.layout === "focus";

    rec.stem.visible = !!this.config.stems;
    rec.stem.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([p.x, 0, p.z, p.x, p.y, p.z], 3)
    );
    rec.stem.material.color.setHex(teamColor(f.team, f.alive));
    rec.stem.material.opacity = f.alive ? 0.4 : 0.15;

    const range = f.radar_range || 50 * NM;
    const hfov = THREE.MathUtils.degToRad(f.radar_hfov || 60);
    const vfov = THREE.MathUtils.degToRad(((f.radar_vfov_up || 40) + (f.radar_vfov_down || 20)) / 2);
    rec.radar.visible = !!f.alive && !!this.config.radar;
    const ac = Math.max(0.01, this.config.aircraftScale);
    rec.radar.scale.set(
      (range * Math.tan(hfov)) / ac,
      range / ac,
      (range * Math.tan(vfov)) / ac
    );
    rec.radar.position.set(0, 0, range / 2);
    rec.radar.material.color.setHex(f.team === 0 ? 0x4aa3ff : 0xff6b5a);
    const hasTrack = (f.tracks || []).some((t) => t.detected);
    rec.radar.material.opacity = f.alive ? (hasTrack ? 0.13 : 0.07) : 0;

    const hpt = (f.tracks || []).find((t) => t.detected && t.id === f.hpt_id);
    if (f.alive && this.config.wez && hpt && hpt.own_r_max > 1) {
      rec.wez.visible = true;
      rec.wez.scale.setScalar(hpt.own_r_max / ac);
    } else {
      rec.wez.visible = false;
    }

    while (rec.tracks.children.length) {
      const ch = rec.tracks.children.pop();
      ch.geometry?.dispose();
      ch.material?.dispose();
    }
    rec.tracks.visible = !!this.config.tracks;
    if (f.alive && this.config.tracks) {
      for (const t of f.tracks || []) {
        if (!t.detected) continue;
        const tgt = byId.get(t.id);
        if (!tgt) continue;
        const tp = this.simToWorld(tgt.pos);
        const isHpt = t.id === f.hpt_id;
        const mat = t.is_missile_support
          ? new THREE.LineDashedMaterial({
              color: 0xc4b5fd,
              dashSize: 8,
              gapSize: 6,
              transparent: true,
              opacity: 0.85,
            })
          : new THREE.LineBasicMaterial({
              color: isHpt ? 0xffd166 : teamColor(f.team, true),
              transparent: true,
              opacity: isHpt ? 0.95 : 0.45,
            });
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([p, tp]), mat);
        if (t.is_missile_support) line.computeLineDistances();
        rec.tracks.add(line);
      }
    }
  }

  _removeFighter(id) {
    const rec = this.fighters.get(id);
    if (!rec) return;
    this.scene.remove(rec.mesh, rec.stem, rec.tracks);
    rec.labelEl.remove();
    this.fighters.delete(id);
  }

  _spawnMissile(m) {
    const mesh = createMissile(m.team);
    this.scene.add(mesh);
    const trailGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]);
    const trail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({
        color: m.team === 0 ? 0x9ad0ff : 0xffb3a8,
        transparent: true,
        opacity: 0.95,
        linewidth: 2,
      })
    );
    this.scene.add(trail);
    trail.visible = false;
    return { mesh, trail, hist: new Map() };
  }

  _updateMissile(rec, m, byId, envKey) {
    const p = this.simToWorld(m.pos);
    rec.mesh.position.copy(p);
    rec.mesh.quaternion.copy(headingQuat(m.hdg, 0));
    rec.mesh.scale.setScalar(this.config.missileScale);
    const plume = rec.mesh.getObjectByName("plume");
    const light = rec.mesh.getObjectByName("plumeLight");
    const halo = rec.mesh.getObjectByName("halo");
    const hot = m.pitbull ? 0xffd166 : m.has_support ? 0xc4b5fd : rec.mesh.userData.team === 0 ? 0x66e0ff : 0xff8a4a;
    if (plume?.material) {
      plume.material.emissive.setHex(hot);
      plume.material.color.setHex(hot);
      plume.material.emissiveIntensity = m.pitbull ? 3.2 : 2.0;
    }
    if (halo?.material) {
      halo.material.color.setHex(hot);
      halo.material.opacity = m.pitbull ? 0.8 : 0.5;
    }
    if (light) {
      light.color.setHex(hot);
      light.intensity = m.pitbull ? 3.2 : 1.8;
    }

    const key = `${envKey}:${m.id}`;
    let hist = rec.hist.get(key) || [];
    hist.push(p.clone());
    if (hist.length > TRAIL_LEN) hist = hist.slice(-TRAIL_LEN);
    rec.hist.set(key, hist);
    if (hist.length >= 2) {
      rec.trail.geometry.setFromPoints(hist);
      rec.trail.visible = true;
      rec.trail.material.color.setHex(hot);
    }
    this.trails.set(key, rec.trail);

    rec.mesh.userData.target = byId.get(m.target_id)?.id;
  }

  _removeMissile(id) {
    const rec = this.missiles.get(id);
    if (!rec) return;
    this.scene.remove(rec.mesh, rec.trail);
    rec.trail.geometry.dispose();
    rec.trail.material.dispose();
    this.missiles.delete(id);
  }

  _visiblePanes() {
    const n = this.snapshots.length;
    if (!n) return [];
    if (!this.splitEnabled || this.config.layout === "focus") {
      const i = Math.min(this.focusIndex, n - 1);
      return [{ index: i, snap: this.snapshots[i] }];
    }
    return this.snapshots.slice(0, SPLIT_CAP).map((snap, index) => ({ index, snap }));
  }

  _paneRect(i, count, width, height) {
    const { cols, rows } = gridFor(count);
    const col = i % cols;
    const row = Math.floor(i / cols);
    const pw = width / cols;
    const ph = height / rows;
    return { x: col * pw, y: height - (row + 1) * ph, w: pw, h: ph, col, row, cols, rows };
  }

  _updateCamera(dt) {
    if (this.cameraMode === "orbit") {
      this.controls.update(dt);
      return;
    }
    const rec = this.followId != null ? this.fighters.get(this.followId) : null;
    if (this.cameraMode === "chase" && rec && this.config.layout === "focus") {
      const back = new THREE.Vector3(0, 10, -52).applyQuaternion(rec.mesh.quaternion);
      const desired = rec.mesh.position.clone().add(back);
      this.camera.position.lerp(desired, 1 - Math.pow(0.001, dt));
      const look = rec.mesh.position.clone().add(new THREE.Vector3(0, 4, 22).applyQuaternion(rec.mesh.quaternion));
      this.camera.lookAt(look);
      return;
    }
    if (this.cameraMode === "tactical") {
      const c = this._centroid();
      const desired = new THREE.Vector3(c.x, 920, c.z + 8);
      this.camera.position.lerp(desired, 0.08);
      this.camera.lookAt(c.x, 20, c.z);
    }
  }

  _centroid() {
    const pts = [...this.fighters.values()].map((r) => r.mesh.position);
    if (!pts.length) return new THREE.Vector3(0, 80, 0);
    const c = new THREE.Vector3();
    pts.forEach((p) => c.add(p));
    return c.multiplyScalar(1 / pts.length);
  }

  _syncLabels() {
    const split = this.splitEnabled && this.config.layout === "split";
    this.labelRenderer.domElement.style.display = split ? "none" : "block";
  }

  _drawPanes(panes, width, height) {
    const split = panes.length > 1;
    this.paneLayer.classList.toggle("hidden", !split && panes.length === 0);
    while (this.paneLayer.childElementCount > panes.length) {
      this.paneLayer.lastChild.remove();
    }
    panes.forEach((pane, i) => {
      let el = this.paneLayer.children[i];
      if (!el) {
        el = document.createElement("button");
        el.type = "button";
        el.className = "pane-tag";
        this.paneLayer.appendChild(el);
        el.addEventListener("click", (ev) => {
          ev.stopPropagation();
          if (this.onFocus) this.onFocus(pane.index);
        });
      }
      const r = this._paneRect(i, panes.length, width, height);
      el.style.left = `${(r.x / width) * 100}%`;
      el.style.top = `${((height - r.y - r.h) / height) * 100}%`;
      el.style.width = `${(r.w / width) * 100}%`;
      el.style.height = `${(r.h / height) * 100}%`;
      const s = pane.snap || {};
      el.textContent = `ENV ${pane.index + 1}  ·  step ${s.action_step ?? 0}  ·  ${s.end ?? "—"}`;
      el.classList.toggle("solo", panes.length === 1);
    });
  }

  _onClick(ev) {
    const panes = this._visiblePanes();
    if (panes.length <= 1 || !this.onFocus) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const { cols, rows } = gridFor(panes.length);
    const col = Math.min(cols - 1, Math.floor((x / rect.width) * cols));
    const row = Math.min(rows - 1, Math.floor((y / rect.height) * rows));
    const i = row * cols + col;
    if (panes[i]) this.onFocus(panes[i].index);
  }

  _loop() {
    this._raf = requestAnimationFrame(this._loop);
    const dt = this.clock.getDelta();
    this._updateCamera(dt);
    this._syncLabels();

    const width = this.canvas.clientWidth || 900;
    const height = Math.max(320, this.hudRoot.clientHeight || 560);
    const panes = this._visiblePanes();
    this._drawPanes(panes, width, height);

    if (!panes.length) {
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, width, height);
      this.renderer.render(this.scene, this.camera);
      return;
    }

    if (panes.length === 1) {
      this.apply(panes[0].snap, panes[0].index);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, width, height);
      this.renderer.render(this.scene, this.camera);
      if (this.config.labels) this.labelRenderer.render(this.scene, this.camera);
      return;
    }

    this.renderer.setScissorTest(true);
    panes.forEach((pane, i) => {
      this.apply(pane.snap, pane.index);
      const r = this._paneRect(i, panes.length, width, height);
      this.camera.aspect = r.w / r.h;
      this.camera.updateProjectionMatrix();
      this.renderer.setViewport(r.x, r.y, r.w, r.h);
      this.renderer.setScissor(r.x, r.y, r.w, r.h);
      this.renderer.render(this.scene, this.camera);
    });
    this.renderer.setScissorTest(false);
  }

  resize() {
    const w = this.hudRoot.clientWidth || this.canvas.clientWidth || 900;
    const h = Math.max(320, this.hudRoot.clientHeight || 560);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    this.labelRenderer.setSize(w, h);
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this.controls.dispose();
    this.labelRenderer.domElement.remove();
    this.paneLayer.remove();
    this.renderer.dispose();
  }
}
