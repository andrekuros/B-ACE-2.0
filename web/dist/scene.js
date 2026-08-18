import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import {
  createFighter,
  createMissile,
  createRadarVolume,
  createAltitudeStem,
  createWezRing,
} from "./models.js";
import { createLabel, createLegend, formatFighterLabel } from "./hud.js";

const NM = 18.52;
const ALT_SCALE = 4.0;
const TRAIL_LEN = 28;

function simToWorld(pos) {
  return new THREE.Vector3(pos[0], pos[1] * ALT_SCALE, pos[2]);
}

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

export class CombatScene {
  constructor(canvas, { hudRoot, legendHost } = {}) {
    this.canvas = canvas;
    this.hudRoot = hudRoot || canvas.parentElement;
    this.cameraMode = "orbit";
    this.followId = null;
    this.clock = new THREE.Clock();
    this.fighters = new Map();
    this.missiles = new Map();
    this.trails = new Map();
    this.snapshot = null;
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

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(w, h);
    this.labelRenderer.domElement.className = "label-layer";
    this.hudRoot.appendChild(this.labelRenderer.domElement);

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

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(this.hudRoot);
    this.resize();
    this._loop = this._loop.bind(this);
    this._raf = requestAnimationFrame(this._loop);
  }

  _lights() {
    this.scene.add(new THREE.HemisphereLight(0x8eb0c8, 0x1a2a22, 0.85));
    const sun = new THREE.DirectionalLight(0xf2efe6, 1.35);
    sun.position.set(400, 720, 180);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 80;
    sun.shadow.camera.far = 2000;
    sun.shadow.camera.left = -400;
    sun.shadow.camera.right = 400;
    sun.shadow.camera.top = 400;
    sun.shadow.camera.bottom = -400;
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0x4aa3ff, 0.22);
    fill.position.set(-300, 120, -200);
    this.scene.add(fill);
  }

  _theater() {
    const sea = new THREE.Mesh(
      new THREE.CircleGeometry(90 * NM, 72),
      new THREE.MeshStandardMaterial({
        color: 0x10202c,
        metalness: 0.35,
        roughness: 0.72,
      })
    );
    sea.rotation.x = -Math.PI / 2;
    sea.receiveShadow = true;
    this.scene.add(sea);

    const grid = new THREE.GridHelper(80 * NM, 16, 0x2a4254, 0x173040);
    grid.position.y = 0.4;
    grid.material.transparent = true;
    grid.material.opacity = 0.45;
    this.scene.add(grid);

    const ringMat = new THREE.LineBasicMaterial({ color: 0x3d5c72, transparent: true, opacity: 0.55 });
    for (const nm of [10, 20, 30, 40, 50, 60]) {
      const pts = [];
      for (let i = 0; i <= 96; i++) {
        const a = (i / 96) * Math.PI * 2;
        pts.push(new THREE.Vector3(Math.sin(a) * nm * NM, 0.8, Math.cos(a) * nm * NM));
      }
      this.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
    }

    const axis = new THREE.Group();
    const mk = (from, to, color) => {
      const g = new THREE.BufferGeometry().setFromPoints([from, to]);
      return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
    };
    axis.add(mk(new THREE.Vector3(-70 * NM, 1, 0), new THREE.Vector3(70 * NM, 1, 0), 0x4aa3ff));
    axis.add(mk(new THREE.Vector3(0, 1, -70 * NM), new THREE.Vector3(0, 1, 70 * NM), 0xff6b5a));
    this.scene.add(axis);

    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(3500, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0x12202c, side: THREE.BackSide })
    );
    this.scene.add(sky);
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    this.controls.enabled = mode === "orbit";
    if (mode === "tactical") {
      this.camera.position.set(0, 980, 40);
      this.controls.target.set(0, 40, 0);
      this.camera.lookAt(this.controls.target);
    }
  }

  setFollowId(id) {
    this.followId = id;
  }

  apply(snapshot) {
    this.snapshot = snapshot || null;
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
      this._updateMissile(rec, m, byId);
    }
    for (const id of [...this.missiles.keys()]) {
      if (!seenM.has(id)) this._removeMissile(id);
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
    label.position.set(0, 4.2, 0);
    mesh.add(radar, wez, label);
    this.scene.add(mesh, stem, tracks);
    return { mesh, radar, stem, wez, tracks, labelEl, label };
  }

  _updateFighter(rec, f, byId) {
    const p = simToWorld(f.pos);
    rec.mesh.position.copy(p);
    rec.mesh.quaternion.copy(headingQuat(f.hdg, f.pitch));
    rec.mesh.visible = true;
    rec.mesh.traverse((ch) => {
      if (ch.isMesh && ch.material && "opacity" in ch.material && ch.name !== "radar") {
        ch.material.transparent = !f.alive;
        if (ch.name !== "exhaust") ch.material.opacity = f.alive ? 1 : 0.28;
      }
    });

    const exhaust = rec.mesh.getObjectByName("exhaust");
    if (exhaust?.material) {
      exhaust.material.emissiveIntensity = f.alive ? 0.7 + 0.5 * Math.sin(this.clock.elapsedTime * 8) : 0.05;
    }
    const glow = rec.mesh.getObjectByName("exhaustLight");
    if (glow) glow.intensity = f.alive ? 1.2 : 0;

    rec.labelEl.textContent = formatFighterLabel(f);
    rec.labelEl.classList.toggle("dead", !f.alive);
    rec.labelEl.classList.toggle("blue", f.team === 0);
    rec.labelEl.classList.toggle("red", f.team === 1);

    rec.stem.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([p.x, 0, p.z, p.x, p.y, p.z], 3)
    );
    rec.stem.material.color.setHex(teamColor(f.team, f.alive));
    rec.stem.material.opacity = f.alive ? 0.4 : 0.15;

    const range = f.radar_range || 50 * NM;
    const hfov = THREE.MathUtils.degToRad(f.radar_hfov || 60);
    const vfov = THREE.MathUtils.degToRad(((f.radar_vfov_up || 40) + (f.radar_vfov_down || 20)) / 2);
    rec.radar.visible = !!f.alive;
    rec.radar.scale.set(range * Math.tan(hfov), range, range * Math.tan(vfov));
    rec.radar.position.set(0, 0, range / 2);
    rec.radar.material.color.setHex(f.team === 0 ? 0x4aa3ff : 0xff6b5a);
    const hasTrack = (f.tracks || []).some((t) => t.detected);
    rec.radar.material.opacity = f.alive ? (hasTrack ? 0.13 : 0.07) : 0;

    const hpt = (f.tracks || []).find((t) => t.detected && t.id === f.hpt_id);
    if (f.alive && hpt && hpt.own_r_max > 1) {
      rec.wez.visible = true;
      rec.wez.scale.setScalar(hpt.own_r_max);
      rec.wez.position.set(0, 0, 0);
    } else {
      rec.wez.visible = false;
    }

    while (rec.tracks.children.length) {
      const ch = rec.tracks.children.pop();
      ch.geometry?.dispose();
      ch.material?.dispose();
    }
    if (f.alive) {
      for (const t of f.tracks || []) {
        if (!t.detected) continue;
        const tgt = byId.get(t.id);
        if (!tgt) continue;
        const tp = simToWorld(tgt.pos);
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
    this.trails.set(m.id, []);
    const trailGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]);
    const trail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ color: m.team === 0 ? 0x9ad0ff : 0xffb3a8, transparent: true, opacity: 0.7 })
    );
    this.scene.add(trail);
    trail.visible = false;
    return { mesh, trail };
  }

  _updateMissile(rec, m, byId) {
    const p = simToWorld(m.pos);
    rec.mesh.position.copy(p);
    rec.mesh.quaternion.copy(headingQuat(m.hdg, 0));
    const plume = rec.mesh.getObjectByName("plume");
    const light = rec.mesh.getObjectByName("plumeLight");
    const hot = m.pitbull ? 0xffd166 : m.has_support ? 0xc4b5fd : 0xffffff;
    if (plume?.material) {
      plume.material.emissive.setHex(hot);
      plume.material.color.setHex(hot);
      plume.material.emissiveIntensity = m.pitbull ? 2.2 : 1.2;
    }
    if (light) {
      light.color.setHex(hot);
      light.intensity = m.pitbull ? 2.0 : 0.8;
    }
    rec.trail.material.color.setHex(m.pitbull ? 0xffd166 : m.has_support ? 0xc4b5fd : 0xffffff);

    let hist = this.trails.get(m.id) || [];
    hist.push(p.clone());
    if (hist.length > TRAIL_LEN) hist = hist.slice(-TRAIL_LEN);
    this.trails.set(m.id, hist);
    if (hist.length >= 2) {
      rec.trail.geometry.setFromPoints(hist);
      rec.trail.visible = true;
    }

    const tgt = byId.get(m.target_id);
    rec.mesh.userData.target = tgt?.id;
  }

  _removeMissile(id) {
    const rec = this.missiles.get(id);
    if (!rec) return;
    this.scene.remove(rec.mesh, rec.trail);
    rec.trail.geometry.dispose();
    rec.trail.material.dispose();
    this.missiles.delete(id);
    this.trails.delete(id);
  }

  _updateCamera(dt) {
    if (this.cameraMode === "orbit") {
      this.controls.update(dt);
      return;
    }
    const rec = this.followId != null ? this.fighters.get(this.followId) : null;
    if (this.cameraMode === "chase" && rec) {
      const back = new THREE.Vector3(0, 8, -42).applyQuaternion(rec.mesh.quaternion);
      const desired = rec.mesh.position.clone().add(back);
      this.camera.position.lerp(desired, 1 - Math.pow(0.001, dt));
      const look = rec.mesh.position.clone().add(new THREE.Vector3(0, 3, 18).applyQuaternion(rec.mesh.quaternion));
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

  _loop() {
    this._raf = requestAnimationFrame(this._loop);
    const dt = this.clock.getDelta();
    this._updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
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
    this.renderer.dispose();
  }
}
