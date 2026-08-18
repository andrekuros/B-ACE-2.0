import * as THREE from "three";

const BLUE = {
  skin: 0x3d8fff,
  accent: 0x9fd0ff,
  exhaust: 0x66e0ff,
};
const RED = {
  skin: 0xff5a4a,
  accent: 0xffb3a8,
  exhaust: 0xff8a4a,
};

function mat(color, extras = {}) {
  return new THREE.MeshStandardMaterial({
    color,
    metalness: 0.18,
    roughness: 0.42,
    ...extras,
  });
}

/** Thin vertical triangle in the XZ plane, apex toward +Z (nose). */
function deltaWing(span, chord) {
  const geo = new THREE.BufferGeometry();
  const s = span / 2;
  const verts = new Float32Array([
    0, 0.04, chord * 0.15,
    s, 0, -chord * 0.85,
    0, 0, -chord,
    0, -0.04, chord * 0.15,
    0, 0, -chord,
    s, 0, -chord * 0.85,
  ]);
  geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geo.computeVertexNormals();
  return geo;
}

/** Stylized fighter: vertical slab fuselage + pyramid/delta wings. +Z is nose. */
export function createFighter(team) {
  const pal = team === 0 ? BLUE : RED;
  const root = new THREE.Group();
  root.name = "fighter";

  const bodyMat = mat(pal.skin, { metalness: 0.22, roughness: 0.38 });
  const accentMat = mat(pal.accent, { emissive: pal.skin, emissiveIntensity: 0.18 });

  const fuse = new THREE.Mesh(new THREE.BoxGeometry(0.55, 2.4, 11.5), bodyMat);
  fuse.castShadow = true;
  root.add(fuse);

  const nose = new THREE.Mesh(new THREE.ConeGeometry(1.15, 5.2, 4), accentMat);
  nose.rotation.x = Math.PI / 2;
  nose.position.z = 7.6;
  nose.castShadow = true;
  root.add(nose);

  const canopy = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.55, 0),
    new THREE.MeshStandardMaterial({
      color: 0xdef4ff,
      emissive: 0x88c8ff,
      emissiveIntensity: 0.35,
      metalness: 0.1,
      roughness: 0.12,
      transparent: true,
      opacity: 0.85,
    })
  );
  canopy.position.set(0, 1.05, 2.2);
  canopy.scale.set(0.7, 0.55, 1.4);
  root.add(canopy);

  const wingL = new THREE.Mesh(deltaWing(13, 8.5), bodyMat);
  const wingR = new THREE.Mesh(deltaWing(13, 8.5), bodyMat);
  wingR.scale.x = -1;
  wingL.castShadow = true;
  wingR.castShadow = true;
  root.add(wingL, wingR);

  const tail = new THREE.Mesh(new THREE.ConeGeometry(0.7, 3.4, 3), accentMat);
  tail.position.set(0, 2.4, -4.2);
  tail.rotation.x = 0.18;
  root.add(tail);

  const stab = new THREE.Mesh(new THREE.ConeGeometry(1.6, 3.2, 3), bodyMat);
  stab.rotation.set(Math.PI / 2, 0, Math.PI / 2);
  stab.position.set(0, 0.15, -5.4);
  root.add(stab);

  const exhaust = new THREE.Mesh(
    new THREE.CircleGeometry(0.72, 16),
    new THREE.MeshStandardMaterial({
      color: pal.exhaust,
      emissive: pal.exhaust,
      emissiveIntensity: 1.6,
      side: THREE.DoubleSide,
    })
  );
  exhaust.position.z = -5.85;
  exhaust.name = "exhaust";
  root.add(exhaust);

  const glow = new THREE.PointLight(pal.exhaust, 1.6, 36, 2);
  glow.position.z = -6.4;
  glow.name = "exhaustLight";
  root.add(glow);

  root.userData.team = team;
  root.userData.palette = pal;
  return root;
}

/** Large, high-contrast BVRAAM. +Z is nose. */
export function createMissile(team) {
  const pal = team === 0 ? BLUE : RED;
  const root = new THREE.Group();

  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.42, 0.42, 9.2, 10),
    new THREE.MeshStandardMaterial({
      color: pal.accent,
      emissive: pal.skin,
      emissiveIntensity: 0.85,
      metalness: 0.35,
      roughness: 0.25,
    })
  );
  body.rotation.x = Math.PI / 2;
  root.add(body);

  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(0.42, 2.2, 8),
    new THREE.MeshStandardMaterial({
      color: 0xfff4c2,
      emissive: pal.skin,
      emissiveIntensity: 0.7,
    })
  );
  nose.rotation.x = Math.PI / 2;
  nose.position.z = 5.6;
  root.add(nose);

  const finGeo = new THREE.ConeGeometry(1.15, 2.4, 3);
  for (const rot of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
    const fin = new THREE.Mesh(finGeo, mat(pal.skin, { emissive: pal.skin, emissiveIntensity: 0.4 }));
    fin.position.z = -3.4;
    fin.rotation.z = rot;
    fin.rotation.x = Math.PI / 2;
    root.add(fin);
  }

  const plume = new THREE.Mesh(
    new THREE.ConeGeometry(0.85, 5.5, 10),
    new THREE.MeshStandardMaterial({
      color: pal.exhaust,
      emissive: pal.exhaust,
      emissiveIntensity: 2.4,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    })
  );
  plume.rotation.x = -Math.PI / 2;
  plume.position.z = -7.4;
  plume.name = "plume";
  root.add(plume);

  const halo = new THREE.Sprite(
    new THREE.SpriteMaterial({
      color: pal.exhaust,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    })
  );
  halo.scale.set(6.5, 6.5, 1);
  halo.name = "halo";
  root.add(halo);

  const light = new THREE.PointLight(pal.exhaust, 2.4, 48, 2);
  light.position.z = -6.8;
  light.name = "plumeLight";
  root.add(light);

  root.userData.team = team;
  return root;
}

export function createRadarVolume() {
  const geo = new THREE.CylinderGeometry(1, 0.02, 1, 28, 1, true);
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({
      color: 0x4aa3ff,
      transparent: true,
      opacity: 0.09,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
  );
  mesh.rotation.x = Math.PI / 2;
  mesh.position.z = 0.5;
  mesh.name = "radar";
  mesh.renderOrder = 1;
  return mesh;
}

export function createAltitudeStem() {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, 1, 0], 3));
  const line = new THREE.Line(
    geo,
    new THREE.LineBasicMaterial({ color: 0x8aa0b5, transparent: true, opacity: 0.35 })
  );
  line.name = "stem";
  return line;
}

export function createWezRing() {
  const geo = new THREE.RingGeometry(0.96, 1.0, 64);
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({
      color: 0xffd166,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.name = "wez";
  mesh.visible = false;
  return mesh;
}

export const DEFAULT_VIEW_CONFIG = {
  layout: "split",
  camera: "orbit",
  radar: true,
  tracks: true,
  wez: true,
  labels: true,
  stems: true,
  aircraftScale: 3,
  missileScale: 4,
  altScale: 4,
};
