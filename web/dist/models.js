import * as THREE from "three";

const BLUE = {
  skin: 0x1c3d5c,
  accent: 0x4aa3ff,
  trim: 0xc9d6e2,
  exhaust: 0x66e0ff,
};
const RED = {
  skin: 0x5a221c,
  accent: 0xff6b5a,
  trim: 0xe8d4c8,
  exhaust: 0xff8a4a,
};

function mat(color, extras = {}) {
  return new THREE.MeshStandardMaterial({
    color,
    metalness: 0.72,
    roughness: 0.32,
    ...extras,
  });
}

/** Stylized 4th-gen fighter. Local +Z is the nose. Length ~20 GDM. */
export function createFighter(team) {
  const pal = team === 0 ? BLUE : RED;
  const root = new THREE.Group();
  root.name = "fighter";

  const bodyMat = mat(pal.skin);
  const accentMat = mat(pal.accent, { metalness: 0.4, roughness: 0.28 });
  const trimMat = mat(pal.trim, { metalness: 0.85, roughness: 0.18 });
  const glassMat = new THREE.MeshStandardMaterial({
    color: 0x8ec8ff,
    metalness: 0.1,
    roughness: 0.05,
    transparent: true,
    opacity: 0.42,
    envMapIntensity: 1.2,
  });

  const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 1.05, 10.5, 16), bodyMat);
  fuselage.rotation.x = Math.PI / 2;
  fuselage.position.z = -0.4;
  fuselage.castShadow = true;
  root.add(fuselage);

  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.7, 4.2, 16), accentMat);
  nose.rotation.x = Math.PI / 2;
  nose.position.z = 6.7;
  root.add(nose);

  const radome = new THREE.Mesh(new THREE.SphereGeometry(0.42, 12, 10), trimMat);
  radome.position.z = 8.6;
  root.add(radome);

  const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.72, 12, 10, 0, Math.PI * 2, 0, Math.PI * 0.55), glassMat);
  canopy.position.set(0, 0.55, 2.4);
  canopy.scale.set(0.85, 0.7, 1.35);
  root.add(canopy);

  const wingGeo = new THREE.BoxGeometry(11.5, 0.12, 3.4);
  const wing = new THREE.Mesh(wingGeo, bodyMat);
  wing.position.set(0, -0.05, -0.6);
  wing.castShadow = true;
  root.add(wing);

  const strakeL = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.08, 2.8), accentMat);
  strakeL.position.set(-1.6, 0.05, 2.2);
  strakeL.rotation.y = 0.35;
  const strakeR = strakeL.clone();
  strakeR.position.x *= -1;
  strakeR.rotation.y *= -1;
  root.add(strakeL, strakeR);

  const intakeGeo = new THREE.BoxGeometry(0.9, 0.55, 2.4);
  const inL = new THREE.Mesh(intakeGeo, trimMat);
  inL.position.set(-0.95, -0.35, 1.6);
  const inR = inL.clone();
  inR.position.x *= -1;
  root.add(inL, inR);

  const stab = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.1, 1.5), bodyMat);
  stab.position.set(0, 0.15, -5.4);
  root.add(stab);

  const tailGeo = new THREE.BoxGeometry(0.12, 2.1, 1.6);
  const tL = new THREE.Mesh(tailGeo, accentMat);
  tL.position.set(-0.85, 1.15, -5.5);
  tL.rotation.z = 0.18;
  const tR = tL.clone();
  tR.position.x *= -1;
  tR.rotation.z *= -1;
  root.add(tL, tR);

  const exhaust = new THREE.Mesh(
    new THREE.CylinderGeometry(0.55, 0.42, 1.4, 12),
    new THREE.MeshStandardMaterial({
      color: 0x111111,
      emissive: pal.exhaust,
      emissiveIntensity: 0.85,
      metalness: 0.4,
      roughness: 0.5,
    })
  );
  exhaust.rotation.x = Math.PI / 2;
  exhaust.position.z = -6.4;
  exhaust.name = "exhaust";
  root.add(exhaust);

  const glow = new THREE.PointLight(pal.exhaust, 1.4, 28, 2);
  glow.position.z = -7.2;
  glow.name = "exhaustLight";
  root.add(glow);

  root.userData.team = team;
  root.userData.palette = pal;
  return root;
}

/** Slender BVRAAM. Local +Z is the nose. */
export function createMissile(team) {
  const pal = team === 0 ? BLUE : RED;
  const root = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.16, 3.4, 10),
    mat(0xd8dde3, { metalness: 0.8, roughness: 0.22 })
  );
  body.rotation.x = Math.PI / 2;
  root.add(body);

  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.7, 10), mat(pal.accent));
  nose.rotation.x = Math.PI / 2;
  nose.position.z = 2.05;
  root.add(nose);

  const finGeo = new THREE.BoxGeometry(0.9, 0.05, 0.45);
  for (const rot of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
    const fin = new THREE.Mesh(finGeo, mat(pal.skin));
    fin.position.z = -1.35;
    fin.rotation.z = rot;
    root.add(fin);
  }

  const plume = new THREE.Mesh(
    new THREE.ConeGeometry(0.22, 1.6, 8),
    new THREE.MeshStandardMaterial({
      color: pal.exhaust,
      emissive: pal.exhaust,
      emissiveIntensity: 1.4,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    })
  );
  plume.rotation.x = -Math.PI / 2;
  plume.position.z = -2.4;
  plume.name = "plume";
  root.add(plume);

  const light = new THREE.PointLight(pal.exhaust, 0.9, 18, 2);
  light.position.z = -2.6;
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
