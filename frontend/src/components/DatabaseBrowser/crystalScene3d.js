/**
 * Imperative Three.js scene for the crystal-structure viewer — real shaded
 * spheres (InstancedMesh), cylinder bonds, translucent coordination polyhedra
 * (convex hulls), a unit-cell box, lighting, OrbitControls and hover picking.
 *
 * `three` is only pulled when this module is dynamically imported (by
 * CrystalStructureViewer), so it stays out of the Database page's initial bundle.
 * WebGL means this is not unit-tested; the pure math lives in crystalScene.js.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';

import { sceneBounds, cellEdges, atomSphereRadius } from './crystalScene';

const BOND_RADIUS = 0.11;      // Å
const AXIS = new THREE.Vector3(0, 1, 0);

/**
 * Mount a crystal scene into `container`.
 * @param {HTMLElement} container
 * @param {{onHover?: (atom|null, x:number, y:number)=>void, background?: string}} opts
 * @returns handle: { update, setOptions, resetView, screenshot, dispose }
 */
export function createCrystalScene(container, { onHover, background = '#21222c' } = {}) {
  const w = Math.max(1, container.clientWidth || 480);
  const h = Math.max(1, container.clientHeight || 360);

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h);
  renderer.setClearColor(new THREE.Color(background), 1);
  container.appendChild(renderer.domElement);
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 5000);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;

  // Lighting — ambient fill + a key and a back directional for even shading.
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(1, 1.2, 1.4);
  scene.add(key);
  const back = new THREE.DirectionalLight(0xffffff, 0.35);
  back.position.set(-1, -0.6, -1);
  scene.add(back);

  // Groups so toggles are just group.visible flips (no rebuild).
  const atomsGroup = new THREE.Group();
  const bondsGroup = new THREE.Group();
  const cellGroup = new THREE.Group();
  const polyGroup = new THREE.Group();
  const axesGroup = new THREE.Group();
  scene.add(atomsGroup, bondsGroup, cellGroup, polyGroup, axesGroup);

  // Shared geometries (disposed once at teardown).
  const sphereGeo = new THREE.SphereGeometry(1, 24, 16);
  const cylGeo = new THREE.CylinderGeometry(1, 1, 1, 12);

  let atomMeshes = [];               // InstancedMesh per element (for picking + hide)
  let initialCam = null;             // {pos, target} for resetView
  let current = { showBonds: true, showCell: true, showPolyhedra: true, hidden: new Set() };

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  // disposeGeometry=false for the atom/bond InstancedMeshes: their geometry is the
  // SHARED sphereGeo/cylGeo (disposed once at teardown), so disposing it here would
  // break the next rebuild. child.dispose?.() frees InstancedMesh instance buffers.
  function clearGroup(group, disposeGeometry = true) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.dispose?.();
      if (disposeGeometry) child.geometry?.dispose?.();
      if (child.material) {
        (Array.isArray(child.material) ? child.material : [child.material]).forEach((m) => m.dispose());
      }
    }
  }

  function buildAtoms(atoms) {
    clearGroup(atomsGroup, false);   // shared sphereGeo — don't dispose it
    atomMeshes = [];
    const byEl = new Map();
    for (const a of atoms) {
      if (!byEl.has(a.element)) byEl.set(a.element, []);
      byEl.get(a.element).push(a);
    }
    const dummy = new THREE.Object3D();
    for (const [element, list] of byEl) {
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(list[0].color), roughness: 0.5, metalness: 0.25,
      });
      const mesh = new THREE.InstancedMesh(sphereGeo, mat, list.length);
      mesh.userData = { element, atoms: list };
      list.forEach((a, i) => {
        const s = atomSphereRadius(a.radius);
        dummy.position.set(a.cart[0], a.cart[1], a.cart[2]);
        dummy.scale.set(s, s, s);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
      atomsGroup.add(mesh);
      atomMeshes.push(mesh);
    }
  }

  function buildBonds(bonds) {
    clearGroup(bondsGroup, false);   // shared cylGeo — don't dispose it
    if (!bonds?.length) return;
    const mat = new THREE.MeshStandardMaterial({ color: 0xa9d5e0, roughness: 0.6, metalness: 0.1 });
    const mesh = new THREE.InstancedMesh(cylGeo, mat, bonds.length);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const mid = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    const m = new THREE.Matrix4();
    bonds.forEach(([p1, p2], i) => {
      a.set(p1[0], p1[1], p1[2]);
      b.set(p2[0], p2[1], p2[2]);
      dir.subVectors(b, a);
      const len = dir.length() || 1e-6;
      mid.addVectors(a, b).multiplyScalar(0.5);
      quat.setFromUnitVectors(AXIS, dir.clone().normalize());
      scl.set(BOND_RADIUS, len, BOND_RADIUS);
      m.compose(mid, quat, scl);
      mesh.setMatrixAt(i, m);
    });
    mesh.instanceMatrix.needsUpdate = true;
    bondsGroup.add(mesh);
  }

  function buildCell(cellVectors) {
    clearGroup(cellGroup);
    if (!cellVectors) return;
    const pts = [];
    for (const [p, q] of cellEdges(cellVectors)) pts.push(...p, ...q);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x8890a6, transparent: true, opacity: 0.55 });
    cellGroup.add(new THREE.LineSegments(geo, mat));
  }

  function buildPolyhedra(polyhedra) {
    clearGroup(polyGroup);
    if (!polyhedra?.length) return;
    for (const poly of polyhedra) {
      const verts = poly.vertices.map((v) => new THREE.Vector3(v[0], v[1], v[2]));
      if (verts.length < 4) continue;
      let geo;
      try {
        geo = new ConvexGeometry(verts);
      } catch {
        continue;                      // coplanar / degenerate hull → skip
      }
      const color = new THREE.Color(poly.color);
      const face = new THREE.MeshStandardMaterial({
        color, transparent: true, opacity: 0.32, side: THREE.DoubleSide,
        depthWrite: false, roughness: 0.55, metalness: 0.0,
      });
      polyGroup.add(new THREE.Mesh(geo, face));
      const edgeMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 });
      polyGroup.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat));
    }
  }

  function frameCamera(atoms) {
    const { center, radius } = sceneBounds(atoms.map((a) => a.cart));
    const c = new THREE.Vector3(center[0], center[1], center[2]);
    const dist = (radius / Math.sin((Math.PI * camera.fov) / 360)) * 1.35;
    const pos = c.clone().add(new THREE.Vector3(0.8, 0.5, 1).normalize().multiplyScalar(dist));
    camera.position.copy(pos);
    camera.near = Math.max(0.02, dist / 100);
    camera.far = dist * 100;
    camera.updateProjectionMatrix();
    controls.target.copy(c);
    controls.update();
    initialCam = { pos: pos.clone(), target: c.clone() };

    clearGroup(axesGroup);
    const axes = new THREE.AxesHelper(Math.max(1.2, radius * 0.28));
    axes.position.copy(c.clone().sub(new THREE.Vector3(radius, radius, radius)));
    axesGroup.add(axes);
  }

  function applyVisibility() {
    bondsGroup.visible = current.showBonds;
    cellGroup.visible = current.showCell;
    polyGroup.visible = current.showPolyhedra;
    for (const mesh of atomMeshes) mesh.visible = !current.hidden.has(mesh.userData.element);
  }

  function update(payload, opts) {
    current = { ...current, ...opts, hidden: new Set(opts?.hidden || current.hidden) };
    buildAtoms(payload.atoms);
    buildBonds(payload.bonds);
    buildCell(payload.cell_vectors);
    buildPolyhedra(payload.polyhedra);
    frameCamera(payload.atoms);
    applyVisibility();
  }

  function setOptions(opts) {
    current = { ...current, ...opts, hidden: new Set(opts?.hidden || current.hidden) };
    applyVisibility();
  }

  function resetView() {
    if (!initialCam) return;
    camera.position.copy(initialCam.pos);
    controls.target.copy(initialCam.target);
    controls.update();
  }

  /** The current view as a PNG data URL, rendered for print rather than grabbed
   * off the screen.
   *
   * Capturing the on-screen buffer and letting the export dialog enlarge it gave
   * a soft, visibly polygonal image: the canvas is only a few hundred pixels
   * wide, and the shared geometries carry just enough detail for that size.
   * So for the capture the scene is re-rendered LARGE and with finer geometry,
   * and everything is put back afterwards.
   *
   * Same camera, same visible elements — only the sampling changes.
   */
  function toDataURL({ targetLongSide = 4096 } = {}) {
    const cw = Math.max(1, container.clientWidth);
    const ch = Math.max(1, container.clientHeight);

    // How far we may go up: the wish, what the GPU will allocate, and a sane cap.
    let maxBuffer = 4096;
    try {
      const gl = renderer.getContext();
      maxBuffer = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) || maxBuffer;
    } catch { /* keep the conservative default */ }
    const scale = Math.max(1, Math.min(
      targetLongSide / Math.max(cw, ch),
      maxBuffer / Math.max(cw, ch),
      8,
    ));

    // 24x16 spheres and 12-sided cylinders read as faceted once they cover a
    // thousand pixels. Swap in finer ones for the shot only — rebuilding the
    // InstancedMeshes would be far more disruptive than swapping their geometry.
    const fineSphere = new THREE.SphereGeometry(1, 96, 64);
    const fineCyl = new THREE.CylinderGeometry(1, 1, 1, 48);
    const swapped = [];
    for (const m of atomsGroup.children) { swapped.push([m, m.geometry]); m.geometry = fineSphere; }
    for (const m of bondsGroup.children) { swapped.push([m, m.geometry]); m.geometry = fineCyl; }

    const prevRatio = renderer.getPixelRatio();
    try {
      // Size the buffer directly and leave the CSS size alone, so the on-screen
      // canvas does not flash at a different size mid-capture.
      renderer.setPixelRatio(1);
      renderer.setSize(Math.round(cw * scale), Math.round(ch * scale), false);
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL('image/png');
    } finally {
      for (const [m, g] of swapped) m.geometry = g;
      fineSphere.dispose();
      fineCyl.dispose();
      renderer.setPixelRatio(prevRatio);
      renderer.setSize(cw, ch);
      renderer.render(scene, camera);
    }
  }

  function screenshot() {
    renderer.render(scene, camera);
    renderer.domElement.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'crystal-structure.png';
      a.click();
      URL.revokeObjectURL(url);
    }, 'image/png');
  }

  // Hover picking.
  function onPointerMove(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(atomMeshes.filter((m) => m.visible), false);
    if (hits.length && hits[0].instanceId != null) {
      const atom = hits[0].object.userData.atoms[hits[0].instanceId];
      onHover?.(atom, ev.clientX, ev.clientY);
    } else {
      onHover?.(null, ev.clientX, ev.clientY);
    }
  }
  const onPointerLeave = () => onHover?.(null, 0, 0);
  if (onHover) {
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerleave', onPointerLeave);
  }

  // Render loop.
  let raf = 0;
  const tick = () => {
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  // Resize.
  const ro = new ResizeObserver(() => {
    const cw = Math.max(1, container.clientWidth);
    const ch = Math.max(1, container.clientHeight);
    renderer.setSize(cw, ch);
    camera.aspect = cw / ch;
    camera.updateProjectionMatrix();
  });
  ro.observe(container);

  function dispose() {
    cancelAnimationFrame(raf);
    ro.disconnect();
    renderer.domElement.removeEventListener('pointermove', onPointerMove);
    renderer.domElement.removeEventListener('pointerleave', onPointerLeave);
    clearGroup(atomsGroup, false);   // shared geo disposed once below
    clearGroup(bondsGroup, false);
    clearGroup(cellGroup);
    clearGroup(polyGroup);
    clearGroup(axesGroup);
    sphereGeo.dispose();
    cylGeo.dispose();
    controls.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement);
  }

  return { update, setOptions, resetView, screenshot, toDataURL, dispose };
}
