import { WebGL2SplatRenderer } from "./renderers/webgl2.js";
import { WebGPUSplatRenderer } from "./renderers/webgpu.js";

const SCENE_MANIFEST_URL = "./assets/scenes/lego-mini.json";

const canvas = document.querySelector("#viewer-canvas");
const sceneTitle = document.querySelector("#scene-title");
const rendererLabel = document.querySelector("#renderer-label");
const statusBadge = document.querySelector("#status-badge");
const alphaControl = document.querySelector("#alpha-control");
const autorotateControl = document.querySelector("#autorotate-control");
const fpsCounter = document.querySelector("#fps-counter");

// Binary format v5: geometry (11 f32/splat) + SH (48 f16/splat)
const GEOM_FLOATS = 11;   // geometry: pos/opacity/quat/scale (origIdx removed, use instance ID)
const SH_FLOATS   = 48;   // SH: sh_r[16] sh_g[16] sh_b[16], stored as float16

const state = {
  scene: null,
  renderer: null,
  width: 0,
  height: 0,
  // Camera
  pointer: {
    active: false,
    x: 0,
    y: 0,
  },
  camera: {
    target: [0, 0, 0],
    yaw: 0,
    pitch: 0,
    distance: 2.5,
    fovY: Math.PI / 4,
    right: new Float32Array(3),
    up: new Float32Array(3),
    eye: new Float32Array(3),
    forward: new Float32Array(3),
    viewProjection: new Float32Array(16),
    view: new Float32Array(16),
    focal: 0,
  },
  renderOptions: {
    alphaScale: 1.0,
  },
  fps: {
    lastTime: 0,
    frames: 0,
    value: 0,
  },
};

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function sub(out, a, b) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
  return out;
}

function addScaled(out, a, b, scale) {
  out[0] = a[0] + b[0] * scale;
  out[1] = a[1] + b[1] * scale;
  out[2] = a[2] + b[2] * scale;
  return out;
}

function normalize(out, input) {
  const length = Math.hypot(input[0], input[1], input[2]) || 1.0;
  out[0] = input[0] / length;
  out[1] = input[1] / length;
  out[2] = input[2] / length;
  return out;
}

function cross(out, a, b) {
  const ax = a[0], ay = a[1], az = a[2];
  const bx = b[0], by = b[1], bz = b[2];
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

function perspective(out, fovY, aspect, near, far) {
  const f = 1.0 / Math.tan(fovY / 2.0);
  const nf = 1.0 / (near - far);
  out[0] = f / aspect;
  out[1] = 0; out[2] = 0; out[3] = 0;
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
  out[8] = 0; out[9] = 0;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[12] = 0; out[13] = 0;
  out[14] = (2 * far * near) * nf;
  out[15] = 0;
  return out;
}

function lookAt(out, eye, target, up) {
  const zAxis = normalize(new Float32Array(3), sub(new Float32Array(3), eye, target));
  const xAxis = normalize(new Float32Array(3), cross(new Float32Array(3), up, zAxis));
  const yAxis = cross(new Float32Array(3), zAxis, xAxis);
  out[0] = xAxis[0]; out[1] = yAxis[0]; out[2] = zAxis[0]; out[3] = 0;
  out[4] = xAxis[1]; out[5] = yAxis[1]; out[6] = zAxis[1]; out[7] = 0;
  out[8] = xAxis[2]; out[9] = yAxis[2]; out[10] = zAxis[2]; out[11] = 0;
  out[12] = -(xAxis[0]*eye[0] + xAxis[1]*eye[1] + xAxis[2]*eye[2]);
  out[13] = -(yAxis[0]*eye[0] + yAxis[1]*eye[1] + yAxis[2]*eye[2]);
  out[14] = -(zAxis[0]*eye[0] + zAxis[1]*eye[1] + zAxis[2]*eye[2]);
  out[15] = 1;
  return out;
}

function multiplyMatrices(out, a, b) {
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

function updateCamera(width, height) {
  const aspect = Math.max(width / Math.max(height, 1), 1e-3);
  const projection = perspective(new Float32Array(16), state.camera.fovY, aspect, 0.01, 100.0);

  const cosPitch = Math.cos(state.camera.pitch);
  const eyeDirection = new Float32Array([
    Math.sin(state.camera.yaw) * cosPitch,
    Math.sin(state.camera.pitch),
    Math.cos(state.camera.yaw) * cosPitch,
  ]);

  addScaled(state.camera.eye, state.camera.target, eyeDirection, state.camera.distance);
  normalize(state.camera.forward, sub(new Float32Array(3), state.camera.target, state.camera.eye));
  normalize(state.camera.right, cross(new Float32Array(3), state.camera.forward, new Float32Array([0, 1, 0])));
  normalize(state.camera.up, cross(new Float32Array(3), state.camera.right, state.camera.forward));

  const view = lookAt(new Float32Array(16), state.camera.eye, state.camera.target, state.camera.up);
  multiplyMatrices(state.camera.viewProjection, projection, view);
  state.camera.view = view;
  state.camera.focal = height * 0.5 * projection[5];
}

function resizeRenderer() {
  const width = Math.floor(canvas.clientWidth * window.devicePixelRatio);
  const height = Math.floor(canvas.clientHeight * window.devicePixelRatio);
  if (width === state.width && height === state.height) return;
  state.width = width;
  state.height = height;
  state.renderer.resize(width, height);
}


// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function setStatus(message) {
  statusBadge.textContent = message;
}

async function chooseRenderer(scene) {
  const webgpu = await WebGPUSplatRenderer.create(canvas, scene);
  if (webgpu) return webgpu;
  const webgl = await WebGL2SplatRenderer.create(canvas, scene);
  if (webgl) return webgl;
  throw new Error("This browser does not expose WebGPU or WebGL2.");
}

function bindControls() {
  alphaControl.addEventListener("input", () => {
    state.renderOptions.alphaScale = Number(alphaControl.value);
    state.renderer.setRenderOptions(state.renderOptions);
  });

  canvas.addEventListener("pointerdown", (event) => {
    state.pointer.active = true;
    state.pointer.x = event.clientX;
    state.pointer.y = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.pointer.active) return;
    const dx = event.clientX - state.pointer.x;
    const dy = event.clientY - state.pointer.y;
    state.pointer.x = event.clientX;
    state.pointer.y = event.clientY;
    state.camera.yaw -= dx * 0.006;
    state.camera.pitch = clamp(state.camera.pitch - dy * 0.006, -1.25, 1.25);
  });

  const releasePointer = () => { state.pointer.active = false; };
  canvas.addEventListener("pointerup", releasePointer);
  canvas.addEventListener("pointercancel", releasePointer);

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const scale = Math.exp(event.deltaY * 0.001);
    state.camera.distance = clamp(state.camera.distance * scale, 0.8, 20.0);
  }, { passive: false });
}

// ---------------------------------------------------------------------------
// Scene load
// ---------------------------------------------------------------------------

async function loadScene() {
  const manifest = await fetch(SCENE_MANIFEST_URL).then((r) => r.json());
  const binaryUrl = new URL(manifest.binary, new URL(SCENE_MANIFEST_URL, window.location.href)).toString();
  const buffer = await fetch(binaryUrl).then((r) => r.arrayBuffer());
  manifest.binaryUrl = binaryUrl;
  const N = manifest.splatCount;
  // Split into two separate ArrayBuffers so each can be transferred independently
  manifest.geomSplats = new Float32Array(buffer.slice(0, N * GEOM_FLOATS * 4));
  // SH stored as float16 (2 bytes each) in v5; pass raw uint16 bits to renderers
  manifest.shSplats   = new Uint16Array(buffer.slice(N * GEOM_FLOATS * 4));
  return manifest;
}

function initializeScene(scene) {
  state.scene = scene;
  state.camera.target = scene.bounds.center.slice();
  state.camera.yaw = scene.camera.yaw;
  state.camera.pitch = scene.camera.pitch;
  state.camera.distance = scene.camera.distance;
  state.camera.fovY = scene.camera.fovY;
  state.renderOptions.alphaScale = scene.render.alphaScale;
  alphaControl.value = String(scene.render.alphaScale);
  sceneTitle.textContent = `${scene.title} · ${scene.splatCount.toLocaleString()} splats`;
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

function animateFrame(time) {
  resizeRenderer();

  if (autorotateControl.checked && !state.pointer.active) {
    state.camera.yaw += 0.0014;
  }

  // FPS counter
  state.fps.frames += 1;
  if (time - state.fps.lastTime >= 500) {
    state.fps.value = Math.round(state.fps.frames * 1000 / (time - state.fps.lastTime));
    state.fps.frames = 0;
    state.fps.lastTime = time;
    fpsCounter.textContent = `${state.fps.value} FPS`;
  }

  updateCamera(state.width, state.height);

  state.renderer.render(state.camera);
  window.__viewerInfo = {
    renderer: state.renderer.label,
    scene: state.scene.slug,
    splatCount: state.scene.splatCount,
    ready: true,
  };
  requestAnimationFrame(animateFrame);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  try {
    setStatus("Loading scene asset…");
    const scene = await loadScene();
    initializeScene(scene);

    setStatus("Selecting renderer…");
    state.renderer = await chooseRenderer(scene);
    rendererLabel.textContent = `Renderer: ${state.renderer.label}`;
    state.renderer.setRenderOptions(state.renderOptions);

    bindControls();
    resizeRenderer();
    updateCamera(state.width || 1, state.height || 1);

    // Upload geometry data directly — no sort needed (Mobile-GS uses order-independent blending)
    state.renderer.updateGeometryData(scene.geomSplats);

    // Upload static SH data to GPU
    state.renderer.initSHData(scene.shSplats);

    setStatus(`Ready · ${state.renderer.label}`);
    requestAnimationFrame(animateFrame);
  } catch (error) {
    console.error(error);
    rendererLabel.textContent = "Renderer: unavailable";
    setStatus(error.message);
    window.__viewerInfo = { renderer: "unsupported", ready: false, error: error.message };
  }
}

boot();
