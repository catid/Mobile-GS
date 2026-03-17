import { splitOpacityPhiWeights } from "./opacity_phi.js";

const GEOM_FLOATS = 11;
const SH_FLOATS = 12;
const SH_TEX_WIDTH = 2048;
const UNIFORM_BUFFER_SIZE = 176;
const WORKGROUP_SIZE = 64;

const WGSL_NET = `
struct CameraUniforms {
  viewProj : mat4x4<f32>,
  view     : mat4x4<f32>,
  viewport : vec4<f32>,
  params   : vec4<f32>,
  eye      : vec4<f32>,
};

struct ScalarBuffer {
  values : array<f32>,
};

struct NetOutputBuffer {
  values : array<vec2<f32>>,
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var shTex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read> geom : ScalarBuffer;
@group(0) @binding(3) var<storage, read_write> netOut : NetOutputBuffer;
@group(0) @binding(4) var<storage, read> w0 : ScalarBuffer;
@group(0) @binding(5) var<storage, read> b0 : ScalarBuffer;
@group(0) @binding(6) var<storage, read> w1 : ScalarBuffer;
@group(0) @binding(7) var<storage, read> b1 : ScalarBuffer;
@group(0) @binding(8) var<storage, read> w2 : ScalarBuffer;
@group(0) @binding(9) var<storage, read> b2 : ScalarBuffer;
@group(0) @binding(10) var<storage, read> wPhi : ScalarBuffer;
@group(0) @binding(11) var<storage, read> bPhi : ScalarBuffer;
@group(0) @binding(12) var<storage, read> wOpacity : ScalarBuffer;
@group(0) @binding(13) var<storage, read> bOpacity : ScalarBuffer;

fn getGeom(index : u32, component : u32) -> f32 {
  return geom.values[index * 11u + component];
}

fn getSHTexel(index : u32, texel : u32) -> vec4<f32> {
  let g = index * 3u + texel;
  return textureLoad(shTex, vec2<i32>(i32(g % ${SH_TEX_WIDTH}u), i32(g / ${SH_TEX_WIDTH}u)), 0);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= arrayLength(&netOut.values)) {
    return;
  }

  let sh0 = getSHTexel(index, 0u);
  let sh1 = getSHTexel(index, 1u);
  let sh2 = getSHTexel(index, 2u);

  var input : array<f32, 24>;
  input[0] = sh0.x;
  input[1] = sh0.y;
  input[2] = sh0.z;
  input[3] = sh0.w;
  input[4] = sh1.x;
  input[5] = sh1.y;
  input[6] = sh1.z;
  input[7] = sh1.w;
  input[8] = sh2.x;
  input[9] = sh2.y;
  input[10] = sh2.z;
  input[11] = sh2.w;

  var shNormSq = 0.0;
  for (var i = 0u; i < 12u; i = i + 1u) {
    shNormSq = shNormSq + input[i] * input[i];
  }
  let shInvNorm = inverseSqrt(max(shNormSq, 1e-8));
  for (var i = 0u; i < 12u; i = i + 1u) {
    input[i] = input[i] * shInvNorm;
  }

  let pos = vec3<f32>(getGeom(index, 0u), getGeom(index, 1u), getGeom(index, 2u));
  let quat = vec4<f32>(getGeom(index, 4u), getGeom(index, 5u), getGeom(index, 6u), getGeom(index, 7u));
  let scale = vec3<f32>(getGeom(index, 8u), getGeom(index, 9u), getGeom(index, 10u));
  let viewdir = normalize(pos - camera.eye.xyz);
  let scaleNorm = normalize(scale);

  input[12] = viewdir.x;
  input[13] = viewdir.y;
  input[14] = viewdir.z;
  input[15] = scaleNorm.x;
  input[16] = scaleNorm.y;
  input[17] = scaleNorm.z;
  input[18] = quat.x;
  input[19] = quat.y;
  input[20] = quat.z;
  input[21] = quat.w;
  input[22] = 0.0;
  input[23] = 0.0;

  var layer0 : array<f32, 256>;
  for (var row = 0u; row < 256u; row = row + 1u) {
    var sum = b0.values[row];
    for (var col = 0u; col < 22u; col = col + 1u) {
      sum = sum + w0.values[row * 22u + col] * input[col];
    }
    layer0[row] = max(sum, 0.0);
  }

  var layer1 : array<f32, 128>;
  for (var row = 0u; row < 128u; row = row + 1u) {
    var sum = b1.values[row];
    for (var col = 0u; col < 256u; col = col + 1u) {
      sum = sum + w1.values[row * 256u + col] * layer0[col];
    }
    layer1[row] = max(sum, 0.0);
  }

  var layer2 : array<f32, 64>;
  for (var row = 0u; row < 64u; row = row + 1u) {
    var sum = b2.values[row];
    for (var col = 0u; col < 128u; col = col + 1u) {
      sum = sum + w2.values[row * 128u + col] * layer1[col];
    }
    layer2[row] = max(sum, 0.0);
  }

  var phiValue = bPhi.values[0];
  var opacityValue = bOpacity.values[0];
  for (var col = 0u; col < 64u; col = col + 1u) {
    phiValue = phiValue + wPhi.values[col] * layer2[col];
    opacityValue = opacityValue + wOpacity.values[col] * layer2[col];
  }

  netOut.values[index] = vec2<f32>(max(phiValue, 0.0), 1.0 / (1.0 + exp(-opacityValue)));
}
`;

const WGSL_SPLAT = `
struct CameraUniforms {
  viewProj : mat4x4<f32>,
  view     : mat4x4<f32>,
  viewport : vec4<f32>,
  params   : vec4<f32>,
  eye      : vec4<f32>,
};

struct NetOutputBuffer {
  values : array<vec2<f32>>,
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var shTex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read> netOut : NetOutputBuffer;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) conic : vec3<f32>,
  @location(1) centerPix : vec2<f32>,
  @location(2) color : vec4<f32>,
  @location(3) weight : f32,
};

const quad = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
);

fn quatToMat(q : vec4<f32>) -> mat3x3<f32> {
  let qw = q.x;
  let qx = q.y;
  let qy = q.z;
  let qz = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy)),
    vec3<f32>(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx)),
    vec3<f32>(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)),
  );
}

fn getSHTexel(index : u32, texel : u32) -> vec4<f32> {
  let g = index * 3u + texel;
  return textureLoad(shTex, vec2<i32>(i32(g % ${SH_TEX_WIDTH}u), i32(g / ${SH_TEX_WIDTH}u)), 0);
}

fn evalSH1(index : u32, dir : vec3<f32>) -> vec3<f32> {
  let t0 = getSHTexel(index, 0u);
  let t1 = getSHTexel(index, 1u);
  let t2 = getSHTexel(index, 2u);

  let c0 = vec3<f32>(t0.x, t0.y, t0.z);
  let c1 = vec3<f32>(t0.w, t1.x, t1.y);
  let c2 = vec3<f32>(t1.z, t1.w, t2.x);
  let c3 = vec3<f32>(t2.y, t2.z, t2.w);

  let x = dir.x;
  let y = dir.y;
  let z = dir.z;
  let c = 0.4886025119029199;
  var color = c0 * 0.28209479177387814;
  color = color + c1 * (c * (-y));
  color = color + c2 * (c * z);
  color = color + c3 * (c * (-x));
  return max(color + 0.5, vec3<f32>(0.0));
}

@vertex
fn vsMain(
  @builtin(vertex_index) vi : u32,
  @builtin(instance_index) instIdx : u32,
  @location(0) posOpacity : vec4<f32>,
  @location(1) quat : vec4<f32>,
  @location(2) scaleXYZ : vec3<f32>,
) -> VertexOutput {
  var out : VertexOutput;
  let pos = posOpacity.xyz;
  let scale = scaleXYZ;
  let phiOpacity = netOut.values[instIdx];
  let phi = phiOpacity.x;
  let opacity = phiOpacity.y;

  let pView = camera.view * vec4<f32>(pos, 1.0);
  let fwd = -pView.z;
  if (fwd < 0.01) {
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  let R = quatToMat(quat);
  let M = mat3x3<f32>(W * (R[0] * scale.x), W * (R[1] * scale.y), W * (R[2] * scale.z));
  let Sv = M * transpose(M);

  let invD = 1.0 / fwd;
  let invD2 = invD * invD;
  let tx = pView.x;
  let ty = pView.y;
  let J0 = vec3<f32>(camera.viewport.z * invD, 0.0, camera.viewport.z * tx * invD2);
  let J1 = vec3<f32>(0.0, camera.viewport.z * invD, camera.viewport.z * ty * invD2);

  let cov00 = dot(J0, Sv * J0) + 0.3;
  let cov01 = dot(J0, Sv * J1);
  let cov11 = dot(J1, Sv * J1) + 0.3;
  let det = cov00 * cov11 - cov01 * cov01;
  if (det <= 0.0) {
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  let invDet = 1.0 / det;
  let conic = vec3<f32>(cov11 * invDet, -cov01 * invDet, cov00 * invDet);
  let trace = cov00 + cov11;
  let disc = max(0.0, trace * trace - 4.0 * det);
  let radius = min(3.0 * sqrt(0.5 * (trace + sqrt(disc))), 1024.0);

  let clip = camera.viewProj * vec4<f32>(pos, 1.0);
  let invW = 1.0 / clip.w;
  let cx = (clip.x * invW * 0.5 + 0.5) * camera.viewport.x;
  let cy = (clip.y * invW * 0.5 + 0.5) * camera.viewport.y;
  let local = quad[vi];

  out.position = vec4<f32>(
    (cx + local.x * radius) / camera.viewport.x * 2.0 - 1.0,
    1.0 - (cy + local.y * radius) / camera.viewport.y * 2.0,
    clip.z * invW,
    1.0
  );
  out.conic = conic;
  out.centerPix = vec2<f32>(cx, cy);
  out.color = vec4<f32>(evalSH1(instIdx, normalize(pos - camera.eye.xyz)), opacity);

  let maxScale = max(scale.x, max(scale.y, scale.z));
  out.weight = exp(min(maxScale / fwd, 20.0)) + phi / (fwd * fwd) + phi * phi;
  return out;
}

fn splatAlpha(in : VertexOutput) -> f32 {
  let d = vec2<f32>(in.position.x - in.centerPix.x, in.centerPix.y - in.position.y);
  let power = 0.5 * (in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
  if (power > 8.0) {
    return -1.0;
  }
  return min(0.99, in.color.a * exp(-power) * camera.params.x);
}

@fragment
fn fsAccum(in : VertexOutput) -> @location(0) vec4<f32> {
  let alpha = splatAlpha(in);
  let alphaWeight = alpha * in.weight;
  if (alphaWeight < 0.00001) {
    discard;
  }
  return vec4<f32>(in.color.rgb * alphaWeight, alphaWeight);
}

@fragment
fn fsLogT(in : VertexOutput) -> @location(0) vec4<f32> {
  let alpha = splatAlpha(in);
  if (alpha < 0.00001) {
    discard;
  }
  return vec4<f32>(log(1.0 - alpha), 0.0, 0.0, 0.0);
}
`;

const WGSL_COMPOSE = `
@group(0) @binding(0) var accumTex : texture_2d<f32>;
@group(0) @binding(1) var<uniform> bgColor : vec4<f32>;
@group(0) @binding(2) var logTTex : texture_2d<f32>;

@vertex
fn vsCompose(@builtin(vertex_index) index : u32) -> @builtin(position) vec4<f32> {
  let corners = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fsCompose(@builtin(position) fragPos : vec4<f32>) -> @location(0) vec4<f32> {
  let coords = vec2<i32>(i32(fragPos.x), i32(fragPos.y));
  let accum = textureLoad(accumTex, coords, 0);
  let logT = textureLoad(logTTex, coords, 0).r;
  let wFg = accum.a;
  let colorFg = select(vec3<f32>(0.0), accum.rgb / wFg, wFg > 0.0001);
  return vec4<f32>(mix(bgColor.rgb, colorFg, 1.0 - exp(logT)), 1.0);
}
`;

function createStorageBuffer(device, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
  const buffer = device.createBuffer({
    size: Math.max(4, data.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

export class WebGPUSplatRenderer {
  static async create(canvas, scene) {
    if (!navigator.gpu || !scene.opacityPhi) return null;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;
      const supportsFloat32Blend = adapter.features.has("float32-blendable");
      const device = await adapter.requestDevice({
        requiredFeatures: supportsFloat32Blend ? ["float32-blendable"] : [],
      });
      const format = navigator.gpu.getPreferredCanvasFormat();
      const renderer = new WebGPUSplatRenderer(canvas, scene, device, format, supportsFloat32Blend);
      await renderer.initialize();
      const context = canvas.getContext("webgpu");
      if (!context) return null;
      context.configure({ device, format, alphaMode: "opaque" });
      renderer.context = context;
      return renderer;
    } catch (error) {
      console.warn("WebGPU renderer unavailable", error);
      return null;
    }
  }

  constructor(canvas, scene, device, format, supportsFloat32Blend) {
    if (scene.shDegree !== 1 || scene.shFloats !== SH_FLOATS) {
      throw new Error("Corrected WebGPU renderer currently expects SH1 exports");
    }
    this.canvas = canvas;
    this.scene = scene;
    this.device = device;
    this.format = format;
    this.context = null;
    this.accumFormat = supportsFloat32Blend ? "rgba32float" : "rgba16float";
    this.logTFormat = supportsFloat32Blend ? "rgba32float" : "rgba16float";
    this.renderOptions = { alphaScale: scene.render.alphaScale };
    this.instanceCount = 0;
    this._accumWidth = 0;
    this._accumHeight = 0;
    this.netLayout = splitOpacityPhiWeights(scene.opacityPhiWeights, scene.opacityPhi);
  }

  async initialize() {
    const device = this.device;
    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.bgBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.bgBuffer, 0, new Float32Array(this.scene.render.backgroundTop));

    this.netOutBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.instanceBuffer = device.createBuffer({
      size: GEOM_FLOATS * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.w0Buffer = createStorageBuffer(device, this.netLayout.w0);
    this.b0Buffer = createStorageBuffer(device, this.netLayout.b0);
    this.w1Buffer = createStorageBuffer(device, this.netLayout.w1);
    this.b1Buffer = createStorageBuffer(device, this.netLayout.b1);
    this.w2Buffer = createStorageBuffer(device, this.netLayout.w2);
    this.b2Buffer = createStorageBuffer(device, this.netLayout.b2);
    this.wPhiBuffer = createStorageBuffer(device, this.netLayout.wPhi);
    this.bPhiBuffer = createStorageBuffer(device, this.netLayout.bPhi);
    this.wOpacityBuffer = createStorageBuffer(device, this.netLayout.wOpacity);
    this.bOpacityBuffer = createStorageBuffer(device, this.netLayout.bOpacity);

    const netShader = device.createShaderModule({ code: WGSL_NET });
    const splatShader = device.createShaderModule({ code: WGSL_SPLAT });
    const composeShader = device.createShaderModule({ code: WGSL_COMPOSE });

    this.netBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    this.splatBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.composeBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      ],
    });

    this.netPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.netBindGroupLayout] }),
      compute: { module: netShader, entryPoint: "main" },
    });

    const vertexState = {
      module: splatShader,
      entryPoint: "vsMain",
      buffers: [{
        arrayStride: GEOM_FLOATS * 4,
        stepMode: "instance",
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x4" },
          { shaderLocation: 1, offset: 16, format: "float32x4" },
          { shaderLocation: 2, offset: 32, format: "float32x3" },
        ],
      }],
    };
    const additiveBlend = {
      color: { srcFactor: "one", dstFactor: "one", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
    };

    this.accumPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.splatBindGroupLayout] }),
      vertex: vertexState,
      fragment: {
        module: splatShader,
        entryPoint: "fsAccum",
        targets: [{ format: this.accumFormat, blend: additiveBlend }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
    });

    this.logTPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.splatBindGroupLayout] }),
      vertex: vertexState,
      fragment: {
        module: splatShader,
        entryPoint: "fsLogT",
        targets: [{ format: this.logTFormat, blend: additiveBlend }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
    });

    this.composePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.composeBindGroupLayout] }),
      vertex: { module: composeShader, entryPoint: "vsCompose" },
      fragment: {
        module: composeShader,
        entryPoint: "fsCompose",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  get label() {
    return "webgpu";
  }

  _ensureAccumTextures(width, height) {
    if (width === this._accumWidth && height === this._accumHeight) return;
    this._accumWidth = width;
    this._accumHeight = height;

    this.accumTexture?.destroy();
    this.logTTexture?.destroy();
    this.accumTexture = this.device.createTexture({
      size: [width, height],
      format: this.accumFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.logTTexture = this.device.createTexture({
      size: [width, height],
      format: this.logTFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.accumView = this.accumTexture.createView();
    this.logTView = this.logTTexture.createView();
    this.composeBindGroup = this.device.createBindGroup({
      layout: this.composeBindGroupLayout,
      entries: [
        { binding: 0, resource: this.accumView },
        { binding: 1, resource: { buffer: this.bgBuffer } },
        { binding: 2, resource: this.logTView },
      ],
    });
  }

  _rebuildBindGroups() {
    if (!this.shTexture) return;
    this.netBindGroup = this.device.createBindGroup({
      layout: this.netBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.shTexture.createView() },
        { binding: 2, resource: { buffer: this.instanceBuffer } },
        { binding: 3, resource: { buffer: this.netOutBuffer } },
        { binding: 4, resource: { buffer: this.w0Buffer } },
        { binding: 5, resource: { buffer: this.b0Buffer } },
        { binding: 6, resource: { buffer: this.w1Buffer } },
        { binding: 7, resource: { buffer: this.b1Buffer } },
        { binding: 8, resource: { buffer: this.w2Buffer } },
        { binding: 9, resource: { buffer: this.b2Buffer } },
        { binding: 10, resource: { buffer: this.wPhiBuffer } },
        { binding: 11, resource: { buffer: this.bPhiBuffer } },
        { binding: 12, resource: { buffer: this.wOpacityBuffer } },
        { binding: 13, resource: { buffer: this.bOpacityBuffer } },
      ],
    });

    this.splatBindGroup = this.device.createBindGroup({
      layout: this.splatBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.shTexture.createView() },
        { binding: 2, resource: { buffer: this.netOutBuffer } },
      ],
    });
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this._ensureAccumTextures(width, height);
  }

  initSHData(shData) {
    const texels = (shData.length / SH_FLOATS) * 3;
    const height = Math.ceil(texels / SH_TEX_WIDTH);
    const padded = new Uint16Array(SH_TEX_WIDTH * height * 4);
    padded.set(shData);
    this.shTexture?.destroy();
    this.shTexture = this.device.createTexture({
      size: [SH_TEX_WIDTH, height],
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.shTexture },
      padded.buffer,
      { offset: 0, bytesPerRow: SH_TEX_WIDTH * 8, rowsPerImage: height },
      { width: SH_TEX_WIDTH, height },
    );
    this._rebuildBindGroups();
  }

  updateGeometryData(geomData) {
    if (this.instanceBuffer.size !== geomData.byteLength) {
      this.instanceBuffer.destroy();
      this.instanceBuffer = this.device.createBuffer({
        size: geomData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.instanceBuffer, 0, geomData.buffer, geomData.byteOffset, geomData.byteLength);
    this.instanceCount = geomData.length / GEOM_FLOATS;

    const netBytes = this.instanceCount * 8;
    if (this.netOutBuffer.size !== netBytes) {
      this.netOutBuffer.destroy();
      this.netOutBuffer = this.device.createBuffer({
        size: netBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this._rebuildBindGroups();
  }

  updateSceneData(geomData) {
    return this.updateGeometryData(geomData);
  }

  setRenderOptions(options) {
    this.renderOptions = options;
  }

  render(cameraState) {
    if (!this.netBindGroup || !this.splatBindGroup || !this.context) return;

    const packed = new Float32Array(44);
    packed.set(cameraState.viewProjection, 0);
    packed.set(cameraState.view, 16);
    packed[32] = this.canvas.width;
    packed[33] = this.canvas.height;
    packed[34] = cameraState.focal;
    packed[36] = this.renderOptions.alphaScale;
    packed[40] = cameraState.eye[0];
    packed[41] = cameraState.eye[1];
    packed[42] = cameraState.eye[2];
    this.device.queue.writeBuffer(this.uniformBuffer, 0, packed);

    const encoder = this.device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.netPipeline);
    computePass.setBindGroup(0, this.netBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(this.instanceCount / WORKGROUP_SIZE));
    computePass.end();

    const accumPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.accumView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    accumPass.setPipeline(this.accumPipeline);
    accumPass.setBindGroup(0, this.splatBindGroup);
    accumPass.setVertexBuffer(0, this.instanceBuffer);
    accumPass.draw(6, this.instanceCount);
    accumPass.end();

    const logTPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.logTView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    logTPass.setPipeline(this.logTPipeline);
    logTPass.setBindGroup(0, this.splatBindGroup);
    logTPass.setVertexBuffer(0, this.instanceBuffer);
    logTPass.draw(6, this.instanceCount);
    logTPass.end();

    const composePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    composePass.setPipeline(this.composePipeline);
    composePass.setBindGroup(0, this.composeBindGroup);
    composePass.draw(3);
    composePass.end();

    this.device.queue.submit([encoder.finish()]);
  }
}
