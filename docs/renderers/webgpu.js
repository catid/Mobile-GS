import { packRowsToRGBA, splitOpacityPhiWeights } from "./opacity_phi.js";

const GEOM_FLOATS = 11;
const SH_FLOATS = 12;
const SH_TEX_WIDTH = 2048;
const INPUT_DIM = 22;
const HIDDEN0 = 256;
const HIDDEN1 = 128;
const HIDDEN2 = 64;
const INPUT_BLOCKS = 6;
const HIDDEN0_BLOCKS = HIDDEN0 / 4;
const HIDDEN1_BLOCKS = HIDDEN1 / 4;
const HIDDEN2_BLOCKS = HIDDEN2 / 4;
const B0_OFFSET = 0;
const B1_OFFSET = B0_OFFSET + HIDDEN0;
const B2_OFFSET = B1_OFFSET + HIDDEN1;
const BPHI_OFFSET = B2_OFFSET + HIDDEN2;
const BOPACITY_OFFSET = BPHI_OFFSET + 1;
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
@group(0) @binding(4) var w0Tex : texture_2d<f32>;
@group(0) @binding(5) var w1Tex : texture_2d<f32>;
@group(0) @binding(6) var w2Tex : texture_2d<f32>;
@group(0) @binding(7) var wPhiTex : texture_2d<f32>;
@group(0) @binding(8) var wOpacityTex : texture_2d<f32>;
@group(0) @binding(9) var<storage, read> biases : ScalarBuffer;

fn getGeom(index : u32, component : u32) -> f32 {
  return geom.values[index * 11u + component];
}

fn getSHTexel(index : u32, texel : u32) -> vec4<f32> {
  let g = index * 3u + texel;
  return textureLoad(shTex, vec2<i32>(i32(g % ${SH_TEX_WIDTH}u), i32(g / ${SH_TEX_WIDTH}u)), 0);
}

fn getWeightTexel(tex : texture_2d<f32>, row : u32, block : u32) -> vec4<f32> {
  return textureLoad(tex, vec2<i32>(i32(block), i32(row)), 0);
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

  var input : array<vec4<f32>, ${INPUT_BLOCKS}>;
  input[0] = sh0;
  input[1] = sh1;
  input[2] = sh2;
  var shNormSq = 0.0;
  shNormSq = dot(input[0], input[0]) + dot(input[1], input[1]) + dot(input[2], input[2]);
  let shInvNorm = inverseSqrt(max(shNormSq, 1e-8));
  input[0] = input[0] * shInvNorm;
  input[1] = input[1] * shInvNorm;
  input[2] = input[2] * shInvNorm;

  let pos = vec3<f32>(getGeom(index, 0u), getGeom(index, 1u), getGeom(index, 2u));
  let quat = vec4<f32>(getGeom(index, 4u), getGeom(index, 5u), getGeom(index, 6u), getGeom(index, 7u));
  let scale = vec3<f32>(getGeom(index, 8u), getGeom(index, 9u), getGeom(index, 10u));
  let viewdir = normalize(pos - camera.eye.xyz);
  let scaleNorm = normalize(scale);

  input[3] = vec4<f32>(viewdir.x, viewdir.y, viewdir.z, scaleNorm.x);
  input[4] = vec4<f32>(scaleNorm.y, scaleNorm.z, quat.x, quat.y);
  input[5] = vec4<f32>(quat.z, quat.w, 0.0, 0.0);

  var layer0 : array<f32, ${HIDDEN0}>;
  for (var row = 0u; row < ${HIDDEN0}u; row = row + 1u) {
    var sum = biases.values[${B0_OFFSET}u + row];
    for (var block = 0u; block < ${INPUT_BLOCKS}u; block = block + 1u) {
      sum = sum + dot(getWeightTexel(w0Tex, row, block), input[block]);
    }
    layer0[row] = max(sum, 0.0);
  }

  var layer0Vec : array<vec4<f32>, ${HIDDEN0_BLOCKS}>;
  for (var block = 0u; block < ${HIDDEN0_BLOCKS}u; block = block + 1u) {
    let base = block * 4u;
    layer0Vec[block] = vec4<f32>(layer0[base], layer0[base + 1u], layer0[base + 2u], layer0[base + 3u]);
  }

  var layer1 : array<f32, ${HIDDEN1}>;
  for (var row = 0u; row < ${HIDDEN1}u; row = row + 1u) {
    var sum = biases.values[${B1_OFFSET}u + row];
    for (var block = 0u; block < ${HIDDEN0_BLOCKS}u; block = block + 1u) {
      sum = sum + dot(getWeightTexel(w1Tex, row, block), layer0Vec[block]);
    }
    layer1[row] = max(sum, 0.0);
  }

  var layer1Vec : array<vec4<f32>, ${HIDDEN1_BLOCKS}>;
  for (var block = 0u; block < ${HIDDEN1_BLOCKS}u; block = block + 1u) {
    let base = block * 4u;
    layer1Vec[block] = vec4<f32>(layer1[base], layer1[base + 1u], layer1[base + 2u], layer1[base + 3u]);
  }

  var layer2 : array<f32, ${HIDDEN2}>;
  for (var row = 0u; row < ${HIDDEN2}u; row = row + 1u) {
    var sum = biases.values[${B2_OFFSET}u + row];
    for (var block = 0u; block < ${HIDDEN1_BLOCKS}u; block = block + 1u) {
      sum = sum + dot(getWeightTexel(w2Tex, row, block), layer1Vec[block]);
    }
    layer2[row] = max(sum, 0.0);
  }

  var layer2Vec : array<vec4<f32>, ${HIDDEN2_BLOCKS}>;
  for (var block = 0u; block < ${HIDDEN2_BLOCKS}u; block = block + 1u) {
    let base = block * 4u;
    layer2Vec[block] = vec4<f32>(layer2[base], layer2[base + 1u], layer2[base + 2u], layer2[base + 3u]);
  }

  var phiValue = biases.values[${BPHI_OFFSET}u];
  var opacityValue = biases.values[${BOPACITY_OFFSET}u];
  for (var block = 0u; block < ${HIDDEN2_BLOCKS}u; block = block + 1u) {
    let activations = layer2Vec[block];
    phiValue = phiValue + dot(getWeightTexel(wPhiTex, 0u, block), activations);
    opacityValue = opacityValue + dot(getWeightTexel(wOpacityTex, 0u, block), activations);
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

struct SplatOutputs {
  @location(0) accum : vec4<f32>,
  @location(1) logT : vec4<f32>,
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
    (cy + local.y * radius) / camera.viewport.y * 2.0 - 1.0,
    clip.z * invW * 0.5 + 0.5,
    1.0
  );
  out.conic = conic;
  out.centerPix = vec2<f32>(cx, cy);
  out.color = vec4<f32>(evalSH1(instIdx, normalize(pos - camera.eye.xyz)), opacity);

  let maxScale = max(scale.x, max(scale.y, scale.z));
  out.weight = exp(min(maxScale / fwd, 20.0)) + phi / (fwd * fwd) + phi * phi;
  return out;
}

fn splatAlpha(in : VertexOutput, fragCoord : vec2<f32>) -> f32 {
  let d = fragCoord - in.centerPix;
  let power = 0.5 * (in.conic.x * d.x * d.x + 2.0 * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
  if (power > 8.0) {
    return -1.0;
  }
  return min(0.99, in.color.a * exp(-power) * camera.params.x);
}

@fragment
fn fsSplat(in : VertexOutput) -> SplatOutputs {
  let fragCoord = vec2<f32>(in.position.x, camera.viewport.y - in.position.y);
  let alpha = splatAlpha(in, fragCoord);
  if (alpha < 0.00001) {
    discard;
  }
  let alphaWeight = alpha * in.weight;
  var out : SplatOutputs;
  if (alphaWeight >= 0.00001) {
    out.accum = vec4<f32>(in.color.rgb * alphaWeight, alphaWeight);
  } else {
    out.accum = vec4<f32>(0.0);
  }
  out.logT = vec4<f32>(log(1.0 - alpha), 0.0, 0.0, 0.0);
  return out;
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

function createFloatTexture(device, width, height, data) {
  const texture = device.createTexture({
    size: [width, height],
    format: "rgba32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture },
    data,
    { offset: 0, bytesPerRow: width * 16, rowsPerImage: height },
    { width, height },
  );
  return texture;
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
    this.uniformData = new Float32Array(44);
    this.instanceCount = 0;
    this._accumWidth = 0;
    this._accumHeight = 0;
    this._netDirty = true;
    this._lastNetEye = new Float32Array([Number.NaN, Number.NaN, Number.NaN]);
    this.netLayout = splitOpacityPhiWeights(scene.opacityPhiWeights, scene.opacityPhi);
    if (this.netLayout.inputDim !== INPUT_DIM ||
        this.netLayout.hiddenDims[0] !== HIDDEN0 ||
        this.netLayout.hiddenDims[1] !== HIDDEN1 ||
        this.netLayout.hiddenDims[2] !== HIDDEN2) {
      throw new Error("Unsupported opacity_phi network shape for WebGPU renderer");
    }
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

    const w0 = packRowsToRGBA(this.netLayout.w0, HIDDEN0, INPUT_DIM);
    const w1 = packRowsToRGBA(this.netLayout.w1, HIDDEN1, HIDDEN0);
    const w2 = packRowsToRGBA(this.netLayout.w2, HIDDEN2, HIDDEN1);
    const wPhi = packRowsToRGBA(this.netLayout.wPhi, 1, HIDDEN2);
    const wOpacity = packRowsToRGBA(this.netLayout.wOpacity, 1, HIDDEN2);
    this.w0Texture = createFloatTexture(device, w0.texWidth, HIDDEN0, w0.packed);
    this.w1Texture = createFloatTexture(device, w1.texWidth, HIDDEN1, w1.packed);
    this.w2Texture = createFloatTexture(device, w2.texWidth, HIDDEN2, w2.packed);
    this.wPhiTexture = createFloatTexture(device, wPhi.texWidth, 1, wPhi.packed);
    this.wOpacityTexture = createFloatTexture(device, wOpacity.texWidth, 1, wOpacity.packed);
    const biasData = new Float32Array(BOPACITY_OFFSET + 1);
    biasData.set(this.netLayout.b0, B0_OFFSET);
    biasData.set(this.netLayout.b1, B1_OFFSET);
    biasData.set(this.netLayout.b2, B2_OFFSET);
    biasData.set(this.netLayout.bPhi, BPHI_OFFSET);
    biasData.set(this.netLayout.bOpacity, BOPACITY_OFFSET);
    this.biasBuffer = createStorageBuffer(device, biasData);

    const netShader = device.createShaderModule({ code: WGSL_NET });
    const splatShader = device.createShaderModule({ code: WGSL_SPLAT });
    const composeShader = device.createShaderModule({ code: WGSL_COMPOSE });

    this.netBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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

    this.splatPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.splatBindGroupLayout] }),
      vertex: vertexState,
      fragment: {
        module: splatShader,
        entryPoint: "fsSplat",
        targets: [
          { format: this.accumFormat, blend: additiveBlend },
          { format: this.logTFormat, blend: additiveBlend },
        ],
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
        { binding: 4, resource: this.w0Texture.createView() },
        { binding: 5, resource: this.w1Texture.createView() },
        { binding: 6, resource: this.w2Texture.createView() },
        { binding: 7, resource: this.wPhiTexture.createView() },
        { binding: 8, resource: this.wOpacityTexture.createView() },
        { binding: 9, resource: { buffer: this.biasBuffer } },
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
    this._netDirty = true;

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

    const packed = this.uniformData;
    packed.fill(0);
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
    const eye = cameraState.eye;
    const needsNetPass = this._netDirty ||
      eye[0] !== this._lastNetEye[0] ||
      eye[1] !== this._lastNetEye[1] ||
      eye[2] !== this._lastNetEye[2];
    if (needsNetPass) {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.netPipeline);
      computePass.setBindGroup(0, this.netBindGroup);
      computePass.dispatchWorkgroups(Math.ceil(this.instanceCount / WORKGROUP_SIZE));
      computePass.end();
      this._lastNetEye[0] = eye[0];
      this._lastNetEye[1] = eye[1];
      this._lastNetEye[2] = eye[2];
      this._netDirty = false;
    }

    const splatPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.accumView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: this.logTView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    splatPass.setPipeline(this.splatPipeline);
    splatPass.setBindGroup(0, this.splatBindGroup);
    splatPass.setVertexBuffer(0, this.instanceBuffer);
    splatPass.draw(6, this.instanceCount);
    splatPass.end();

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
