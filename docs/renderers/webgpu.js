// Mobile-GS two-pass WebGPU renderer — matches _ms CUDA blending formula (phi=0):
//   Pass 1: MRT with two render targets
//     target 0 (rgba16float): accumulate (color*alpha_w, alpha_w) additively
//     target 1 (r16float):    accumulate log(1-alpha_clamped) additively
//   Pass 2: compose
//     w_fg     = accum.a
//     C_fg     = accum.rgb / w_fg
//     T        = exp(logT_accum)  -- = prod(1-alpha_i), exact transmittance
//     coverage = 1 - T            -- matches CUDA formula exactly
//     output   = mix(bg, C_fg, coverage)
//   alpha = min(0.99, opacity * exp(-power) * alphaScale)  -- clamped as in CUDA
//   weight = exp(min(max_scale/depth, 20))                 -- in color only, not T
//
// Uniform buffer layout (176 bytes = 44 floats):
//   offset   0 : mat4x4  viewProj
//   offset  64 : mat4x4  view
//   offset 128 : vec4    viewport  (width, height, focal, 0)
//   offset 144 : vec4    params    (alphaScale, 0, 0, 0)
//   offset 160 : vec4    eye       (ex, ey, ez, 0)
const UNIFORM_BUFFER_SIZE = 176;

// Geometry vertex buffer layout (11 floats = 44 bytes per splat, v5 format):
//   offset  0 : vec4  posOpacity  (x, y, z, opacity)
//   offset 16 : vec4  quat        (qw, qx, qy, qz)
//   offset 32 : vec3  scale       (sx, sy, sz)  — origIdx removed, use instance_index
const GEOM_FLOATS = 11;

// ---------------------------------------------------------------------------
// Pass 1: splat accumulation shader
// ---------------------------------------------------------------------------
const WGSL_SPLAT = `
struct CameraUniforms {
  viewProj : mat4x4<f32>,
  view     : mat4x4<f32>,
  viewport : vec4<f32>,
  params   : vec4<f32>,
  eye      : vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
// SH stored as rgba16float texture: width=2048, each splat occupies 12 texels (4 f16 each)
@group(0) @binding(1) var shTex : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position  : vec4<f32>,
  @location(0)       conic     : vec3<f32>,
  @location(1)       centerPix : vec2<f32>,
  @location(2)       color     : vec4<f32>,  // rgb=SH color, a=opacity
  @location(3)       weight    : f32,         // exp(min(max_scale/depth, 20))
};

struct FragOut {
  @location(0) accum : vec4<f32>,  // color*alpha_w, alpha_w (w_fg)
  @location(1) logT  : f32,        // log(1-alpha_clamped) — negative, sums to log(T)
};

const quad = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
  vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
);

fn quatToMat(q: vec4<f32>) -> mat3x3<f32> {
  let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0-2.0*(qy*qy+qz*qz), 2.0*(qx*qy+qw*qz), 2.0*(qx*qz-qw*qy)),
    vec3<f32>(2.0*(qx*qy-qw*qz), 1.0-2.0*(qx*qx+qz*qz), 2.0*(qy*qz+qw*qx)),
    vec3<f32>(2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1.0-2.0*(qx*qx+qy*qy)),
  );
}

// group=0..11: each splat has 12 texels of 4 f16 values = 48 SH coefficients
fn getSH(origIdx: u32, group: u32) -> vec4<f32> {
  let g = origIdx * 12u + group;
  return textureLoad(shTex, vec2<i32>(i32(g % 2048u), i32(g / 2048u)), 0);
}

fn evalSH3(dir: vec3<f32>,
    r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, r3: vec4<f32>,
    g0: vec4<f32>, g1: vec4<f32>, g2: vec4<f32>, g3: vec4<f32>,
    b0: vec4<f32>, b1: vec4<f32>, b2: vec4<f32>, b3: vec4<f32>,
) -> vec3<f32> {
  let x = dir.x; let y = dir.y; let z = dir.z;
  var col = vec3<f32>(r0.x, g0.x, b0.x) * 0.28209479177387814;
  let c1 = 0.4886025119029199;
  col += vec3<f32>(r0.y, g0.y, b0.y) * (c1 * (-y));
  col += vec3<f32>(r0.z, g0.z, b0.z) * (c1 * z);
  col += vec3<f32>(r0.w, g0.w, b0.w) * (c1 * (-x));
  let xx = x*x; let yy = y*y; let zz = z*z;
  let xy = x*y; let yz = y*z; let xz = x*z;
  col += vec3<f32>(r1.x, g1.x, b1.x) * ( 1.0925484305920792 * xy);
  col += vec3<f32>(r1.y, g1.y, b1.y) * (-1.0925484305920792 * yz);
  col += vec3<f32>(r1.z, g1.z, b1.z) * ( 0.31539156525252005 * (2.0*zz-xx-yy));
  col += vec3<f32>(r1.w, g1.w, b1.w) * (-1.0925484305920792 * xz);
  col += vec3<f32>(r2.x, g2.x, b2.x) * ( 0.5462742152960396 * (xx-yy));
  col += vec3<f32>(r2.y, g2.y, b2.y) * (-0.5900435899266435 * y * (3.0*xx-yy));
  col += vec3<f32>(r2.z, g2.z, b2.z) * ( 2.890611442640554  * xy * z);
  col += vec3<f32>(r2.w, g2.w, b2.w) * (-0.4570457994644658 * y * (4.0*zz-xx-yy));
  col += vec3<f32>(r3.x, g3.x, b3.x) * ( 0.3731763325901154 * z * (2.0*zz-3.0*xx-3.0*yy));
  col += vec3<f32>(r3.y, g3.y, b3.y) * (-0.4570457994644658 * x * (4.0*zz-xx-yy));
  col += vec3<f32>(r3.z, g3.z, b3.z) * ( 1.445305721320277  * z * (xx-yy));
  col += vec3<f32>(r3.w, g3.w, b3.w) * (-0.5900435899266435 * x * (xx-3.0*yy));
  return max(col + 0.5, vec3<f32>(0.0));
}

@vertex
fn vsMain(
  @builtin(vertex_index)   vi      : u32,
  @builtin(instance_index) instIdx : u32,
  @location(0) posOpacity : vec4<f32>,
  @location(1) quat       : vec4<f32>,
  @location(2) scaleXYZ   : vec3<f32>,  // v5: no origIdx, use instIdx instead
) -> VertexOutput {
  var out: VertexOutput;
  let pos     = posOpacity.xyz;
  let opacity = posOpacity.w;
  let scale   = scaleXYZ;
  let origIdx = instIdx;

  let p_view = camera.view * vec4<f32>(pos, 1.0);
  let fwd    = -p_view.z;
  if (fwd < 0.01) { out.position = vec4<f32>(0.0,0.0,2.0,1.0); return out; }

  let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  let R = quatToMat(quat);
  let M = mat3x3<f32>(W*(R[0]*scale.x), W*(R[1]*scale.y), W*(R[2]*scale.z));
  let Sv = M * transpose(M);

  let focal = camera.viewport.z;
  let inv_d  = 1.0 / fwd;
  let inv_d2 = inv_d * inv_d;
  let tx = p_view.x; let ty = p_view.y;
  let J0 = vec3<f32>(focal*inv_d, 0.0, focal*tx*inv_d2);
  let J1 = vec3<f32>(0.0, focal*inv_d, focal*ty*inv_d2);

  var cov00 = dot(J0, Sv*J0) + 0.3;
  let cov01 = dot(J0, Sv*J1);
  var cov11 = dot(J1, Sv*J1) + 0.3;
  let det = cov00*cov11 - cov01*cov01;
  if (det <= 0.0) { out.position = vec4<f32>(0.0,0.0,2.0,1.0); return out; }
  let inv_det = 1.0 / det;
  let conic   = vec3<f32>(cov11*inv_det, -cov01*inv_det, cov00*inv_det);

  let trace   = cov00 + cov11;
  let disc    = max(0.0, trace*trace - 4.0*det);
  let lambda1 = 0.5*(trace + sqrt(disc));
  let radius  = min(3.0*sqrt(lambda1), 1024.0);

  let p_clip = camera.viewProj * vec4<f32>(pos, 1.0);
  let inv_pw = 1.0 / p_clip.w;
  let ndc    = p_clip.xy * inv_pw;
  let width  = camera.viewport.x;
  let height = camera.viewport.y;
  let cx     = (ndc.x * 0.5 + 0.5) * width;
  let cy     = (0.5 - ndc.y * 0.5) * height;

  let local = quad[vi];
  out.position  = vec4<f32>((cx + local.x*radius)/width*2.0-1.0,
                              1.0-(cy + local.y*radius)/height*2.0,
                              p_clip.z*inv_pw, 1.0);
  out.conic     = conic;
  out.centerPix = vec2<f32>(cx, cy);

  let dir = normalize(pos - camera.eye.xyz);
  let sh_r0 = getSH(origIdx,  0u); let sh_r1 = getSH(origIdx,  1u);
  let sh_r2 = getSH(origIdx,  2u); let sh_r3 = getSH(origIdx,  3u);
  let sh_g0 = getSH(origIdx,  4u); let sh_g1 = getSH(origIdx,  5u);
  let sh_g2 = getSH(origIdx,  6u); let sh_g3 = getSH(origIdx,  7u);
  let sh_b0 = getSH(origIdx,  8u); let sh_b1 = getSH(origIdx,  9u);
  let sh_b2 = getSH(origIdx, 10u); let sh_b3 = getSH(origIdx, 11u);
  let rgb = evalSH3(dir, sh_r0,sh_r1,sh_r2,sh_r3, sh_g0,sh_g1,sh_g2,sh_g3, sh_b0,sh_b1,sh_b2,sh_b3);
  out.color = vec4<f32>(rgb, opacity);

  let max_scale = max(scale.x, max(scale.y, scale.z));
  out.weight = exp(min(max_scale / fwd, 20.0));
  return out;
}

@fragment
fn fsMain(in: VertexOutput) -> FragOut {
  let d = in.position.xy - in.centerPix;
  let power = 0.5*(in.conic.x*d.x*d.x + 2.0*in.conic.y*d.x*d.y + in.conic.z*d.y*d.y);
  if (power > 8.0) { discard; }
  // alpha clamped at 0.99 as in the CUDA kernel — affects both T and w_fg
  let alpha = min(0.99, in.color.a * exp(-power) * camera.params.x);
  let alpha_w = alpha * in.weight;
  if (alpha_w < 0.00001) { discard; }
  var out: FragOut;
  // target 0: color accumulation (with weight) and w_fg
  out.accum = vec4<f32>(in.color.rgb * alpha_w, alpha_w);
  // target 1: log-transmittance (weight-independent, as in CUDA T = prod(1-alpha))
  out.logT  = log(1.0 - alpha);
  return out;
}
`;

// ---------------------------------------------------------------------------
// Pass 2: composition shader
// ---------------------------------------------------------------------------
const WGSL_COMPOSE = `
@group(0) @binding(0) var accumTex : texture_2d<f32>;
@group(0) @binding(1) var<uniform> bgColor : vec4<f32>;
@group(0) @binding(2) var logTTex : texture_2d<f32>;

@vertex
fn vsCompose(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let corners = array<vec2<f32>,3>(vec2(-1,-1), vec2(3,-1), vec2(-1,3));
  return vec4<f32>(corners[vi], 0.0, 1.0);
}

@fragment
fn fsCompose(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
  let coords = vec2<i32>(i32(fragPos.x), i32(fragPos.y));
  let accum    = textureLoad(accumTex, coords, 0);
  let logT     = textureLoad(logTTex,  coords, 0).r;
  let w_fg     = accum.a;
  let C_fg     = select(vec3<f32>(0.0), accum.rgb / w_fg, w_fg > 0.0001);
  // T = prod(1-alpha_i); cleared to 0 → exp(0)=1 (fully transparent) before splats
  let T        = exp(logT);
  let coverage = 1.0 - T;
  let color    = mix(bgColor.rgb, C_fg, coverage);
  return vec4<f32>(color, 1.0);
}
`;

export class WebGPUSplatRenderer {
  static async create(canvas, scene) {
    if (!navigator.gpu) return null;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;
      const device  = await adapter.requestDevice();
      const format  = navigator.gpu.getPreferredCanvasFormat();
      // Delay canvas context acquisition until AFTER initialize() succeeds,
      // so a failed init doesn't prevent WebGL2 from claiming the canvas.
      const renderer = new WebGPUSplatRenderer(canvas, scene, device, format);
      await renderer.initialize();
      // All init succeeded — now claim the canvas context
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

  constructor(canvas, scene, device, format) {
    this.canvas  = canvas;
    this.scene   = scene;
    this.device  = device;
    this.context = null;  // set by create() after initialize() succeeds
    this.format  = format;
    this.instanceCount = 0;
    this.renderOptions = { alphaScale: scene.render.alphaScale };
    this._accumWidth  = 0;
    this._accumHeight = 0;
  }

  async initialize() {
    const device = this.device;
    // Note: canvas context is acquired AFTER initialize() by create() to avoid
    // preventing WebGL2 fallback if something here fails.

    // ── Splat program ────────────────────────────────────────────────────
    const splatShader = device.createShaderModule({ code: WGSL_SPLAT });

    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // SH stored as rgba16float texture (v5 format)
    this.shTexture = null;  // created in initSHData()
    this.instanceBuffer = device.createBuffer({
      size: GEOM_FLOATS * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this.splatBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: "float" } },
      ],
    });

    this.splatPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.splatBGL] }),
      vertex: {
        module: splatShader, entryPoint: "vsMain",
        buffers: [{
          arrayStride: 44, stepMode: "instance",  // v5: 11 floats = 44 bytes
          attributes: [
            { shaderLocation: 0, offset:  0, format: "float32x4" },  // pos+opacity
            { shaderLocation: 1, offset: 16, format: "float32x4" },  // quat
            { shaderLocation: 2, offset: 32, format: "float32x3" },  // scale only
          ],
        }],
      },
      fragment: {
        module: splatShader, entryPoint: "fsMain",
        targets: [
          {
            format: "rgba16float",
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
          {
            // log-transmittance: accumulate log(1-alpha) additively
            // clear=0 → T=exp(0)=1; each splat contributes negative value → T→0
            format: "r16float",
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
    });

    // ── Compose program ─────────────────────────────────────────────────
    const composeShader = device.createShaderModule({ code: WGSL_COMPOSE });

    const bg = scene.render.backgroundTop;
    this.bgBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.bgBuffer, 0,
      new Float32Array([bg[0], bg[1], bg[2], bg[3]]));

    this.composeBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      ],
    });

    this.composePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.composeBGL] }),
      vertex:   { module: composeShader, entryPoint: "vsCompose" },
      fragment: {
        module: composeShader, entryPoint: "fsCompose",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this._rebuildSplatBindGroup();
    // accumulation textures created on first resize()
  }

  _rebuildSplatBindGroup() {
    if (!this.shTexture) return;  // defer until initSHData() provides the texture
    this.splatBindGroup = this.device.createBindGroup({
      layout: this.splatBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.shTexture.createView() },
      ],
    });
  }

  _ensureAccumTexture(width, height) {
    if (width === this._accumWidth && height === this._accumHeight) return;
    this._accumWidth  = width;
    this._accumHeight = height;
    if (this.accumTexture) this.accumTexture.destroy();
    if (this.logTTexture)  this.logTTexture.destroy();
    this.accumTexture = this.device.createTexture({
      size: [width, height],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.logTTexture = this.device.createTexture({
      size: [width, height],
      format: "r16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.accumView = this.accumTexture.createView();
    this.logTView  = this.logTTexture.createView();
    this.composeBindGroup = this.device.createBindGroup({
      layout: this.composeBGL,
      entries: [
        { binding: 0, resource: this.accumView },
        { binding: 1, resource: { buffer: this.bgBuffer } },
        { binding: 2, resource: this.logTView },
      ],
    });
  }

  get label() { return "webgpu"; }

  resize(width, height) {
    this.canvas.width  = width;
    this.canvas.height = height;
    this._ensureAccumTexture(width, height);
  }

  initSHData(shData) {
    // shData: Uint16Array of raw float16 bits, N*48 elements (v5 format)
    const device = this.device;
    const N = shData.length / 48;
    const SH_TEX_WIDTH = 2048;
    const totalTexels  = N * 12;
    const height = Math.ceil(totalTexels / SH_TEX_WIDTH);

    if (this.shTexture) this.shTexture.destroy();
    this.shTexture = device.createTexture({
      size: [SH_TEX_WIDTH, height],
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Pad data to full rows
    const padded = new Uint16Array(SH_TEX_WIDTH * height * 4);
    padded.set(shData);

    device.queue.writeTexture(
      { texture: this.shTexture },
      padded.buffer,
      { offset: 0, bytesPerRow: SH_TEX_WIDTH * 8, rowsPerImage: height },
      { width: SH_TEX_WIDTH, height },
    );
    this._rebuildSplatBindGroup();
  }

  updateGeometryData(geomData) {
    const device = this.device;
    if (!this.instanceBuffer || this.instanceBuffer.size !== geomData.byteLength) {
      if (this.instanceBuffer) this.instanceBuffer.destroy();
      this.instanceBuffer = device.createBuffer({
        size: geomData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(this.instanceBuffer, 0, geomData.buffer, geomData.byteOffset, geomData.byteLength);
    this.instanceCount = geomData.length / GEOM_FLOATS;
  }

  updateSceneData(geomData) { return this.updateGeometryData(geomData); }

  setRenderOptions(options) { this.renderOptions = options; }

  render(cameraState) {
    if (!this.splatBindGroup) return;  // wait for initSHData()
    const device = this.device;
    const W = this.canvas.width;
    const H = this.canvas.height;
    this._ensureAccumTexture(W, H);

    // Pack uniforms
    const packed = new Float32Array(44);
    packed.set(cameraState.viewProjection, 0);
    packed.set(cameraState.view, 16);
    packed[32] = W; packed[33] = H; packed[34] = cameraState.focal; packed[35] = 0;
    packed[36] = this.renderOptions.alphaScale;
    packed[40] = cameraState.eye[0]; packed[41] = cameraState.eye[1];
    packed[42] = cameraState.eye[2]; packed[43] = 0;
    device.queue.writeBuffer(this.uniformBuffer, 0, packed.buffer);

    const encoder = device.createCommandEncoder();

    // ── Pass 1: accumulate splats (MRT) ──────────────────────────────────
    const accumPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.accumView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: this.logTView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },  // clear to 0 → T=exp(0)=1
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    accumPass.setPipeline(this.splatPipeline);
    accumPass.setBindGroup(0, this.splatBindGroup);
    accumPass.setVertexBuffer(0, this.instanceBuffer);
    accumPass.draw(6, this.instanceCount);
    accumPass.end();

    // ── Pass 2: compose to canvas ────────────────────────────────────────
    const composePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    composePass.setPipeline(this.composePipeline);
    composePass.setBindGroup(0, this.composeBindGroup);
    composePass.draw(3);
    composePass.end();

    device.queue.submit([encoder.finish()]);
  }
}
