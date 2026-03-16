// Uniform buffer layout (176 bytes = 44 floats):
//   offset   0 : mat4x4  viewProj
//   offset  64 : mat4x4  view
//   offset 128 : vec4    viewport  (width, height, focal, 0)
//   offset 144 : vec4    params    (alphaScale, 0, 0, 0)
//   offset 160 : vec4    eye       (ex, ey, ez, 0)
const UNIFORM_BUFFER_SIZE = 176;

// Geometry vertex buffer layout (12 floats = 48 bytes per splat):
//   offset  0 : vec4  posOpacity  (x, y, z, opacity)
//   offset 16 : vec4  quat        (qw, qx, qy, qz)
//   offset 32 : vec4  scaleIdx    (sx, sy, sz, float(originalIndex))
const GEOM_FLOATS = 12;

const WGSL_SHADER = `
struct CameraUniforms {
  viewProj : mat4x4<f32>,
  view     : mat4x4<f32>,
  viewport : vec4<f32>,
  params   : vec4<f32>,
  eye      : vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera : CameraUniforms;
@group(0) @binding(1) var<storage, read> sh_data : array<f32>;

struct VertexOutput {
  @builtin(position) position  : vec4<f32>,
  @location(0)       conic     : vec3<f32>,
  @location(1)       centerPix : vec2<f32>,
  @location(2)       color     : vec4<f32>,
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

fn getSH(origIdx: u32, ofs: u32) -> vec4<f32> {
  let b = origIdx * 48u + ofs;
  return vec4<f32>(sh_data[b], sh_data[b+1u], sh_data[b+2u], sh_data[b+3u]);
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
  @builtin(vertex_index) vi : u32,
  @location(0) posOpacity : vec4<f32>,
  @location(1) quat       : vec4<f32>,
  @location(2) scaleIdx   : vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  let pos     = posOpacity.xyz;
  let opacity = posOpacity.w;
  let scale   = scaleIdx.xyz;
  let origIdx = u32(scaleIdx.w);

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
  let sh_r0 = getSH(origIdx,  0u); let sh_r1 = getSH(origIdx,  4u);
  let sh_r2 = getSH(origIdx,  8u); let sh_r3 = getSH(origIdx, 12u);
  let sh_g0 = getSH(origIdx, 16u); let sh_g1 = getSH(origIdx, 20u);
  let sh_g2 = getSH(origIdx, 24u); let sh_g3 = getSH(origIdx, 28u);
  let sh_b0 = getSH(origIdx, 32u); let sh_b1 = getSH(origIdx, 36u);
  let sh_b2 = getSH(origIdx, 40u); let sh_b3 = getSH(origIdx, 44u);
  let rgb = evalSH3(dir, sh_r0,sh_r1,sh_r2,sh_r3, sh_g0,sh_g1,sh_g2,sh_g3, sh_b0,sh_b1,sh_b2,sh_b3);
  out.color = vec4<f32>(rgb, opacity);
  return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let d = in.position.xy - in.centerPix;
  let power = 0.5*(in.conic.x*d.x*d.x + 2.0*in.conic.y*d.x*d.y + in.conic.z*d.y*d.y);
  if (power > 8.0) { discard; }
  let alpha = in.color.a * exp(-power) * camera.params.x;
  if (alpha < 0.00392) { discard; }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;

export class WebGPUSplatRenderer {
  static async create(canvas, scene) {
    if (!navigator.gpu) {
      return null;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return null;
      }

      const device = await adapter.requestDevice();
      const context = canvas.getContext("webgpu");
      if (!context) {
        return null;
      }
      const format = navigator.gpu.getPreferredCanvasFormat();
      const renderer = new WebGPUSplatRenderer(canvas, scene, device, context, format);
      await renderer.initialize();
      return renderer;
    } catch (error) {
      console.warn("WebGPU renderer unavailable", error);
      return null;
    }
  }

  constructor(canvas, scene, device, context, format) {
    this.canvas = canvas;
    this.scene = scene;
    this.device = device;
    this.context = context;
    this.format = format;
    this.instanceCount = 0;
    this.renderOptions = {
      alphaScale: scene.render.alphaScale,
    };
  }

  async initialize() {
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    const shader = this.device.createShaderModule({ code: WGSL_SHADER });

    this.uniformBuffer = this.device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Placeholder SH storage buffer (1 element); replaced by initSHData()
    this.shBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.instanceBuffer = this.device.createBuffer({
      size: GEOM_FLOATS * 4,   // minimum allocation; will be replaced on first data upload
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: shader,
        entryPoint: "vsMain",
        buffers: [
          {
            arrayStride: 48,   // 12 floats * 4 bytes = 48 bytes per splat
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset:  0, format: "float32x4" },  // posOpacity
              { shaderLocation: 1, offset: 16, format: "float32x4" },  // quat
              { shaderLocation: 2, offset: 32, format: "float32x4" },  // scaleIdx
            ],
          },
        ],
      },
      fragment: {
        module: shader,
        entryPoint: "fsMain",
        targets: [
          {
            format: this.format,
            blend: {
              color: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
    });

    this._rebuildBindGroup();

    this.clearValue = {
      r: this.scene.render.backgroundTop[0],
      g: this.scene.render.backgroundTop[1],
      b: this.scene.render.backgroundTop[2],
      a: this.scene.render.backgroundTop[3],
    };
  }

  _rebuildBindGroup() {
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.shBuffer } },
      ],
    });
  }

  get label() {
    return "webgpu";
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  initSHData(shData) {
    // Create/resize storage buffer and upload SH data, then rebuild bind group
    if (!this.shBuffer || this.shBuffer.size !== shData.byteLength) {
      if (this.shBuffer) this.shBuffer.destroy();
      this.shBuffer = this.device.createBuffer({
        size: shData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(
      this.shBuffer, 0,
      shData.buffer, shData.byteOffset, shData.byteLength,
    );
    this._rebuildBindGroup();
  }

  updateGeometryData(geomData) {
    // Reuse the existing buffer if same size; only reallocate when it changes.
    if (!this.instanceBuffer || this.instanceBuffer.size !== geomData.byteLength) {
      if (this.instanceBuffer) this.instanceBuffer.destroy();
      this.instanceBuffer = this.device.createBuffer({
        size: geomData.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(
      this.instanceBuffer, 0,
      geomData.buffer, geomData.byteOffset, geomData.byteLength,
    );
    this.instanceCount = geomData.length / GEOM_FLOATS;
  }

  // Alias for backwards compatibility
  updateSceneData(geomData) {
    return this.updateGeometryData(geomData);
  }

  setRenderOptions(options) {
    this.renderOptions = options;
  }

  render(cameraState) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const focal = cameraState.focal;

    // Pack uniforms: viewProj(16) + view(16) + viewport(4) + params(4) + eye(4) = 44 floats
    const packed = new Float32Array(44);
    packed.set(cameraState.viewProjection, 0);
    packed.set(cameraState.view, 16);
    packed[32] = W;
    packed[33] = H;
    packed[34] = focal;
    packed[35] = 0.0;
    packed[36] = this.renderOptions.alphaScale;
    // packed[37..39] = 0 (default)
    packed[40] = cameraState.eye[0];
    packed[41] = cameraState.eye[1];
    packed[42] = cameraState.eye[2];
    packed[43] = 0.0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, packed.buffer);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: this.clearValue,
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.instanceBuffer);
    pass.draw(6, this.instanceCount);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }
}
