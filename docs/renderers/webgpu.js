// Uniform buffer layout (160 bytes = 40 floats):
//   offset   0 : mat4x4  viewProj
//   offset  64 : mat4x4  view
//   offset 128 : vec4    viewport  (width, height, focal, 0)
//   offset 144 : vec4    params    (alphaScale, 0, 0, 0)
const UNIFORM_BUFFER_SIZE = 160;

// Instance buffer layout (64 bytes = 16 floats per splat):
//   offset  0 : vec4  posOpacity  (x, y, z, opacity)
//   offset 16 : vec4  quat        (qw, qx, qy, qz)
//   offset 32 : vec4  scalePad    (sx, sy, sz, 0)
//   offset 48 : vec4  colorPad    (r, g, b, 0)
const FLOATS_PER_SPLAT = 16;

const WGSL_SHADER = `
struct CameraUniforms {
  viewProj : mat4x4<f32>,
  view     : mat4x4<f32>,
  viewport : vec4<f32>,   // width, height, focal, 0
  params   : vec4<f32>,   // alphaScale, 0, 0, 0
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct VertexOutput {
  @builtin(position) position  : vec4<f32>,
  @location(0)       conic     : vec3<f32>,
  @location(1)       centerPix : vec2<f32>,
  @location(2)       color     : vec4<f32>,  // rgb + opacity
};

const quad = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>( 1.0,  1.0),
);

// Convert unit quaternion (qw, qx, qy, qz) to column-major rotation matrix.
fn quatToMat(q: vec4<f32>) -> mat3x3<f32> {
  let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0*(qy*qy + qz*qz),  2.0*(qx*qy + qw*qz),  2.0*(qx*qz - qw*qy)),
    vec3<f32>(2.0*(qx*qy - qw*qz),  1.0 - 2.0*(qx*qx + qz*qz),  2.0*(qy*qz + qw*qx)),
    vec3<f32>(2.0*(qx*qz + qw*qy),  2.0*(qy*qz - qw*qx),  1.0 - 2.0*(qx*qx + qy*qy)),
  );
}

@vertex
fn vsMain(
  @builtin(vertex_index) vertexIndex : u32,
  @location(0) posOpacity : vec4<f32>,  // x y z opacity
  @location(1) quat       : vec4<f32>,  // qw qx qy qz
  @location(2) scalePad   : vec4<f32>,  // sx sy sz 0
  @location(3) colorPad   : vec4<f32>,  // r  g  b  0
) -> VertexOutput {
  var out: VertexOutput;

  let pos     = posOpacity.xyz;
  let opacity = posOpacity.w;
  let scale   = scalePad.xyz;
  let color   = colorPad.rgb;

  // ---- View-space position -----------------------------------------------
  let p_view = camera.view * vec4<f32>(pos, 1.0);
  let fwd    = -p_view.z;            // positive depth for visible splats
  if (fwd < 0.01) {
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  // ---- View-space covariance  Sigma_v = M * M^T  where M = W * R * S -----
  // W = upper-left 3×3 of view matrix (world → view rotation)
  let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
  let R = quatToMat(quat);
  let M = mat3x3<f32>(
    W * (R[0] * scale.x),
    W * (R[1] * scale.y),
    W * (R[2] * scale.z),
  );
  let Sv = M * transpose(M);

  // ---- Jacobian of perspective projection (pixel space) ------------------
  let focal  = camera.viewport.z;
  let inv_d  = 1.0 / fwd;
  let inv_d2 = inv_d * inv_d;
  let tx = p_view.x;
  let ty = p_view.y;
  let J0 = vec3<f32>(focal * inv_d,  0.0,           focal * tx * inv_d2);
  let J1 = vec3<f32>(0.0,            focal * inv_d, focal * ty * inv_d2);

  // ---- 2D screen-space covariance  Sigma_2D = J Sv J^T ------------------
  let SvJ0  = Sv * J0;
  let SvJ1  = Sv * J1;
  var cov00 = dot(J0, SvJ0) + 0.3;  // low-frequency regularisation
  let cov01 = dot(J0, SvJ1);
  var cov11 = dot(J1, SvJ1) + 0.3;

  // ---- Conic = inverse of Sigma_2D ---------------------------------------
  let det = cov00 * cov11 - cov01 * cov01;
  if (det <= 0.0) {
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    return out;
  }
  let inv_det = 1.0 / det;
  let conic   = vec3<f32>(cov11 * inv_det, -cov01 * inv_det, cov00 * inv_det);

  // ---- Bounding radius in pixels -----------------------------------------
  let trace   = cov00 + cov11;
  let disc    = max(0.0, trace * trace - 4.0 * det);
  let lambda1 = 0.5 * (trace + sqrt(disc));
  let radius  = min(3.0 * sqrt(lambda1), 1024.0);

  // ---- Project center (WebGPU pixel coords: y = 0 at top) ---------------
  let p_clip = camera.viewProj * vec4<f32>(pos, 1.0);
  let inv_pw = 1.0 / p_clip.w;
  let ndc    = p_clip.xy * inv_pw;
  let width  = camera.viewport.x;
  let height = camera.viewport.y;
  let cx     = (ndc.x *  0.5 + 0.5) * width;
  let cy     = (0.5   - ndc.y * 0.5) * height;  // y flipped: 0 at top

  // ---- Billboard quad vertex ---------------------------------------------
  let local = quad[vertexIndex];
  let vx    = cx + local.x * radius;
  let vy    = cy + local.y * radius;

  out.position  = vec4<f32>(vx / width * 2.0 - 1.0,
                             1.0 - vy / height * 2.0,
                             p_clip.z * inv_pw, 1.0);
  out.conic     = conic;
  out.centerPix = vec2<f32>(cx, cy);
  out.color     = vec4<f32>(color, opacity);
  return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
  // in.position.xy: WebGPU fragment coords (y = 0 at top), same convention as centerPix
  let d     = in.position.xy - in.centerPix;
  let power = 0.5 * (in.conic.x * d.x * d.x
                   + 2.0 * in.conic.y * d.x * d.y
                   + in.conic.z * d.y * d.y);
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

    this.instanceBuffer = this.device.createBuffer({
      size: 64,   // minimum allocation; will be replaced on first data upload
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
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
            arrayStride: FLOATS_PER_SPLAT * 4,   // 64 bytes per splat
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset:  0, format: "float32x4" },  // posOpacity
              { shaderLocation: 1, offset: 16, format: "float32x4" },  // quat
              { shaderLocation: 2, offset: 32, format: "float32x4" },  // scalePad
              { shaderLocation: 3, offset: 48, format: "float32x4" },  // colorPad
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

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    this.clearValue = {
      r: this.scene.render.backgroundTop[0],
      g: this.scene.render.backgroundTop[1],
      b: this.scene.render.backgroundTop[2],
      a: this.scene.render.backgroundTop[3],
    };
  }

  get label() {
    return "webgpu";
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  updateSceneData(interleaved) {
    // Reuse the existing buffer if same size; only reallocate when it changes.
    if (!this.instanceBuffer || this.instanceBuffer.size !== interleaved.byteLength) {
      if (this.instanceBuffer) this.instanceBuffer.destroy();
      this.instanceBuffer = this.device.createBuffer({
        size: interleaved.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(
      this.instanceBuffer, 0,
      interleaved.buffer, interleaved.byteOffset, interleaved.byteLength,
    );
    this.instanceCount = interleaved.length / FLOATS_PER_SPLAT;
  }

  setRenderOptions(options) {
    this.renderOptions = options;
  }

  render(cameraState) {
    const W = this.canvas.width;
    const H = this.canvas.height;
    const focal = cameraState.focal;

    // Pack uniforms: viewProj(16) + view(16) + viewport(4) + params(4) = 40 floats
    const packed = new Float32Array(40);
    packed.set(cameraState.viewProjection, 0);
    packed.set(cameraState.view, 16);
    packed[32] = W;
    packed[33] = H;
    packed[34] = focal;
    packed[35] = 0.0;
    packed[36] = this.renderOptions.alphaScale;
    // packed[37..39] = 0 (default)
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
