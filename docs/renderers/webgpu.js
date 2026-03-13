const UNIFORM_BUFFER_SIZE = 112;

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
      pointScale: scene.render.pointScale,
      alphaScale: scene.render.alphaScale,
    };
  }

  async initialize() {
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    const shader = this.device.createShaderModule({
      code: `
struct CameraUniforms {
  viewProj: mat4x4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
};

const quad = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>( 1.0,  1.0),
);

@vertex
fn vsMain(
  @builtin(vertex_index) vertexIndex: u32,
  @location(0) centerRadius: vec4<f32>,
  @location(1) colorAlpha: vec4<f32>,
) -> VertexOutput {
  let local = quad[vertexIndex];
  let worldPosition =
    centerRadius.xyz +
    (camera.cameraRight.xyz * local.x + camera.cameraUp.xyz * local.y) *
      centerRadius.w *
      camera.params.x;

  var out: VertexOutput;
  out.position = camera.viewProj * vec4<f32>(worldPosition, 1.0);
  out.local = local;
  out.color = colorAlpha;
  return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(in.local, in.local);
  if (radiusSquared > 1.0) {
    discard;
  }

  let falloff = exp(-radiusSquared * 3.5);
  let alpha = in.color.a * falloff * camera.params.y;
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
      `,
    });

    this.uniformBuffer = this.device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.instanceBuffer = this.device.createBuffer({
      size: 32,
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
            arrayStride: 32,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x4" },
              { shaderLocation: 1, offset: 16, format: "float32x4" },
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
      multisample: {
        count: 1,
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
    if (this.instanceBuffer) {
      this.instanceBuffer.destroy();
    }
    this.instanceBuffer = this.device.createBuffer({
      size: interleaved.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.instanceBuffer, 0, interleaved.buffer, interleaved.byteOffset, interleaved.byteLength);
    this.instanceCount = interleaved.length / 8;
  }

  setRenderOptions(options) {
    this.renderOptions = options;
  }

  render(cameraState) {
    const packed = new Float32Array(28);
    packed.set(cameraState.viewProjection, 0);
    packed.set([...cameraState.right, 0.0], 16);
    packed.set([...cameraState.up, 0.0], 20);
    packed.set([this.renderOptions.pointScale, this.renderOptions.alphaScale, 0.0, 0.0], 24);
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
