const VERTEX_SOURCE = `#version 300 es
precision highp float;

layout(location = 0) in vec4 aCenterRadius;
layout(location = 1) in vec4 aColorAlpha;

uniform mat4 uViewProj;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uPointScale;

out vec2 vLocal;
out vec4 vColor;

vec2 quadVertex(int index) {
  if (index == 0) return vec2(-1.0, -1.0);
  if (index == 1) return vec2( 1.0, -1.0);
  if (index == 2) return vec2(-1.0,  1.0);
  if (index == 3) return vec2(-1.0,  1.0);
  if (index == 4) return vec2( 1.0, -1.0);
  return vec2(1.0, 1.0);
}

void main() {
  vec2 local = quadVertex(gl_VertexID);
  vec3 worldPosition =
    aCenterRadius.xyz +
    (uCameraRight * local.x + uCameraUp * local.y) *
      aCenterRadius.w *
      uPointScale;

  gl_Position = uViewProj * vec4(worldPosition, 1.0);
  vLocal = local;
  vColor = aColorAlpha;
}
`;

const FRAGMENT_SOURCE = `#version 300 es
precision highp float;

in vec2 vLocal;
in vec4 vColor;

uniform float uAlphaScale;

out vec4 outColor;

void main() {
  float radiusSquared = dot(vLocal, vLocal);
  if (radiusSquared > 1.0) {
    discard;
  }

  float falloff = exp(-radiusSquared * 3.5);
  float alpha = vColor.a * falloff * uAlphaScale;
  outColor = vec4(vColor.rgb * alpha, alpha);
}
`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(log ?? "Shader compilation failed");
  }
  return shader;
}

function linkProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(log ?? "Program link failed");
  }
  return program;
}

export class WebGL2SplatRenderer {
  static async create(canvas, scene) {
    try {
      const gl = canvas.getContext("webgl2", { antialias: true, alpha: true, premultipliedAlpha: true });
      if (!gl) {
        return null;
      }
      return new WebGL2SplatRenderer(canvas, scene, gl);
    } catch (error) {
      console.warn("WebGL2 renderer unavailable", error);
      return null;
    }
  }

  constructor(canvas, scene, gl) {
    this.canvas = canvas;
    this.scene = scene;
    this.gl = gl;
    this.program = linkProgram(gl, VERTEX_SOURCE, FRAGMENT_SOURCE);
    this.renderOptions = {
      pointScale: scene.render.pointScale,
      alphaScale: scene.render.alphaScale,
    };

    this.locations = {
      viewProj: gl.getUniformLocation(this.program, "uViewProj"),
      cameraRight: gl.getUniformLocation(this.program, "uCameraRight"),
      cameraUp: gl.getUniformLocation(this.program, "uCameraUp"),
      pointScale: gl.getUniformLocation(this.program, "uPointScale"),
      alphaScale: gl.getUniformLocation(this.program, "uAlphaScale"),
    };

    this.instanceBuffer = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 32, 0);
    gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 16);
    gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(
      scene.render.backgroundTop[0],
      scene.render.backgroundTop[1],
      scene.render.backgroundTop[2],
      scene.render.backgroundTop[3],
    );
  }

  get label() {
    return "webgl2";
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
  }

  updateSceneData(interleaved) {
    this.instanceCount = interleaved.length / 8;
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.instanceBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, interleaved, this.gl.DYNAMIC_DRAW);
  }

  setRenderOptions(options) {
    this.renderOptions = options;
  }

  render(cameraState) {
    const gl = this.gl;
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this.locations.viewProj, false, cameraState.viewProjection);
    gl.uniform3fv(this.locations.cameraRight, cameraState.right);
    gl.uniform3fv(this.locations.cameraUp, cameraState.up);
    gl.uniform1f(this.locations.pointScale, this.renderOptions.pointScale);
    gl.uniform1f(this.locations.alphaScale, this.renderOptions.alphaScale);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);
    gl.bindVertexArray(null);
  }
}
