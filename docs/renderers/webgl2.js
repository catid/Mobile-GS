// Instance buffer layout (64 bytes = 16 floats per splat):
//   offset  0 : vec4  posOpacity  (x, y, z, opacity)
//   offset 16 : vec4  quat        (qw, qx, qy, qz)
//   offset 32 : vec4  scalePad    (sx, sy, sz, 0)
//   offset 48 : vec4  colorPad    (r,  g,  b,  0)
const FLOATS_PER_SPLAT = 16;

const VERTEX_SOURCE = `#version 300 es
precision highp float;

layout(location = 0) in vec4 aPosOpacity;   // x y z opacity
layout(location = 1) in vec4 aQuat;          // qw qx qy qz
layout(location = 2) in vec4 aScalePad;      // sx sy sz 0
layout(location = 3) in vec4 aColorPad;      // r  g  b  0

uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uFocal;
uniform float uWidth;
uniform float uHeight;

out vec3 vConic;
out vec2 vCenterPix;  // in gl_FragCoord space: y=0 at BOTTOM
out vec4 vColor;      // rgb + opacity

vec2 quadVertex(int idx) {
  if (idx == 0) return vec2(-1.0, -1.0);
  if (idx == 1) return vec2( 1.0, -1.0);
  if (idx == 2) return vec2(-1.0,  1.0);
  if (idx == 3) return vec2(-1.0,  1.0);
  if (idx == 4) return vec2( 1.0, -1.0);
  return vec2( 1.0,  1.0);
}

// Unit quaternion (qw, qx, qy, qz) → column-major 3×3 rotation matrix.
mat3 quatToMat(vec4 q) {
  float qw = q.x, qx = q.y, qy = q.z, qz = q.w;
  return mat3(
    1.0 - 2.0*(qy*qy + qz*qz),  2.0*(qx*qy + qw*qz),  2.0*(qx*qz - qw*qy),
    2.0*(qx*qy - qw*qz),  1.0 - 2.0*(qx*qx + qz*qz),  2.0*(qy*qz + qw*qx),
    2.0*(qx*qz + qw*qy),  2.0*(qy*qz - qw*qx),  1.0 - 2.0*(qx*qx + qy*qy)
  );
}

void main() {
  vec3  pos     = aPosOpacity.xyz;
  float opacity = aPosOpacity.w;
  vec3  scale   = aScalePad.xyz;
  vec3  color   = aColorPad.rgb;

  // ---- View-space position -----------------------------------------------
  vec4  p_view4 = uView * vec4(pos, 1.0);
  float fwd     = -p_view4.z;       // positive depth for visible splats
  if (fwd < 0.01) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vConic = vec3(0.0); vCenterPix = vec2(0.0); vColor = vec4(0.0);
    return;
  }

  // ---- View-space covariance  Sigma_v = M * M^T  where M = W * R * S -----
  mat3 R  = quatToMat(aQuat);
  mat3 W  = mat3(uView);             // upper-left 3×3 of view matrix
  mat3 M  = mat3(
    W * (R[0] * scale.x),
    W * (R[1] * scale.y),
    W * (R[2] * scale.z)
  );
  mat3 Sv = M * transpose(M);

  // ---- Jacobian of perspective projection (pixel space) ------------------
  float inv_d  = 1.0 / fwd;
  float inv_d2 = inv_d * inv_d;
  float tx = p_view4.x, ty = p_view4.y;
  vec3 J0 = vec3(uFocal * inv_d,  0.0,           uFocal * tx * inv_d2);
  vec3 J1 = vec3(0.0,             uFocal * inv_d, uFocal * ty * inv_d2);

  // ---- 2D screen-space covariance  Sigma_2D = J Sv J^T ------------------
  vec3  SvJ0  = Sv * J0;
  vec3  SvJ1  = Sv * J1;
  float cov00 = dot(J0, SvJ0) + 0.3;   // low-frequency regularisation
  float cov01 = dot(J0, SvJ1);
  float cov11 = dot(J1, SvJ1) + 0.3;

  // ---- Conic = inverse of Sigma_2D ---------------------------------------
  float det = cov00 * cov11 - cov01 * cov01;
  if (det <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vConic = vec3(0.0); vCenterPix = vec2(0.0); vColor = vec4(0.0);
    return;
  }
  float inv_det = 1.0 / det;
  vec3 conic = vec3(cov11 * inv_det, -cov01 * inv_det, cov00 * inv_det);

  // ---- Bounding radius in pixels -----------------------------------------
  float trace   = cov00 + cov11;
  float disc    = max(0.0, trace * trace - 4.0 * det);
  float lambda1 = 0.5 * (trace + sqrt(disc));
  float radius  = min(3.0 * sqrt(lambda1), 1024.0);

  // ---- Project center (gl_FragCoord space: y = 0 at BOTTOM) --------------
  vec4  p_clip = uViewProj * vec4(pos, 1.0);
  float inv_pw = 1.0 / p_clip.w;
  vec2  ndc    = p_clip.xy * inv_pw;
  float cx     = (ndc.x * 0.5 + 0.5) * uWidth;
  float cy_gl  = (ndc.y * 0.5 + 0.5) * uHeight;   // y=0 at bottom (gl convention)

  // ---- Billboard quad vertex ---------------------------------------------
  vec2  local = quadVertex(gl_VertexID);
  float vx    = cx     + local.x * radius;
  float vy_gl = cy_gl  + local.y * radius;

  gl_Position = vec4(vx / uWidth * 2.0 - 1.0,
                     vy_gl / uHeight * 2.0 - 1.0,
                     p_clip.z * inv_pw, 1.0);
  vConic     = conic;
  vCenterPix = vec2(cx, cy_gl);
  vColor     = vec4(color, opacity);
}
`;

const FRAGMENT_SOURCE = `#version 300 es
precision highp float;

in vec3 vConic;
in vec2 vCenterPix;   // y=0 at bottom (gl_FragCoord convention)
in vec4 vColor;

uniform float uAlphaScale;

out vec4 outColor;

void main() {
  // gl_FragCoord.y is 0 at bottom — consistent with vCenterPix
  vec2  d     = gl_FragCoord.xy - vCenterPix;
  float power = 0.5 * (vConic.x * d.x * d.x
                     + 2.0 * vConic.y * d.x * d.y
                     + vConic.z * d.y * d.y);
  if (power > 8.0) discard;
  float alpha = vColor.a * exp(-power) * uAlphaScale;
  if (alpha < 0.00392) discard;
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
      const gl = canvas.getContext("webgl2", {
        antialias: false,
        alpha: true,
        premultipliedAlpha: true,
      });
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
    this.scene  = scene;
    this.gl     = gl;
    this.program = linkProgram(gl, VERTEX_SOURCE, FRAGMENT_SOURCE);
    this.renderOptions = { alphaScale: scene.render.alphaScale };

    this.locations = {
      viewProj:   gl.getUniformLocation(this.program, "uViewProj"),
      view:       gl.getUniformLocation(this.program, "uView"),
      focal:      gl.getUniformLocation(this.program, "uFocal"),
      width:      gl.getUniformLocation(this.program, "uWidth"),
      height:     gl.getUniformLocation(this.program, "uHeight"),
      alphaScale: gl.getUniformLocation(this.program, "uAlphaScale"),
    };

    this.instanceBuffer = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);

    const stride = FLOATS_PER_SPLAT * 4;   // 64 bytes
    // attr 0: posOpacity  (offset 0)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(0, 1);
    // attr 1: quat  (offset 16)
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(1, 1);
    // attr 2: scalePad  (offset 32)
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 32);
    gl.vertexAttribDivisor(2, 1);
    // attr 3: colorPad  (offset 48)
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, stride, 48);
    gl.vertexAttribDivisor(3, 1);

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
    this.canvas.width  = width;
    this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
  }

  updateSceneData(interleaved) {
    this.instanceCount = interleaved.length / FLOATS_PER_SPLAT;
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
    gl.uniformMatrix4fv(this.locations.view,     false, cameraState.view);
    gl.uniform1f(this.locations.focal,      cameraState.focal);
    gl.uniform1f(this.locations.width,      this.canvas.width);
    gl.uniform1f(this.locations.height,     this.canvas.height);
    gl.uniform1f(this.locations.alphaScale, this.renderOptions.alphaScale);

    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);
    gl.bindVertexArray(null);
  }
}
