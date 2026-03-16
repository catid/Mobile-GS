// Instance buffer layout (60 floats = 240 bytes per splat, SH3 format):
//   offset   0 : vec4  posOpacity  (x, y, z, opacity)
//   offset  16 : vec4  quat        (qw, qx, qy, qz)
//   offset  32 : vec4  scalePad    (sx, sy, sz, 0)
//   offset  48 : vec4  sh_r[0..3]  (dc_r, rest_r_0..2)
//   offset  64 : vec4  sh_r[4..7]
//   offset  80 : vec4  sh_r[8..11]
//   offset  96 : vec4  sh_r[12..15]
//   offset 112 : vec4  sh_g[0..3]  ... (same pattern)
//   offset 176 : vec4  sh_b[0..3]  ... (same pattern)
const FLOATS_PER_SPLAT = 60;

const VERTEX_SOURCE = `#version 300 es
precision highp float;

layout(location =  0) in vec4 aPosOpacity;   // x y z opacity
layout(location =  1) in vec4 aQuat;          // qw qx qy qz
layout(location =  2) in vec4 aScalePad;      // sx sy sz 0
layout(location =  3) in vec4 aSH_R0;         // sh_r[0..3]
layout(location =  4) in vec4 aSH_R1;
layout(location =  5) in vec4 aSH_R2;
layout(location =  6) in vec4 aSH_R3;
layout(location =  7) in vec4 aSH_G0;         // sh_g[0..3]
layout(location =  8) in vec4 aSH_G1;
layout(location =  9) in vec4 aSH_G2;
layout(location = 10) in vec4 aSH_G3;
layout(location = 11) in vec4 aSH_B0;         // sh_b[0..3]
layout(location = 12) in vec4 aSH_B1;
layout(location = 13) in vec4 aSH_B2;
layout(location = 14) in vec4 aSH_B3;

uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uFocal;
uniform float uWidth;
uniform float uHeight;
uniform vec3 uEye;

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

// Evaluate SH degree-3 (16 coefficients per channel) given unit view direction.
vec3 evalSH3(vec3 dir,
    vec4 r0, vec4 r1, vec4 r2, vec4 r3,
    vec4 g0, vec4 g1, vec4 g2, vec4 g3,
    vec4 b0, vec4 b1, vec4 b2, vec4 b3) {
  float x = dir.x, y = dir.y, z = dir.z;

  // L0
  vec3 col = vec3(r0.x, g0.x, b0.x) * 0.28209479177387814;

  // L1
  float c1 = 0.4886025119029199;
  col += vec3(r0.y, g0.y, b0.y) * (c1 * (-y));
  col += vec3(r0.z, g0.z, b0.z) * (c1 * z);
  col += vec3(r0.w, g0.w, b0.w) * (c1 * (-x));

  // L2
  float xx = x*x, yy = y*y, zz = z*z;
  float xy = x*y, yz = y*z, xz = x*z;
  col += vec3(r1.x, g1.x, b1.x) * ( 1.0925484305920792 * xy);
  col += vec3(r1.y, g1.y, b1.y) * (-1.0925484305920792 * yz);
  col += vec3(r1.z, g1.z, b1.z) * ( 0.31539156525252005 * (2.0*zz - xx - yy));
  col += vec3(r1.w, g1.w, b1.w) * (-1.0925484305920792 * xz);
  col += vec3(r2.x, g2.x, b2.x) * ( 0.5462742152960396 * (xx - yy));

  // L3
  col += vec3(r2.y, g2.y, b2.y) * (-0.5900435899266435 * y * (3.0*xx - yy));
  col += vec3(r2.z, g2.z, b2.z) * ( 2.890611442640554  * xy * z);
  col += vec3(r2.w, g2.w, b2.w) * (-0.4570457994644658 * y * (4.0*zz - xx - yy));
  col += vec3(r3.x, g3.x, b3.x) * ( 0.3731763325901154 * z * (2.0*zz - 3.0*xx - 3.0*yy));
  col += vec3(r3.y, g3.y, b3.y) * (-0.4570457994644658 * x * (4.0*zz - xx - yy));
  col += vec3(r3.z, g3.z, b3.z) * ( 1.445305721320277  * z * (xx - yy));
  col += vec3(r3.w, g3.w, b3.w) * (-0.5900435899266435 * x * (xx - 3.0*yy));

  return max(col + 0.5, vec3(0.0));
}

void main() {
  vec3  pos     = aPosOpacity.xyz;
  float opacity = aPosOpacity.w;
  vec3  scale   = aScalePad.xyz;

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

  // ---- SH3 color evaluation (view-dependent) -----------------------------
  vec3 dir = normalize(pos - uEye);
  vec3 rgb = evalSH3(dir,
    aSH_R0, aSH_R1, aSH_R2, aSH_R3,
    aSH_G0, aSH_G1, aSH_G2, aSH_G3,
    aSH_B0, aSH_B1, aSH_B2, aSH_B3);
  vColor = vec4(rgb, opacity);
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
      eye:        gl.getUniformLocation(this.program, "uEye"),
      alphaScale: gl.getUniformLocation(this.program, "uAlphaScale"),
    };

    this.instanceBuffer = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);

    const stride = FLOATS_PER_SPLAT * 4;   // 240 bytes

    // attrs 0-2: posOpacity, quat, scalePad (offsets 0, 16, 32)
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 4, gl.FLOAT, false, stride, i * 16);
      gl.vertexAttribDivisor(i, 1);
    }
    // attrs 3-6: sh_r (offsets 48, 64, 80, 96)
    for (let i = 0; i < 4; i++) {
      gl.enableVertexAttribArray(3 + i);
      gl.vertexAttribPointer(3 + i, 4, gl.FLOAT, false, stride, 48 + i * 16);
      gl.vertexAttribDivisor(3 + i, 1);
    }
    // attrs 7-10: sh_g (offsets 112, 128, 144, 160)
    for (let i = 0; i < 4; i++) {
      gl.enableVertexAttribArray(7 + i);
      gl.vertexAttribPointer(7 + i, 4, gl.FLOAT, false, stride, 112 + i * 16);
      gl.vertexAttribDivisor(7 + i, 1);
    }
    // attrs 11-14: sh_b (offsets 176, 192, 208, 224)
    for (let i = 0; i < 4; i++) {
      gl.enableVertexAttribArray(11 + i);
      gl.vertexAttribPointer(11 + i, 4, gl.FLOAT, false, stride, 176 + i * 16);
      gl.vertexAttribDivisor(11 + i, 1);
    }

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
    if (this._instanceBufferSize === interleaved.byteLength) {
      this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, interleaved);
    } else {
      this.gl.bufferData(this.gl.ARRAY_BUFFER, interleaved, this.gl.DYNAMIC_DRAW);
      this._instanceBufferSize = interleaved.byteLength;
    }
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
    gl.uniform3fv(this.locations.eye,       cameraState.eye);
    gl.uniform1f(this.locations.alphaScale, this.renderOptions.alphaScale);

    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);
    gl.bindVertexArray(null);
  }
}
