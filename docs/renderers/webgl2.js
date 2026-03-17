import { packRowsToRGBA, splitOpacityPhiWeights } from "./opacity_phi.js";

const GEOM_FLOATS = 11;
const SH_FLOATS = 12;
const SH_TEX_WIDTH = 2048;
const INPUT_DIM = 22;
const HIDDEN0 = 256;
const HIDDEN1 = 128;
const HIDDEN2 = 64;

const NET_VERT = `#version 300 es
precision highp float;
precision highp sampler2D;

layout(location = 0) in vec4 aPosOpacity;
layout(location = 1) in vec4 aQuat;
layout(location = 2) in vec3 aScale;

uniform vec3 uEye;
uniform highp sampler2D uSHData;
uniform highp sampler2D uW0;
uniform highp sampler2D uW1;
uniform highp sampler2D uW2;
uniform highp sampler2D uWPhi;
uniform highp sampler2D uWOpacity;
uniform float uB0[${HIDDEN0}];
uniform float uB1[${HIDDEN1}];
uniform float uB2[${HIDDEN2}];
uniform float uBPhi;
uniform float uBOpacity;

out float vPhi;
out float vOpacity;

vec4 getSHTexel(int splatIndex, int texelIndex) {
  int g = splatIndex * 3 + texelIndex;
  return texelFetch(uSHData, ivec2(g % ${SH_TEX_WIDTH}, g / ${SH_TEX_WIDTH}), 0);
}

vec4 getWeightTexel(highp sampler2D tex, int row, int block) {
  return texelFetch(tex, ivec2(block, row), 0);
}

void main() {
  int splatIndex = gl_VertexID;
  vec4 sh0 = getSHTexel(splatIndex, 0);
  vec4 sh1 = getSHTexel(splatIndex, 1);
  vec4 sh2 = getSHTexel(splatIndex, 2);

  float inputVec[24];
  inputVec[0] = sh0.x;
  inputVec[1] = sh0.y;
  inputVec[2] = sh0.z;
  inputVec[3] = sh0.w;
  inputVec[4] = sh1.x;
  inputVec[5] = sh1.y;
  inputVec[6] = sh1.z;
  inputVec[7] = sh1.w;
  inputVec[8] = sh2.x;
  inputVec[9] = sh2.y;
  inputVec[10] = sh2.z;
  inputVec[11] = sh2.w;

  float shNormSq = 0.0;
  for (int i = 0; i < 12; i++) {
    shNormSq += inputVec[i] * inputVec[i];
  }
  float shInvNorm = inversesqrt(max(shNormSq, 1e-8));
  for (int i = 0; i < 12; i++) {
    inputVec[i] *= shInvNorm;
  }

  vec3 viewdir = normalize(aPosOpacity.xyz - uEye);
  vec3 scaleNorm = normalize(aScale);
  inputVec[12] = viewdir.x;
  inputVec[13] = viewdir.y;
  inputVec[14] = viewdir.z;
  inputVec[15] = scaleNorm.x;
  inputVec[16] = scaleNorm.y;
  inputVec[17] = scaleNorm.z;
  inputVec[18] = aQuat.x;
  inputVec[19] = aQuat.y;
  inputVec[20] = aQuat.z;
  inputVec[21] = aQuat.w;
  inputVec[22] = 0.0;
  inputVec[23] = 0.0;

  float layer0[${HIDDEN0}];
  for (int row = 0; row < ${HIDDEN0}; row++) {
    float sum = uB0[row];
    for (int block = 0; block < 6; block++) {
      vec4 w = getWeightTexel(uW0, row, block);
      int base = block * 4;
      sum += dot(w, vec4(inputVec[base], inputVec[base + 1], inputVec[base + 2], inputVec[base + 3]));
    }
    layer0[row] = max(sum, 0.0);
  }

  float layer1[${HIDDEN1}];
  for (int row = 0; row < ${HIDDEN1}; row++) {
    float sum = uB1[row];
    for (int block = 0; block < ${HIDDEN0 / 4}; block++) {
      vec4 w = getWeightTexel(uW1, row, block);
      int base = block * 4;
      sum += dot(w, vec4(layer0[base], layer0[base + 1], layer0[base + 2], layer0[base + 3]));
    }
    layer1[row] = max(sum, 0.0);
  }

  float layer2[${HIDDEN2}];
  for (int row = 0; row < ${HIDDEN2}; row++) {
    float sum = uB2[row];
    for (int block = 0; block < ${HIDDEN1 / 4}; block++) {
      vec4 w = getWeightTexel(uW2, row, block);
      int base = block * 4;
      sum += dot(w, vec4(layer1[base], layer1[base + 1], layer1[base + 2], layer1[base + 3]));
    }
    layer2[row] = max(sum, 0.0);
  }

  float phiValue = uBPhi;
  float opacityValue = uBOpacity;
  for (int block = 0; block < ${HIDDEN2 / 4}; block++) {
    vec4 wPhi = getWeightTexel(uWPhi, 0, block);
    vec4 wOpacity = getWeightTexel(uWOpacity, 0, block);
    int base = block * 4;
    vec4 activations = vec4(layer2[base], layer2[base + 1], layer2[base + 2], layer2[base + 3]);
    phiValue += dot(wPhi, activations);
    opacityValue += dot(wOpacity, activations);
  }

  vPhi = max(phiValue, 0.0);
  vOpacity = 1.0 / (1.0 + exp(-opacityValue));
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
`;

const NET_FRAG = `#version 300 es
precision highp float;
void main() {}
`;

const SPLAT_VERT = `#version 300 es
precision highp float;
precision highp sampler2D;

layout(location = 0) in vec4 aPosOpacity;
layout(location = 1) in vec4 aQuat;
layout(location = 2) in vec3 aScale;
layout(location = 3) in vec2 aNet;

uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uFocal;
uniform float uWidth;
uniform float uHeight;
uniform vec3 uEye;
uniform highp sampler2D uSHData;

out vec3 vConic;
out vec2 vCenterPix;
out vec4 vColor;
out float vWeight;

vec2 quadVertex(int idx) {
  if (idx == 0) return vec2(-1.0, -1.0);
  if (idx == 1) return vec2(1.0, -1.0);
  if (idx == 2) return vec2(-1.0, 1.0);
  if (idx == 3) return vec2(-1.0, 1.0);
  if (idx == 4) return vec2(1.0, -1.0);
  return vec2(1.0, 1.0);
}

mat3 quatToMat(vec4 q) {
  float qw = q.x;
  float qx = q.y;
  float qy = q.z;
  float qz = q.w;
  return mat3(
    1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy),
    2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx),
    2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)
  );
}

vec4 getSHTexel(int splatIndex, int texelIndex) {
  int g = splatIndex * 3 + texelIndex;
  return texelFetch(uSHData, ivec2(g % ${SH_TEX_WIDTH}, g / ${SH_TEX_WIDTH}), 0);
}

vec3 evalSH1(int splatIndex, vec3 dir) {
  vec4 t0 = getSHTexel(splatIndex, 0);
  vec4 t1 = getSHTexel(splatIndex, 1);
  vec4 t2 = getSHTexel(splatIndex, 2);

  vec3 c0 = vec3(t0.x, t0.y, t0.z);
  vec3 c1 = vec3(t0.w, t1.x, t1.y);
  vec3 c2 = vec3(t1.z, t1.w, t2.x);
  vec3 c3 = vec3(t2.y, t2.z, t2.w);

  float x = dir.x;
  float y = dir.y;
  float z = dir.z;
  float c = 0.4886025119029199;
  vec3 color = c0 * 0.28209479177387814;
  color += c1 * (c * (-y));
  color += c2 * (c * z);
  color += c3 * (c * (-x));
  return max(color + 0.5, vec3(0.0));
}

void main() {
  vec3 pos = aPosOpacity.xyz;
  vec3 scale = aScale;
  float phi = aNet.x;
  float opacity = aNet.y;
  int splatIndex = gl_InstanceID;

  vec4 pv = uView * vec4(pos, 1.0);
  float fwd = -pv.z;
  if (fwd < 0.01) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vConic = vec3(0.0);
    vCenterPix = vec2(0.0);
    vColor = vec4(0.0);
    vWeight = 0.0;
    return;
  }

  mat3 R = quatToMat(aQuat);
  mat3 W = mat3(uView);
  mat3 M = mat3(W * (R[0] * scale.x), W * (R[1] * scale.y), W * (R[2] * scale.z));
  mat3 Sv = M * transpose(M);

  float invD = 1.0 / fwd;
  float invD2 = invD * invD;
  float tx = pv.x;
  float ty = pv.y;
  vec3 J0 = vec3(uFocal * invD, 0.0, uFocal * tx * invD2);
  vec3 J1 = vec3(0.0, uFocal * invD, uFocal * ty * invD2);
  float cov00 = dot(J0, Sv * J0) + 0.3;
  float cov01 = dot(J0, Sv * J1);
  float cov11 = dot(J1, Sv * J1) + 0.3;
  float det = cov00 * cov11 - cov01 * cov01;
  if (det <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vConic = vec3(0.0);
    vCenterPix = vec2(0.0);
    vColor = vec4(0.0);
    vWeight = 0.0;
    return;
  }

  float invDet = 1.0 / det;
  vec3 conic = vec3(cov11 * invDet, -cov01 * invDet, cov00 * invDet);
  float trace = cov00 + cov11;
  float disc = max(0.0, trace * trace - 4.0 * det);
  float radius = min(3.0 * sqrt(0.5 * (trace + sqrt(disc))), 1024.0);

  vec4 clip = uViewProj * vec4(pos, 1.0);
  float invW = 1.0 / clip.w;
  float cx = (clip.x * invW * 0.5 + 0.5) * uWidth;
  float cy = (clip.y * invW * 0.5 + 0.5) * uHeight;
  vec2 local = quadVertex(gl_VertexID);

  gl_Position = vec4(
    (cx + local.x * radius) / uWidth * 2.0 - 1.0,
    (cy + local.y * radius) / uHeight * 2.0 - 1.0,
    clip.z * invW,
    1.0
  );

  vConic = conic;
  vCenterPix = vec2(cx, cy);
  vColor = vec4(evalSH1(splatIndex, normalize(pos - uEye)), opacity);

  float maxScale = max(scale.x, max(scale.y, scale.z));
  vWeight = exp(min(maxScale / fwd, 20.0)) + phi / (fwd * fwd) + phi * phi;
}
`;

const SPLAT_FRAG = `#version 300 es
precision highp float;

in vec3 vConic;
in vec2 vCenterPix;
in vec4 vColor;
in float vWeight;

uniform float uAlphaScale;

out vec4 outAccum;

void main() {
  vec2 d = gl_FragCoord.xy - vCenterPix;
  float power = 0.5 * (vConic.x * d.x * d.x + 2.0 * vConic.y * d.x * d.y + vConic.z * d.y * d.y);
  if (power > 8.0) discard;
  float alpha = min(0.99, vColor.a * exp(-power) * uAlphaScale);
  float alphaW = alpha * vWeight;
  if (alphaW < 0.00001) discard;
  outAccum = vec4(vColor.rgb * alphaW, alphaW);
}
`;

const LOGT_FRAG = `#version 300 es
precision highp float;

in vec3 vConic;
in vec2 vCenterPix;
in vec4 vColor;

uniform float uAlphaScale;

out vec4 outLogT;

void main() {
  vec2 d = gl_FragCoord.xy - vCenterPix;
  float power = 0.5 * (vConic.x * d.x * d.x + 2.0 * vConic.y * d.x * d.y + vConic.z * d.y * d.y);
  if (power > 8.0) discard;
  float alpha = min(0.99, vColor.a * exp(-power) * uAlphaScale);
  if (alpha < 0.00001) discard;
  outLogT = vec4(log(1.0 - alpha), 0.0, 0.0, 0.0);
}
`;

const COMPOSE_VERT = `#version 300 es
void main() {
  vec2 p[3];
  p[0] = vec2(-1.0, -1.0);
  p[1] = vec2(3.0, -1.0);
  p[2] = vec2(-1.0, 3.0);
  gl_Position = vec4(p[gl_VertexID], 0.0, 1.0);
}
`;

const COMPOSE_FRAG = `#version 300 es
precision highp float;

uniform highp sampler2D uAccum;
uniform highp sampler2D uLogT;
uniform vec3 uBgColor;

out vec4 outColor;

void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uAccum, 0));
  vec4 accum = texture(uAccum, uv);
  float logT = texture(uLogT, uv).r;
  float wFg = accum.a;
  vec3 colorFg = (wFg > 0.0001) ? accum.rgb / wFg : vec3(0.0);
  float T = exp(logT);
  outColor = vec4(mix(uBgColor, colorFg, 1.0 - T), 1.0);
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

function linkProgram(gl, vertexSource, fragmentSource, varyings) {
  const program = gl.createProgram();
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  if (varyings) {
    gl.transformFeedbackVaryings(program, varyings, gl.INTERLEAVED_ATTRIBS);
  }
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

function createFloatTexture(gl, width, height, data) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

export class WebGL2SplatRenderer {
  static async create(canvas, scene) {
    if (!scene.opacityPhi) return null;
    try {
      const gl = canvas.getContext("webgl2", {
        antialias: false,
        alpha: true,
        premultipliedAlpha: true,
      });
      if (!gl) return null;
      if (!gl.getExtension("EXT_color_buffer_float")) return null;
      return new WebGL2SplatRenderer(canvas, scene, gl);
    } catch (error) {
      console.warn("WebGL2 renderer unavailable", error);
      return null;
    }
  }

  constructor(canvas, scene, gl) {
    if (scene.shDegree !== 1 || scene.shFloats !== SH_FLOATS) {
      throw new Error("Corrected WebGL2 renderer currently expects SH1 exports");
    }
    this.canvas = canvas;
    this.scene = scene;
    this.gl = gl;
    this.renderOptions = { alphaScale: scene.render.alphaScale };
    this._bg = scene.render.backgroundTop.slice(0, 3);
    this._instanceBufferSize = 0;
    this._netBufferSize = 0;
    this._fboWidth = 0;
    this._fboHeight = 0;
    this.instanceCount = 0;

    const layout = splitOpacityPhiWeights(scene.opacityPhiWeights, scene.opacityPhi);
    if (layout.inputDim !== INPUT_DIM ||
        layout.hiddenDims[0] !== HIDDEN0 ||
        layout.hiddenDims[1] !== HIDDEN1 ||
        layout.hiddenDims[2] !== HIDDEN2) {
      throw new Error("Unsupported opacity_phi network shape for WebGL2 renderer");
    }

    this.netLayout = layout;
    this.netProgram = linkProgram(gl, NET_VERT, NET_FRAG, ["vPhi", "vOpacity"]);
    this.splatProgram = linkProgram(gl, SPLAT_VERT, SPLAT_FRAG);
    this.logTProgram = linkProgram(gl, SPLAT_VERT, LOGT_FRAG);
    this.composeProgram = linkProgram(gl, COMPOSE_VERT, COMPOSE_FRAG);

    this.netLoc = {
      eye: gl.getUniformLocation(this.netProgram, "uEye"),
      shData: gl.getUniformLocation(this.netProgram, "uSHData"),
      w0: gl.getUniformLocation(this.netProgram, "uW0"),
      w1: gl.getUniformLocation(this.netProgram, "uW1"),
      w2: gl.getUniformLocation(this.netProgram, "uW2"),
      wPhi: gl.getUniformLocation(this.netProgram, "uWPhi"),
      wOpacity: gl.getUniformLocation(this.netProgram, "uWOpacity"),
      b0: gl.getUniformLocation(this.netProgram, "uB0"),
      b1: gl.getUniformLocation(this.netProgram, "uB1"),
      b2: gl.getUniformLocation(this.netProgram, "uB2"),
      bPhi: gl.getUniformLocation(this.netProgram, "uBPhi"),
      bOpacity: gl.getUniformLocation(this.netProgram, "uBOpacity"),
    };

    this.splatLoc = {
      viewProj: gl.getUniformLocation(this.splatProgram, "uViewProj"),
      view: gl.getUniformLocation(this.splatProgram, "uView"),
      focal: gl.getUniformLocation(this.splatProgram, "uFocal"),
      width: gl.getUniformLocation(this.splatProgram, "uWidth"),
      height: gl.getUniformLocation(this.splatProgram, "uHeight"),
      eye: gl.getUniformLocation(this.splatProgram, "uEye"),
      alphaScale: gl.getUniformLocation(this.splatProgram, "uAlphaScale"),
      shData: gl.getUniformLocation(this.splatProgram, "uSHData"),
    };

    this.logTLoc = {
      viewProj: gl.getUniformLocation(this.logTProgram, "uViewProj"),
      view: gl.getUniformLocation(this.logTProgram, "uView"),
      focal: gl.getUniformLocation(this.logTProgram, "uFocal"),
      width: gl.getUniformLocation(this.logTProgram, "uWidth"),
      height: gl.getUniformLocation(this.logTProgram, "uHeight"),
      eye: gl.getUniformLocation(this.logTProgram, "uEye"),
      alphaScale: gl.getUniformLocation(this.logTProgram, "uAlphaScale"),
      shData: gl.getUniformLocation(this.logTProgram, "uSHData"),
    };

    this.composeLoc = {
      accum: gl.getUniformLocation(this.composeProgram, "uAccum"),
      logT: gl.getUniformLocation(this.composeProgram, "uLogT"),
      bgColor: gl.getUniformLocation(this.composeProgram, "uBgColor"),
    };

    this.instanceBuffer = gl.createBuffer();
    this.netBuffer = gl.createBuffer();
    this.transformFeedback = gl.createTransformFeedback();
    this.composeVao = gl.createVertexArray();
    this._createVaos();
    this._createWeightTextures();

    this.accumFBO = gl.createFramebuffer();
    this.logTFBO = gl.createFramebuffer();
    this.accumTexture = gl.createTexture();
    this.logTTexture = gl.createTexture();
  }

  _createVaos() {
    const gl = this.gl;

    this.netVao = gl.createVertexArray();
    gl.bindVertexArray(this.netVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    const stride = GEOM_FLOATS * 4;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 16);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 32);
    gl.bindVertexArray(null);

    this.renderVao = gl.createVertexArray();
    gl.bindVertexArray(this.renderVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(0, 1);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 32);
    gl.vertexAttribDivisor(2, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.netBuffer);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 8, 0);
    gl.vertexAttribDivisor(3, 1);
    gl.bindVertexArray(null);
  }

  _createWeightTextures() {
    const gl = this.gl;
    const w0 = packRowsToRGBA(this.netLayout.w0, HIDDEN0, INPUT_DIM);
    const w1 = packRowsToRGBA(this.netLayout.w1, HIDDEN1, HIDDEN0);
    const w2 = packRowsToRGBA(this.netLayout.w2, HIDDEN2, HIDDEN1);
    const wPhi = packRowsToRGBA(this.netLayout.wPhi, 1, HIDDEN2);
    const wOpacity = packRowsToRGBA(this.netLayout.wOpacity, 1, HIDDEN2);

    this.w0Texture = createFloatTexture(gl, w0.texWidth, HIDDEN0, w0.packed);
    this.w1Texture = createFloatTexture(gl, w1.texWidth, HIDDEN1, w1.packed);
    this.w2Texture = createFloatTexture(gl, w2.texWidth, HIDDEN2, w2.packed);
    this.wPhiTexture = createFloatTexture(gl, wPhi.texWidth, 1, wPhi.packed);
    this.wOpacityTexture = createFloatTexture(gl, wOpacity.texWidth, 1, wOpacity.packed);
  }

  get label() {
    return "webgl2";
  }

  _ensureFBO(width, height) {
    if (width === this._fboWidth && height === this._fboHeight) return;
    const gl = this.gl;
    this._fboWidth = width;
    this._fboHeight = height;

    const setupTexture = (texture) => {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    };

    setupTexture(this.accumTexture);
    setupTexture(this.logTTexture);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.accumTexture, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.logTFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.logTTexture, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
    this._ensureFBO(width, height);
  }

  initSHData(shData) {
    const gl = this.gl;
    const totalTexels = (shData.length / SH_FLOATS) * 3;
    const height = Math.ceil(totalTexels / SH_TEX_WIDTH);
    const texData = new Uint16Array(SH_TEX_WIDTH * height * 4);
    texData.set(shData);

    if (!this.shTexture) this.shTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.shTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, SH_TEX_WIDTH, height, 0, gl.RGBA, gl.HALF_FLOAT, texData);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  updateGeometryData(geomData) {
    const gl = this.gl;
    this.instanceCount = geomData.length / GEOM_FLOATS;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    if (this._instanceBufferSize === geomData.byteLength) {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, geomData);
    } else {
      gl.bufferData(gl.ARRAY_BUFFER, geomData, gl.DYNAMIC_DRAW);
      this._instanceBufferSize = geomData.byteLength;
    }

    const netBytes = this.instanceCount * 2 * 4;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.netBuffer);
    if (this._netBufferSize !== netBytes) {
      gl.bufferData(gl.ARRAY_BUFFER, netBytes, gl.DYNAMIC_DRAW);
      this._netBufferSize = netBytes;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  updateSceneData(geomData) {
    return this.updateGeometryData(geomData);
  }

  setRenderOptions(options) {
    this.renderOptions = options;
  }

  _bindWeightTextures(program) {
    const gl = this.gl;
    const bindings = [
      { texture: this.shTexture, location: gl.getUniformLocation(program, "uSHData"), unit: 1 },
      { texture: this.w0Texture, location: gl.getUniformLocation(program, "uW0"), unit: 2 },
      { texture: this.w1Texture, location: gl.getUniformLocation(program, "uW1"), unit: 3 },
      { texture: this.w2Texture, location: gl.getUniformLocation(program, "uW2"), unit: 4 },
      { texture: this.wPhiTexture, location: gl.getUniformLocation(program, "uWPhi"), unit: 5 },
      { texture: this.wOpacityTexture, location: gl.getUniformLocation(program, "uWOpacity"), unit: 6 },
    ];

    for (const binding of bindings) {
      gl.activeTexture(gl.TEXTURE0 + binding.unit);
      gl.bindTexture(gl.TEXTURE_2D, binding.texture);
      gl.uniform1i(binding.location, binding.unit);
    }
  }

  _runNetPass(cameraState) {
    const gl = this.gl;
    gl.useProgram(this.netProgram);
    gl.bindVertexArray(this.netVao);
    gl.uniform3fv(this.netLoc.eye, cameraState.eye);
    gl.uniform1fv(this.netLoc.b0, this.netLayout.b0);
    gl.uniform1fv(this.netLoc.b1, this.netLayout.b1);
    gl.uniform1fv(this.netLoc.b2, this.netLayout.b2);
    gl.uniform1f(this.netLoc.bPhi, this.netLayout.bPhi[0]);
    gl.uniform1f(this.netLoc.bOpacity, this.netLayout.bOpacity[0]);
    this._bindWeightTextures(this.netProgram);

    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.netBuffer);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.transformFeedback);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.instanceCount);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
    gl.bindVertexArray(null);
  }

  _bindRenderShared(program, loc, cameraState) {
    const gl = this.gl;
    gl.useProgram(program);
    gl.uniformMatrix4fv(loc.viewProj, false, cameraState.viewProjection);
    gl.uniformMatrix4fv(loc.view, false, cameraState.view);
    gl.uniform1f(loc.focal, cameraState.focal);
    gl.uniform1f(loc.width, this.canvas.width);
    gl.uniform1f(loc.height, this.canvas.height);
    gl.uniform3fv(loc.eye, cameraState.eye);
    gl.uniform1f(loc.alphaScale, this.renderOptions.alphaScale);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.shTexture);
    gl.uniform1i(loc.shData, 1);
  }

  render(cameraState) {
    const gl = this.gl;
    const width = this.canvas.width;
    const height = this.canvas.height;
    this._ensureFBO(width, height);

    this._runNetPass(cameraState);

    gl.bindVertexArray(this.renderVao);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.DEPTH_TEST);
    this._bindRenderShared(this.splatProgram, this.splatLoc, cameraState);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.logTFBO);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this._bindRenderShared(this.logTProgram, this.logTLoc, cameraState);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);

    gl.bindVertexArray(null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.composeVao);
    gl.useProgram(this.composeProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.accumTexture);
    gl.uniform1i(this.composeLoc.accum, 0);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.logTTexture);
    gl.uniform1i(this.composeLoc.logT, 2);
    gl.uniform3fv(this.composeLoc.bgColor, this._bg);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}
