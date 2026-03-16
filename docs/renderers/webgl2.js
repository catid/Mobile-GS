// Mobile-GS two-pass renderer — matches the _ms CUDA blending formula (phi=0):
//   Pass 1: accumulate (color * alpha_w, alpha_w) additively into RGBA16F FBO
//   Pass 2: compose — C/w_fg * coverage + bg*(1-coverage), coverage=1-exp(-w_fg)
//   weight = exp(min(max_scale / depth, 20))  — geometry-based, no MLP needed
//
// Geometry vertex buffer layout (12 floats = 48 bytes per splat):
//   offset  0 : vec4  posOpacity  (x, y, z, opacity)
//   offset 16 : vec4  quat        (qw, qx, qy, qz)
//   offset 32 : vec4  scaleIdx    (sx, sy, sz, float(originalIndex))
// SH data stored in RGBA32F texture: width=2048, each splat occupies 12 texels
const GEOM_FLOATS = 12;

// ---------------------------------------------------------------------------
// Pass 1: splat accumulation into float FBO
// ---------------------------------------------------------------------------
const SPLAT_VERT = `#version 300 es
precision highp float;
precision highp sampler2D;

layout(location = 0) in vec4 aPosOpacity;
layout(location = 1) in vec4 aQuat;
layout(location = 2) in vec4 aScaleIdx;

uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uFocal;
uniform float uWidth;
uniform float uHeight;
uniform vec3 uEye;
uniform highp sampler2D uSHData;

out vec3 vConic;
out vec2 vCenterPix;
out vec4 vColor;    // rgb = SH color, a = opacity
out float vWeight;  // Mobile-GS weight = exp(min(max_scale/depth, 20))

vec2 quadVertex(int idx) {
  if (idx==0) return vec2(-1,-1); if (idx==1) return vec2(1,-1);
  if (idx==2) return vec2(-1,1);  if (idx==3) return vec2(-1,1);
  if (idx==4) return vec2(1,-1);  return vec2(1,1);
}

mat3 quatToMat(vec4 q) {
  float qw=q.x,qx=q.y,qy=q.z,qz=q.w;
  return mat3(
    1.0-2.0*(qy*qy+qz*qz), 2.0*(qx*qy+qw*qz), 2.0*(qx*qz-qw*qy),
    2.0*(qx*qy-qw*qz), 1.0-2.0*(qx*qx+qz*qz), 2.0*(qy*qz+qw*qx),
    2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1.0-2.0*(qx*qx+qy*qy)
  );
}

vec4 getSH(int origIdx, int group) {
  int g = origIdx * 12 + group;
  return texelFetch(uSHData, ivec2(g % 2048, g / 2048), 0);
}

vec3 evalSH3(vec3 dir,
    vec4 r0,vec4 r1,vec4 r2,vec4 r3,
    vec4 g0,vec4 g1,vec4 g2,vec4 g3,
    vec4 b0,vec4 b1,vec4 b2,vec4 b3) {
  float x=dir.x,y=dir.y,z=dir.z;
  vec3 col=vec3(r0.x,g0.x,b0.x)*0.28209479177387814;
  float c1=0.4886025119029199;
  col+=vec3(r0.y,g0.y,b0.y)*(c1*(-y));
  col+=vec3(r0.z,g0.z,b0.z)*(c1*z);
  col+=vec3(r0.w,g0.w,b0.w)*(c1*(-x));
  float xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z;
  col+=vec3(r1.x,g1.x,b1.x)*(1.0925484305920792*xy);
  col+=vec3(r1.y,g1.y,b1.y)*(-1.0925484305920792*yz);
  col+=vec3(r1.z,g1.z,b1.z)*(0.31539156525252005*(2.0*zz-xx-yy));
  col+=vec3(r1.w,g1.w,b1.w)*(-1.0925484305920792*xz);
  col+=vec3(r2.x,g2.x,b2.x)*(0.5462742152960396*(xx-yy));
  col+=vec3(r2.y,g2.y,b2.y)*(-0.5900435899266435*y*(3.0*xx-yy));
  col+=vec3(r2.z,g2.z,b2.z)*(2.890611442640554*xy*z);
  col+=vec3(r2.w,g2.w,b2.w)*(-0.4570457994644658*y*(4.0*zz-xx-yy));
  col+=vec3(r3.x,g3.x,b3.x)*(0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy));
  col+=vec3(r3.y,g3.y,b3.y)*(-0.4570457994644658*x*(4.0*zz-xx-yy));
  col+=vec3(r3.z,g3.z,b3.z)*(1.445305721320277*z*(xx-yy));
  col+=vec3(r3.w,g3.w,b3.w)*(-0.5900435899266435*x*(xx-3.0*yy));
  return max(col+0.5,vec3(0.0));
}

void main() {
  vec3 pos=aPosOpacity.xyz; float opacity=aPosOpacity.w; vec3 scale=aScaleIdx.xyz;
  int origIdx=int(aScaleIdx.w);
  vec4 pv=uView*vec4(pos,1.0); float fwd=-pv.z;
  if(fwd<0.01){gl_Position=vec4(0,0,2,1);vConic=vec3(0);vCenterPix=vec2(0);vColor=vec4(0);vWeight=0.0;return;}
  mat3 R=quatToMat(aQuat); mat3 W=mat3(uView);
  mat3 M=mat3(W*(R[0]*scale.x),W*(R[1]*scale.y),W*(R[2]*scale.z));
  mat3 Sv=M*transpose(M);
  float inv_d=1.0/fwd,inv_d2=inv_d*inv_d,tx=pv.x,ty=pv.y;
  vec3 J0=vec3(uFocal*inv_d,0,uFocal*tx*inv_d2),J1=vec3(0,uFocal*inv_d,uFocal*ty*inv_d2);
  float cov00=dot(J0,Sv*J0)+0.3,cov01=dot(J0,Sv*J1),cov11=dot(J1,Sv*J1)+0.3;
  float det=cov00*cov11-cov01*cov01;
  if(det<=0.0){gl_Position=vec4(0,0,2,1);vConic=vec3(0);vCenterPix=vec2(0);vColor=vec4(0);vWeight=0.0;return;}
  float inv_det=1.0/det;
  vec3 conic=vec3(cov11*inv_det,-cov01*inv_det,cov00*inv_det);
  float tr=cov00+cov11,disc=max(0.0,tr*tr-4.0*det);
  float radius=min(3.0*sqrt(0.5*(tr+sqrt(disc))),1024.0);
  vec4 pc=uViewProj*vec4(pos,1.0); float inv_pw=1.0/pc.w;
  float cx=(pc.x*inv_pw*0.5+0.5)*uWidth,cy_gl=(pc.y*inv_pw*0.5+0.5)*uHeight;
  vec2 lv=quadVertex(gl_VertexID);
  gl_Position=vec4((cx+lv.x*radius)/uWidth*2.0-1.0,(cy_gl+lv.y*radius)/uHeight*2.0-1.0,pc.z*inv_pw,1.0);
  vConic=conic; vCenterPix=vec2(cx,cy_gl);
  vec4 r0=getSH(origIdx,0),r1=getSH(origIdx,1),r2=getSH(origIdx,2),r3=getSH(origIdx,3);
  vec4 gg0=getSH(origIdx,4),gg1=getSH(origIdx,5),gg2=getSH(origIdx,6),gg3=getSH(origIdx,7);
  vec4 b0=getSH(origIdx,8),b1=getSH(origIdx,9),b2=getSH(origIdx,10),b3=getSH(origIdx,11);
  vec3 dir=normalize(pos-uEye);
  vColor=vec4(evalSH3(dir,r0,r1,r2,r3,gg0,gg1,gg2,gg3,b0,b1,b2,b3),opacity);
  float max_scale=max(scale.x,max(scale.y,scale.z));
  vWeight=exp(min(max_scale/fwd,20.0));
}
`;

const SPLAT_FRAG = `#version 300 es
precision highp float;

in vec3 vConic;
in vec2 vCenterPix;
in vec4 vColor;
in float vWeight;

uniform float uAlphaScale;

out vec4 outAccum;  // rgb = color * alpha_w,  a = alpha_w (w_fg)

void main() {
  vec2 d = gl_FragCoord.xy - vCenterPix;
  float power = 0.5*(vConic.x*d.x*d.x + 2.0*vConic.y*d.x*d.y + vConic.z*d.y*d.y);
  if (power > 8.0) discard;
  float alpha_w = vColor.a * exp(-power) * uAlphaScale * vWeight;
  if (alpha_w < 0.00001) discard;
  outAccum = vec4(vColor.rgb * alpha_w, alpha_w);
}
`;

// ---------------------------------------------------------------------------
// Pass 2: compose accumulation buffer with background
// ---------------------------------------------------------------------------
const COMPOSE_VERT = `#version 300 es
void main() {
  // Full-screen triangle — no vertex buffer needed
  vec2 p[3]; p[0]=vec2(-1,-1); p[1]=vec2(3,-1); p[2]=vec2(-1,3);
  gl_Position = vec4(p[gl_VertexID], 0, 1);
}
`;

const COMPOSE_FRAG = `#version 300 es
precision highp float;
uniform highp sampler2D uAccum;
uniform vec3 uBgColor;
out vec4 outColor;
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uAccum, 0));
  vec4 accum = texture(uAccum, uv);
  float w_fg = accum.a;
  vec3 C_fg = (w_fg > 0.0001) ? accum.rgb / w_fg : vec3(0.0);
  float coverage = 1.0 - exp(-w_fg);
  outColor = vec4(mix(uBgColor, C_fg, coverage), 1.0);
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
      if (!gl) return null;
      // Need float renderable textures for accumulation FBO
      if (!gl.getExtension("EXT_color_buffer_float") &&
          !gl.getExtension("EXT_color_buffer_half_float")) {
        console.warn("WebGL2: no float color buffer support");
        return null;
      }
      return new WebGL2SplatRenderer(canvas, scene, gl);
    } catch (error) {
      console.warn("WebGL2 renderer unavailable", error);
      return null;
    }
  }

  constructor(canvas, scene, gl) {
    this.canvas  = canvas;
    this.scene   = scene;
    this.gl      = gl;
    this.renderOptions = { alphaScale: scene.render.alphaScale };
    this.shTexture     = null;
    this.instanceCount = 0;
    this._instanceBufferSize = 0;
    this._fboWidth  = 0;
    this._fboHeight = 0;

    // Determine best float format for the accumulation FBO
    this._floatFmt = gl.getExtension("EXT_color_buffer_float")
      ? gl.RGBA32F : gl.RGBA16F;
    this._floatType = this._floatFmt === gl.RGBA32F ? gl.FLOAT : gl.HALF_FLOAT;

    // Splat accumulation program
    this.splatProgram   = linkProgram(gl, SPLAT_VERT, SPLAT_FRAG);
    this.composeProgram = linkProgram(gl, COMPOSE_VERT, COMPOSE_FRAG);

    // Splat program uniform locations
    this.splatLoc = {
      viewProj:   gl.getUniformLocation(this.splatProgram, "uViewProj"),
      view:       gl.getUniformLocation(this.splatProgram, "uView"),
      focal:      gl.getUniformLocation(this.splatProgram, "uFocal"),
      width:      gl.getUniformLocation(this.splatProgram, "uWidth"),
      height:     gl.getUniformLocation(this.splatProgram, "uHeight"),
      eye:        gl.getUniformLocation(this.splatProgram, "uEye"),
      alphaScale: gl.getUniformLocation(this.splatProgram, "uAlphaScale"),
      shData:     null,  // set in initSHData
    };

    // Compose program uniform locations
    this.composeLoc = {
      accum:   gl.getUniformLocation(this.composeProgram, "uAccum"),
      bgColor: gl.getUniformLocation(this.composeProgram, "uBgColor"),
    };

    // Geometry vertex buffer + VAO
    this.instanceBuffer = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    const stride = GEOM_FLOATS * 4;
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 4, gl.FLOAT, false, stride, i * 16);
      gl.vertexAttribDivisor(i, 1);
    }
    gl.bindVertexArray(null);

    // Accumulation FBO (created on first resize)
    this.accumFBO     = gl.createFramebuffer();
    this.accumTexture = gl.createTexture();

    // Background colour
    this._bg = scene.render.backgroundTop.slice(0, 3);
  }

  get label() { return "webgl2"; }

  _ensureFBO(width, height) {
    if (width === this._fboWidth && height === this._fboHeight) return;
    const gl = this.gl;
    this._fboWidth  = width;
    this._fboHeight = height;
    gl.bindTexture(gl.TEXTURE_2D, this.accumTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, this._floatFmt, width, height, 0,
                  gl.RGBA, this._floatType, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, this.accumTexture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  resize(width, height) {
    this.canvas.width  = width;
    this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
    this._ensureFBO(width, height);
  }

  initSHData(shData) {
    const gl = this.gl;
    const N = shData.length / 48;
    const SH_TEX_WIDTH = 2048;
    const totalTexels  = N * 12;
    const height = Math.ceil(totalTexels / SH_TEX_WIDTH);
    const texData = new Float32Array(SH_TEX_WIDTH * height * 4);
    texData.set(shData);
    if (!this.shTexture) this.shTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.shTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, SH_TEX_WIDTH, height, 0,
                  gl.RGBA, gl.FLOAT, texData);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.splatLoc.shData = gl.getUniformLocation(this.splatProgram, "uSHData");
  }

  updateGeometryData(geomData) {
    this.instanceCount = geomData.length / GEOM_FLOATS;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    if (this._instanceBufferSize === geomData.byteLength) {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, geomData);
    } else {
      gl.bufferData(gl.ARRAY_BUFFER, geomData, gl.DYNAMIC_DRAW);
      this._instanceBufferSize = geomData.byteLength;
    }
  }

  updateSceneData(geomData) { return this.updateGeometryData(geomData); }

  setRenderOptions(options) { this.renderOptions = options; }

  render(cameraState) {
    const gl = this.gl;
    const W  = this.canvas.width;
    const H  = this.canvas.height;
    this._ensureFBO(W, H);

    // ── Pass 1: accumulate splats into float FBO ─────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.DEPTH_TEST);

    gl.useProgram(this.splatProgram);
    gl.bindVertexArray(this.vao);

    gl.uniformMatrix4fv(this.splatLoc.viewProj, false, cameraState.viewProjection);
    gl.uniformMatrix4fv(this.splatLoc.view,     false, cameraState.view);
    gl.uniform1f(this.splatLoc.focal,      cameraState.focal);
    gl.uniform1f(this.splatLoc.width,      W);
    gl.uniform1f(this.splatLoc.height,     H);
    gl.uniform3fv(this.splatLoc.eye,       cameraState.eye);
    gl.uniform1f(this.splatLoc.alphaScale, this.renderOptions.alphaScale);

    if (this.shTexture && this.splatLoc.shData !== null) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.shTexture);
      gl.uniform1i(this.splatLoc.shData, 1);
    }

    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.instanceCount);
    gl.bindVertexArray(null);

    // ── Pass 2: compose with background ──────────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);

    gl.useProgram(this.composeProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.accumTexture);
    gl.uniform1i(this.composeLoc.accum,   0);
    gl.uniform3fv(this.composeLoc.bgColor, this._bg);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
