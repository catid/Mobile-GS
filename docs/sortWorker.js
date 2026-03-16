/**
 * Web Worker: depth-sorts Gaussian splats off the main thread.
 *
 * Protocol
 * --------
 * main → worker  { type:'init',   splats: Float32Array }   (transferable)
 * main → worker  { type:'sort',   eye: [x,y,z], forward: [x,y,z] }
 * worker → main  { type:'sorted', buffer: ArrayBuffer }    (transferable)
 * main → worker  { type:'returnBuffer', buffer: ArrayBuffer } (transferable)
 *
 * The worker owns the splat buffer between sort calls.
 */

const FLOATS_PER_SPLAT = 12;

let splats     = null;  // Float32Array — original unsorted splat data
let sortedBuf  = null;  // Float32Array — output (ping-pong with main thread)
let sortIdxA   = null;  // Int32Array   — sort index buffer A
let sortIdxB   = null;  // Int32Array   — sort index buffer B (radix temp)
let fBits      = null;  // Int32Array   — bit-cast float depths
let depths     = null;  // Float32Array — per-splat depths

self.onmessage = function (e) {
  const msg = e.data;

  if (msg.type === "init") {
    splats = msg.splats;
    const n = splats.length / FLOATS_PER_SPLAT;
    sortedBuf = new Float32Array(splats.length);
    sortIdxA  = new Int32Array(n);
    sortIdxB  = new Int32Array(n);
    fBits     = new Int32Array(n);
    depths    = new Float32Array(n);
    for (let i = 0; i < n; i++) sortIdxA[i] = i;
    self.postMessage({ type: "ready", count: n });
    return;
  }

  if (msg.type === "sort") {
    if (!sortedBuf) {
      // Buffer still in transit back from main thread — skip this request.
      // Main thread will see sortPending=false and sortNeeded=true, so it
      // will call requestSort() again on the next frame.
      self.postMessage({ type: "skipped" });
      return;
    }

    const { eye, forward } = msg;
    const n = splats.length / FLOATS_PER_SPLAT;

    // 1. Compute projected depth for every splat
    for (let i = 0; i < n; i++) {
      const b = i * FLOATS_PER_SPLAT;
      depths[i] =
        (splats[b]     - eye[0]) * forward[0] +
        (splats[b + 1] - eye[1]) * forward[1] +
        (splats[b + 2] - eye[2]) * forward[2];
    }

    // 2. Radix sort indices by depth descending (back-to-front)
    radixSortDesc(n);

    // 3. Scatter splat data into output buffer
    const idx = sortIdxA;
    for (let w = 0; w < n; w++) {
      const src = idx[w] * FLOATS_PER_SPLAT;
      const dst = w     * FLOATS_PER_SPLAT;
      sortedBuf.set(splats.subarray(src, src + FLOATS_PER_SPLAT), dst);
    }

    // 4. Transfer output to main thread (zero-copy)
    const outBuf = sortedBuf.buffer;
    sortedBuf = null;
    self.postMessage({ type: "sorted", buffer: outBuf }, [outBuf]);
    return;
  }

  if (msg.type === "returnBuffer") {
    // Main thread finished uploading; reclaim for next sort
    sortedBuf = new Float32Array(msg.buffer);
    return;
  }
};

// ---------------------------------------------------------------------------
// Radix sort — 4 × 8-bit passes, descending on float depth
// Uses a reusable DataView over a single 4-byte scratch ArrayBuffer.
// ---------------------------------------------------------------------------
const _scratch  = new ArrayBuffer(4);
const _f32view  = new Float32Array(_scratch);
const _i32view  = new Int32Array(_scratch);

function radixSortDesc(n) {
  const BUCKETS = 256;
  const hist    = new Int32Array(BUCKETS);

  // Bit-cast float depths to sortable int32 (IEEE 754 trick)
  for (let i = 0; i < n; i++) {
    _f32view[0] = depths[i];
    let bits = _i32view[0];
    // If sign bit is set: flip all bits; else flip only sign bit.
    fBits[i] = bits < 0 ? (~bits | 0) : (bits ^ 0x80000000);
  }

  let src = sortIdxA;
  let dst = sortIdxB;

  for (let pass = 0; pass < 4; pass++) {
    const shift = pass * 8;

    // Count frequencies
    hist.fill(0);
    for (let i = 0; i < n; i++) {
      hist[(fBits[src[i]] >>> shift) & 0xff]++;
    }

    // Prefix sum descending (bucket 255 → 0 so output is back-to-front)
    let total = 0;
    for (let b = BUCKETS - 1; b >= 0; b--) {
      const h = hist[b]; hist[b] = total; total += h;
    }

    // Scatter
    for (let i = 0; i < n; i++) {
      const b = (fBits[src[i]] >>> shift) & 0xff;
      dst[hist[b]++] = src[i];
    }

    // Swap src/dst
    const tmp = src; src = dst; dst = tmp;
  }

  // After 4 passes the result is always in sortIdxA (even pass count)
  if (src !== sortIdxA) {
    sortIdxA.set(src);
  }
}
