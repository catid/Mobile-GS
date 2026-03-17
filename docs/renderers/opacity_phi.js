export function splitOpacityPhiWeights(flatWeights, metadata) {
  if (!metadata?.hiddenDims || metadata.hiddenDims.length !== 3) {
    throw new Error("opacityPhi.hiddenDims must contain exactly three layer sizes");
  }

  const inputDim = metadata.inputDim;
  const [hidden0, hidden1, hidden2] = metadata.hiddenDims;
  let offset = 0;

  function take(count) {
    const slice = flatWeights.subarray(offset, offset + count);
    if (slice.length !== count) {
      throw new Error(`opacity_phi weight blob truncated at offset ${offset}`);
    }
    offset += count;
    return slice;
  }

  const layout = {
    inputDim,
    hiddenDims: [hidden0, hidden1, hidden2],
    w0: take(hidden0 * inputDim),
    b0: take(hidden0),
    w1: take(hidden1 * hidden0),
    b1: take(hidden1),
    w2: take(hidden2 * hidden1),
    b2: take(hidden2),
    wPhi: take(hidden2),
    bPhi: take(1),
    wOpacity: take(hidden2),
    bOpacity: take(1),
  };

  if (offset !== flatWeights.length) {
    throw new Error(`opacity_phi weight blob has ${flatWeights.length - offset} trailing values`);
  }

  return layout;
}

export function packRowsToRGBA(weights, rows, cols) {
  const texWidth = Math.ceil(cols / 4);
  const packed = new Float32Array(texWidth * rows * 4);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const packedIndex = row * texWidth * 4 + Math.floor(col / 4) * 4 + (col % 4);
      packed[packedIndex] = weights[row * cols + col];
    }
  }
  return {
    texWidth,
    packed,
  };
}
