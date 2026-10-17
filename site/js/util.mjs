// Pure helpers of the neural engine (no onnxruntime import, so they can be unit-tested in node).
export const SIZE = 64;
export const PIX = SIZE * SIZE;
export const N = 3 * PIX;
// Context frames (and actions) the model takes: 4 for v2. A live export: the page sets it from the model meta before anything is allocated.
export let K = 4;
export function setContextLength(k) { K = k; }

export function buildSigmas(steps, smin = 2e-3, smax = 5, rho = 7) {
  const a = Math.pow(smax, 1 / rho), b = Math.pow(smin, 1 / rho);
  const s = [];
  for (let i = 0; i < steps; i++) {
    const l = steps === 1 ? 0 : i / (steps - 1);
    s.push(Math.pow(a + l * (b - a), rho));
  }
  s.push(0);
  return s;
}

// mulberry32 + Box-Muller, so that capture runs are reproducible
export class Rng {
  constructor(seed = 1) { this.s = seed >>> 0; this.spare = null; }
  next() {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  randn() {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next(), r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
}

// helpers shared with the app
export function rgbToChw(rgb, out = new Float32Array(N)) {
  for (let p = 0; p < PIX; p++) for (let c = 0; c < 3; c++) out[c * PIX + p] = rgb[p * 3 + c] / 127.5 - 1;
  return out;
}

export function psnr(a, b) {
  let se = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; se += d * d; }
  const mse = se / a.length;
  return mse < 1e-9 ? 99 : 10 * Math.log10((255 * 255) / mse);
}

// Euler sampler of the EDM ODE, in place on x (sigma_0 noise). nfe(x, sigma) -> D(x; sigma) as a Float32Array.
export async function eulerSample(x, sigmas, nfe) {
  for (let i = 0; i < sigmas.length - 1; i++) {
    const sigma = sigmas[i];
    const D = await nfe(x, sigma);
    const dt = (sigmas[i + 1] - sigma) / sigma;
    for (let j = 0; j < x.length; j++) x[j] += (x[j] - D[j]) * dt;
  }
  return x;
}

// Clamp to [-1, 1], round to the 8-bit grid; writes CHW floats (the next context frame) and HWC bytes (the canvas).
export function quantiseFrame(x, chwOut, rgbOut) {
  for (let c = 0; c < 3; c++) {
    for (let p = 0; p < PIX; p++) {
      const v = Math.min(1, Math.max(-1, x[c * PIX + p]));
      const b = Math.floor((v + 1) * 127.5 + 0.5);
      chwOut[c * PIX + p] = b / 127.5 - 1;
      rgbOut[p * 3 + c] = b;
    }
  }
}

// Share of pixels whose colour is the floor/ceiling colour of their row (within `tol` grey levels, max over the channels): the
// per-frame input of the flat-wall metric (model/flatwall.py). rgb: HWC bytes [H*W*3]; bgRows: the game's row colours [H*3] (World.bg).
export function backgroundFraction(rgb, bgRows, tol = 4) {
  let n = 0;
  for (let y = 0; y < SIZE; y++) {
    const r = bgRows[y * 3], g = bgRows[y * 3 + 1], b = bgRows[y * 3 + 2];
    for (let x = 0; x < SIZE; x++) {
      const o = (y * SIZE + x) * 3;
      if (Math.abs(rgb[o] - r) <= tol && Math.abs(rgb[o + 1] - g) <= tol && Math.abs(rgb[o + 2] - b) <= tol) n++;
    }
  }
  return n / PIX;
}
