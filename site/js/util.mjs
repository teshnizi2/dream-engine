// Pure helpers of the neural engine (no onnxruntime import, so they can be unit-tested in node).
export const SIZE = 64;
export const PIX = SIZE * SIZE;
export const N = 3 * PIX;
export const K = 4;

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
