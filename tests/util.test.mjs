// Unit tests for the pure helpers of the browser engine (site/js/util.mjs): the EDM sigma schedule must equal the
// Python one used in training, the RNG must be reproducible and standard normal, PSNR/format helpers must be exact.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildSigmas, Rng, rgbToChw, psnr, N, PIX } from '../site/js/util.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(fs.readFileSync(path.join(here, 'fixtures/parity.json'), 'utf8'));

test('sigma schedule equals the python build_sigmas (rho 7, 2e-3 .. 5) for 1, 2 and 3 steps', () => {
  for (const steps of [1, 2, 3]) {
    const js = buildSigmas(steps), py = fx.sigmas[String(steps)];
    assert.equal(js.length, py.length);
    js.forEach((v, i) => assert.ok(Math.abs(v - py[i]) <= 1e-6 * Math.max(1, Math.abs(py[i])), `steps ${steps} idx ${i}: ${v} vs ${py[i]}`));
    assert.equal(js[js.length - 1], 0);
  }
});

test('rng is reproducible and standard normal', () => {
  const a = new Rng(7), b = new Rng(7);
  for (let i = 0; i < 50; i++) assert.equal(a.randn(), b.randn());
  const r = new Rng(1);
  let s = 0, s2 = 0;
  const n = 200000;
  for (let i = 0; i < n; i++) { const v = r.randn(); s += v; s2 += v * v; }
  assert.ok(Math.abs(s / n) < 0.01, `mean ${s / n}`);
  assert.ok(Math.abs(s2 / n - 1) < 0.02, `var ${s2 / n}`);
});

test('rgbToChw maps HWC bytes to CHW floats in [-1, 1]', () => {
  const rgb = new Uint8Array(N);
  rgb[0] = 0; rgb[1] = 255; rgb[2] = 51;                  // first pixel
  const chw = rgbToChw(rgb);
  assert.equal(chw[0], -1); assert.equal(chw[PIX], 1); assert.ok(Math.abs(chw[2 * PIX] - (51 / 127.5 - 1)) < 1e-7);
});

test('psnr: identical frames cap at 99 dB, a constant offset of 10 levels gives 20 log10(255/10)', () => {
  const a = new Uint8Array(N).fill(100), b = new Uint8Array(N).fill(110);
  assert.equal(psnr(a, a), 99);
  assert.ok(Math.abs(psnr(a, b) - 20 * Math.log10(25.5)) < 1e-9);
});

import { eulerSample, quantiseFrame } from '../site/js/util.mjs';

test('euler sampler returns the clean frame for a perfect denoiser, for 1, 2 and 3 steps', async () => {
  const x0 = Float32Array.from({ length: N }, (_, i) => Math.sin(i * 0.37));
  for (const steps of [1, 2, 3]) {
    const sg = buildSigmas(steps);
    const rng = new Rng(3);
    const x = Float32Array.from({ length: N }, () => rng.randn() * sg[0]);
    let calls = 0;
    await eulerSample(x, sg, async () => { calls++; return x0; });
    assert.equal(calls, steps);
    let maxd = 0;
    for (let i = 0; i < N; i++) maxd = Math.max(maxd, Math.abs(x[i] - x0[i]));
    assert.ok(maxd < 1e-5, `steps ${steps}: max diff ${maxd}`);
  }
});

test('quantiseFrame clamps to [-1, 1], rounds to the 8-bit grid and writes both CHW floats and HWC bytes', () => {
  const x = new Float32Array(N);
  x[0] = 1.7; x[1] = -3; x[2] = 0.0; x[PIX] = 0.5;            // red of pixels 0,1,2 and green of pixel 0
  const chw = new Float32Array(N), rgb = new Uint8Array(N);
  quantiseFrame(x, chw, rgb);
  assert.equal(rgb[0], 255); assert.equal(rgb[3], 0); assert.equal(rgb[6], Math.floor(127.5 + 0.5));
  assert.equal(rgb[1], Math.floor((0.5 + 1) * 127.5 + 0.5));      // green of pixel 0 (HWC order)
  assert.equal(chw[0], 1); assert.equal(chw[1], -1);
  assert.ok(Math.abs(chw[PIX] - (rgb[1] / 127.5 - 1)) < 1e-7);
});
