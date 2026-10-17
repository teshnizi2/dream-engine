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
