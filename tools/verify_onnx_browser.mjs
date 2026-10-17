// Runs the PyTorch test vectors (site/models/testvec.*) through the exported ONNX models in headless Edge:
// WebGPU fp32/fp16 (with and without graph capture) and WASM fp32; writes results/onnx_browser_verification.json.
//   node tools/verify_onnx_browser.mjs
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer, rootFromArgs } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = rootFromArgs(path.resolve(here, '..'));
const srv = await startServer(root, 0);
const browser = await puppeteer.launch({ executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: 'new', args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'] });
const configs = [
  { model: 'dream_fp32', ep: 'webgpu', capture: false }, { model: 'dream_fp32', ep: 'webgpu', capture: true },
  { model: 'dream_fp16', ep: 'webgpu', capture: false }, { model: 'dream_fp16', ep: 'webgpu', capture: true },
  { model: 'dream_fp32', ep: 'wasm', capture: false },
];
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${srv.port}/tools/selftest.html?configs=${encodeURIComponent(JSON.stringify(configs))}`);
  await page.waitForFunction('window.__bench', { timeout: 600000 });
  const r = await page.evaluate(() => window.__bench);
  const out = { script: 'tools/verify_onnx_browser.mjs', date: new Date().toISOString(), adapter: await page.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? { description: a.info?.description, features: [...a.features].filter((f) => f.includes('f16')) } : null; }), ...r };
  fs.mkdirSync(path.join(root, 'results'), { recursive: true });
  fs.writeFileSync(path.join(root, 'results/onnx_browser_verification.json'), JSON.stringify(out, null, 1));
  for (const c of r.cases || []) console.log(c.model, c.ep, c.capture ? 'captured' : '', c.error ? 'ERROR ' + c.error.slice(0, 100) : `max abs diff vs PyTorch ${c.max_abs_diff.toExponential(2)}`);
} finally { await browser.close(); srv.close(); }
