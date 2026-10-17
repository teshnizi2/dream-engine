// Runs tools/bench.html in headless Microsoft Edge with WebGPU and prints the result JSON.
//   node tools/bench_run.mjs --model /path/to/model.onnx [--ep webgpu|wasm] [--layout NHWC|NCHW] [--n 40] [--headed]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']);
  return acc;
}, []));
const modelAbs = path.resolve(args.model || 'tools/bench_models/benchA_fp32.onnx');
// serve a temp root containing tools/ and the model
const root = path.resolve(here, '..');
const srv = await startServer(root, 0);
const rel = path.relative(root, modelAbs);
let modelUrl;
if (!rel.startsWith('..')) modelUrl = '/' + rel;
else {
  // copy outside-the-project models into a gitignored scratch folder inside the served root
  fs.mkdirSync(path.join(root, 'tools/bench_models'), { recursive: true });
  const dst = path.join(root, 'tools/bench_models', path.basename(modelAbs));
  fs.copyFileSync(modelAbs, dst);
  modelUrl = '/tools/bench_models/' + path.basename(modelAbs);
}
const EDGE = '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge';
const browser = await puppeteer.launch({
  executablePath: EDGE, headless: args.headed ? false : 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features',
         '--no-first-run', '--no-default-browser-check'],
});
try {
  const page = await browser.newPage();
  const prof = [];
  page.on('console', (m) => {
    const t = m.text();
    if (args.profile && t.includes('[profiling]')) prof.push(t);
    else if (args.verbose) console.log('[page]', t);
  });
  const qs = new URLSearchParams({ models: args.models || '', model: modelUrl, ep: args.ep || 'webgpu', layout: args.layout || 'NHWC', n: args.n || '40',
    opt: args.opt || 'all', threads: args.threads || '1', ...(args.profile ? { profile: '1' } : {}) });
  await page.goto(`http://127.0.0.1:${srv.port}/tools/${args.page || 'bench.html'}?${qs}`);
  await page.waitForFunction('window.__bench', { timeout: +(args.timeout || 300000) });
  const r = await page.evaluate(() => window.__bench);
  if (args.profile) {
    // aggregate GPU kernel times per op type: lines look like  [profiling] kernel "id|Op|name" ... execution time: N ns
    const agg = {}; let total = 0, n = 0;
    for (const t of prof) {
      const m = t.match(/kernel "[^|]*\|([^|]+)\|/); const e = t.match(/execution time: (\d+) ns/);
      if (m && e) { agg[m[1]] = agg[m[1]] || { ms: 0, n: 0 }; agg[m[1]].ms += +e[1] / 1e6; agg[m[1]].n++; total += +e[1] / 1e6; n++; }
    }
    r.profile = { kernels_logged: n, total_ms_logged: total, by_op: Object.fromEntries(Object.entries(agg).sort((a, b) => b[1].ms - a[1].ms).slice(0, 14)) };
    r.profile_sample = prof.slice(0, 2);
  }
  console.log(JSON.stringify(r, null, 1));
} finally {
  await browser.close();
  srv.close();
}
