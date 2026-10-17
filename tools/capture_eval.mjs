// Runs window.__capture.oneStepEval in the page (headless Edge + WebGPU) and prints the JSON.
//   node tools/capture_eval.mjs --n 96 --steps 3 [--dtype fp16|fp32] [--backend wasm]
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer, rootFromArgs } from './serve.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => { if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return acc; }, []));
const srv = await startServer(rootFromArgs(path.resolve(here, '..')), 0);
const browser = await puppeteer.launch({ executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: 'new', args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'] });
try {
  const page = await browser.newPage();
  const q = new URLSearchParams({ capture: '1', steps: args.steps || '3', ...(args.dtype ? { dtype: args.dtype } : {}), ...(args.backend ? { backend: args.backend } : {}) });
  await page.goto(`http://127.0.0.1:${srv.port}/site/index.html?${q}`);
  await page.waitForFunction('window.__captureReady === true', { timeout: 180000 });
  let r;
  if (args.windows) { const wins = JSON.parse(fs.readFileSync(args.windows, 'utf8')); r = await page.evaluate(async (w) => window.__capture.oneStepFromWindows(w), wins); }
  else r = await page.evaluate(async (n) => window.__capture.oneStepEval(n, 7), +(args.n || 96));
  console.log(JSON.stringify(r));
} finally { await browser.close(); srv.close(); }
