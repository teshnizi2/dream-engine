// Where does one dream frame go? Per-frame time split into JS preparation, the model calls (upload + run + readback) and
// JS post-processing, for 1, 2 and 3 denoising steps (60 frames each, back-to-back, headless Edge + WebGPU).
//   node tools/browser_profile.mjs --out results/browser_profile.json
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => { if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return acc; }, []));
const srv = await startServer(path.resolve(here, '..'), 0);
const browser = await puppeteer.launch({ executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: 'new', args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'] });
const out = { script: 'tools/browser_profile.mjs', date: new Date().toISOString(), frames_per_setting: 60, settings: {} };
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${srv.port}/site/index.html?capture=1${args.dtype ? '&dtype=' + args.dtype : ''}`);
  await page.waitForFunction('window.__captureReady === true', { timeout: 180000 });
  out.info = await page.evaluate(() => window.__capture.info());
  for (const steps of [1, 2, 3]) {
    await page.evaluate((s) => window.__capture.steps(s), steps);
    await page.evaluate(async () => { for (let i = 0; i < 6; i++) await window.__capture.tick([1, 0, 0]); window.__capture.profile(); });
    await page.evaluate(async () => { for (let i = 0; i < 60; i++) await window.__capture.tick([1, 0, 0]); });
    out.settings[steps] = await page.evaluate(() => window.__capture.profile());
    console.log(steps, JSON.stringify(out.settings[steps]));
  }
} finally { await browser.close(); srv.close(); }
if (args.out) fs.writeFileSync(path.resolve(args.out), JSON.stringify(out, null, 1));
