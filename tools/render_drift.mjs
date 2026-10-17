// Renders media/drift.png (1080x1350) from the per-frame curves that eval.py saves in results/eval_<name>.json.
//   node tools/render_drift.mjs --eval results/eval_full1_local.json [--out media/drift.png] [--scale 1]
// The figure page (tools/drift_figure.html) uses the site's fonts and tokens; every number on it is read from the json.
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer, rootFromArgs } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = rootFromArgs(path.resolve(here, '..'));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => { if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return acc; }, []));
if (!args.eval) { console.error('usage: node tools/render_drift.mjs --eval results/eval_<name>.json [--out media/drift.png] [--scale 1]'); process.exit(2); }
const rel = path.relative(root, path.resolve(args.eval));
const out = path.resolve(args.out || path.join(root, 'media/drift.png'));
fs.mkdirSync(path.dirname(out), { recursive: true });

const srv = await startServer(root, 0);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: 'new',
  args: ['--hide-scrollbars', '--force-color-profile=srgb'],
});
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)));
  await page.setViewport({ width: 1080, height: 1350, deviceScaleFactor: +(args.scale || 1) });
  await page.goto(`http://127.0.0.1:${srv.port}/tools/drift_figure.html?eval=${encodeURIComponent(rel)}`);
  await page.waitForFunction('window.__figReady === true', { timeout: 60000 });
  await page.screenshot({ path: out, type: 'png' });
  console.log('wrote', out, (fs.statSync(out).size / 1e3).toFixed(0), 'kB');
} finally { await browser.close(); srv.close(); }
