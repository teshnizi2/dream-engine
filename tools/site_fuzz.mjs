// Random interaction fuzz of the demo page (keys, buttons, resizes, drawer, steps) for N seconds; fails on any page error.
//   node tools/site_fuzz.mjs [--seconds 25] [--seed 1]
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer, rootFromArgs } from './serve.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => { if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return acc; }, []));
let seed = +(args.seed || 1);
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const pick = (a) => a[Math.floor(rnd() * a.length)];
const srv = await startServer(rootFromArgs(path.resolve(here, '..')), 0);
const browser = await puppeteer.launch({ executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: 'new', args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'] });
const errors = [];
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 860 });
  page.on('pageerror', (e) => errors.push('pageerror: ' + String(e).slice(0, 200)));
  page.on('console', (m) => { if (m.type() === 'error' && !/404/.test(m.text())) errors.push('console: ' + m.text().slice(0, 200)); });
  await page.goto(`http://127.0.0.1:${srv.port}/site/index.html`);
  // interact already while the model is loading
  await page.keyboard.down('KeyW'); await page.click('#btn-about'); await page.keyboard.press('Escape'); await page.keyboard.up('KeyW');
  await page.waitForFunction(() => !document.body.classList.contains('loading'), { timeout: 180000 });
  const keys = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowLeft', 'ArrowDown', 'ArrowRight'];
  const buttons = ['#btn-wake', '#btn-dreamonly', '#btn-scan', '#btn-pause', '#btn-auto', '#btn-autowake', '.seg-b[data-steps="1"]', '.seg-b[data-steps="2"]', '.seg-b[data-steps="3"]', '#btn-about'];
  const sizes = [[1280, 860], [900, 700], [375, 812], [760, 600], [1600, 900], [320, 640]];
  const t0 = Date.now();
  let n = 0, down = new Set();
  while (Date.now() - t0 < 1000 * +(args.seconds || 25)) {
    const r = rnd();
    try {
      if (r < 0.35) { const k = pick(keys); if (down.has(k)) { await page.keyboard.up(k); down.delete(k); } else { await page.keyboard.down(k); down.add(k); } }
      else if (r < 0.75) { const b = pick(buttons); const el = await page.$(b); if (el && await el.isIntersectingViewport()) await el.click(); }
      else if (r < 0.85) { const [w, h] = pick(sizes); await page.setViewport({ width: w, height: h }); }
      else if (r < 0.9) { await page.keyboard.press(pick(['KeyR', 'KeyT', 'KeyP', 'Digit1', 'Digit2', 'Digit3', 'Escape'])); }
      else { await page.evaluate(() => window.scrollTo(0, Math.random() * 400)); }
    } catch (e) { /* element detached during resize etc. */ }
    n++;
    await new Promise((r2) => setTimeout(r2, 40 + rnd() * 160));
  }
  for (const k of down) await page.keyboard.up(k);
  const ticks = await page.evaluate(() => window.__ticks || 0);
  console.log(`${n} random interactions, ${ticks} simulation ticks, ${errors.length} errors`);
  if (errors.length) console.log(errors.slice(0, 5).join('\n'));
  if (!ticks) errors.push('no ticks');
} finally { await browser.close(); srv.close(); }
process.exit(errors.length ? 1 : 0);
