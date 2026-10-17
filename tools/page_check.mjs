// Load the demo in headless Microsoft Edge (WebGPU on), optionally hold keys, screenshot, print readouts + console errors.
//   node tools/page_check.mjs --path "site/index.html?steps=3" --out /tmp/shot.png [--width 1280 --height 900 --hold KeyW:2500 --wait 2000 --mobile]
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']);
  return acc;
}, []));
const root = path.resolve(here, '..');
const srv = await startServer(root, 0);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: args.headed ? false : 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist', '--no-first-run', '--no-default-browser-check'],
});
const logs = [];
try {
  const page = await browser.newPage();
  const w = +(args.width || 1280), h = +(args.height || 900);
  await page.setViewport({ width: w, height: h, deviceScaleFactor: +(args.dpr || 1), isMobile: !!args.mobile, hasTouch: !!args.mobile });
  page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) logs.push(`[${m.type()}] ${m.text()}`.slice(0, 300)); });
  page.on('pageerror', (e) => logs.push('[pageerror] ' + String(e).slice(0, 300)));
  await page.goto(`http://127.0.0.1:${srv.port}/${args.path || 'site/index.html'}`);
  await page.waitForFunction(() => !document.body.classList.contains('loading'), { timeout: +(args.timeout || 120000) });
  await new Promise((r) => setTimeout(r, +(args.wait || 1500)));
  for (const hold of String(args.hold || '').split(',').filter(Boolean)) {
    const [code, ms] = hold.split(':');
    await page.keyboard.down(code);
    await new Promise((r) => setTimeout(r, +ms));
    await page.keyboard.up(code);
  }
  if (args.click) { await page.click(args.click); await new Promise((r) => setTimeout(r, 400)); }
  await new Promise((r) => setTimeout(r, +(args.after || 300)));
  const state = await page.evaluate(() => Object.fromEntries([...document.querySelectorAll('.deck .cell')].map((c) => [c.querySelector('.lab').textContent, c.querySelector('.val').textContent])));
  console.log(JSON.stringify(state));
  if (args.out) await page.screenshot({ path: args.out, fullPage: !!args.full });
  console.log('console issues:', logs.length ? '\n' + logs.join('\n') : 'none');
} finally {
  await browser.close();
  srv.close();
}
