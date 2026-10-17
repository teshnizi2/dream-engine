// Measures what a user sees: loads the real page in headless Edge (WebGPU), lets it run the scripted autoplay tour
// for each sampler setting and reads the live readouts (ms/frame, fps) plus an independent frame counter.
// The GPU of this Mac is shared with other jobs, so the system-wide GPU utilisation is recorded before and after.
//   node tools/browser_fps.mjs --out results/browser_fps.json [--dtype fp16|fp32] [--seconds 8]
import path from 'node:path';
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => { if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return acc; }, []));
const gpuUtil = () => { try { const o = execSync('ioreg -r -d 1 -w0 -c AGXAccelerator', { encoding: 'utf8' }); const m = o.match(/"Device Utilization %"=(\d+)/); return m ? +m[1] : null; } catch (e) { return null; } };
const srv = await startServer(root, 0);
const browser = await puppeteer.launch({ executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: 'new', args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'] });
const result = { script: 'tools/browser_fps.mjs', date: new Date().toISOString(), seconds_per_setting: +(args.seconds || 8), runs: [] };
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 860 });
  const q = new URLSearchParams({ ...(args.dtype ? { dtype: args.dtype } : {}), ...(args.backend ? { backend: args.backend } : {}) });
  await page.goto(`http://127.0.0.1:${srv.port}/site/index.html?${q}`);
  await page.waitForFunction(() => !document.body.classList.contains('loading'), { timeout: 180000 });
  result.userAgent = await page.evaluate(() => navigator.userAgent);
  result.backend = await page.evaluate(() => document.getElementById('r-backend').textContent);
  result.params = await page.evaluate(() => document.getElementById('r-params').textContent);
  await page.click('#btn-auto');
  // Other jobs share this GPU. ioreg's utilisation includes our own page, so pause the page, wait a moment and sample:
  // that reading is "everyone else". Wait (up to maxMs) for it to be <= 30% three times in a row before a window.
  const othersUtil = async () => {
    await page.click('#btn-pause'); await new Promise((r) => setTimeout(r, 1200));
    const xs = []; for (let i = 0; i < 3; i++) { xs.push(gpuUtil()); await new Promise((r) => setTimeout(r, 400)); }
    await page.click('#btn-pause');
    return Math.round(xs.reduce((u, v) => u + (v ?? 0), 0) / xs.length);
  };
  const waitQuiet = async (maxMs) => {
    const t = Date.now(); let last = null;
    while (Date.now() - t < maxMs) { last = await othersUtil(); if (last <= 30) return { quiet: true, others: last }; await new Promise((r) => setTimeout(r, 2000)); }
    return { quiet: false, others: last };
  };
  const repeats = +(args.repeats || 3);
  for (const steps of [1, 2, 3, ...(repeats > 1 ? [1, 2, 3].flatMap((s) => Array(repeats - 1).fill(s)) : [])]) {
    await page.click(`.seg-b[data-steps="${steps}"]`);
    await new Promise((r) => setTimeout(r, 1500));
    const q = args['no-wait'] ? { quiet: null, others: await othersUtil() } : await waitQuiet(+(args['quiet-wait'] || 60000));
    await new Promise((r) => setTimeout(r, 1000));
    const g0 = gpuUtil();
    const before = await page.evaluate(() => window.__ticks || 0);
    const t0 = Date.now();
    const samples = [];
    while (Date.now() - t0 < result.seconds_per_setting * 1000) {
      await new Promise((r) => setTimeout(r, 500));
      samples.push(await page.evaluate(() => ({ ms: parseFloat(document.getElementById('r-ms').textContent), fps: parseFloat(document.getElementById('r-fps').textContent) })));
    }
    const after = await page.evaluate(() => window.__ticks || 0);
    const dt = (Date.now() - t0) / 1000;
    const g1 = gpuUtil();
    const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
    result.runs.push({ steps, ms_per_frame_median: med(samples.map((s) => s.ms)), fps_readout_median: med(samples.map((s) => s.fps)),
      fps_counted: (after - before) / dt, gpu_util_during_start_pct: g0, gpu_util_during_end_pct: g1, gpu_util_others_before_pct: q.others, quiet_gpu_at_start: q.quiet });
    console.log(JSON.stringify(result.runs[result.runs.length - 1]));
  }
  // summary per setting: median of the windows, and the best (quietest) window
  const med2 = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
  result.summary = {};
  for (const st of [1, 2, 3]) { const rs = result.runs.filter((r) => r.steps === st); result.summary[st] = { windows: rs.length, ms_per_frame_median: med2(rs.map((r) => r.ms_per_frame_median)), ms_per_frame_best: Math.min(...rs.map((r) => r.ms_per_frame_median)), fps_counted_median: med2(rs.map((r) => r.fps_counted)), fps_counted_best: Math.max(...rs.map((r) => r.fps_counted)) }; }
} finally { await browser.close(); srv.close(); }
if (args.out) { fs.writeFileSync(path.resolve(args.out), JSON.stringify(result, null, 1)); console.log('wrote', args.out); }
