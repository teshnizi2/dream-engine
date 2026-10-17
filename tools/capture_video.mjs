// Deterministic capture of the demo page: replays the scripted tour on both panels, one game tick per video frame,
// screenshots the 1080x1350 poster layout and encodes an H.264 mp4 with ffmpeg.
//   node tools/capture_video.mjs --out media/dream-1080x1350.mp4 [--fps 24] [--max 480] [--steps 3] [--seed 1234]
//        [--ms 45 --fps-label 22] [--wake-every 0 [--wake-until TICK]] [--keep-frames] [--still media/hero.png --still-at 300]
//        [--notes "0=caption one|336=caption two"]   (caption changes at the given ticks; **bold** is allowed)
import path from 'node:path';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { startServer, rootFromArgs } from './serve.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = rootFromArgs(path.resolve(here, '..'));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => { if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return acc; }, []));
const FPS = +(args.fps || 24);
const tour = JSON.parse(fs.readFileSync(path.join(root, 'site/js/tour.json'), 'utf8')).actions;
const N = Math.min(tour.length, +(args.max || tour.length));
const SQUARE = !!args.square;
const W = +(args.width || 1080), H = +(args.height || (SQUARE ? 1080 : 1350));
const framesDir = path.join(root, 'media/frames');
fs.rmSync(framesDir, { recursive: true, force: true });
fs.mkdirSync(framesDir, { recursive: true });

const srv = await startServer(root, 0);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist', '--hide-scrollbars', '--force-color-profile=srgb'],
});
const t0 = Date.now();
try {
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  const q = new URLSearchParams({ capture: '1', steps: args.steps || '3', seed: args.seed || '1234', ...(args.dtype ? { dtype: args.dtype } : {}),
    ...(args.ms ? { ms: args.ms } : {}), ...(args.note ? { note: args.note } : {}), ...(SQUARE ? { square: '1' } : {}), ...(args['fps-label'] ? { fps: args['fps-label'] } : {}) });
  page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 200)));
  await page.goto(`http://127.0.0.1:${srv.port}/site/index.html?${q}`);
  await page.waitForFunction('window.__captureReady === true', { timeout: 180000 });
  await page.evaluate(() => document.fonts.ready);
  const info = await page.evaluate(() => window.__capture.info());
  console.log('model/backend:', JSON.stringify(info));
  const stills = (args['still-at'] || '').split(',').filter(Boolean).map(Number);
  const wakeEvery = +(args['wake-every'] || 0), wakeUntil = +(args['wake-until'] || Infinity);
  const notes = new Map((args.notes || '').split('|').filter(Boolean).map((s) => { const i = s.indexOf('='); return [+s.slice(0, i), s.slice(i + 1)]; }));
  const psnrs = [], bgDream = [], bgReal = [];
  for (let i = 0; i < N; i++) {
    if (notes.has(i)) await page.evaluate((t) => window.__capture.setNote(t), notes.get(i));
    if (wakeEvery && i > 0 && i % wakeEvery === 0 && i < wakeUntil) await page.evaluate(() => window.__capture.wake());
    const r = await page.evaluate((a) => window.__capture.tick(a), tour[i]);
    psnrs.push(r.psnr); bgDream.push(r.bgDream); bgReal.push(r.bgReal);
    const file = path.join(framesDir, `frame_${String(i).padStart(5, '0')}.png`);
    await page.screenshot({ path: file, type: 'png' });
    if (stills.includes(i) && args.still) fs.copyFileSync(file, path.resolve(args.still.replace(/(\.png)?$/, `_${i}.png`)));
    if (i % 60 === 0) console.log(`tick ${i}/${N}  psnr ${r.psnr.toFixed(1)}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  fs.mkdirSync(path.join(root, 'results'), { recursive: true });
  fs.writeFileSync(path.join(root, 'results/video_capture_psnr.json'), JSON.stringify({ script: 'tools/capture_video.mjs', ticks: N, steps: +(args.steps || 3), seed: +(args.seed || 1234), psnr_per_tick: psnrs, bg_dream_per_tick: bgDream, bg_real_per_tick: bgReal, wake_every: wakeEvery, model: info }, null, 0));
} finally { await browser.close(); srv.close(); }
if (args.out) {
  const out = path.resolve(args.out);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(framesDir, 'frame_%05d.png'),
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: 'inherit' });
  console.log('wrote', out, (fs.statSync(out).size / 1e6).toFixed(1), 'MB');
}
if (!args['keep-frames']) fs.rmSync(framesDir, { recursive: true, force: true });
