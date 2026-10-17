// Dream Engine: REAL (JS raycaster) next to DREAM (diffusion model in onnxruntime-web), same keypresses.
import { World, stateFromSpecStart } from './game.mjs';
import { Dream, Rng, rgbToChw, psnr, N, PIX, K } from './dream.mjs';

const $ = (id) => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const CAPTURE = qs.get('capture') === '1';
const TICK_HZ = 30;                               // game steps per second at most (lockstep with the dream)
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const body = document.body;

const cvReal = $('cv-real'), cvDream = $('cv-dream'), cvDrift = $('cv-drift');
const ctxReal = cvReal.getContext('2d'), ctxDream = cvDream.getContext('2d'), ctxDrift = cvDrift.getContext('2d');
const imgReal = ctxReal.createImageData(64, 64), imgDream = ctxDream.createImageData(64, 64);
const cvMap = $('cv-map'), ctxMap = cvMap.getContext('2d');

let spec, world, real, dream, meta = {};
const realRGB = new Uint8Array(N);
const frameHist = [];                              // last K real frames (CHW floats), newest last
const actHist = [];                                // actions that produced them
let steps = 3, framesSince = 0, paused = false, ready = false, autoplay = false, tourIdx = 0, tour = [];
let autowake = qs.get('autowake') === '1';
let flash = 0;                                      // capture mode: frames left of the wake-up flash (deterministic, unlike a CSS animation)
const AUTOWAKE_FRAMES = 60;                         // 2 s at the 30 Hz game tick
const psnrHist = [];
const wakeMarks = [];                              // indexes into psnrHist where the dream was (re)seeded from the real game
const tickTimes = [];
let emaMs = 0;
const held = new Set();

// ------------------------------------------------------------------ helpers
const actionFromKeys = (k) => [
  k.has('F') && !k.has('B') ? 1 : k.has('B') && !k.has('F') ? 2 : 0,
  k.has('Q') && !k.has('E') ? 1 : k.has('E') && !k.has('Q') ? 2 : 0,
  k.has('L') && !k.has('R') ? 1 : k.has('R') && !k.has('L') ? 2 : 0,
];
const keysFromAction = (a) => {
  const s = new Set();
  if (a[0] === 1) s.add('F'); if (a[0] === 2) s.add('B');
  if (a[1] === 1) s.add('Q'); if (a[1] === 2) s.add('E');
  if (a[2] === 1) s.add('L'); if (a[2] === 2) s.add('R');
  return s;
};
const drawRgb = (ctx, img, rgb) => {
  const d = img.data;
  for (let i = 0, j = 0; i < PIX; i++, j += 4) { d[j] = rgb[i * 3]; d[j + 1] = rgb[i * 3 + 1]; d[j + 2] = rgb[i * 3 + 2]; d[j + 3] = 255; }
  ctx.putImageData(img, 0, 0);
};
const toast = (msg, ms = 6000) => {
  const t = $('toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, ms);
};
const fmtParams = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(0) + 'k' : String(n);
const nextFrame = () => new Promise((r) => requestAnimationFrame(r));

// ------------------------------------------------------------------ layout
function fitScale() {
  let s;
  const w = innerWidth, h = innerHeight;
  const dreamOnly = body.classList.contains('dreamonly');
  document.documentElement.style.removeProperty('--sr');
  if (CAPTURE) s = 7;
  else if (w <= 760) {
    s = Math.max(3, Math.min(dreamOnly ? 6 : 5, Math.floor((w - 72) / 64)));
    if (!dreamOnly) document.documentElement.style.setProperty('--sr', Math.max(3, s - 1));     // the real panel a notch smaller so both fit on one screen
  }
  else {
    const panels = dreamOnly ? 1 : 2;
    const keysCol = w >= 1060 ? 132 + 52 : 0;                 // keys sit between the panels on wide screens
    const avail = Math.min(w, 1180) - 32 - keysCol - (panels === 2 ? 26 : 0) - panels * 26;
    const byW = Math.floor(avail / panels / 64);
    const byH = Math.max(5, Math.floor((h - 345) / 64));
    s = Math.max(4, Math.min(dreamOnly ? 9 : 8, byW, byH + (dreamOnly ? 1 : 0)));
  }
  document.documentElement.style.setProperty('--s', s);
}

// ------------------------------------------------------------------ static while the model loads
let staticTimer = null;
function drawStatic() {
  const d = imgDream.data;
  for (let j = 0; j < d.length; j += 4) { const v = (Math.random() * 90) | 0; d[j] = v; d[j + 1] = v; d[j + 2] = v + 10; d[j + 3] = 255; }
  ctxDream.putImageData(imgDream, 0, 0);
}
function startStatic() { drawStatic(); if (!reduced) staticTimer = setInterval(drawStatic, 110); }
function stopStatic() { clearInterval(staticTimer); staticTimer = null; }

// ------------------------------------------------------------------ real game + history
function pushHistory(rgb, action) {
  frameHist.push(rgbToChw(rgb)); actHist.push(action.slice());
  while (frameHist.length > K) { frameHist.shift(); actHist.shift(); }
}
function resetReal() {
  real = stateFromSpecStart(spec);
  frameHist.length = 0; actHist.length = 0;
  world.render(real, realRGB);
  for (let k = 0; k < K; k++) pushHistory(realRGB, [0, 0, 0]);
  drawRgb(ctxReal, imgReal, realRGB);
}
function wake() {
  dream.reset(frameHist, actHist);
  framesSince = 0; wakeMarks.push(psnrHist.length);
  const p = $('panel-dream');
  if (CAPTURE) flash = 7; else { p.classList.remove('woke'); void p.offsetWidth; p.classList.add('woke'); }
  updateReadouts();
  drawDrift();
}

// ------------------------------------------------------------------ one game tick: real step + dream step
async function tick(action) {
  real = world.step(real, action);
  world.render(real, realRGB);
  pushHistory(realRGB, action);
  const out = await dream.step(action);
  framesSince++; window.__ticks = (window.__ticks || 0) + 1;
  const p = psnr(realRGB, out.rgb);
  psnrHist.push(Math.min(p, 50));
  if (psnrHist.length > 400) { psnrHist.shift(); for (let i = 0; i < wakeMarks.length; i++) wakeMarks[i]--; while (wakeMarks.length && wakeMarks[0] < 0) wakeMarks.shift(); }
  drawRgb(ctxReal, imgReal, realRGB);
  drawRgb(ctxDream, imgDream, out.rgb);
  if (CAPTURE) $('panel-dream').classList.toggle('flash', flash-- > 0);
  const now = performance.now();
  tickTimes.push(now); while (tickTimes.length && now - tickTimes[0] > 1000) tickTimes.shift();
  emaMs = emaMs ? emaMs * 0.85 + out.ms * 0.15 : out.ms;
  return { psnr: p, ms: out.ms };
}

// ------------------------------------------------------------------ readouts
function updateReadouts() {
  const last = psnrHist.length ? psnrHist[psnrHist.length - 1] : null;
  $('r-ms').innerHTML = emaMs ? `${(CAPTURE && qs.get('ms') ? +qs.get('ms') : emaMs).toFixed(1)}<small>ms</small>` : '–';
  const fps = CAPTURE && qs.get('fps') ? +qs.get('fps') : (tickTimes.length > 2 ? tickTimes.length : 0);
  $('r-fps').textContent = fps ? String(Math.round(fps)) : '–';
  $('r-steps').textContent = String(steps);
  $('r-psnr').innerHTML = last === null ? '–' : `${last.toFixed(1)}<small>dB</small>`;
  $('r-since').textContent = String(framesSince);
  $('dream-foot').textContent = `frames since wake-up ${framesSince}`;
  const sx = real ? real.x.toFixed(2) : '0.00', sy = real ? real.y.toFixed(2) : '0.00';
  const deg = real ? ((Math.atan2(real.dy, real.dx) * 180 / Math.PI + 360) % 360).toFixed(0) : '0';
  $('pose').textContent = `x ${sx} · y ${sy} · heading ${deg}°`;
  drawMap();
}

function drawDrift() {
  const c = cvDrift, ctx = ctxDrift, W = c.width, H = c.height;
  const dpr = devicePixelRatio || 1;
  const cw = Math.round(c.clientWidth * dpr), ch = Math.round(c.clientHeight * dpr);
  if (cw && (c.width !== cw || c.height !== ch)) { c.width = cw; c.height = ch; }
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  const lo = 10, hi = 50, y = (v) => h - 6 - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (h - 14);
  ctx.strokeStyle = '#2e3a52'; ctx.lineWidth = 1; ctx.fillStyle = '#8893ab'; ctx.font = `${10 * dpr}px "JetBrains Mono", monospace`;
  for (const g of [20, 30, 40]) { ctx.beginPath(); ctx.moveTo(0, y(g) + .5); ctx.lineTo(w, y(g) + .5); ctx.globalAlpha = .55; ctx.stroke(); ctx.globalAlpha = 1; ctx.fillText(String(g), 4, y(g) - 3); }
  if (psnrHist.length < 2) return;
  const n = 400, step = (w - 8) / (n - 1), x0 = 4;
  ctx.save(); ctx.strokeStyle = 'rgba(245,158,11,.45)'; ctx.setLineDash([3 * dpr, 4 * dpr]); ctx.lineWidth = dpr;
  for (const m of wakeMarks) { const mx = x0 + m * step; ctx.beginPath(); ctx.moveTo(mx, 2); ctx.lineTo(mx, h); ctx.stroke(); }
  ctx.restore();
  ctx.beginPath();
  psnrHist.forEach((v, i) => { const px = x0 + i * step; i ? ctx.lineTo(px, y(v)) : ctx.moveTo(px, y(v)); });
  const lx = x0 + (psnrHist.length - 1) * step;
  ctx.lineTo(lx, h); ctx.lineTo(x0, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h); grad.addColorStop(0, 'rgba(245,158,11,.35)'); grad.addColorStop(1, 'rgba(245,158,11,0)');
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  psnrHist.forEach((v, i) => { const px = x0 + i * step; i ? ctx.lineTo(px, y(v)) : ctx.moveTo(px, y(v)); });
  ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.6 * dpr; ctx.lineJoin = 'round'; ctx.stroke();
}

// top-down minimap of the real player (REAL panel): walls coloured like the map, amber dot + heading tick
let mapLayer = null;
function drawMap() {
  if (!spec || !real) return;
  const S = cvMap.width, cell = S / spec.size;
  if (!mapLayer) {
    mapLayer = document.createElement('canvas'); mapLayer.width = mapLayer.height = S;
    const m = mapLayer.getContext('2d');
    m.fillStyle = '#0b0e14'; m.fillRect(0, 0, S, S);
    for (let r = 0; r < spec.size; r++) for (let c = 0; c < spec.size; c++) {
      const ch = spec.map[r][c];
      if (ch === '.') continue;
      const rgb = spec.walls[String(spec.wall_chars[ch])].rgb;
      m.fillStyle = `rgb(${rgb[0] * 0.8 | 0},${rgb[1] * 0.8 | 0},${rgb[2] * 0.8 | 0})`;
      m.fillRect(c * cell, r * cell, cell, cell);
    }
  }
  ctxMap.drawImage(mapLayer, 0, 0);
  const px = real.x * cell, py = real.y * cell;
  ctxMap.strokeStyle = '#f59e0b'; ctxMap.lineWidth = 1.6;
  ctxMap.beginPath(); ctxMap.moveTo(px, py); ctxMap.lineTo(px + real.dx * cell * 1.6, py + real.dy * cell * 1.6); ctxMap.stroke();
  ctxMap.fillStyle = '#ffffff'; ctxMap.beginPath(); ctxMap.arc(px, py, Math.max(1.4, cell * 0.38), 0, 7); ctxMap.fill();
}

function setKeyLights(set) {
  document.querySelectorAll('.keys .key').forEach((el) => el.classList.toggle('on', set.has(el.dataset.k)));
  document.querySelectorAll('.tp').forEach((el) => el.classList.toggle('on', set.has(el.dataset.k)));
}

// ------------------------------------------------------------------ input
const KEYMAP = { KeyW: 'F', ArrowUp: 'F', KeyS: 'B', ArrowDown: 'B', KeyA: 'L', ArrowLeft: 'L', KeyD: 'R', ArrowRight: 'R', KeyQ: 'Q', KeyE: 'E' };
addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target instanceof HTMLElement && e.target.closest('button') && (e.code === 'Space' || e.code === 'Enter')) return;
  const k = KEYMAP[e.code];
  if (k) { held.add(k); setKeyLights(held); if (autoplay) setAutoplay(false); e.preventDefault(); return; }
  if (e.code === 'KeyR') { $('btn-wake').click(); e.preventDefault(); }
  else if (e.code === 'KeyT') { $('btn-dreamonly').click(); e.preventDefault(); }
  else if (e.code === 'KeyP') { $('btn-pause').click(); e.preventDefault(); }
  else if (e.code === 'Escape') closeDrawer();
  else if (['Digit1', 'Digit2', 'Digit3'].includes(e.code)) setSteps(+e.code.slice(-1));
});
addEventListener('keyup', (e) => { const k = KEYMAP[e.code]; if (k) { held.delete(k); setKeyLights(held); } });
addEventListener('blur', () => { held.clear(); setKeyLights(held); });
document.querySelectorAll('.tp').forEach((b) => {
  const down = (e) => { e.preventDefault(); held.add(b.dataset.k); setKeyLights(held); b.setPointerCapture?.(e.pointerId); };
  const up = () => { held.delete(b.dataset.k); setKeyLights(held); };
  b.addEventListener('pointerdown', down); b.addEventListener('pointerup', up); b.addEventListener('pointercancel', up); b.addEventListener('lostpointercapture', up);
});

// ------------------------------------------------------------------ controls
function setSteps(n) {
  steps = n; dream?.setSteps(n);
  document.querySelectorAll('.seg-b').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.steps === n)));
  tickTimes.length = 0; emaMs = 0; updateReadouts();
}
document.querySelectorAll('.seg-b').forEach((b) => b.addEventListener('click', () => setSteps(+b.dataset.steps)));
$('btn-wake').addEventListener('click', () => { if (ready) wake(); });
$('btn-dreamonly').addEventListener('click', (e) => {
  const on = body.classList.toggle('dreamonly');
  e.currentTarget.setAttribute('aria-pressed', String(on)); fitScale(); drawDrift();
});
$('btn-scan').addEventListener('click', (e) => { const on = body.classList.toggle('scanlines'); e.currentTarget.setAttribute('aria-pressed', String(on)); });
$('btn-pause').addEventListener('click', (e) => { paused = !paused; e.currentTarget.setAttribute('aria-pressed', String(paused)); });
function setAutoplay(on) {
  autoplay = on; $('btn-auto').setAttribute('aria-pressed', String(on));
  if (on && ready) { resetReal(); wake(); tourIdx = 0; }
  if (!on) setKeyLights(held);
}
$('btn-auto').addEventListener('click', () => setAutoplay(!autoplay));
$('btn-autowake').addEventListener('click', (e) => { autowake = !autowake; e.currentTarget.setAttribute('aria-pressed', String(autowake)); });
if (autowake) $('btn-autowake').setAttribute('aria-pressed', 'true');
document.querySelectorAll('.tp').forEach((b) => b.addEventListener('pointerdown', () => { if (autoplay) setAutoplay(false); }));
const drawer = $('drawer'), scrim = $('scrim');
function openDrawer() { drawer.classList.add('open'); drawer.setAttribute('aria-hidden', 'false'); scrim.hidden = false; $('btn-about').setAttribute('aria-expanded', 'true'); drawer.focus({ preventScroll: true }); }
function closeDrawer() { drawer.classList.remove('open'); drawer.setAttribute('aria-hidden', 'true'); scrim.hidden = true; $('btn-about').setAttribute('aria-expanded', 'false'); }
$('btn-about').addEventListener('click', () => (drawer.classList.contains('open') ? closeDrawer() : openDrawer()));
$('btn-close').addEventListener('click', closeDrawer);
scrim.addEventListener('click', closeDrawer);
addEventListener('resize', () => { fitScale(); drawDrift(); });
addEventListener('load', fitScale);
if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => fitScale()).observe(document.documentElement);   // viewport changes that fire no resize event (phones)
document.addEventListener('visibilitychange', () => { if (document.hidden) { held.clear(); setKeyLights(held); } });

// ------------------------------------------------------------------ loading
async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = +r.headers.get('content-length') || 0;
  if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(); const buf = new Uint8Array(total); let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; buf.set(value, got); got += value.length; onProgress(got / total, total); }
  return buf;
}

async function pickModel() {
  const forced = qs.get('dtype');
  let f16 = false;
  try { const ad = navigator.gpu && await navigator.gpu.requestAdapter(); f16 = !!ad && ad.features.has('shader-f16'); } catch (e) { /* no WebGPU */ }
  const order = forced ? [forced] : (f16 ? ['fp16', 'fp32'] : ['fp32']);
  for (const d of order) {
    const url = `models/dream_${d}.onnx`;
    try { const head = await fetch(url, { method: 'HEAD' }); if (head.ok) return { url, dtype: d }; } catch (e) { /* try next */ }
  }
  throw new Error('no model file found in models/');
}

async function boot() {
  body.classList.toggle('capture', CAPTURE);
  body.classList.toggle('square', CAPTURE && qs.get('square') === '1');
  fitScale();
  startStatic();
  const veilText = $('veil-text');
  try {
    spec = await (await fetch('js/spec.json')).json();
    try { meta = await (await fetch('models/dream_meta.json')).json(); } catch (e) { meta = {}; }
    try { tour = (await (await fetch('js/tour.json')).json()).actions; } catch (e) { tour = []; }
    try {
      const ev = await (await fetch('models/eval_summary.json')).json();
      const span = document.querySelector('[data-eval="drift"]');
      if (span && ev.psnr_at) {
        const g = (k) => (ev.psnr_at[k] !== undefined ? ev.psnr_at[k].toFixed(1) : null);
        let t = `Measured on the game's own exploration trajectories (${ev.episodes} episodes): the dream matches the real frame at ${g('1')} dB after 1 frame, ${g('16')} dB after 16, ${g('32')} dB after 32 and ${g('128')} dB after 128; an unrelated frame scores about ${ev.floor.toFixed(0)} dB.`;
        if (ev.validity) t += ` Even where it no longer matches, its frames stay game-like: ${ev.validity.dream_mean.toFixed(1)} dB to the nearest real frame, against ${ev.validity.real_mean.toFixed(1)} dB for real frames.`;
        span.textContent = t + ' ';
      }
    } catch (e) { /* optional */ }
    world = new World(spec);
    resetReal();
    $('r-params').textContent = meta.params ? fmtParams(meta.params) : '–';
    const choice = await pickModel();
    veilText.textContent = 'loading model…';
    const bytes = await fetchWithProgress(choice.url, (f, total) => { veilText.textContent = `loading model ${(f * 100).toFixed(0)}% of ${(total / 1e6).toFixed(0)} MB`; });
    veilText.textContent = 'compiling…';
    dream = await Dream.create({ modelBytes: bytes, meta, backend: qs.get('backend') || 'auto', capture: qs.get('capture_graph') !== '0', seed: +(qs.get('seed') || 1234), onStatus: (s) => { veilText.textContent = s; } });
    dream.info.dtype = choice.dtype;
    const steps0 = +(qs.get('steps') || meta.default_steps || 3);
    setSteps(Math.min(3, Math.max(1, steps0)));
    $('r-backend').innerHTML = `${dream.info.backend === 'webgpu' ? 'WebGPU' : 'WASM'}<small>${choice.dtype}${dream.info.graphCapture ? ' · captured' : ''}</small>`;
    $('dream-sub').textContent = `neural net · ${meta.params ? fmtParams(meta.params) + ' params · ' : ''}no game code`;
    stopStatic();
    ready = true;
    body.classList.remove('loading');
    fitScale();
    wake();
    if (dream.info.backend === 'wasm') toast('WebGPU is not available here, so the dream runs on the slow WASM path. Expect a few frames per second.');
    if (CAPTURE) installCapture(); else run();
  } catch (err) {
    console.error(err);
    stopStatic(); body.classList.remove('loading');
    veilText.textContent = 'could not wake the model';
    toast(`Could not load the model: ${err.message || err}`, 20000);
  }
}

// ------------------------------------------------------------------ main loop
async function run() {
  let hudAt = 0;
  for (;;) {
    if (paused || document.hidden) { await nextFrame(); continue; }
    const t0 = performance.now();
    if (autowake && framesSince >= AUTOWAKE_FRAMES) wake();
    let a;
    if (autoplay && tour.length) {
      if (tourIdx >= tour.length) { resetReal(); wake(); tourIdx = 0; }
      a = tour[tourIdx++]; setKeyLights(keysFromAction(a));
    } else a = actionFromKeys(held);
    await tick(a);
    const now = performance.now();
    if (now - hudAt > 150) { hudAt = now; updateReadouts(); drawDrift(); }
    const budget = 1000 / TICK_HZ;
    do { await nextFrame(); } while (performance.now() - t0 < budget - 1.5);      // always yield: paint + key events
  }
}

// ------------------------------------------------------------------ capture mode (scripted, deterministic)
function installCapture() {
  if (qs.get('note')) $('capnote').innerHTML = qs.get('note').replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  window.__capture = {
    async tick(spec) {                              // spec: array [move, strafe, turn]
      const a = Array.isArray(spec) ? spec : actionFromKeys(new Set(String(spec).split('+').filter(Boolean)));
      setKeyLights(keysFromAction(a));
      const r = await tick(a);
      updateReadouts(); drawDrift();
      return r;
    },
    wake() { wake(); },
    steps(n) { setSteps(n); },
    // consistency check of the whole browser pipeline: 1-step prediction PSNR on random windows of the JS game,
    // against the copy-last-frame baseline on the same windows (python: results/eval_*.json uses the same recipe)
    // same, on windows exported from python (poses of frame 0 and the 4 actions): exact cross-check of the pipeline
    async oneStepFromWindows(wins) {
      const rgb = new Uint8Array(N), tgt = new Uint8Array(N), last = new Uint8Array(N);
      let sd = 0, sc = 0;
      for (const w of wins) {
        let st = { x: w.pose[0], y: w.pose[1], dx: w.pose[2], dy: w.pose[3] };
        const fr = [], ac = [];
        world.render(st, rgb); fr.push(rgbToChw(rgb)); ac.push([0, 0, 0]);
        for (let k = 0; k < 3; k++) { st = world.step(st, w.actions[k]); world.render(st, rgb); fr.push(rgbToChw(rgb)); ac.push(w.actions[k]); }
        last.set(rgb);
        st = world.step(st, w.actions[3]); world.render(st, tgt);
        dream.reset(fr, ac);
        const out = await dream.step(w.actions[3]);
        sd += Math.min(100, psnr(out.rgb, tgt)); sc += Math.min(100, psnr(last, tgt));
      }
      return { n: wins.length, dream_psnr: sd / wins.length, copy_last_psnr: sc / wins.length, steps, ctx_sigma: dream.ctxSigma, backend: dream.info.backend };
    },
    async oneStepEval(n = 64, seed = 7) {
      const rng = new Rng(seed), cells = world.freeCells();
      const pick = (p1, p2) => (rng.next() < p1 ? 0 : (rng.next() < 0.5 ? 1 : 2));
      const rnd = () => [rng.next() < 0.78 ? 1 : (rng.next() < 0.5 ? 0 : 2), pick(0.9), pick(0.5)];
      let sd = 0, sc = 0;
      const rgb = new Uint8Array(N), last = new Uint8Array(N), tgt = new Uint8Array(N);
      for (let i = 0; i < n; i++) {
        const c = cells[Math.floor(rng.next() * cells.length)], ang = rng.next() * 2 * Math.PI;
        let st = { x: c[0] + 0.25 + 0.5 * rng.next(), y: c[1] + 0.25 + 0.5 * rng.next(), dx: Math.cos(ang), dy: Math.sin(ang) };
        const fr = [], ac = [];
        world.render(st, rgb); fr.push(rgbToChw(rgb)); ac.push(rnd());
        for (let k = 1; k < K; k++) { const a = rnd(); st = world.step(st, a); world.render(st, rgb); fr.push(rgbToChw(rgb)); ac.push(a); }
        last.set(rgb);
        const a3 = rnd(); st = world.step(st, a3); world.render(st, tgt);
        dream.reset(fr, ac);
        const out = await dream.step(a3);
        sd += Math.min(60, psnr(out.rgb, tgt)); sc += Math.min(60, psnr(last, tgt));
      }
      return { n, dream_psnr: sd / n, copy_last_psnr: sc / n, steps, ctx_sigma: dream.ctxSigma, backend: dream.info.backend };
    },
    info: () => ({ ...dream.info, params: meta.params, steps }),
    state: () => ({ x: real.x, y: real.y, dx: real.dx, dy: real.dy, framesSince }),
  };
  window.__captureReady = true;
}

boot();
