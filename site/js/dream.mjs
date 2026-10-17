// The neural game engine: an onnxruntime-web session plus the EDM sampler. There is no game code in here:
// given the last 4 frames and the last 4 actions it imagines the next frame, then feeds it back in.
//
//   frame ~ D(x; sigma | context frames, actions)   (3 Euler steps by default, sigmas from the Karras schedule)
//
// The ONNX graph (export.py) has static shapes (batch 1): x [1,3,64,64], sigma [1], ctx [1,12,64,64],
// act [1,4,9] (one-hot), ctx_sigma [1] -> denoised [1,3,64,64].

import * as ort from 'onnxruntime-web';

export const SIZE = 64;
export const PIX = SIZE * SIZE;
export const N = 3 * PIX;
export const K = 4;

export function buildSigmas(steps, smin = 2e-3, smax = 5, rho = 7) {
  const a = Math.pow(smax, 1 / rho), b = Math.pow(smin, 1 / rho);
  const s = [];
  for (let i = 0; i < steps; i++) {
    const l = steps === 1 ? 0 : i / (steps - 1);
    s.push(Math.pow(a + l * (b - a), rho));
  }
  s.push(0);
  return s;
}

// mulberry32 + Box-Muller, so that capture runs are reproducible
export class Rng {
  constructor(seed = 1) { this.s = seed >>> 0; this.spare = null; }
  next() {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  randn() {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next(), r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
}

const POOL = 1 << 20;

export class Dream {
  static async create({ modelBytes, meta = {}, backend = 'auto', capture = true, seed = 1234, onStatus = () => {} }) {
    const d = new Dream();
    d.rng = new Rng(seed);
    d.meta = meta;
    d.pool = new Float32Array(POOL);
    for (let i = 0; i < POOL; i++) d.pool[i] = d.rng.randn();
    d.frames = Array.from({ length: K }, () => new Float32Array(N));
    d.acts = Array.from({ length: K }, () => [0, 0, 0]);
    d.ctxArr = new Float32Array(K * N);
    d.actArr = new Float32Array(K * 9);
    d.xArr = new Float32Array(N);
    d.sigArr = new Float32Array(4);
    d.csArr = new Float32Array(4);
    d.steps = 3;
    d.sigmas = buildSigmas(3);
    d.ctxSigma = meta.ctx_sigma ?? 0.1;
    d.initScale = meta.sampler_init === 'unit' ? 'unit' : 'edm';
    d.lastMs = 0;
    await d._open(modelBytes, backend, capture, onStatus);
    return d;
  }

  async _open(modelBytes, backend, capture, onStatus) {
    ort.env.wasm.numThreads = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    ort.env.wasm.proxy = false;
    const t0 = performance.now();
    let adapter = null;
    if (backend !== 'wasm' && navigator.gpu) {
      try { adapter = await navigator.gpu.requestAdapter(); } catch (e) { adapter = null; }
    }
    this.info = { backend: 'wasm', graphCapture: false, adapter: null, modelBytes: modelBytes.byteLength };
    const tryWebgpu = async (cap) => {
      onStatus(cap ? 'compiling WebGPU graph (captured)' : 'compiling WebGPU graph');
      const opts = { executionProviders: [{ name: 'webgpu', preferredLayout: 'NHWC' }], graphOptimizationLevel: 'all' };
      if (cap) { opts.enableGraphCapture = true; opts.preferredOutputLocation = { denoised: 'gpu-buffer' }; }
      this.session = await ort.InferenceSession.create(modelBytes, opts);
      this.info.backend = 'webgpu';
      this.info.graphCapture = cap;
      this.info.adapter = adapter && adapter.info ? (adapter.info.description || adapter.info.vendor || 'WebGPU') : 'WebGPU';
      if (cap) this._initGpuIo();
      await this._warmup();
    };
    if (adapter) {
      try { await tryWebgpu(capture); }
      catch (e1) {
        console.warn('WebGPU with graph capture failed, retrying without capture', e1);
        try { this.session = null; this.gpu = null; await tryWebgpu(false); }
        catch (e2) { console.warn('WebGPU failed, falling back to wasm', e2); this.session = null; this.gpu = null; this.info.backend = 'wasm'; }
      }
    }
    if (!this.session) {
      onStatus('compiling WASM graph (slow path)');
      this.session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      this.info.backend = 'wasm';
      this.info.graphCapture = false;
      await this._warmup();
    }
    this.info.loadMs = performance.now() - t0;
  }

  _initGpuIo() {
    const device = ort.env.webgpu.device;
    const mk = (arr, dims) => {
      const size = Math.ceil(arr.byteLength / 16) * 16;
      const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      return { buf, tensor: ort.Tensor.fromGpuBuffer(buf, { dataType: 'float32', dims }) };
    };
    this.gpu = {
      device,
      x: mk(this.xArr, [1, 3, SIZE, SIZE]),
      sigma: mk(this.sigArr.subarray(0, 1), [1]),
      ctx: mk(this.ctxArr, [1, K * 3, SIZE, SIZE]),
      act: mk(this.actArr, [1, K, 9]),
      cs: mk(this.csArr.subarray(0, 1), [1]),
    };
    this.gpuFeeds = { x: this.gpu.x.tensor, sigma: this.gpu.sigma.tensor, ctx: this.gpu.ctx.tensor, act: this.gpu.act.tensor, ctx_sigma: this.gpu.cs.tensor };
  }

  async _warmup() {
    const saved = this.frames.map((f) => f.slice());
    for (let i = 0; i < 3; i++) await this.step([0, 0, 0]);
    this.frames.forEach((f, k) => f.set(saved[k]));
    this.acts = Array.from({ length: K }, () => [0, 0, 0]);
  }

  setSteps(n) { this.steps = n; this.sigmas = buildSigmas(n); }
  setCtxSigma(s) { this.ctxSigma = s; }

  // frames: K Float32Arrays [3*64*64] in [-1,1] (CHW), actions: K triples (move, strafe, turn)
  reset(frames, actions) {
    for (let k = 0; k < K; k++) { this.frames[k].set(frames[k]); this.acts[k] = actions[k].slice(); }
  }

  _noise(out, scale, offset = 0, n = out.length) {
    const start = Math.floor(this.rng.next() * (POOL - n));
    const p = this.pool;
    for (let i = 0; i < n; i++) out[offset + i] = p[start + i] * scale;
  }

  async _nfe(x, sigma) {
    this.sigArr[0] = sigma;
    if (this.gpu) {
      const q = this.gpu.device.queue;
      q.writeBuffer(this.gpu.x.buf, 0, x);
      q.writeBuffer(this.gpu.sigma.buf, 0, this.sigArr, 0, 1);
      const out = await this.session.run(this.gpuFeeds);
      return await out.denoised.getData();
    }
    const feeds = {
      x: new ort.Tensor('float32', x, [1, 3, SIZE, SIZE]),
      sigma: new ort.Tensor('float32', this.sigArr.slice(0, 1), [1]),
      ctx: new ort.Tensor('float32', this.ctxArr, [1, K * 3, SIZE, SIZE]),
      act: new ort.Tensor('float32', this.actArr, [1, K, 9]),
      ctx_sigma: new ort.Tensor('float32', this.csArr.slice(0, 1), [1]),
    };
    const out = await this.session.run(feeds);
    return out.denoised.data;
  }

  // Imagine the next frame given the action taken now. Returns { chw (quantised, [-1,1]), rgb (HWC bytes), ms }.
  async step(action) {
    const t0 = performance.now();
    this.acts.push(action.slice());
    this.acts.shift();
    const cs = this.ctxSigma;
    for (let k = 0; k < K; k++) {
      this.ctxArr.set(this.frames[k], k * N);
      if (cs > 0) {
        // add noise to the context (the model was trained to see and correct a noisy context)
        const tmp = this.xArr; // reuse as scratch
        this._noise(tmp, cs);
        const o = k * N;
        for (let i = 0; i < N; i++) this.ctxArr[o + i] += tmp[i];
      }
    }
    this.actArr.fill(0);
    for (let k = 0; k < K; k++) {
      const a = this.acts[k];
      this.actArr[k * 9 + a[0]] = 1;
      this.actArr[k * 9 + 3 + a[1]] = 1;
      this.actArr[k * 9 + 6 + a[2]] = 1;
    }
    this.csArr[0] = cs;
    if (this.gpu) {
      const q = this.gpu.device.queue;
      q.writeBuffer(this.gpu.ctx.buf, 0, this.ctxArr);
      q.writeBuffer(this.gpu.act.buf, 0, this.actArr);
      q.writeBuffer(this.gpu.cs.buf, 0, this.csArr, 0, 1);
    }
    const sg = this.sigmas;
    const x = this.xArr;
    this._noise(x, this.initScale === 'edm' ? sg[0] : 1.0);
    for (let i = 0; i < this.steps; i++) {
      const sigma = sg[i];
      const D = await this._nfe(x, sigma);
      const dt = (sg[i + 1] - sigma) / sigma;
      for (let j = 0; j < N; j++) x[j] += (x[j] - D[j]) * dt;
    }
    // clamp, quantise to the 8-bit grid, and keep the quantised frame as the next context frame
    const next = this.frames[0];            // recycle the oldest frame's storage
    const rgb = new Uint8Array(N);
    for (let c = 0; c < 3; c++) {
      for (let p = 0; p < PIX; p++) {
        const v = Math.min(1, Math.max(-1, x[c * PIX + p]));
        const b = Math.floor((v + 1) * 127.5 + 0.5);
        next[c * PIX + p] = b / 127.5 - 1;
        rgb[p * 3 + c] = b;
      }
    }
    this.frames.push(this.frames.shift());   // the recycled array becomes the newest frame
    this.lastMs = performance.now() - t0;
    return { chw: next, rgb, ms: this.lastMs };
  }
}

// helpers shared with the app
export function rgbToChw(rgb, out = new Float32Array(N)) {
  for (let p = 0; p < PIX; p++) for (let c = 0; c < 3; c++) out[c * PIX + p] = rgb[p * 3 + c] / 127.5 - 1;
  return out;
}

export function psnr(a, b) {
  let se = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; se += d * d; }
  const mse = se / a.length;
  return mse < 1e-9 ? 99 : 10 * Math.log10((255 * 255) / mse);
}
