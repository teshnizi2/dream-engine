// The neural game engine: an onnxruntime-web session plus the EDM sampler. There is no game code in here:
// given the last K frames and the last K actions (K = 4 in v2, read from the model meta) it imagines the next frame, then feeds it back in.
//
//   frame ~ D(x; sigma | context frames, actions)   (3 Euler steps by default, sigmas from the Karras schedule)
//
// The ONNX graph (export.py) has static shapes (batch 1): x [1,3,64,64], sigma [1], ctx [1,3K,64,64],
// act [1,K,9] (one-hot), ctx_sigma [1] -> denoised [1,3,64,64].

import { SIZE, PIX, N, K, setContextLength, buildSigmas, Rng, eulerSample, quantiseFrame } from './util.mjs';
export { SIZE, PIX, N, K, setContextLength, buildSigmas, Rng, rgbToChw, psnr, eulerSample, quantiseFrame, backgroundFraction } from './util.mjs';

const POOL = 1 << 20;

// onnxruntime-web is loaded on demand: the WebGPU build (which carries a 28 MB wasm for its CPU path) only when the page
// will use WebGPU, the plain WASM build (about half the size) otherwise. The import map in index.html names both.
let ort = null;
const ortLoads = {};
export function loadOrt(kind) {
  const names = kind === 'wasm' ? ['onnxruntime-web/wasm', 'onnxruntime-web'] : ['onnxruntime-web'];
  return (ortLoads[kind] ||= (async () => {
    let err;
    for (const n of names) { try { return await import(n); } catch (e) { err = e; } }
    throw err;
  })());
}

export class Dream {
  static async create({ modelBytes, meta = {}, backend = 'auto', capture = true, seed = 1234, onStatus = () => {}, wasmFallback = true }) {
    const d = new Dream();
    setContextLength(meta.config?.num_ctx ?? 4);          // the graph's context length (v2: 4); must be set before the buffers below
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
    await d._open(modelBytes, backend, capture, onStatus, wasmFallback);
    return d;
  }

  async _open(modelBytes, backend, capture, onStatus, wasmFallback = true) {
    const t0 = performance.now();
    let adapter = null;
    if (backend !== 'wasm' && navigator.gpu) {
      try { adapter = await navigator.gpu.requestAdapter(); } catch (e) { adapter = null; }
    }
    ort = await loadOrt(adapter ? 'webgpu' : 'wasm');
    // WASM threads need cross-origin isolation (SharedArrayBuffer); without it onnxruntime-web runs single-threaded
    const th = new URLSearchParams(location.search).get('threads');
    ort.env.wasm.numThreads = th ? +th : ((typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) ? Math.min(4, navigator.hardwareConcurrency || 2) : 1);
    ort.env.wasm.proxy = !adapter;                // CPU inference runs in a worker so a slow device keeps a responsive page
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
        catch (e2) {
          console.warn('WebGPU failed, falling back to wasm', e2); this.session = null; this.gpu = null; this.info.backend = 'wasm';
          if (!wasmFallback) { const e = new Error('WebGPU failed to start on this device'); e.code = 'webgpu-failed'; throw e; }
          ort = await loadOrt('wasm'); ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = true;
        }
      }
    }
    if (!this.session) {
      console.log('[dream] creating wasm session');
      onStatus('compiling WASM graph (slow path)');
      this.session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      this.info.backend = 'wasm';
      this.info.graphCapture = false;
      await this._warmup();
    }
    this.info.loadMs = performance.now() - t0;
    console.log('[dream] ready', this.info.backend, Math.round(this.info.loadMs), 'ms');
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
    console.log('[dream] warmup');
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
    // copies: in proxy (worker) mode onnxruntime-web transfers the input buffers, which would detach ours
    const feeds = {
      x: new ort.Tensor('float32', x.slice(), [1, 3, SIZE, SIZE]),
      sigma: new ort.Tensor('float32', this.sigArr.slice(0, 1), [1]),
      ctx: new ort.Tensor('float32', this.ctxArr.slice(), [1, K * 3, SIZE, SIZE]),
      act: new ort.Tensor('float32', this.actArr.slice(), [1, K, 9]),
      ctx_sigma: new ort.Tensor('float32', this.csArr.slice(0, 1), [1]),
    };
    const out = await this.session.run(feeds);
    return out.denoised.data;
  }

  // Imagine the next frame given the action taken now. Returns { chw (quantised, [-1,1]), rgb (HWC bytes), ms }.
  async step(action) {
    const t0 = performance.now();
    const P = this.prof || (this.prof = { prep: 0, nfe: 0, post: 0, n: 0 });
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
    const tp = performance.now();
    const sg = this.sigmas;
    const x = this.xArr;
    this._noise(x, this.initScale === 'edm' ? sg[0] : 1.0);
    const tn = performance.now();
    await eulerSample(x, sg, (xx, sigma) => this._nfe(xx, sigma));
    const tq = performance.now();
    // clamp, quantise to the 8-bit grid, and keep the quantised frame as the next context frame
    const next = this.frames[0];            // recycle the oldest frame's storage
    const rgb = new Uint8Array(N);
    quantiseFrame(x, next, rgb);
    this.frames.push(this.frames.shift());   // the recycled array becomes the newest frame
    this.lastMs = performance.now() - t0;
    P.prep += tp - t0; P.nfe += tq - tn; P.post += performance.now() - tq; P.n++;
    return { chw: next, rgb, ms: this.lastMs };
  }
}

