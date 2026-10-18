# Loading and fallbacks on phones

Measured with headless Chromium at 390x844 with touch emulation, onnxruntime-web served from a local copy (no real WebGPU was available, so the WebGPU path was only exercised with a simulated adapter that fails to start).

| Visitor | Before | After |
|---|---|---|
| Browser without WebGPU | downloaded about 82 MB before the dream moved (28 MB preloaded runtime that is never used, 27 MB runtime actually used, 27 MB model); 4 fps at 2 steps, inference on the page thread | about 0.6 MB to the first screen (recording and poster, lazy); the live model only on tap: about 42 MB (14 MB plain WASM runtime, 27 MB fp32 model), 6 fps at 1 step in a worker |
| WebGPU | unchanged model and runtime (13.8 MB fp16, 28 MB runtime wasm from the CDN), now fetched in parallel with the page | |
| Low memory (deviceMemory <= 2), data saver, 2g/3g | downloaded everything | recording first, one line saying why, button to run anyway |
| WebGPU fails to start, or the runtime cannot load | `could not wake the model` and a toast | recording plus one line saying why; button to try the CPU |

Speed numbers in the README are unchanged. The WASM rows above are from the sandbox CPU, not a phone; steps default to 1 on WASM (steps 1 to 3 remain selectable).
