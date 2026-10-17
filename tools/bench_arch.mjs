// Speed of the two candidate architectures in the browser (random weights: speed does not depend on the weights):
// A = DIAMOND size (widths 64x4, 4.47M params), B = widths 32-64-96-128 (6.77M, used). One forward pass = one denoising
// step, including upload and readback, WebGPU with graph capture, 3 interleaved repetitions each.
//   node tools/bench_arch.mjs --out results/browser_arch_bench.json
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => { if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : 'true']); return acc; }, []));
const models = [['A_diamond_64x4', 'benchA_fp16'], ['B_32-64-96-128', 'benchB_fp16'], ['A_diamond_64x4', 'benchA_fp32'], ['B_32-64-96-128', 'benchB_fp32']];
const out = { script: 'tools/bench_arch.mjs', date: new Date().toISOString(), runs: [] };
for (let rep = 0; rep < 3; rep++) {
  for (const [arch, file] of models) {
    const r = JSON.parse(execFileSync('node', [path.join(here, 'bench_run.mjs'), '--page', 'bench_capture.html', '--model', path.join(here, 'bench_models', file + '.onnx'), '--capture', '1'], { encoding: 'utf8' }).replace(/^[^{]*/, ''));
    out.runs.push({ arch, file, nfe_ms_median: r.nfe_ms_median, nfe_ms_p10: r.nfe_ms_p10, error: r.error });
    console.log(arch, file, r.nfe_ms_median?.toFixed(1), r.error ? 'ERROR' : '');
  }
}
out.summary = {};
for (const [arch, file] of models) { const v = out.runs.filter((r) => r.file === file && !r.error).map((r) => r.nfe_ms_median).sort((a, b) => a - b); out.summary[file] = { arch, median_of_runs_ms: v[Math.floor(v.length / 2)], best_ms: v[0] }; }
if (args.out) fs.writeFileSync(path.resolve(args.out), JSON.stringify(out, null, 1));
