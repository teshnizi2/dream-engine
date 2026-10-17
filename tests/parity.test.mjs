// Parity test: the JS raycaster must reproduce the Python reference.
//   python tests/make_parity_fixture.py && node --test tests/
// Requirements (BRIEF): >= 99.5% identical pixels on fixed trajectories; max abs diff is reported.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { World } from '../game/raycaster.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const spec = JSON.parse(fs.readFileSync(path.join(here, '../game/spec.json'), 'utf8'));
const fx = JSON.parse(fs.readFileSync(path.join(here, 'fixtures/parity.json'), 'utf8'));
const blob = new Uint8Array(fs.readFileSync(path.join(here, 'fixtures/parity_frames.bin')));
const FS = spec.width * spec.height * 3;
const world = new World(spec);

function compare(a, b) {
  let same = 0, maxd = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d === 0) same++;
    if (d > maxd) maxd = d;
  }
  return { same, total: a.length, maxd };
}

const results = {};
const report = (name, r) => {
  console.log(`  [${name}] identical pixels: ${(100 * r.same / r.total).toFixed(4)}%  max abs diff: ${r.maxd}`);
  results[name] = { frames: r.total / FS, identical_pixel_pct: 100 * r.same / r.total, max_abs_diff: r.maxd };
  fs.mkdirSync(path.join(here, '../results'), { recursive: true });
  fs.writeFileSync(path.join(here, '../results/game_parity.json'),
    JSON.stringify({ script: 'tests/parity.test.mjs + tests/make_parity_fixture.py', fixture_seed: 1234, ...results }, null, 1));
};

test('trajectory states and frames match the Python reference', () => {
  const N = fx.n_traj, T = fx.T;
  let st = [];
  for (let n = 0; n < N; n++) st.push({ x: fx.init.x[n], y: fx.init.y[n], dx: fx.init.dx[n], dy: fx.init.dy[n] });
  const out = new Uint8Array(FS);
  let same = 0, total = 0, maxd = 0, maxState = 0;
  for (let t = 0; t <= T; t++) {
    if (t > 0) st = st.map((s, n) => world.step(s, fx.actions[t - 1][n]));
    for (let n = 0; n < N; n++) {
      for (const k of ['x', 'y', 'dx', 'dy']) maxState = Math.max(maxState, Math.abs(st[n][k] - fx.states[k][t][n]));
      world.render(st[n], out);
      const ref = blob.subarray((t * N + n) * FS, (t * N + n + 1) * FS);
      const r = compare(out, ref);
      same += r.same; total += r.total; maxd = Math.max(maxd, r.maxd);
    }
  }
  report('trajectories', { same, total, maxd });
  console.log(`  max |state diff| after ${T} steps: ${maxState}`);
  results.max_state_diff_after_100_steps = maxState;
  assert.ok(maxState < 1e-12, `state drift ${maxState}`);
  assert.ok(same / total >= 0.995, `identical pixels ${same / total}`);
});

test('random poses (incl. axis-aligned headings and cell-boundary positions) match', () => {
  const base = (fx.T + 1) * fx.n_traj;
  const out = new Uint8Array(FS);
  let same = 0, total = 0, maxd = 0;
  for (let i = 0; i < fx.n_pose; i++) {
    const s = { x: fx.pose.x[i], y: fx.pose.y[i], dx: fx.pose.dx[i], dy: fx.pose.dy[i] };
    world.render(s, out);
    const ref = blob.subarray((base + i) * FS, (base + i + 1) * FS);
    const r = compare(out, ref);
    same += r.same; total += r.total; maxd = Math.max(maxd, r.maxd);
  }
  report('random poses', { same, total, maxd });
  assert.ok(same / total >= 0.995);
});

test('exact-zero ray direction components match', () => {
  const base = (fx.T + 1) * fx.n_traj + fx.n_pose;
  const out = new Uint8Array(FS);
  for (let i = 0; i < 4; i++) {
    const s = { x: fx.edge.x[i], y: fx.edge.y[i], dx: fx.edge.dx[i], dy: fx.edge.dy[i] };
    world.render(s, out);
    const r = compare(out, blob.subarray((base + i) * FS, (base + i + 1) * FS));
    report(`edge ${i}`, r);
    assert.ok(r.same / r.total >= 0.995);
  }
});

test('noop returns the identical state values', () => {
  const s = { x: 8.5, y: 7.5, dx: 0.6, dy: 0.8 };
  assert.deepEqual(world.step(s, [0, 0, 0]), s);
});
