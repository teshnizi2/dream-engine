// JavaScript twin of game/raycaster.py. Same spec.json, same arithmetic, same expression order, so
// that frames are bit-identical to the Python reference (see tests/parity.test.mjs).
// Only + - * / floor abs sqrt on float64 are used in the simulation: no sin/cos/atan.
//
// Coordinates: x east (map column), y south (map row). Heading (1,0) looks east; a positive rotation
// turns right. Camera plane = (-dy, dx) * plane. Action = [move, strafe, turn], each 0|1|2:
//   move: none/forward/back   strafe: none/left/right   turn: none/left/right

const frac = (a) => a - Math.floor(a);

// pattern outputs (written into these to avoid allocations in the pixel loop)
let PK = 0, PM = 1, PEM = 0;

function pattern(name, p, u, v) {
  switch (name) {
    case 'blocks': {
      const vr = v * p[0];
      const rr = Math.floor(vr);
      const odd = rr - 2.0 * Math.floor(rr * 0.5);
      const uu = u * p[1] + 0.5 * odd;
      PK = (vr - rr < p[2] || frac(uu) < p[3]) ? 1 : 0; PM = 1; PEM = 0;
      return;
    }
    case 'frame': {
      const t = p[0];
      PK = (u < t || u >= 1.0 - t || v < t || v >= 1.0 - t) ? 1 : 0; PM = 1; PEM = 0;
      return;
    }
    case 'pillar': {
      const t = 2.0 * u - 1.0;
      PM = 0.6 + 0.4 * (1.0 - t * t);
      PK = frac(v * p[0]) < p[1] ? 1 : 0; PEM = 1;
      return;
    }
    case 'windows': {
      const a = frac(u * p[0]);
      PK = (a >= p[1] && a < p[2] && v >= p[3] && v < p[4]) ? 1 : 0; PM = 1; PEM = PK;
      return;
    }
    case 'moss': {
      const w = Math.floor(u * p[0]) * 7.0;
      const r = w - 5.0 * Math.floor(w / 5.0);
      PK = r < 2.0 ? 1 : 0; PM = 0.8 + 0.2 * v; PEM = 0;
      return;
    }
    case 'dado': {
      PK = v >= p[0] ? 1 : 0;
      PM = (v >= p[0] && v < p[0] + p[1]) ? 1.3 : 1.0; PEM = 0;
      return;
    }
    default:
      throw new Error('unknown pattern ' + name);
  }
}

export function stateFromSpecStart(spec) {
  const s = spec.start;
  return { x: s.x, y: s.y, dx: s.dx, dy: s.dy };
}

export class World {
  constructor(spec) {
    this.spec = spec;
    this.N = spec.size; this.W = spec.width; this.H = spec.height;
    const N = this.N;
    this.grid = new Int8Array(N * N);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const ch = spec.map[r][c];
        this.grid[r * N + c] = ch === '.' ? 0 : spec.wall_chars[ch];
      }
    }
    let nid = 0;
    for (const k of Object.keys(spec.walls)) nid = Math.max(nid, +k + 1);
    this.rgb1 = Array.from({ length: nid }, () => [0, 0, 0]);
    this.rgb2 = Array.from({ length: nid }, () => [0, 0, 0]);
    this.emit = new Float64Array(nid);
    this.pat = new Array(nid).fill(null);
    for (const [k, w] of Object.entries(spec.walls)) {
      this.rgb1[+k] = w.rgb; this.rgb2[+k] = w.rgb2; this.emit[+k] = w.emit;
      this.pat[+k] = [w.pattern, w.p];
    }
    this.plane = spec.plane; this.fogK = spec.fog_k; this.fog = spec.fog_color;
    this.sideShade = spec.side_shade;
    this.camx = new Float64Array(this.W);
    for (let i = 0; i < this.W; i++) this.camx[i] = (2.0 * (i + 0.5)) / this.W - 1.0;
    this.bg = this._backgroundRows();
    // per-column scratch
    this._perp = new Float64Array(this.W);
    this._side = new Int8Array(this.W);
    this._wid = new Int8Array(this.W);
    this._u = new Float64Array(this.W);
  }

  _backgroundRows() {
    const H = this.H, half = H / 2.0, sp = this.spec, fog = this.fog;
    const out = new Uint8Array(H * 3);
    for (let y = 0; y < H; y++) {
      const yy = (y + 0.5) - half;
      const s = Math.abs(yy) / half;
      const rd = half / Math.abs(yy);
      const t = rd * this.fogK;
      const f = 1.0 / (1.0 + t * t);
      const A = yy < 0 ? sp.ceiling.horizon : sp.floor.horizon;
      const Bc = yy < 0 ? sp.ceiling.top : sp.floor.bottom;
      for (let c = 0; c < 3; c++) {
        let col = A[c] + (Bc[c] - A[c]) * s;
        col = fog[c] + (col - fog[c]) * f;
        out[y * 3 + c] = Math.min(255, Math.max(0, Math.floor(col + 0.5)));
      }
    }
    return out;
  }

  freeCells() {
    const cells = [];
    for (let r = 0; r < this.N; r++) for (let c = 0; c < this.N; c++) if (this.grid[r * this.N + c] === 0) cells.push([c, r]);
    return cells;
  }

  cell(cx, cy) {
    const N = this.N;
    cx = cx < 0 ? 0 : cx > N - 1 ? N - 1 : cx;
    cy = cy < 0 ? 0 : cy > N - 1 ? N - 1 : cy;
    return this.grid[cy * N + cx];
  }

  // DDA for one ray. Writes perp/side/wid/u into slot i of the scratch arrays.
  _cast(st, camx, i) {
    const x = st.x, y = st.y;
    const px = -st.dy * this.plane, py = st.dx * this.plane;
    const rdx = st.dx + px * camx;
    const rdy = st.dy + py * camx;
    let mapx = Math.floor(x), mapy = Math.floor(y);
    const ddx = Math.abs(1.0 / rdx), ddy = Math.abs(1.0 / rdy);
    const stepx = rdx < 0 ? -1 : 1, stepy = rdy < 0 ? -1 : 1;
    let sdx = rdx < 0 ? (x - mapx) * ddx : (mapx + 1.0 - x) * ddx;
    let sdy = rdy < 0 ? (y - mapy) * ddy : (mapy + 1.0 - y) * ddy;
    let side = 0, wid = 0;
    for (let it = 0; it < 2 * this.N + 2; it++) {
      if (sdx < sdy) { sdx += ddx; mapx += stepx; side = 0; }
      else { sdy += ddy; mapy += stepy; side = 1; }
      const c = this.cell(mapx, mapy);
      if (c > 0) { wid = c; break; }
    }
    let perp = side === 0 ? sdx - ddx : sdy - ddy;
    if (perp < 1e-4) perp = 1e-4;
    let wx = side === 0 ? y + perp * rdy : x + perp * rdx;
    wx = wx - Math.floor(wx);
    const flip = (side === 0 && rdx > 0) || (side === 1 && rdy < 0);
    this._perp[i] = perp; this._side[i] = side; this._wid[i] = wid; this._u[i] = flip ? 1.0 - wx : wx;
  }

  distanceAhead(st) {
    this._cast(st, 0.0, 0);
    return this._perp[0];
  }

  // Renders into `out` (Uint8Array/Uint8ClampedArray, length H*W*3, RGB row-major).
  render(st, out) {
    const H = this.H, W = this.W, half = H / 2.0, fog = this.fog;
    for (let i = 0; i < W; i++) this._cast(st, this.camx[i], i);
    for (let i = 0; i < W; i++) {
      const perp = this._perp[i];
      const lineh = H / perp;
      const top = half - lineh * 0.5;
      const bot = half + lineh * 0.5;
      const wid = this._wid[i], u = this._u[i];
      const shade = this._side[i] === 1 ? this.sideShade : 1.0;
      const t = perp * this.fogK;
      const fw = 1.0 / (1.0 + t * t);
      const rgb1 = this.rgb1[wid], rgb2 = this.rgb2[wid];
      const pat = this.pat[wid];
      const emit = this.emit[wid];
      for (let y = 0; y < H; y++) {
        const o = (y * W + i) * 3;
        const yc = y + 0.5;
        if (yc >= top && yc < bot) {
          const v = (yc - top) / lineh;
          pattern(pat[0], pat[1], u, v);
          const e = emit * PEM;
          const f2 = fw + (1.0 - fw) * e;
          for (let c = 0; c < 3; c++) {
            const base = rgb1[c] + (rgb2[c] - rgb1[c]) * PK;
            let col = base * PM;
            col = col * shade;
            col = fog[c] + (col - fog[c]) * f2;
            out[o + c] = Math.min(255, Math.max(0, Math.floor(col + 0.5)));
          }
        } else {
          out[o] = this.bg[y * 3]; out[o + 1] = this.bg[y * 3 + 1]; out[o + 2] = this.bg[y * 3 + 2];
        }
      }
    }
    return out;
  }

  // One game step. Turn first, then move along the new heading, with per-axis wall collision.
  step(st, a) {
    const sp = this.spec;
    const move = a[0], strafe = a[1], turn = a[2];
    let dx = st.dx, dy = st.dy;
    if (turn !== 0) {
      const ss = sp.turn_sin * (turn === 2 ? 1.0 : -1.0);
      const rdx = st.dx * sp.turn_cos - st.dy * ss;
      const rdy = st.dx * ss + st.dy * sp.turn_cos;
      const norm = Math.sqrt(rdx * rdx + rdy * rdy);
      dx = rdx / norm; dy = rdy / norm;
    }
    const vf = move === 1 ? 1.0 : move === 2 ? -1.0 : 0.0;
    const vs = strafe === 2 ? 1.0 : strafe === 1 ? -1.0 : 0.0;
    const fwd = vf * sp.move_speed;
    const sd = vs * sp.strafe_speed;
    const mx = dx * fwd + (-dy) * sd;
    const my = dy * fwd + dx * sd;
    const R = sp.radius;
    const sx = mx > 0 ? 1.0 : mx < 0 ? -1.0 : 0.0;
    const nx = st.x + mx;
    const x = this.cell(Math.floor(nx + sx * R), Math.floor(st.y)) === 0 ? nx : st.x;
    const sy = my > 0 ? 1.0 : my < 0 ? -1.0 : 0.0;
    const ny = st.y + my;
    const y = this.cell(Math.floor(x), Math.floor(ny + sy * R)) === 0 ? ny : st.y;
    return { x, y, dx, dy };
  }
}
