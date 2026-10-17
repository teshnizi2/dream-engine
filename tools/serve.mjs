// Tiny static file server for local testing and automation (no dependencies).
// Sends COOP/COEP headers so SharedArrayBuffer (multi-threaded wasm) is available, correct MIME types for
// .mjs/.wasm/.onnx, and Range-less full responses. Usage: import { startServer } from './serve.mjs'.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.css': 'text/css', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.bin': 'application/octet-stream',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.mp4': 'video/mp4', '.woff2': 'font/woff2', '.ico': 'image/x-icon',
};

export function startServer(root, port = 0) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, 'http://x');
      let p = path.normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, '');
      let file = path.join(root, p);
      if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
      if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
      if (!fs.existsSync(file)) { res.writeHead(404); return res.end('not found: ' + p); }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'credentialless',
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(file).pipe(res);
    });
    server.listen(port, '127.0.0.1', () => resolve({ server, port: server.address().port, close: () => server.close() }));
  });
}

if (process.argv[1] && process.argv[1].endsWith('serve.mjs')) {
  const root = path.resolve(process.argv[2] || '.');
  const port = +(process.argv[3] || 8765);
  const s = await startServer(root, port);
  console.log(`serving ${root} on http://127.0.0.1:${s.port}/ (COOP/COEP on)`);
}
