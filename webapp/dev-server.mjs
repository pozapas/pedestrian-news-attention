// Local test server: serves public/ and the /api handlers against a local
// folder instead of Vercel Blob. Not deployed (see .vercelignore).
//   node dev-server.mjs            -> http://localhost:3217/code/test-token
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
process.env.MOCK_BLOB_DIR ||= path.join(here, ".mockblob");
process.env.CODER_TOKENS ||= JSON.stringify({ "test-token": "T00", "c1-token": "C01", "c2-token": "C02" });
process.env.TICKET_SECRET ||= "local-secret";
process.env.ADMIN_KEY ||= "local-admin";

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css" };
const PORT = Number(process.env.PORT || 3217);

http.createServer(async (req, res) => {
  const u = new URL(req.url, "http://x");
  try {
    const m = u.pathname.match(/^\/api\/(\w+)$/);
    if (m) {
      const mod = await import(`./api/${m[1]}.js`);
      req.query = Object.fromEntries(u.searchParams);
      return await mod.default(req, res);
    }
    let f = u.pathname === "/" || u.pathname.startsWith("/code/") ? "/index.html" : u.pathname;
    const body = await fs.readFile(path.join(here, "public", path.normalize(f)));
    res.setHeader("Content-Type", TYPES[path.extname(f)] || "application/octet-stream");
    res.end(body);
  } catch (e) {
    res.statusCode = e.code === "ENOENT" || e.code === "ERR_MODULE_NOT_FOUND" ? 404 : 500;
    res.end(String(e.message));
  }
}).listen(PORT, () => console.log(`dev server on http://localhost:${PORT}/code/test-token`));
