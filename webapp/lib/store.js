// Storage wrapper. On Vercel this uses a PRIVATE Blob store, so nothing written
// here is readable without the server-side token. When MOCK_BLOB_DIR is set it
// writes to a local folder instead, which is how the handlers are tested.
import fs from "node:fs/promises";
import path from "node:path";
import { put, list, get } from "@vercel/blob";

const MOCK = process.env.MOCK_BLOB_DIR;

export async function putJson(pathname, obj) {
  const body = JSON.stringify(obj);
  if (MOCK) {
    const f = path.join(MOCK, pathname);
    await fs.mkdir(path.dirname(f), { recursive: true });
    await fs.writeFile(f, body);
    return;
  }
  await put(pathname, body, {
    access: "private",
    addRandomSuffix: false,
    contentType: "application/json",
  });
}

export async function listPrefix(prefix) {
  if (MOCK) {
    const out = [];
    async function walk(dir) {
      let ents = [];
      try { ents = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }
      for (const e of ents) {
        const f = path.join(dir, e.name);
        if (e.isDirectory()) await walk(f);
        else out.push(path.relative(MOCK, f).split(path.sep).join("/"));
      }
    }
    await walk(MOCK);
    return out.filter((p) => p.startsWith(prefix)).map((pathname) => ({ pathname }));
  }
  const all = [];
  let cursor;
  do {
    const r = await list({ prefix, cursor, limit: 1000 });
    all.push(...r.blobs.map((b) => ({ pathname: b.pathname })));
    cursor = r.hasMore ? r.cursor : undefined;
  } while (cursor);
  return all;
}

export async function getJson(pathname) {
  if (MOCK) return JSON.parse(await fs.readFile(path.join(MOCK, pathname), "utf8"));
  const r = await get(pathname, { access: "private" });
  if (!r || r.statusCode !== 200) return null;
  return JSON.parse(await new Response(r.stream).text());
}
