// POST /api/finish {t, name, statements:[true,true,true]} -> coder statement
import { coderFor, ORDER, answeredIndices, send, readBody } from "../lib/core.js";
import { putJson } from "../lib/store.js";

export default async function handler(req, res) {
  if (req.method !== "POST") return send(res, 405, { error: "Use POST." });
  const body = await readBody(req);
  const coder = coderFor(body.t);
  if (!coder) return send(res, 403, { error: "This link is not valid." });
  const done = await answeredIndices(coder);
  if (done.size < ORDER[coder].length)
    return send(res, 400, { error: "Some items are not finished yet." });
  const name = String(body.name || "").trim().slice(0, 120);
  const st = Array.isArray(body.statements) ? body.statements : [];
  if (!name || st.length !== 3 || !st.every((x) => x === true))
    return send(res, 400, { error: "Please confirm all three statements and type your name." });
  const now = new Date().toISOString();
  await putJson(`finish/${coder}_${Date.now()}.json`, {
    coder_id: coder, name, statements_confirmed: true, submitted_at: now,
  });
  send(res, 200, { ok: true });
}
