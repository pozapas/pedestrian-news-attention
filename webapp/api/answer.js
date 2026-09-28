// POST /api/answer {t, ticket, answers} -> stores one answer with server times
import { coderFor, ORDER, verify, validate, clean, answerPath, send, readBody } from "../lib/core.js";
import { putJson } from "../lib/store.js";

// Browser-reported counts of copy events and of the page losing focus while the
// item was open. Clamped, since the client can send anything.
const count = (v) => Math.min(9999, Math.max(0, Math.floor(Number(v) || 0)));

export default async function handler(req, res) {
  if (req.method !== "POST") return send(res, 405, { error: "Use POST." });
  const body = await readBody(req);
  const coder = coderFor(body.t);
  if (!coder) return send(res, 403, { error: "This link is not valid." });
  const tk = verify(body.ticket);
  if (!tk || tk.c !== coder || ORDER[coder][tk.i] !== tk.code)
    return send(res, 400, { error: "This item has expired. Please reload the page." });
  const err = validate(body.answers);
  if (err) return send(res, 400, { error: err });
  const submitted = new Date();
  const record = {
    coder_id: coder, index: tk.i, item_code: tk.code,
    ...clean(body.answers),
    served_at: tk.s, submitted_at: submitted.toISOString(),
    seconds_on_item: Math.round((submitted - new Date(tk.s)) / 1000),
    copy_events: count(body.activity?.copy),
    focus_losses: count(body.activity?.blur),
    user_agent: String(req.headers["user-agent"] || "").slice(0, 200),
  };
  await putJson(answerPath(coder, tk.i, tk.code), record);
  send(res, 200, { ok: true, next_index: tk.i + 1 });
}
