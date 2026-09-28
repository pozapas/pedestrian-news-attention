// GET /api/next?t=TOKEN&i=INDEX -> one item, with a signed ticket recording
// the server time it was shown
import { coderFor, ORDER, ITEMS, sign, send, query } from "../lib/core.js";

export default async function handler(req, res) {
  const q = query(req);
  const coder = coderFor(q.t);
  if (!coder) return send(res, 403, { error: "This link is not valid." });
  const order = ORDER[coder];
  const i = Number(q.i);
  if (!Number.isInteger(i) || i < 0 || i >= order.length)
    return send(res, 400, { error: "Item not found." });
  const code = order[i];
  const it = ITEMS[code];
  const ticket = sign({ c: coder, i, code, s: new Date().toISOString() });
  send(res, 200, {
    index: i, total: order.length,
    item: { code, published: it.published, outlet: it.outlet, text: it.text },
    ticket,
  });
}
