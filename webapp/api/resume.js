// GET /api/resume?t=TOKEN -> where this coder should continue
import { coderFor, ORDER, answeredIndices, send, query } from "../lib/core.js";
import { listPrefix } from "../lib/store.js";

export default async function handler(req, res) {
  const coder = coderFor(query(req).t);
  if (!coder) return send(res, 403, { error: "This link is not valid." });
  const total = ORDER[coder].length;
  const done = await answeredIndices(coder);
  let next = 0;
  while (next < total && done.has(next)) next++;
  const finished = (await listPrefix(`finish/${coder}_`)).length > 0;
  send(res, 200, { coder, total, answered: done.size, next_index: next, finished });
}
