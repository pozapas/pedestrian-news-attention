// GET /api/export?admin=KEY                 -> CSV, latest answer per coder and item
// GET /api/export?admin=KEY&format=summary  -> JSON progress and pacing per coder
// Add &include_test=1 to include the T00 test link.
import crypto from "node:crypto";
import { FIELDS, ORDER, send, query } from "../lib/core.js";
import { listPrefix, getJson } from "../lib/store.js";

function adminOk(k) {
  const want = process.env.ADMIN_KEY || "";
  if (!want || typeof k !== "string") return false;
  const a = Buffer.from(k), b = Buffer.from(want);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

async function mapLimit(arr, n, fn) {
  const out = new Array(arr.length);
  let i = 0;
  await Promise.all(Array.from({ length: Math.min(n, arr.length) }, async () => {
    while (i < arr.length) { const j = i++; out[j] = await fn(arr[j]); }
  }));
  return out;
}

const COLS = ["coder_id", "index", "item_code", ...FIELDS, "coder_notes",
  "served_at", "submitted_at", "seconds_on_item", "first_seconds_on_item",
  "copy_events", "focus_losses", "revisions"];

function csvCell(v) {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

export default async function handler(req, res) {
  const q = query(req);
  if (!adminOk(q.admin)) return send(res, 403, { error: "Not authorised." });
  const includeTest = q.include_test === "1";

  const rows = (await listPrefix("answers/")).filter(
    (r) => includeTest || !r.pathname.startsWith("answers/T00/"));
  const attempts = new Map();        // coder|index -> [{pathname, ms}]
  for (const r of rows) {
    const m = r.pathname.match(/^answers\/([^/]+)\/(\d{3})_[^_]+_(\d+)\.json$/);
    if (!m) continue;
    const key = m[1] + "|" + Number(m[2]);
    if (!attempts.has(key)) attempts.set(key, []);
    attempts.get(key).push({ pathname: r.pathname, ms: Number(m[3]) });
  }
  const recs = await mapLimit([...attempts.values()], 20, async (list) => {
    list.sort((a, b) => a.ms - b.ms);
    const all = (await Promise.all(list.map((x) => getJson(x.pathname)))).filter(Boolean);
    if (!all.length) return null;
    const rec = all[all.length - 1];              // the latest answer counts
    // a revision re-times the item, so keep the time of the first attempt, and
    // sum the activity counts over every attempt
    return { ...rec, revisions: all.length - 1,
      first_seconds_on_item: all[0].seconds_on_item,
      copy_events: all.reduce((s, r) => s + (r.copy_events || 0), 0),
      focus_losses: all.reduce((s, r) => s + (r.focus_losses || 0), 0) };
  });
  const data = recs.filter(Boolean).sort((a, b) =>
    a.coder_id.localeCompare(b.coder_id) || a.index - b.index);

  if (q.format === "summary") {
    const fin = (await listPrefix("finish/")).map((r) => r.pathname);
    const out = {};
    for (const coder of Object.keys(ORDER)) {
      if (coder === "T00" && !includeTest) continue;
      const d = data.filter((r) => r.coder_id === coder);
      const secs = d.map((r) => r.first_seconds_on_item).sort((a, b) => a - b);
      const med = secs.length ? secs[Math.floor(secs.length / 2)] : null;
      out[coder] = {
        answered: d.length, total: ORDER[coder].length,
        median_seconds_per_item: med,
        items_under_30_seconds: secs.filter((s) => s < 30).length,
        items_with_copy: d.filter((r) => r.copy_events > 0).length,
        items_with_focus_loss: d.filter((r) => r.focus_losses > 0).length,
        first_submitted: d.length ? d.map((r) => r.submitted_at).sort()[0] : null,
        last_submitted: d.length ? d.map((r) => r.submitted_at).sort().slice(-1)[0] : null,
        statement_submitted: fin.some((p) => p.startsWith(`finish/${coder}_`)),
      };
    }
    return send(res, 200, out);
  }

  const lines = [COLS.join(",")];
  for (const r of data) lines.push(COLS.map((c) => csvCell(r[c])).join(","));
  res.statusCode = 200;
  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.setHeader("Content-Disposition", 'attachment; filename="coding_export.csv"');
  res.setHeader("Cache-Control", "no-store");
  res.end("﻿" + lines.join("\r\n"));
}
