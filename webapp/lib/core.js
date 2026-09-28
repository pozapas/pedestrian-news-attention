// Shared server logic: coder tokens, signed item tickets, answer validation.
import crypto from "node:crypto";
import { ITEMS, ORDER } from "./items.js";
import { listPrefix } from "./store.js";

export { ITEMS, ORDER };

export const FIELDS = [
  "is_pedestrian_fatal_crash", "n_pedestrians_killed", "victim_age",
  "victim_gender", "victim_named", "driver_fled", "charges_mentioned", "lighting",
];

const ALLOWED = {
  is_pedestrian_fatal_crash: ["yes", "no"],
  victim_gender: ["male", "female", "mixed", "not_stated"],
  victim_named: ["yes", "no"],
  driver_fled: ["yes", "no", "not_stated"],
  charges_mentioned: ["filed", "none_or_pending", "not_stated"],
  lighting: ["daylight", "dark", "dawn_or_dusk", "not_stated"],
};

export function coderFor(token) {
  let map = {};
  try { map = JSON.parse(process.env.CODER_TOKENS || "{}"); } catch { map = {}; }
  if (!token || !Object.prototype.hasOwnProperty.call(map, token)) return null;
  const coder = map[token];
  return ORDER[coder] ? coder : null;
}

// The server time an item was shown travels inside a signed ticket, so the
// answer can be timed against a server clock without a second write.
function mac(body) {
  return crypto.createHmac("sha256", process.env.TICKET_SECRET || "")
    .update(body).digest("base64url");
}
export function sign(payload) {
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return body + "." + mac(body);
}
export function verify(ticket) {
  if (!process.env.TICKET_SECRET || typeof ticket !== "string") return null;
  const [body, sig] = ticket.split(".");
  if (!body || !sig) return null;
  const a = Buffer.from(sig), b = Buffer.from(mac(body));
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  try { return JSON.parse(Buffer.from(body, "base64url").toString()); } catch { return null; }
}

function intOrNotStated(v, lo, hi) {
  if (v === "not_stated") return true;
  if (!/^\d{1,3}$/.test(String(v))) return false;
  const n = Number(v);
  return n >= lo && n <= hi;
}

// Returns an error message, or null when the answer set is complete and valid.
export function validate(a) {
  if (!a || typeof a !== "object") return "No answers were sent.";
  if (!ALLOWED.is_pedestrian_fatal_crash.includes(a.is_pedestrian_fatal_crash))
    return "Please answer the first question.";
  if (a.is_pedestrian_fatal_crash === "no") return null;
  if (!intOrNotStated(a.n_pedestrians_killed, 1, 99)) return "Pedestrians killed: enter a number or choose Not stated.";
  if (!intOrNotStated(a.victim_age, 0, 120)) return "Victim age: enter a whole number or choose Not stated.";
  for (const f of ["victim_gender", "victim_named", "driver_fled", "charges_mentioned", "lighting"])
    if (!ALLOWED[f].includes(a[f])) return "Please answer every question.";
  return null;
}

export function clean(a) {
  const out = {};
  for (const f of FIELDS) out[f] = a.is_pedestrian_fatal_crash === "no" && f !== "is_pedestrian_fatal_crash" ? "" : String(a[f] ?? "");
  out.coder_notes = String(a.coder_notes ?? "").slice(0, 1000);
  return out;
}

// answers/<coder>/<index>_<code>_<ms>.json -- append-only, latest wins
export function answerPath(coder, index, code) {
  return `answers/${coder}/${String(index).padStart(3, "0")}_${code}_${Date.now()}.json`;
}

export async function answeredIndices(coder) {
  const rows = await listPrefix(`answers/${coder}/`);
  const s = new Set();
  for (const r of rows) {
    const m = r.pathname.match(/\/(\d{3})_/);
    if (m) s.add(Number(m[1]));
  }
  return s;
}

export function send(res, status, obj) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(obj));
}

export async function readBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string") { try { return JSON.parse(req.body); } catch { return {}; } }
  const chunks = [];
  for await (const c of req) chunks.push(c);
  try { return JSON.parse(Buffer.concat(chunks).toString() || "{}"); } catch { return {}; }
}

export function query(req) {
  if (req.query) return req.query;
  const u = new URL(req.url, "http://x");
  return Object.fromEntries(u.searchParams);
}
