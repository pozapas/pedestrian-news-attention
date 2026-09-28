// End-to-end API test against the local mock store. Not deployed.
//   node test-api.mjs
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
process.env.MOCK_BLOB_DIR = path.join(here, ".mocktest");
process.env.CODER_TOKENS = JSON.stringify({ tt: "T00", aa: "C01" });
process.env.TICKET_SECRET = "s1";
process.env.ADMIN_KEY = "adm";
await fs.rm(process.env.MOCK_BLOB_DIR, { recursive: true, force: true });

const { ORDER } = await import("./lib/core.js");
const H = {};
for (const n of ["resume", "next", "answer", "finish", "export"]) H[n] = (await import(`./api/${n}.js`)).default;

function call(name, { method = "GET", q = {}, body } = {}) {
  return new Promise((resolve) => {
    const req = { method, query: q, body, headers: { "user-agent": "test" } };
    const res = { statusCode: 200, h: {}, setHeader(k, v) { this.h[k.toLowerCase()] = v; },
      end(s) { const t = String(s ?? ""); let j = null; try { j = JSON.parse(t); } catch {} resolve({ status: this.statusCode, json: j, text: t, h: this.h }); } };
    H[name](req, res);
  });
}

const YES = { is_pedestrian_fatal_crash: "yes", n_pedestrians_killed: "1", victim_age: "not_stated",
  victim_gender: "female", victim_named: "yes", driver_fled: "not_stated", charges_mentioned: "filed", lighting: "dark" };

// invalid token
assert.equal((await call("resume", { q: { t: "nope" } })).status, 403);
assert.equal((await call("next", { q: { t: "nope", i: "0" } })).status, 403);

// fresh start
let r = await call("resume", { q: { t: "aa" } });
assert.deepEqual(r.json, { coder: "C01", total: ORDER.C01.length, answered: 0, next_index: 0, finished: false });

// item out of range
assert.equal((await call("next", { q: { t: "aa", i: "999" } })).status, 400);

// answer item 0
let n = await call("next", { q: { t: "aa", i: "0" } });
assert.equal(n.status, 200);
assert.equal(n.json.item.code, ORDER.C01[0]);
assert.ok(!("hit_run_flag" in n.json.item) && !("machine" in n.json.item));
r = await call("answer", { method: "POST", body: { t: "aa", ticket: n.json.ticket, answers: YES, activity: { copy: 2, blur: 1 } } });
assert.equal(r.status, 200, r.text);

// tampered ticket, ticket from another coder, incomplete answers, bad age
const [b, sg] = n.json.ticket.split(".");
const forged = Buffer.from(JSON.stringify({ c: "C01", i: 1, code: ORDER.C01[1], s: "2020-01-01T00:00:00Z" })).toString("base64url") + "." + sg;
assert.equal((await call("answer", { method: "POST", body: { t: "aa", ticket: forged, answers: YES } })).status, 400);
const nt = await call("next", { q: { t: "tt", i: "0" } });
assert.equal((await call("answer", { method: "POST", body: { t: "aa", ticket: nt.json.ticket, answers: YES } })).status, 400);
const n1 = await call("next", { q: { t: "aa", i: "1" } });
assert.equal((await call("answer", { method: "POST", body: { t: "aa", ticket: n1.json.ticket, answers: { is_pedestrian_fatal_crash: "yes" } } })).status, 400);
assert.equal((await call("answer", { method: "POST", body: { t: "aa", ticket: n1.json.ticket, answers: { ...YES, victim_age: "44.5" } } })).status, 400);
assert.equal((await call("answer", { method: "POST", body: { t: "aa", ticket: n1.json.ticket, answers: { ...YES, victim_age: "150" } } })).status, 400);
// "no" needs nothing else, and extra fields are blanked
r = await call("answer", { method: "POST", body: { t: "aa", ticket: n1.json.ticket, answers: { is_pedestrian_fatal_crash: "no", victim_age: "40", coder_notes: 'stub, "headline" only' } } });
assert.equal(r.status, 200, r.text);

// finish refused before everything is answered
assert.equal((await call("finish", { method: "POST", body: { t: "aa", name: "X", statements: [true, true, true] } })).status, 400);

// revise item 0 (latest wins, revision counted)
await new Promise((z) => setTimeout(z, 5));
const n0 = await call("next", { q: { t: "aa", i: "0" } });
assert.equal((await call("answer", { method: "POST", body: { t: "aa", ticket: n0.json.ticket, answers: { ...YES, lighting: "daylight" }, activity: { copy: "1e9", blur: -5 } } })).status, 200);

r = await call("resume", { q: { t: "aa" } });
assert.equal(r.json.answered, 2);
assert.equal(r.json.next_index, 2);

// answer the rest
for (let i = 2; i < ORDER.C01.length; i++) {
  const x = await call("next", { q: { t: "aa", i: String(i) } });
  const y = await call("answer", { method: "POST", body: { t: "aa", ticket: x.json.ticket, answers: YES } });
  assert.equal(y.status, 200);
}
r = await call("resume", { q: { t: "aa" } });
assert.equal(r.json.next_index, ORDER.C01.length);
assert.equal(r.json.finished, false);

// finish validation then success
assert.equal((await call("finish", { method: "POST", body: { t: "aa", name: "", statements: [true, true, true] } })).status, 400);
assert.equal((await call("finish", { method: "POST", body: { t: "aa", name: "A B", statements: [true, false, true] } })).status, 400);
assert.equal((await call("finish", { method: "POST", body: { t: "aa", name: "A B", statements: [true, true, true] } })).status, 200);
assert.equal((await call("resume", { q: { t: "aa" } })).json.finished, true);

// export
assert.equal((await call("export", { q: { admin: "wrong" } })).status, 403);
const csv = await call("export", { q: { admin: "adm" } });
assert.equal(csv.status, 200);
const lines = csv.text.replace(/^﻿/, "").split("\r\n");
assert.equal(lines.length, 1 + ORDER.C01.length);
assert.ok(lines[0].startsWith("coder_id,index,item_code,is_pedestrian_fatal_crash"));
const row0 = lines.find((l) => l.startsWith("C01,0,"));
assert.ok(row0.includes(",daylight,") && row0.endsWith(",10001,1,1"), row0); // latest wins; per-attempt counts clamped (1e9 -> 9999, -5 -> 0) and summed; 1 revision
assert.ok(lines[0].endsWith("copy_events,focus_losses,revisions"));
const row1 = lines.find((l) => l.startsWith("C01,1,"));
assert.ok(row1.includes('"stub, ""headline"" only"'), row1);                // CSV quoting
assert.ok(row1.includes(",no,,,,,,,,"), row1);                               // blanked fields
assert.ok(!lines.some((l) => l.startsWith("T00,")));                         // test token excluded
const sum = await call("export", { q: { admin: "adm", format: "summary" } });
assert.equal(sum.json.C01.answered, ORDER.C01.length);
assert.equal(sum.json.C01.statement_submitted, true);
assert.equal(sum.json.C01.items_with_copy, 1);
assert.equal(sum.json.C01.items_with_focus_loss, 1);
assert.equal(sum.json.C02.answered, 0);
console.log("summary:", JSON.stringify(sum.json));
console.log("ALL API TESTS PASSED");
await fs.rm(process.env.MOCK_BLOB_DIR, { recursive: true, force: true });
