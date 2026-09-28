// Coder front end. One article per screen; answers are saved on the server
// after every item, so a coder can close the tab and continue later.

const $ = (s, el = document) => el.querySelector(s);
const app = $("#app");

function tokenFromUrl() {
  const m = location.pathname.match(/^\/code\/([^/]+)/);
  if (m) return decodeURIComponent(m[1]);
  return new URLSearchParams(location.search).get("t") || "";
}
const TOKEN = tokenFromUrl();

// Questions, in the order the coder sees them. Everything after the first
// question is shown only when the first answer is "yes".
const QUESTIONS = [
  { id: "is_pedestrian_fatal_crash", type: "choice",
    label: "Does this article report a crash in which a pedestrian was killed?",
    hint: "A pedestrian is a person on foot, including a person in a wheelchair or on a skateboard or scooter. A cyclist is not a pedestrian. Choose No for injury-only crashes, deaths of drivers or passengers, headline-only stubs, and unrelated articles.",
    options: [["yes", "Yes"], ["no", "No"]] },
  { id: "n_pedestrians_killed", type: "number", min: 1, max: 99,
    label: "How many pedestrians died?",
    hint: "Count only pedestrians who died. Do not count injured people or people who were inside a vehicle." },
  { id: "victim_age", type: "number", min: 0, max: 120,
    label: "How old was the pedestrian who died?",
    hint: "Enter an exact age in years only. “In his 40s” or “elderly” means Not stated. A baby under one year is 0. If several pedestrians died, enter the youngest." },
  { id: "victim_gender", type: "choice",
    label: "What was the gender of the pedestrian who died?",
    hint: "Use what the article says, including he or she. Do not guess from a first name. Choose Mixed only when several pedestrians died and they differ.",
    options: [["male", "Male"], ["female", "Female"], ["mixed", "Mixed"], ["not_stated", "Not stated"]] },
  { id: "victim_named", type: "choice",
    label: "Does the article give the name of the pedestrian who died?",
    hint: "The driver’s name does not count. “Police have not released the name” means No.",
    options: [["yes", "Yes"], ["no", "No"]] },
  { id: "driver_fled", type: "choice",
    label: "Did the driver leave the scene (a hit-and-run)?",
    hint: "Yes if the driver fled or police are searching for the vehicle. No if the driver stayed or cooperated. Not stated if the article says nothing either way.",
    options: [["yes", "Yes"], ["no", "No"], ["not_stated", "Not stated"]] },
  { id: "charges_mentioned", type: "choice",
    label: "What does the article say about criminal charges?",
    hint: "Filed means the article names a charge, for example DUI, manslaughter, or vehicular homicide. Choose the middle option when it says no charges were filed, charges are pending, or the investigation is ongoing. Not mentioned means charges do not come up at all.",
    options: [["filed", "Charges filed"], ["none_or_pending", "None filed or pending"], ["not_stated", "Not mentioned"]] },
  { id: "lighting", type: "choice",
    label: "Was it light or dark when the crash happened?",
    hint: "Use only what the article says about the time or the light. “Just after 10 p.m.” is Dark. “Tuesday morning” with no clock time is Not stated. Do not look up sunrise or sunset.",
    options: [["daylight", "Daylight"], ["dark", "Dark"], ["dawn_or_dusk", "Dawn or dusk"], ["not_stated", "Not stated"]] },
];

const state = { coder: null, total: 0, index: 0, frontier: 0, item: null, ticket: null,
  answers: {}, saved: {}, activity: { copy: 0, blur: 0 } };

// Per-item counts of copying and of leaving the page, disclosed on the
// welcome screen and sent with each answer.
document.addEventListener("copy", () => { if (state.item) state.activity.copy++; });
window.addEventListener("blur", () => { if (state.item) state.activity.blur++; });

// Answers already sent are also kept in this browser, only so that Back can
// show them again after a reload. The server copy is the record.
const LS = "saved:v2:" + TOKEN;
try { state.saved = JSON.parse(localStorage.getItem(LS) || "{}"); } catch { state.saved = {}; }
// The article being worked on is also autosaved in this browser, so answers
// chosen but not yet saved survive a closed tab or a reload.
const DRAFT = "draft:v2:" + TOKEN;
function saveDraft() {
  try { localStorage.setItem(DRAFT, JSON.stringify({ i: state.index, a: state.answers })); } catch { /* storage unavailable */ }
}
function readDraft(i) {
  try { const d = JSON.parse(localStorage.getItem(DRAFT) || "null"); return d && d.i === i ? d.a : null; } catch { return null; }
}
function clearDraft() { try { localStorage.removeItem(DRAFT); } catch { /* storage unavailable */ } }

function remember(i, a) {
  state.saved[i] = a;
  try { localStorage.setItem(LS, JSON.stringify(state.saved)); } catch { /* storage unavailable */ }
}

async function api(path, opts = {}) {
  const r = await fetch(path, { cache: "no-store", ...opts,
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) } });
  let data = {};
  try { data = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw Object.assign(new Error(data.error || "Something went wrong. Please try again."), { status: r.status });
  return data;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setProgress(done) {
  $("#progress").hidden = false;
  $("#guideBtn").hidden = false;
  $("#progressText").textContent = `${done} of ${state.total} done`;
  $("#progressFill").style.width = `${(100 * done) / state.total}%`;
}

function message(title, body) {
  app.innerHTML = `<section class="card narrow"><h1>${esc(title)}</h1><p>${body}</p></section>`;
}

// ---------------------------------------------------------------- screens

function welcome(answered) {
  const left = state.total - answered;
  app.innerHTML = `
  <section class="card narrow">
    <h1>${answered ? "Welcome back" : "Welcome"}</h1>
    ${answered ? `<p>You have finished <strong>${answered}</strong> of ${state.total} articles. You will continue where you stopped.</p>`
      : `<p>You will read <strong>${state.total}</strong> short news articles about traffic crashes and answer up to eight questions about each one. Most articles take two to three minutes.</p>`}
    <ul class="rules">
      <li>Answer only from what the article <strong>says</strong>. When it does not say, choose <strong>Not stated</strong>.</li>
      <li>Do not search for the crash online, and do not use ChatGPT or any other AI tool.</li>
      <li>Each answer is saved when you press <strong>Save and next</strong>. You can close the page at any time and continue later, on this or any other computer, with the same link or access code.</li>
      <li>Short breaks help. Try to work in sessions of about an hour.</li>
      <li>The <strong>Coding guide</strong> button at the top lists the hard cases.</li>
      <li>For the study record, the page notes how long each article is open and whether text is copied or the page is left while an article is open.</li>
    </ul>
    <p class="muted">${left} article${left === 1 ? "" : "s"} left.</p>
    <div class="actions"><button class="primary" id="startBtn">${answered ? "Continue" : "Start"}</button></div>
  </section>`;
  $("#startBtn").onclick = () => load(state.index);
}

async function load(i) {
  app.innerHTML = `<section class="card narrow"><p>Loading article…</p></section>`;
  try {
    const d = await api(`/api/next?t=${encodeURIComponent(TOKEN)}&i=${i}`);
    state.index = d.index; state.item = d.item; state.ticket = d.ticket;
    state.answers = { ...(readDraft(i) || state.saved[i] || {}) };
    state.activity = { copy: 0, blur: 0 };
    renderItem();
    window.scrollTo(0, 0);
  } catch (e) { message("Could not load the article", esc(e.message)); }
}

// Back reaches only the article just before the furthest one reached, so a
// coder can fix a misclick but cannot re-open earlier work.
function canGoBack() { return state.index > 0 && state.index >= state.frontier; }

function renderItem() {
  const it = state.item;
  setProgress(Math.min(state.index, state.total));
  const paras = it.text.split(/\n{2,}/).map((p) => `<p>${esc(p)}</p>`).join("");
  app.innerHTML = `
  <div class="work">
    <article class="card article">
      <div class="meta">Article ${state.index + 1} of ${state.total} · ${esc(it.outlet)} · ${esc(it.published)}</div>
      <div class="text">${paras}</div>
    </article>
    <form class="card form" id="form" autocomplete="off">
      <div id="qs"></div>
      <label class="q notes">
        <span class="label">Note <span class="muted">(optional)</span></span>
        <span class="hint">Only if something was unclear or you had to make a close call.</span>
        <textarea id="notes" rows="2" maxlength="1000"></textarea>
      </label>
      <p class="error" id="err" role="alert"></p>
      <div class="actions">
        <button type="button" class="secondary" id="backBtn" ${canGoBack() ? "" : "disabled"} title="Go back one article to change your answers">Back</button>
        <button type="submit" class="primary" id="saveBtn" disabled>Save and next</button>
      </div>
    </form>
  </div>`;
  $("#notes").value = state.answers.coder_notes || "";
  $("#notes").oninput = (e) => { state.answers.coder_notes = e.target.value; saveDraft(); };
  $("#backBtn").onclick = () => { if (canGoBack()) load(state.index - 1); };
  $("#form").onsubmit = submit;
  renderQuestions();
}

function renderQuestions() {
  const a = state.answers;
  const show = a.is_pedestrian_fatal_crash === "yes" ? QUESTIONS : QUESTIONS.slice(0, 1);
  $("#qs").innerHTML = show.map((q, n) => {
    const head = `<span class="label"><span class="num">${n + 1}</span>${esc(q.label)}</span><span class="hint">${esc(q.hint)}</span>`;
    if (q.type === "choice") {
      return `<fieldset class="q" data-id="${q.id}"><legend class="sr">${esc(q.label)}</legend>${head}
        <div class="opts">${q.options.map(([v, t]) =>
          `<button type="button" class="opt${a[q.id] === v ? " on" : ""}" data-v="${v}" aria-pressed="${a[q.id] === v}">${esc(t)}</button>`).join("")}</div></fieldset>`;
    }
    const ns = a[q.id] === "not_stated";
    const val = ns || a[q.id] === undefined ? "" : a[q.id];
    return `<fieldset class="q" data-id="${q.id}"><legend class="sr">${esc(q.label)}</legend>${head}
      <div class="opts"><input type="number" inputmode="numeric" min="${q.min}" max="${q.max}" step="1" value="${esc(val)}" ${ns ? "disabled" : ""} aria-label="${esc(q.label)}">
      <button type="button" class="opt${ns ? " on" : ""}" data-v="not_stated" aria-pressed="${ns}">Not stated</button></div></fieldset>`;
  }).join("");

  for (const fs of app.querySelectorAll("fieldset.q")) {
    const id = fs.dataset.id;
    const q = QUESTIONS.find((x) => x.id === id);
    for (const b of fs.querySelectorAll("button.opt")) {
      b.onclick = () => {
        if (q.type === "number" && a[id] === "not_stated") delete a[id];
        else a[id] = b.dataset.v;
        if (id === "is_pedestrian_fatal_crash") renderQuestions();
        else refreshField(fs, q);
        check();
      };
    }
    const inp = fs.querySelector("input");
    if (inp) inp.oninput = () => {
      const v = inp.value.trim();
      if (v === "") delete a[id]; else a[id] = v;
      check();
    };
  }
  check();
}

function refreshField(fs, q) {
  const v = state.answers[q.id];
  for (const b of fs.querySelectorAll("button.opt")) {
    const on = b.dataset.v === v;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on);
  }
  const inp = fs.querySelector("input");
  if (inp) {
    inp.disabled = v === "not_stated";
    if (v === "not_stated") inp.value = "";
    else inp.focus();
  }
}

function fieldOk(q, v) {
  if (v === undefined || v === "") return false;
  if (q.type === "choice") return q.options.some(([x]) => x === v);
  if (v === "not_stated") return true;
  if (!/^\d{1,3}$/.test(v)) return false;
  return Number(v) >= q.min && Number(v) <= q.max;
}

function problems() {
  const a = state.answers;
  if (!fieldOk(QUESTIONS[0], a.is_pedestrian_fatal_crash)) return [null];
  if (a.is_pedestrian_fatal_crash === "no") return [];
  const out = [];
  QUESTIONS.slice(1).forEach((q, n) => {
    const v = a[q.id];
    if (q.type === "number" && v !== undefined && v !== "not_stated" && !fieldOk(q, v))
      out.push(`Question ${n + 2}: enter a whole number from ${q.min} to ${q.max}, or choose Not stated.`);
    else if (!fieldOk(q, v)) out.push(null);
  });
  return out;
}

function check() {
  const p = problems();
  saveDraft();
  $("#saveBtn").disabled = p.length > 0;
  // only a mistyped number gets a message; an unanswered question just keeps
  // the button disabled
  $("#err").textContent = p.find((x) => x) || "";
}

async function submit(e) {
  e.preventDefault();
  if (problems().length) return;
  const btn = $("#saveBtn");
  btn.disabled = true; btn.textContent = "Saving…";
  const answers = { ...state.answers };
  try {
    await api("/api/answer", { method: "POST",
      body: JSON.stringify({ t: TOKEN, ticket: state.ticket, answers, activity: state.activity }) });
    remember(state.index, answers);
    clearDraft();
    const next = state.index + 1;
    state.frontier = Math.max(state.frontier, next);
    if (next >= state.total) return resumeAndRoute();
    load(next);
  } catch (err) {
    btn.disabled = false; btn.textContent = "Save and next";
    $("#err").textContent = err.message + " Your answers are still on the screen.";
  }
}

function finishScreen() {
  setProgress(state.total);
  const S = [
    "I read and coded every article myself.",
    "I did not use ChatGPT or any other AI tool, and I did not search for the crashes online.",
    "I did not discuss my answers with the other coder.",
  ];
  app.innerHTML = `
  <section class="card narrow">
    <h1>Last step</h1>
    <p>You have coded all ${state.total} articles. Thank you. Please confirm the statements below and type your name.</p>
    <form id="fin">
      ${S.map((s, i) => `<label class="check"><input type="checkbox" id="s${i}"> <span>${esc(s)}</span></label>`).join("")}
      <label class="q"><span class="label">Your full name</span><input type="text" id="name" maxlength="120"></label>
      <p class="error" id="err" role="alert"></p>
      <div class="actions"><button class="primary" id="finBtn" disabled>Submit</button></div>
    </form>
  </section>`;
  const upd = () => {
    $("#finBtn").disabled = !([0, 1, 2].every((i) => $("#s" + i).checked) && $("#name").value.trim());
  };
  app.querySelectorAll("input").forEach((x) => (x.oninput = upd));
  $("#fin").onsubmit = async (e) => {
    e.preventDefault();
    $("#finBtn").disabled = true;
    try {
      await api("/api/finish", { method: "POST", body: JSON.stringify({
        t: TOKEN, name: $("#name").value.trim(), statements: [0, 1, 2].map((i) => $("#s" + i).checked) }) });
      doneScreen();
    } catch (err) { $("#err").textContent = err.message; upd(); }
  };
}

function doneScreen() {
  setProgress(state.total);
  message("All done", "Your answers and statement have been received. Thank you for your careful work. You can close this page.");
}

async function resumeAndRoute() {
  const r = await api(`/api/resume?t=${encodeURIComponent(TOKEN)}`);
  state.coder = r.coder; state.total = r.total; state.index = r.next_index;
  state.frontier = r.next_index; state.item = null;
  if (r.finished) return doneScreen();
  if (r.next_index >= r.total) return finishScreen();
  // an item skipped earlier (for example after an error) is picked up here
  setProgress(r.answered);
  welcome(r.answered);
}

// ---------------------------------------------------------------- start
$("#guideBtn").onclick = () => $("#guide").showModal();

function signIn(note) {
  app.innerHTML = `
  <section class="card narrow">
    <h1>Sign in</h1>
    ${note ? `<p class="error">${esc(note)}</p>` : ""}
    <p>Enter the access code from your email.</p>
    <form id="login">
      <label class="q"><span class="label">Access code</span>
        <input type="text" id="code" autocomplete="off" autocapitalize="off" spellcheck="false"></label>
      <div class="actions"><button class="primary" id="goBtn" disabled>Continue</button></div>
    </form>
  </section>`;
  let last = "";
  try { last = localStorage.getItem("lastCode") || ""; } catch { /* storage unavailable */ }
  $("#code").value = note ? "" : last;
  const upd = () => { $("#goBtn").disabled = !$("#code").value.trim(); };
  $("#code").oninput = upd; upd();
  $("#login").onsubmit = (e) => {
    e.preventDefault();
    location.href = "/code/" + encodeURIComponent($("#code").value.trim());
  };
}

if (!TOKEN) {
  signIn();
} else {
  resumeAndRoute().then(() => {
    try { localStorage.setItem("lastCode", TOKEN); } catch { /* storage unavailable */ }
  }).catch((e) => {
    if (e.status === 403) signIn("That access code or link is not valid. Please check it and try again.");
    else message("Could not connect", esc(e.message) + " Please reload the page in a minute.");
  });
}
