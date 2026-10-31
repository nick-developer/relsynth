// SPDX-License-Identifier: Apache-2.0
// relsynth web UI: plain JavaScript, no build step, no third-party code.
// Data from the API is only ever inserted with textContent (never innerHTML),
// and the page's Content-Security-Policy allows scripts from this origin only.
"use strict";

const S = { cfg: null, me: null, tenant: null, token: null, dev: null, timers: [] };
const API = "/api/v1";

// ---------------------------------------------------------------- helpers
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of kids.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const $ = (id) => document.getElementById(id);
const keep = (kids) => kids.flat().filter((k) => k !== null && k !== undefined && k !== false);
function fill(el, ...kids) { el.replaceChildren(...keep(kids)); }
function add(el, ...kids) { el.append(...keep(kids)); }
function toast(msg, error) {
  const d = h("div", { class: error ? "error" : "" }, msg);
  $("toast").append(d);
  setTimeout(() => d.remove(), error ? 9000 : 4000);
}
function when(ts) { return ts ? new Date(ts * 1000).toLocaleString() : ""; }
function stateClass(s) {
  return { succeeded: "ok", approved: "ok", failed: "bad", rejected: "bad", cancelled: "muted",
           running: "warn", queued: "muted", pending_approval: "warn" }[s] || "";
}
function pill(s) { return h("span", { class: "pill " + stateClass(s) }, (s || "").replace("_", " ")); }

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (S.token) headers.Authorization = "Bearer " + S.token;
  if (S.dev) headers["X-Relsynth-Dev-User"] = S.dev;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(API + path, { method: opts.method || "GET", headers, body: opts.body });
  if (r.status === 401) { signOut(); throw new Error("signed out"); }
  if (opts.raw) { if (!r.ok) throw new Error(await r.text()); return r; }
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (!r.ok) {
    const d = body && body.detail;
    throw new Error(typeof d === "string" ? d : JSON.stringify(d || body));
  }
  return body;
}
const T = (p) => `/t/${encodeURIComponent(S.tenant)}${p}`;

// ---------------------------------------------------------------- sign-in (OIDC code flow + PKCE)
function b64url(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function randomString(n) { return b64url(crypto.getRandomValues(new Uint8Array(n))); }

async function discovery() {
  const r = await fetch(S.cfg.issuer.replace(/\/$/, "") + "/.well-known/openid-configuration");
  return r.json();
}

async function signIn() {
  const d = await discovery();
  const verifier = randomString(48), state = randomString(16), nonce = randomString(16);
  const challenge = b64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
  sessionStorage.setItem("pkce", JSON.stringify({ verifier, state, nonce }));
  const q = new URLSearchParams({ response_type: "code", client_id: S.cfg.client_id, scope: "openid profile email",
    redirect_uri: location.origin + "/ui/", state, nonce, code_challenge: challenge, code_challenge_method: "S256" });
  location.assign(d.authorization_endpoint + "?" + q);
}

async function finishSignIn() {
  const p = new URLSearchParams(location.search);
  if (!p.get("code")) return false;
  const saved = JSON.parse(sessionStorage.getItem("pkce") || "{}");
  sessionStorage.removeItem("pkce");
  history.replaceState(null, "", "/ui/" + location.hash);
  if (!saved.state || saved.state !== p.get("state")) { toast("sign-in failed: state mismatch", true); return false; }
  const d = await discovery();
  const r = await fetch(d.token_endpoint, { method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ grant_type: "authorization_code", code: p.get("code"), client_id: S.cfg.client_id,
                                redirect_uri: location.origin + "/ui/", code_verifier: saved.verifier }) });
  const tok = await r.json();
  if (!tok.access_token) { toast("sign-in failed: " + (tok.error_description || tok.error), true); return false; }
  S.token = tok.access_token;
  sessionStorage.setItem("token", S.token);       // per tab; cleared when the tab closes
  return true;
}

function signOut() {
  sessionStorage.removeItem("token"); sessionStorage.removeItem("dev");
  S.token = S.dev = null;
  renderLogin();
}

function renderLogin() {
  $("logout").hidden = true; $("who").textContent = "";
  const m = $("main"); fill(m, h("h1", {}, "Sign in"));
  if (S.cfg.auth_mode === "oidc") {
    add(m, h("p", {}, "Sign in with your organisation's account."), h("button", { onclick: signIn }, "Sign in"));
  } else {
    const inp = h("input", { placeholder: "alice; acme:engineer,acme:privacy_officer", size: 50 });
    add(m, h("p", { class: "warn" }, "Development sign-in (never enable in production)."),
      h("div", { class: "row" }, inp, h("button", { onclick: () => {
        S.dev = inp.value; sessionStorage.setItem("dev", S.dev); start(); } }, "Continue")));
  }
}

// ---------------------------------------------------------------- views
function clearTimers() { S.timers.forEach(clearInterval); S.timers = []; }
function can(role) { return (S.me.tenants[S.tenant] || []).some((r) => role.includes(r)); }
const ROLE = { write: ["data_owner", "engineer"], run: ["data_owner", "engineer"], approve: ["data_owner", "privacy_officer"],
               audit: ["data_owner", "privacy_officer", "tenant_admin"], spec: ["data_owner", "privacy_officer", "engineer"] };

async function viewProjects() {
  const m = $("main");
  const rows = await api(T("/projects"));
  const list = h("table", {}, h("tr", {}, h("th", {}, "Project"), h("th", {}, "Description"), h("th", {}, "Created")),
    rows.map((p) => h("tr", {}, h("td", {}, h("a", { href: `#/project/${p.id}` }, p.name)), h("td", {}, p.description),
                       h("td", {}, when(p.created_at)))));
  const name = h("input", { required: true }), desc = h("input");
  const form = can(ROLE.write) ? h("form", { class: "grid card", onsubmit: async (e) => {
    e.preventDefault();
    const p = await api(T("/projects"), { method: "POST", json: { name: name.value, description: desc.value } });
    location.hash = `#/project/${p.id}`; } },
    h("label", {}, "Name", name), h("label", {}, "Description", desc), h("button", {}, "New project")) : null;
  fill(m, h("h1", {}, "Projects"), form, rows.length ? list : h("p", { class: "muted" }, "No projects yet."));
}

async function viewProject(pid) {
  const m = $("main");
  const [p, cons, specs, runs] = await Promise.all([api(T(`/projects/${pid}`)),
    can(ROLE.spec) ? api(T(`/projects/${pid}/connections`)) : [], can(ROLE.spec) ? api(T(`/projects/${pid}/specs`)) : [],
    api(T(`/runs?project_id=${pid}`))]);
  fill(m, h("h1", {}, p.name), h("p", { class: "muted" }, p.description));

  // connections
  add(m, h("h2", {}, "Connections"));
  add(m, h("table", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Kind"), h("th", {}, "Location"),
    h("th", {}, "Secret"), h("th", {}, "")),
    cons.map((c) => h("tr", {}, h("td", {}, c.name), h("td", {}, c.source.kind),
      h("td", {}, c.source.url || c.source.path || ""), h("td", { class: "muted" }, c.secret_ref || "none"),
      h("td", {}, can(ROLE.write) ? h("button", { class: "secondary", onclick: async () => {
        const j = await api(T(`/connections/${c.id}/introspect`), { method: "POST", json: { name: p.name } });
        toast(`Introspection queued (${j.id}); the draft spec appears below when it finishes.`); } }, "Draft spec") : "")))));
  if (can(ROLE.write)) {
    const f = { name: h("input", { required: true }), kind: h("select", {}, ["postgresql", "mssql", "oracle", "mysql",
      "mariadb", "databricks", "fabric", "snowflake", "bigquery", "sqlite", "csv", "parquet"].map((k) => h("option", {}, k))),
      loc: h("input", { placeholder: "postgresql://user@host:5432/db  or a path", required: true }),
      schema: h("input"), secret: h("input", { placeholder: "vault:kv/data/relsynth/acme/pg#password" }),
      cls: h("select", {}, ["", "public", "internal", "confidential", "restricted"].map((k) => h("option", {}, k))) };
    add(m, h("form", { class: "grid card", onsubmit: async (e) => {
      e.preventDefault();
      const fileKind = ["sqlite", "csv", "parquet"].includes(f.kind.value);
      const source = { kind: f.kind.value, [fileKind ? "path" : "url"]: f.loc.value };
      if (f.schema.value) source.schema = f.schema.value;
      await api(T(`/projects/${pid}/connections`), { method: "POST", json: { name: f.name.value, source,
        secret_ref: f.secret.value || null, labels: f.cls.value ? { classification: f.cls.value } : {} } });
      route(); } },
      h("label", {}, "Name", f.name), h("label", {}, "Kind", f.kind), h("label", {}, "URL (no password) or path", f.loc),
      h("label", {}, "Schema", f.schema), h("label", {}, "Secret reference", f.secret), h("label", {}, "Classification", f.cls),
      h("button", {}, "Add connection")));
  }

  // specs
  add(m, h("h2", {}, "Specs"));
  add(m, h("table", {}, h("tr", {}, h("th", {}, "Spec"), h("th", {}, "Version"), h("th", {}, "Review questions"),
    h("th", {}, "Saved")),
    specs.map((s) => h("tr", {}, h("td", {}, h("a", { href: `#/spec/${s.id}` }, s.name)), h("td", { class: "num" }, s.version),
      h("td", {}, (s.review || []).length), h("td", {}, when(s.created_at))))));

  // runs
  add(m, h("h2", {}, "Runs"), runsTable(runs));
  S.timers.push(setInterval(() => { if (runs.some((r) => ["queued", "running"].includes(r.state))) route(); }, 4000));
}

async function viewSpec(sid) {
  const m = $("main");
  const s = await api(T(`/specs/${sid}`));
  const ta = h("textarea", { spellcheck: "false" }); ta.value = s.body;
  const checks = (s.review || []).map((q) => h("input", { type: "checkbox" }));
  const errs = h("ul", { class: "bad" }, (s.errors || []).map((e) => h("li", {}, e)));
  const save = h("button", { onclick: async () => {
    const r = await api(T(`/projects/${s.project_id}/specs`), { method: "POST",
      json: { name: s.name, body: ta.value, connection_id: s.connection_id } });
    toast(`Saved version ${r.version}`); location.hash = `#/spec/${r.id}`; } }, "Save as new version");
  const scale = h("input", { type: "number", step: "0.1", min: "0.01", value: "1" });
  const seed = h("input", { type: "number", placeholder: "random" });
  const fmt = h("select", {}, h("option", {}, "csv"), h("option", {}, "parquet"));
  const runBtn = h("button", { onclick: async () => {
    const body = { spec_id: s.id, scale: Number(scale.value) || null, format: fmt.value };
    if (seed.value) body.seed = Number(seed.value);
    const r = await api(T("/runs"), { method: "POST", json: body });
    location.hash = `#/run/${r.id}`; } }, "Run");
  const gate = () => { runBtn.disabled = checks.some((c) => !c.checked) || (s.errors || []).length > 0; };
  checks.forEach((c) => c.addEventListener("change", gate)); gate();
  fill(m, h("h1", {}, `${s.name} v${s.version}`), h("p", {}, h("a", { href: `#/project/${s.project_id}` }, "Back to project")),
    (s.review || []).length ? h("div", { class: "card" }, h("h2", {}, "Review the introspection"),
      h("p", { class: "muted" }, "Each line is an inference from the data. Check it, edit the spec if it is wrong, and tick it."),
      h("ul", { class: "review" }, s.review.map((q, i) => h("li", {}, h("label", { class: "check" }, checks[i], h("span", {}, q)))))) : null,
    (s.errors || []).length ? h("div", { class: "card" }, h("h2", {}, "Errors"), errs) : null,
    ta, h("div", { class: "row" }, can(ROLE.write) ? save : null),
    can(ROLE.run) ? h("div", { class: "card" }, h("h2", {}, "Run"), h("div", { class: "grid" },
      h("label", {}, "Scale", scale), h("label", {}, "Seed", seed), h("label", {}, "Format", fmt), runBtn)) : null);
}

function runsTable(runs) {
  return h("table", {}, h("tr", {}, h("th", {}, "Run"), h("th", {}, "Kind"), h("th", {}, "State"), h("th", {}, "Release"),
    h("th", {}, "Score"), h("th", {}, "By"), h("th", {}, "Started")),
    runs.map((r) => h("tr", {}, h("td", {}, h("a", { href: `#/run/${r.id}` }, r.id)), h("td", {}, r.kind), h("td", {}, pill(r.state)),
      h("td", {}, r.kind === "run" ? pill(r.release_state) : ""),
      h("td", { class: "num" }, r.summary && r.summary.score !== undefined ? (100 * r.summary.score).toFixed(1) + "%" : ""),
      h("td", {}, r.created_by), h("td", {}, when(r.created_at)))));
}

async function viewRuns() {
  const runs = await api(T("/runs"));
  fill($("main"), h("h1", {}, "Runs"), runsTable(runs));
  S.timers.push(setInterval(() => { if (runs.some((r) => ["queued", "running"].includes(r.state))) route(); }, 4000));
}

async function download(path, name) {
  const r = await api(T(path), { raw: true });
  const url = URL.createObjectURL(await r.blob());
  const a = h("a", { href: url, download: name }); document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

async function viewRun(rid) {
  const m = $("main");
  const r = await api(T(`/runs/${rid}`));
  fill(m, h("h1", {}, `Run ${r.id}`),
    h("p", {}, pill(r.state), " ", r.kind === "run" ? pill(r.release_state) : "", " ",
      h("span", { class: "muted" }, `attempt ${r.attempts} of ${r.max_attempts} · by ${r.created_by} · ${when(r.created_at)}`)));
  if (r.error) add(m, h("div", { class: "card bad" }, r.error));
  if (["queued", "running"].includes(r.state)) {
    if (can(ROLE.run)) add(m, h("button", { class: "danger", onclick: async () => {
      await api(T(`/runs/${rid}/cancel`), { method: "POST" }); route(); } }, "Cancel"));
    S.timers.push(setInterval(route, 3000));
    return;
  }
  if (r.kind === "introspect" && r.summary.spec_id) add(m, h("p", {}, h("a", { href: `#/spec/${r.summary.spec_id}` }, "Open the draft spec")));
  if (r.summary.rows) {
    add(m, h("h2", {}, "Synthetic tables"), h("table", {}, h("tr", {}, h("th", {}, "Table"), h("th", {}, "Rows")),
      Object.entries(r.summary.rows).map(([t, n]) => h("tr", {}, h("td", {}, t), h("td", { class: "num" }, n)))));
  }
  if (r.state === "succeeded" && r.kind === "run") {
    try {
      const rep = await api(T(`/runs/${rid}/report`));
      const sc = rep.scorecard || {};
      add(m, h("h2", {}, `Scorecard: ${sc.passed} of ${sc.total} checks passed`),
        (sc.failed || []).length ? h("table", {}, h("tr", {}, h("th", {}, "Area"), h("th", {}, "Metric"), h("th", {}, "Value"), h("th", {}, "Limit")),
          sc.failed.map((f) => h("tr", {}, h("td", {}, f.area), h("td", {}, f.metric), h("td", { class: "num" }, f.value), h("td", { class: "num" }, f.limit))))
          : h("p", { class: "ok" }, "Every check passed."));
    } catch (e) { add(m, h("p", { class: "muted" }, "No scorecard: " + e.message)); }
    add(m, h("h2", {}, "Release"));
    add(m, h("ul", {}, (r.release.reasons || []).map((x) => h("li", { class: "warn" }, x))));
    add(m, h("table", {}, h("tr", {}, h("th", {}, "Decision"), h("th", {}, "By"), h("th", {}, "Role"), h("th", {}, "Reason"), h("th", {}, "When")),
      r.approvals.map((a) => h("tr", {}, h("td", {}, pill(a.decision === "approve" ? "approved" : "rejected")), h("td", {}, a.approver),
        h("td", {}, a.role), h("td", {}, a.reason), h("td", {}, when(a.created_at))))));
    if (can(ROLE.approve)) {
      const reason = h("input", { placeholder: "reason (required to reject or revoke)", size: 50 });
      const act = (d) => async () => {
        try { await api(T(`/runs/${rid}/approvals`), { method: "POST", json: { decision: d, reason: reason.value } }); route(); }
        catch (e) { toast(e.message, true); } };
      add(m, h("div", { class: "row card" }, reason, h("button", { onclick: act("approve") }, "Approve"),
        h("button", { class: "danger", onclick: act("reject") }, "Reject"), h("button", { class: "secondary", onclick: act("revoke") }, "Revoke")));
    }
    if (r.release.allow) add(m, h("button", { onclick: () => download(`/runs/${rid}/download`, `${rid}.zip`) }, "Download synthetic data"));
  }
}

async function viewCompare() {
  const runs = (await api(T("/runs?state=succeeded"))).filter((r) => r.kind === "run");
  const opt = () => h("select", {}, runs.map((r) => h("option", { value: r.id }, `${r.id} · ${when(r.created_at)}`)));
  const a = opt(), b = opt(); if (runs[1]) b.value = runs[1].id;
  const out = h("div");
  fill($("main"), h("h1", {}, "Compare runs"), h("div", { class: "row card" }, a, b, h("button", { onclick: async () => {
    const c = await api(T(`/compare?a=${a.value}&b=${b.value}`));
    out.replaceChildren(h("p", {}, `A: ${c.a.passed}/${c.a.total} · B: ${c.b.passed}/${c.b.total}`),
      h("table", {}, h("tr", {}, h("th", {}, "Area"), h("th", {}, "Metric"), h("th", {}, "A"), h("th", {}, "B"), h("th", {}, "Limit")),
        c.changed.map((d) => h("tr", {}, h("td", {}, d.area), h("td", {}, d.metric),
          h("td", { class: "num " + (d.a_pass ? "ok" : "bad") }, d.a ?? "–"), h("td", { class: "num " + (d.b_pass ? "ok" : "bad") }, d.b ?? "–"),
          h("td", { class: "num" }, d.limit ?? ""))))); } }, "Compare")), out);
}

async function viewAudit() {
  const [log, v] = await Promise.all([api(T("/audit?limit=500")), api(T("/audit/verify"))]);
  fill($("main"), h("h1", {}, "Audit log"),
    h("p", { class: v.ok ? "ok" : "bad" }, v.ok ? `Hash chain verified: ${v.entries} entries.` : `Chain broken: ${v.error}`),
    h("table", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "When"), h("th", {}, "Who"), h("th", {}, "Action"), h("th", {}, "Target"), h("th", {}, "Outcome")),
      log.slice().reverse().map((e) => h("tr", {}, h("td", { class: "num" }, e.seq), h("td", {}, when(e.ts)), h("td", {}, e.actor),
        h("td", {}, e.action), h("td", {}, e.target), h("td", { class: e.outcome === "ok" ? "" : "bad" }, e.outcome)))));
}

// ---------------------------------------------------------------- router
async function route() {
  clearTimers();
  const [, view, id] = (location.hash || "#/").split("/");
  try {
    if (!S.tenant) { fill($("main"), h("p", {}, "You have no role in any tenant. Ask your administrator.")); return; }
    if (view === "project") await viewProject(id);
    else if (view === "spec") await viewSpec(id);
    else if (view === "run") await viewRun(id);
    else if (view === "runs") await viewRuns();
    else if (view === "compare") await viewCompare();
    else if (view === "audit") await viewAudit();
    else await viewProjects();
  } catch (e) {
    if (e.message !== "signed out") fill($("main"), h("h1", {}, "Something went wrong"), h("p", { class: "bad" }, e.message));
  }
}

async function start() {
  try { S.me = await api("/me"); } catch (e) { renderLogin(); return; }
  $("who").textContent = S.me.name; $("logout").hidden = false;
  const tenants = Object.keys(S.me.tenants).sort();
  S.tenant = localStorage.getItem("tenant");
  if (!tenants.includes(S.tenant)) S.tenant = tenants[0] || null;
  $("tenant").replaceChildren(...tenants.map((t) => h("option", { value: t }, t)));
  if (S.tenant) $("tenant").value = S.tenant;
  route();
}

window.addEventListener("hashchange", route);
document.addEventListener("DOMContentLoaded", async () => {
  $("logout").addEventListener("click", signOut);
  $("tenant").addEventListener("change", (e) => { S.tenant = e.target.value; localStorage.setItem("tenant", S.tenant); location.hash = "#/"; route(); });
  S.cfg = await (await fetch(API + "/ui-config")).json();
  S.token = sessionStorage.getItem("token"); S.dev = sessionStorage.getItem("dev");
  if (S.cfg.auth_mode === "oidc") await finishSignIn();
  if (!S.token && !S.dev) { renderLogin(); return; }
  start();
});
