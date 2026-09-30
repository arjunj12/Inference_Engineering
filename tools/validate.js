#!/usr/bin/env node
/* tools/validate.js — sanity-check all curriculum content without a browser.
 *   node tools/validate.js            # everything
 *   node tools/validate.js m03-build-gpt m04-modern-llms   # only these modules
 * Checks: JS loads, schema fields, unique lesson ids, KaTeX errors, unknown ::viz names,
 * suspicious unescaped dollar signs, resource urls/types.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const root = path.resolve(__dirname, "..");
globalThis.window = globalThis;
globalThis.document = { compatMode: "CSS1Compat", createElement: () => ({}), querySelectorAll: () => [] };

function load(rel) {
  const file = path.join(root, rel);
  if (!fs.existsSync(file)) return false;
  vm.runInThisContext(fs.readFileSync(file, "utf8"), { filename: file });
  return true;
}
["assets/vendor/marked.umd.js", "assets/vendor/katex/katex.min.js", "assets/js/core.js", "assets/js/viz/_kit.js", "content/course.js"].forEach(load);

const CFG = window.COURSE_CONFIG;
const errors = [], warns = [];
(CFG.vizFiles || []).forEach((f) => {
  try { if (!load("assets/js/viz/" + f + ".js")) warns.push("viz file missing: " + f); }
  catch (e) { errors.push("viz file " + f + " failed to load: " + e.message); }
});
(CFG.extraFiles || []).forEach((f) => {
  try { if (!load("content/" + f + ".js")) warns.push("extra file missing: content/" + f + ".js"); }
  catch (e) { errors.push("extra " + f + " failed: " + e.message); }
});
const vizNames = new Set(Object.keys(window.Viz.registry));

const only = process.argv.slice(2);
const ids = [];
CFG.phases.forEach((p) => p.modules.forEach((m) => ids.push(m)));
const targets = only.length ? only : ids;

const KINDS = new Set(["build", "concept", "math", "lab", "read", "deep", "demo"]);
const RTYPES = new Set(["video", "paper", "repo", "article", "book", "course", "docs", "tool", "practice"]);
const HW = new Set(["mac", "colab", "cloud", "multi-gpu", "browser", "any"]);

const render = window.Course.renderMarkdown;
function checkMd(where, src) {
  if (!src) return;
  if (typeof src !== "string") { errors.push(where + ": md is not a string"); return; }
  let html;
  try { html = render(src); } catch (e) { errors.push(where + ": markdown render threw " + e.message); return; }
  const kErr = html.match(/class="katex-error"[^>]*title="([^"]*)"/g);
  if (kErr) kErr.forEach((k) => errors.push(where + ": KaTeX error → " + k.replace(/.*title="/, "").slice(0, 160)));
  if (/MATHTOKEN\d+X|DOLLARTOKENX/.test(html)) errors.push(where + ": leftover math token (unbalanced $?)");
  const re = /^::viz[ \t]+([\w-]+)/gm; let m;
  while ((m = re.exec(src))) if (!vizNames.has(m[1])) warns.push(where + ": unknown viz '" + m[1] + "'");
  const jre = /^::viz[ \t]+[\w-]+[ \t]*(\{.*\})[ \t]*$/gm; let j;
  while ((j = jre.exec(src))) { try { JSON.parse(j[1]); } catch (e) { errors.push(where + ": bad viz JSON " + j[1]); } }
  const stripped = src.replace(/```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`/g, "");
  const cur = stripped.match(/(^|[^\\$])\$\d[\d,.]*\s*(\/|per|an |a |USD|k\b|M\b|B\b|million|billion|hour|hr)/g);
  if (cur) warns.push(where + ": possible unescaped currency " + JSON.stringify(cur.slice(0, 3)) + " (write \\$)");
}

let totalLessons = 0, totalMin = 0;
targets.forEach((id) => {
  const file = "content/modules/" + id + ".js";
  const before = new Set(Object.keys(window.Course.modules));
  try { if (!load(file)) { warns.push("module file missing: " + file); return; } }
  catch (e) { errors.push(file + ": JS error → " + e.message); return; }
  const m = window.Course.modules[id];
  if (!m) {
    const added = Object.keys(window.Course.modules).filter((k) => !before.has(k));
    errors.push(file + ": no module registered with id '" + id + "'" + (added.length ? " (registered: " + added.join(",") + ")" : ""));
    return;
  }
  ["title", "tagline", "goal", "why", "connects"].forEach((f) => { if (!m[f]) warns.push(id + ": missing " + f); });
  if (!m.challenge || !m.challenge.md) errors.push(id + ": missing challenge.md");
  else if (!m.challenge.checklist || !m.challenge.checklist.length) warns.push(id + ": challenge has no checklist");
  if (m.lessons.length < 3) warns.push(id + ": only " + m.lessons.length + " lessons");
  (m.runsOn || []).forEach((h) => { if (!HW.has(h)) warns.push(id + ": unknown runsOn " + h); });
  if (m.demo && !vizNames.has(m.demo.viz)) warns.push(id + ": unknown demo viz '" + m.demo.viz + "'");
  const seen = new Set();
  m.lessons.forEach((l) => {
    totalLessons++; totalMin += l.minutes || 0;
    if (seen.has(l.id)) errors.push(id + ": duplicate lesson id " + l.id);
    seen.add(l.id);
    if (!KINDS.has(l.kind)) errors.push(id + "/" + l.id + ": bad kind " + l.kind);
    if (!l.minutes) warns.push(id + "/" + l.id + ": no minutes");
    if (!l.md || l.md.length < 200) warns.push(id + "/" + l.id + ": very short md (" + (l.md || "").length + " chars)");
    (l.runsOn || []).forEach((h) => { if (!HW.has(h)) warns.push(id + "/" + l.id + ": unknown runsOn " + h); });
    checkMd(id + "/" + l.id, l.md);
    (l.resources || []).forEach((r) => checkRes(id + "/" + l.id, r));
  });
  checkMd(id + ":goal", m.goal); checkMd(id + ":why", m.why); checkMd(id + ":connects", m.connects);
  (m.prereqs || []).forEach((p, i) => { if (!p.title) warns.push(id + ": prereq " + i + " no title"); checkMd(id + ":prereq" + i, p.md); });
  if (m.challenge) { checkMd(id + ":challenge", m.challenge.md); (m.challenge.checklist || []).forEach((c, i) => checkMd(id + ":check" + i, c)); if (m.challenge.stretch) checkMd(id + ":stretch", m.challenge.stretch); }
  (m.interview || []).forEach((q, i) => checkMd(id + ":interview" + i, q));
  (m.resources || []).forEach((r) => checkRes(id, r));
  const chars = JSON.stringify(m).length;
  console.log("✓ " + id.padEnd(26) + String(m.lessons.length).padStart(3) + " lessons  " + String(Math.round(m.lessons.reduce((a, l) => a + (l.minutes || 0), 0) / 60)).padStart(3) + " h  " + Math.round(chars / 1024) + " KB");
});
function checkRes(where, r) {
  if (!r.title || !r.url) errors.push(where + ": resource missing title/url " + JSON.stringify(r));
  else if (!/^https?:\/\//.test(r.url) && !/\.pdf$/.test(r.url)) warns.push(where + ": odd url " + r.url);
  if (r.type && !RTYPES.has(r.type)) warns.push(where + ": bad resource type " + r.type);
}

const ex = window.Course.extras;
if (ex.glossary && ex.glossary.terms) ex.glossary.terms.forEach((t, i) => { if (!t.term || !t.def) errors.push("glossary term " + i + " incomplete"); else checkMd("glossary:" + t.term, t.def); });

console.log("\nviz registered: " + vizNames.size + " | lessons: " + totalLessons + " | ~" + Math.round(totalMin / 60) + " h");
if (warns.length) { console.log("\nWARNINGS (" + warns.length + "):"); warns.forEach((w) => console.log("  ⚠ " + w)); }
if (errors.length) { console.log("\nERRORS (" + errors.length + "):"); errors.forEach((e) => console.log("  ✗ " + e)); process.exit(1); }
console.log("\nNo errors.");
