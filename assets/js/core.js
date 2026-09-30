/* core.js — content registry + markdown/math renderer.
 *
 * Content files call:
 *   Course.module({...})           register a module (see README.md for schema)
 *   Course.extra(name, data)       register extra data (skills map, glossary, …)
 *   MD(function(){/* markdown *\/}) write markdown with NO escaping needed
 *
 * Why MD(function(){/* … *\/})? The markdown lives inside a JS comment, so
 * backticks, ${}, and LaTeX backslashes (\frac, \sum) are all preserved
 * exactly as written. The only thing you cannot write inside is the
 * comment terminator (star-slash). Use // comments in code samples.
 */
(function () {
  "use strict";

  // ---------- MD helper ----------
  function MD(fn) {
    if (typeof fn === "string") return dedent(fn);
    var src = fn.toString();
    var a = src.indexOf("/*"), b = src.lastIndexOf("*/");
    if (a < 0 || b < 0) return "";
    return dedent(src.slice(a + 2, b));
  }
  function dedent(s) {
    var lines = s.replace(/\r\n/g, "\n").split("\n");
    while (lines.length && !lines[0].trim()) lines.shift();
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
    var min = Infinity;
    lines.forEach(function (l) {
      if (!l.trim()) return;
      var m = l.match(/^[ \t]*/)[0].length;
      if (m < min) min = m;
    });
    if (!isFinite(min)) min = 0;
    return lines.map(function (l) { return l.slice(min); }).join("\n");
  }
  window.MD = MD;

  // ---------- Registry ----------
  var modules = {};
  var extras = {};
  var Course = {
    modules: modules,
    extras: extras,
    module: function (m) {
      if (!m || !m.id) { console.error("Course.module: missing id", m); return; }
      m.lessons = m.lessons || [];
      m.lessons.forEach(function (l, i) {
        if (!l.id) l.id = "l" + (i + 1);
        l.key = m.id + "/" + l.id;
        l.kind = l.kind || "concept";
      });
      modules[m.id] = m;
    },
    extra: function (name, data) { extras[name] = data; },
  };
  window.Course = Course;

  // ---------- Markdown + math ----------
  var CALLOUTS = {
    TIP: "Tip", NOTE: "Note", WARNING: "Watch out", IMPORTANT: "Important",
    MATH: "Math", GOAL: "End goal", WHY: "Why it matters", PREREQ: "Prerequisite",
    BUILD: "Build it", REAL: "In the real world", INTUITION: "Intuition", CHECK: "Check yourself",
  };

  function renderMarkdown(src) {
    if (!src) return "";
    var stash = [];
    function hold(s) { stash.push(s); return "\u0000" + (stash.length - 1) + "\u0000"; }

    // 1) protect code (fenced + inline) so math/viz regexes never touch it
    src = src.replace(/(^|\n)(```|~~~)[^\n]*\n[\s\S]*?\n\2[ \t]*(?=\n|$)/g, function (m) { return hold(m); });
    src = src.replace(/``[^\n]+?``|`[^`\n]+`/g, function (m) { return hold(m); });

    // 2) viz directives:  ::viz name {"json":"params"}
    src = src.replace(/^::viz[ \t]+([\w-]+)[ \t]*(\{.*\})?[ \t]*$/gm, function (_, name, json) {
      var p = (json || "{}").replace(/'/g, "&#39;");
      return '\n<div class="viz" data-viz="' + name + "\" data-params='" + p + "'></div>\n";
    });

    // 3) math
    var maths = [];
    function mathTok(tex, display) {
      maths.push({ tex: tex, display: display });
      return "MATHTOKEN" + (maths.length - 1) + "X";
    }
    src = src.replace(/\\\$/g, "DOLLARTOKENX");
    src = src.replace(/\$\$([\s\S]+?)\$\$/g, function (_, t) { return mathTok(t, true); });
    src = src.replace(/\$(?=\S)([^$\n]+?)(?<=\S)\$(?!\d)/g, function (_, t) { return mathTok(t, false); });

    // 4) restore code, render markdown
    src = src.replace(/\u0000(\d+)\u0000/g, function (_, i) { return stash[+i]; });
    // (nested placeholders from inline-inside-fence are impossible because fences are held first)
    var html = window.marked.parse(src, { gfm: true, breaks: false });

    // 5) substitute math
    html = html.replace(/MATHTOKEN(\d+)X/g, function (_, i) {
      var m = maths[+i];
      try {
        return window.katex.renderToString(m.tex, { displayMode: m.display, throwOnError: false, strict: false });
      } catch (e) { return "<code>" + m.tex + "</code>"; }
    });
    html = html.replace(/DOLLARTOKENX/g, "$");
    return html;
  }

  // Post-process a rendered container: callouts, code highlight, links, task lists.
  function enhance(el, keyPrefix, store) {
    // Callouts: > [!TIP] Optional title
    el.querySelectorAll("blockquote").forEach(function (bq) {
      var p = bq.querySelector("p");
      if (!p) return;
      var m = p.innerHTML.match(/^\s*\[!(\w+)\]\s*([^\n<]*)/);
      if (!m) return;
      var type = m[1].toUpperCase();
      if (!CALLOUTS[type]) return;
      p.innerHTML = p.innerHTML.slice(m[0].length).replace(/^\s*(<br>)?/, "");
      if (!p.innerHTML.trim()) p.remove();
      var div = document.createElement("div");
      div.className = "callout callout-" + type.toLowerCase();
      var head = document.createElement("div");
      head.className = "callout-title";
      head.textContent = m[2].trim() || CALLOUTS[type];
      div.appendChild(head);
      while (bq.firstChild) div.appendChild(bq.firstChild);
      bq.replaceWith(div);
    });
    // Code highlight
    if (window.hljs) {
      el.querySelectorAll("pre code").forEach(function (c) {
        var cls = c.className || "";
        if (/language-cuda/.test(cls)) c.className = cls.replace("language-cuda", "language-cpp");
        if (/language-(text|plain|txt)/.test(cls)) return;
        try { window.hljs.highlightElement(c); } catch (e) {}
      });
      el.querySelectorAll("pre").forEach(function (pre) {
        if (pre.querySelector(".copy-btn")) return;
        var b = document.createElement("button");
        b.className = "copy-btn"; b.textContent = "copy";
        b.onclick = function () {
          navigator.clipboard && navigator.clipboard.writeText(pre.querySelector("code").innerText);
          b.textContent = "copied"; setTimeout(function () { b.textContent = "copy"; }, 1200);
        };
        pre.appendChild(b);
      });
    }
    // External links open in new tab
    el.querySelectorAll("a[href^='http']").forEach(function (a) { a.target = "_blank"; a.rel = "noopener"; });
    // Task-list checkboxes -> persistent exercises
    if (keyPrefix && store) {
      el.querySelectorAll('li input[type="checkbox"]').forEach(function (cb, i) {
        var k = keyPrefix + "#t" + i;
        cb.disabled = false;
        cb.checked = !!store.get(k);
        cb.closest("li").classList.add("task");
        cb.addEventListener("change", function () { store.set(k, cb.checked); });
      });
    }
    // Tables: wrap for scroll
    el.querySelectorAll("table").forEach(function (t) {
      if (t.parentElement.classList.contains("table-wrap")) return;
      var w = document.createElement("div"); w.className = "table-wrap";
      t.replaceWith(w); w.appendChild(t);
    });
  }

  Course.renderMarkdown = renderMarkdown;
  Course.enhance = enhance;
})();
