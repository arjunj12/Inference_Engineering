/* app.js — loads content, renders views, tracks progress. */
(function () {
  "use strict";
  var CFG = window.COURSE_CONFIG;
  var Course = window.Course;
  var K = window.Viz.K, h = K.h;
  var main = document.getElementById("main");
  var sidebar = document.getElementById("sidebar");

  // ---------------- progress store ----------------
  var STORE_KEY = "aic-progress-v1";
  var state = {};
  try { state = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { state = {}; }
  var store = {
    get: function (k) { return state[k]; },
    set: function (k, v) {
      if (v) state[k] = v; else delete state[k];
      try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (e) {}
      refreshProgress();
    },
  };

  // ---------------- loading ----------------
  function loadScript(src) {
    return new Promise(function (res) {
      var s = document.createElement("script");
      s.src = src;
      s.onload = function () { res(true); };
      s.onerror = function () { console.warn("Could not load", src); res(false); };
      document.body.appendChild(s);
    });
  }
  function allModuleIds() {
    var ids = [];
    CFG.phases.forEach(function (p) { p.modules.forEach(function (m) { ids.push(m); }); });
    return ids;
  }
  var ORDER = allModuleIds();

  Promise.all(
    (CFG.vizFiles || []).map(function (f) { return loadScript("assets/js/viz/" + f + ".js"); })
      .concat((CFG.extraFiles || []).map(function (f) { return loadScript("content/" + f + ".js"); }))
      .concat(ORDER.map(function (id) { return loadScript("content/modules/" + id + ".js"); }))
  ).then(init);

  // ---------------- helpers ----------------
  function mod(id) { return Course.modules[id]; }
  function phaseOf(id) { return CFG.phases.find(function (p) { return p.modules.indexOf(id) >= 0; }); }
  function md(el, src, keyPrefix) {
    el.innerHTML = Course.renderMarkdown(src || "");
    Course.enhance(el, keyPrefix, store);
    window.Viz.mountAll(el);
    return el;
  }
  function mdEl(tag, cls, src, keyPrefix) { return md(h(tag, { class: cls }), src, keyPrefix); }
  function moduleNum(id) { return String(ORDER.indexOf(id)).padStart(2, "0"); }
  function lessonDone(l) { return !!store.get("done:" + l.key); }
  function moduleProgress(m) {
    if (!m) return 0;
    var total = m.lessons.length + (m.challenge ? 1 : 0);
    if (!total) return 0;
    var done = m.lessons.filter(lessonDone).length + (m.challenge && store.get("done:" + m.id + "/challenge") ? 1 : 0);
    return done / total;
  }
  function overallProgress() {
    var ms = ORDER.map(mod).filter(Boolean);
    if (!ms.length) return 0;
    var tot = 0, done = 0;
    ms.forEach(function (m) {
      var t = m.lessons.length + (m.challenge ? 1 : 0);
      tot += t; done += moduleProgress(m) * t;
    });
    return tot ? done / tot : 0;
  }
  function ring(p, size) {
    size = size || 28;
    var r = size / 2 - 3, c = 2 * Math.PI * r;
    var svg = K.s("svg", { width: size, height: size, viewBox: "0 0 " + size + " " + size, class: "ring" },
      K.s("circle", { cx: size / 2, cy: size / 2, r: r, class: "ring-bg" }),
      K.s("circle", {
        cx: size / 2, cy: size / 2, r: r, class: "ring-fg" + (p >= 0.999 ? " full" : ""),
        "stroke-dasharray": c, "stroke-dashoffset": c * (1 - p),
        transform: "rotate(-90 " + size / 2 + " " + size / 2 + ")",
      }));
    return svg;
  }
  var KIND = {
    build: { label: "Build", icon: "🛠" }, concept: { label: "Concept", icon: "💡" },
    math: { label: "Math", icon: "∑" }, lab: { label: "Lab", icon: "🧪" },
    read: { label: "Read / Watch", icon: "📖" }, deep: { label: "Deep dive", icon: "🔭" },
    demo: { label: "Demo", icon: "▶" },
  };
  var HW = { mac: "🍎 Mac", colab: "☁️ Colab GPU", cloud: "☁️ Cloud GPU", "multi-gpu": "☁️☁️ Multi-GPU", browser: "🌐 Browser", any: "💻 Any" };
  function hwBadges(list) {
    return (list || []).map(function (x) { return h("span", { class: "hw", text: HW[x] || x }); });
  }
  function fmtHours(hr) { return hr ? (hr >= 1 ? hr + " h" : Math.round(hr * 60) + " min") : ""; }
  function lessonHours(m) {
    return m.hours || Math.round(m.lessons.reduce(function (a, l) { return a + (l.minutes || 0); }, 0) / 60);
  }

  // ---------------- resources ----------------
  var RTYPE = { video: "▶ Video", paper: "📄 Paper", repo: "⌥ Repo", article: "✎ Article", book: "📚 Book", course: "🎓 Course", docs: "📘 Docs", tool: "🔧 Tool", practice: "🏋 Practice" };
  function resourceList(list) {
    if (!list || !list.length) return null;
    return h("ul", { class: "res-list" }, list.map(function (r) {
      return h("li", null,
        h("span", { class: "res-type t-" + (r.type || "article"), text: RTYPE[r.type] || r.type || "Link" }),
        h("a", { href: r.url, target: "_blank", rel: "noopener", text: r.title }),
        r.note ? h("span", { class: "res-note", text: " — " + r.note }) : null);
    }));
  }

  // ---------------- module rendering ----------------
  function renderLesson(m, l, opts) {
    opts = opts || {};
    var kind = KIND[l.kind] || KIND.concept;
    var cb = h("input", { type: "checkbox", class: "done-cb", title: "Mark lesson complete" });
    cb.checked = lessonDone(l);
    cb.addEventListener("click", function (e) { e.stopPropagation(); });
    cb.addEventListener("change", function () { store.set("done:" + l.key, cb.checked); det.classList.toggle("is-done", cb.checked); });
    var summary = h("summary", { class: "lesson-sum" },
      cb,
      h("span", { class: "kind kind-" + l.kind, text: kind.icon + " " + kind.label }),
      h("span", { class: "lesson-title", text: l.title }),
      h("span", { class: "lesson-meta" },
        l.optional ? h("span", { class: "pill pill-opt", text: "optional" }) : null,
        hwBadges(l.runsOn),
        l.minutes ? h("span", { class: "mins", text: fmtHours(l.minutes / 60) }) : null));
    var body = h("div", { class: "lesson-body prose" });
    var det = h("details", { class: "lesson kind-" + l.kind + (lessonDone(l) ? " is-done" : ""), id: "lesson-" + l.key.replace("/", "--") }, summary, body);
    var rendered = false;
    function renderBody() {
      if (rendered) return;
      rendered = true;
      if (opts.context) body.appendChild(h("div", { class: "lesson-context" }, opts.context));
      body.appendChild(mdEl("div", "md", l.md, "ex:" + l.key));
      var rl = resourceList(l.resources);
      if (rl) body.appendChild(h("div", { class: "lesson-res" }, h("div", { class: "sub-h", text: "Resources for this lesson" }), rl));
      body.appendChild(h("div", { class: "lesson-foot" },
        K.button(lessonDone(l) ? "✓ Completed" : "Mark lesson complete", function (e) {
          var v = !lessonDone(l); store.set("done:" + l.key, v); cb.checked = v;
          det.classList.toggle("is-done", v); e.target.textContent = v ? "✓ Completed" : "Mark lesson complete";
        }, "primary")));
    }
    det.addEventListener("toggle", function () {
      if (det.open) { renderBody(); window.Viz.mountAll(body); }
    });
    if (opts.open) { det.open = true; renderBody(); }
    return det;
  }

  function renderModuleBody(m) {
    var wrap = h("div", { class: "module-body" });
    // 1. End goal first (top-down)
    if (m.goal || m.demo) {
      var g = h("section", { class: "goal-box" }, h("div", { class: "box-label", text: "🎯 End goal — see it working first" }));
      if (m.goal) g.appendChild(mdEl("div", "prose", m.goal, "ex:" + m.id + "/goal"));
      var demoInGoal = m.demo && new RegExp("^::viz[ \\t]+" + m.demo.viz + "(\\s|$)", "m").test(m.goal || "");
      if (m.demo && !demoInGoal) g.appendChild(mdEl("div", "prose", "::viz " + m.demo.viz + " " + JSON.stringify(m.demo.params || {})));
      wrap.appendChild(g);
    }
    if (m.why) {
      wrap.appendChild(h("section", { class: "why-box" }, h("div", { class: "box-label", text: "🌍 Why this matters in real engineering" }), mdEl("div", "prose", m.why)));
    }
    // 2. prerequisites
    if (m.prereqs && m.prereqs.length) {
      var pr = h("section", { class: "prereq-box" }, h("div", { class: "box-label", text: "🧩 Before you start — prerequisites & refreshers" }));
      m.prereqs.forEach(function (p) {
        var d = h("details", { class: "prereq" }, h("summary", null, h("span", { text: p.title }), p.skipIf ? h("span", { class: "skip-if", text: "skip if: " + p.skipIf }) : null));
        var b = h("div", { class: "prose" });
        d.appendChild(b);
        var done = false;
        d.addEventListener("toggle", function () { if (d.open && !done) { done = true; md(b, p.md); } });
        pr.appendChild(d);
      });
      wrap.appendChild(pr);
    }
    // 3. lessons
    var ls = h("section", { class: "lessons" }, h("div", { class: "box-label", text: "🧭 Path — big picture → components → implementation" }));
    m.lessons.forEach(function (l) { ls.appendChild(renderLesson(m, l)); });
    wrap.appendChild(ls);
    // 4. challenge
    if (m.challenge) {
      var ck = "done:" + m.id + "/challenge";
      var cb = h("input", { type: "checkbox", class: "done-cb" });
      cb.checked = !!store.get(ck);
      cb.addEventListener("change", function () { store.set(ck, cb.checked); });
      var c = h("section", { class: "challenge-box" },
        h("div", { class: "box-label" }, cb, h("span", { text: " 🏁 Section challenge — prove you understand it" })),
        h("h4", { text: m.challenge.title }),
        mdEl("div", "prose", m.challenge.md, "ex:" + m.id + "/challenge"));
      if (m.challenge.checklist) {
        c.appendChild(h("div", { class: "sub-h", text: "Done when…" }));
        c.appendChild(mdEl("div", "prose", m.challenge.checklist.map(function (x) { return "- [ ] " + x; }).join("\n"), "ex:" + m.id + "/challenge-list"));
      }
      if (m.challenge.stretch) c.appendChild(mdEl("div", "prose stretch", "> [!TIP] Stretch goal\n> " + m.challenge.stretch.replace(/\n/g, "\n> ")));
      wrap.appendChild(c);
    }
    // 5. how it connects
    if (m.connects) wrap.appendChild(h("section", { class: "connect-box" }, h("div", { class: "box-label", text: "🔗 How this connects to the bigger picture" }), mdEl("div", "prose", m.connects)));
    // 6. interview
    if (m.interview && m.interview.length) {
      var iv = h("details", { class: "interview" }, h("summary", { text: "💼 Interview-style questions you should now be able to answer (" + m.interview.length + ")" }));
      iv.appendChild(mdEl("div", "prose", m.interview.map(function (q, i) { return (i + 1) + ". " + q; }).join("\n")));
      wrap.appendChild(iv);
    }
    // 7. resources
    if (m.resources && m.resources.length) {
      wrap.appendChild(h("section", { class: "res-box" }, h("div", { class: "box-label", text: "📚 Curated resources" }), resourceList(m.resources)));
    }
    return wrap;
  }

  function moduleHeader(id, m, asLink) {
    var num = moduleNum(id);
    return h("div", { class: "mod-head" },
      h("span", { class: "mod-num", text: num }),
      h("div", { class: "mod-titles" },
        h("div", { class: "mod-title" },
          asLink ? h("a", { href: "#/m/" + id, text: m ? m.title : id }) : (m ? m.title : id),
          m && m.level === "optional" ? h("span", { class: "pill pill-opt", text: "optional" }) : null,
          m && m.capstone ? h("span", { class: "pill pill-cap", text: "capstone" }) : null),
        h("div", { class: "mod-tag", text: m ? m.tagline || "" : "Coming soon — content file not found." }),
        m ? h("div", { class: "mod-meta" },
          h("span", { text: "⏱ " + lessonHours(m) + " h" }),
          h("span", { text: "📘 " + m.lessons.length + " lessons" }),
          m.challenge ? h("span", { text: "🏁 challenge" }) : null,
          hwBadges(m.runsOn)) : null),
      h("span", { class: "mod-ring" }, ring(moduleProgress(m), 34)));
  }

  function moduleCard(id) {
    var m = mod(id);
    var card = h("details", { class: "mod-card", id: "card-" + id });
    var sum = h("summary", null, moduleHeader(id, m, false));
    card.appendChild(sum);
    var body = h("div", { class: "mod-card-body" });
    card.appendChild(body);
    var built = false;
    card.addEventListener("toggle", function () {
      if (card.open && !built && m) {
        built = true;
        body.appendChild(h("div", { class: "focus-link" }, h("a", { href: "#/m/" + id, text: "Open in focus mode →" })));
        body.appendChild(renderModuleBody(m));
      }
      if (card.open) window.Viz.mountAll(body);
    });
    return card;
  }

  // ---------------- views ----------------
  function roadmapSVG() {
    var phases = CFG.phases, n = phases.length;
    var W = 980, H = 150, pad = 56, step = (W - 2 * pad) / (n - 1);
    var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "roadmap-svg" });
    svg.appendChild(K.s("line", { x1: pad, x2: W - pad, y1: 60, y2: 60, class: "rm-track" }));
    phases.forEach(function (p, i) {
      var x = pad + i * step;
      var prog = p.modules.map(mod).filter(Boolean);
      var pp = prog.length ? prog.reduce(function (a, m) { return a + moduleProgress(m); }, 0) / p.modules.length : 0;
      var a = K.s("a", { href: "#phase-" + p.id });
      a.appendChild(K.s("circle", { cx: x, cy: 60, r: 17, class: "rm-node" + (pp >= 0.999 ? " done" : pp > 0 ? " partial" : "") }));
      a.appendChild(K.s("text", { x: x, y: 65, class: "rm-num", "text-anchor": "middle", text: i }));
      var lines = [[]];
      p.title.replace(/&/g, "&").split(" ").forEach(function (w) {
        var cur = lines[lines.length - 1];
        if (cur.length && (cur.join(" ") + " " + w).length > 12) lines.push([w]); else cur.push(w);
      });
      lines.forEach(function (ln, li) {
        a.appendChild(K.s("text", { x: x, y: 98 + li * 15, class: "rm-label", "text-anchor": "middle", text: ln.join(" ") }));
      });
      a.appendChild(K.s("text", { x: x, y: 30, class: "rm-weeks", "text-anchor": "middle", text: p.weeks }));
      svg.appendChild(a);
    });
    return svg;
  }

  function viewHome() {
    var totalH = ORDER.map(mod).filter(Boolean).reduce(function (a, m) { return a + lessonHours(m); }, 0);
    var p = overallProgress();
    var hero = h("section", { class: "hero" },
      h("div", { class: "eyebrow", text: "Curriculum · v" + CFG.version }),
      h("h1", { text: CFG.title + " curriculum" }),
      h("p", { class: "lead", text: CFG.subtitle }),
      h("div", { class: "hero-stats" },
        h("div", null, h("b", { text: ORDER.length }), h("span", { text: "modules" })),
        h("div", null, h("b", { text: CFG.phases.length }), h("span", { text: "phases" })),
        h("div", null, h("b", { text: "~" + totalH + " h" }), h("span", { text: "hands-on time" })),
        h("div", null, h("b", { text: "~" + Math.round(totalH / CFG.hoursPerWeek) + " wks" }), h("span", { text: "at " + CFG.hoursPerWeek + " h/week" })),
        h("div", { class: "hero-prog" }, h("b", { text: Math.round(p * 100) + "%" }), h("span", { text: "your progress" }),
          h("div", { class: "bar" }, h("i", { style: { width: p * 100 + "%" } })))));
    var goal = h("section", { class: "goal-box big" }, h("div", { class: "box-label", text: "🎯 The final goal — what you’ll have built" }), mdEl("div", "prose", CFG.finalGoal));
    var how = h("details", { class: "howto" }, h("summary", { text: "How this curriculum works (read once)" }), mdEl("div", "prose", CFG.howToLearn));
    var rm = h("section", { class: "roadmap" }, h("div", { class: "box-label", text: "Roadmap" }), roadmapSVG());
    var list = h("div", { class: "phases" });
    CFG.phases.forEach(function (ph, i) {
      var sec = h("section", { class: "phase", id: "phase-" + ph.id },
        h("div", { class: "phase-head" },
          h("span", { class: "phase-idx", text: "Phase " + i }),
          h("h2", { text: ph.title }),
          h("span", { class: "phase-weeks", text: ph.weeks })),
        h("p", { class: "phase-blurb", text: ph.blurb }));
      ph.modules.forEach(function (id) { sec.appendChild(moduleCard(id)); });
      list.appendChild(sec);
    });
    return [hero, rm, goal, how, list];
  }

  function viewModule(id, lessonId) {
    var m = mod(id);
    if (!m) return [h("div", { class: "empty-state" }, h("h2", { text: "Module not found" }), h("p", { text: id }))];
    var ph = phaseOf(id), idx = ORDER.indexOf(id);
    var crumbs = h("div", { class: "crumbs" }, h("a", { href: "#/", text: "Curriculum" }), " / ", h("a", { href: "#/#phase-" + ph.id, text: ph.title }));
    var head = h("section", { class: "module-page-head" }, moduleHeader(id, m, false));
    var body = renderModuleBody(m);
    var nav = h("div", { class: "pager" },
      idx > 0 ? h("a", { href: "#/m/" + ORDER[idx - 1], class: "prev" }, h("small", { text: "← Previous" }), h("span", { text: (mod(ORDER[idx - 1]) || {}).title || ORDER[idx - 1] })) : h("span"),
      idx < ORDER.length - 1 ? h("a", { href: "#/m/" + ORDER[idx + 1], class: "next" }, h("small", { text: "Next →" }), h("span", { text: (mod(ORDER[idx + 1]) || {}).title || ORDER[idx + 1] })) : h("span"));
    setTimeout(function () {
      var target = lessonId && document.getElementById("lesson-" + id + "--" + lessonId);
      if (target) { target.open = true; target.scrollIntoView({ behavior: "smooth", block: "start" }); }
      window.Viz.mountAll(main);
    }, 30);
    return [crumbs, head, body, nav];
  }

  function viewMath() {
    var out = [h("section", { class: "hero slim" },
      h("div", { class: "eyebrow", text: "Math track" }),
      h("h1", { text: "All the math, in the order you need it" }),
      h("p", { class: "lead", text: "Every math lesson from the curriculum in one place. Each one starts from zero, includes the prerequisite you need, and explains *why* an ML engineer cares. Use it to review or to pre-learn." }))];
    ORDER.forEach(function (id) {
      var m = mod(id);
      if (!m) return;
      var ls = m.lessons.filter(function (l) { return l.kind === "math"; });
      var pre = (m.prereqs || []).filter(function (p) { return p.math; });
      if (!ls.length && !pre.length) return;
      var sec = h("section", { class: "track-sec" }, h("h3", null, h("span", { class: "mod-num sm", text: moduleNum(id) }), h("a", { href: "#/m/" + id, text: m.title })));
      pre.forEach(function (p) {
        var d = h("details", { class: "prereq" }, h("summary", { text: "Refresher: " + p.title }));
        var b = h("div", { class: "prose" }); d.appendChild(b);
        d.addEventListener("toggle", function () { if (d.open && !b.childNodes.length) md(b, p.md); });
        sec.appendChild(d);
      });
      ls.forEach(function (l) { sec.appendChild(renderLesson(m, l)); });
      out.push(sec);
    });
    return out;
  }

  function viewProjects() {
    var out = [h("section", { class: "hero slim" },
      h("div", { class: "eyebrow", text: "Projects" }),
      h("h1", { text: "Everything you’ll build" }),
      h("p", { class: "lead", text: "Every Build/Lab lesson and every section challenge, in order. This list is your portfolio." }))];
    ORDER.forEach(function (id) {
      var m = mod(id);
      if (!m) return;
      var ls = m.lessons.filter(function (l) { return l.kind === "build" || l.kind === "lab"; });
      var sec = h("section", { class: "track-sec" }, h("h3", null, h("span", { class: "mod-num sm", text: moduleNum(id) }), h("a", { href: "#/m/" + id, text: m.title })));
      ls.forEach(function (l) { sec.appendChild(renderLesson(m, l)); });
      if (m.challenge) {
        sec.appendChild(h("div", { class: "challenge-mini" },
          h("b", { text: "🏁 Challenge: " + m.challenge.title }),
          m.challenge.checklist ? h("ul", null, m.challenge.checklist.map(function (c) { return h("li", { html: Course.renderMarkdown(c).replace(/^<p>|<\/p>\s*$/g, "") }); })) : null));
      }
      out.push(sec);
    });
    return out;
  }

  function viewResources() {
    var all = [];
    ORDER.forEach(function (id) {
      var m = mod(id);
      if (!m) return;
      (m.resources || []).forEach(function (r) { all.push({ r: r, m: m, id: id }); });
      m.lessons.forEach(function (l) { (l.resources || []).forEach(function (r) { all.push({ r: r, m: m, id: id, l: l }); }); });
    });
    var seen = {};
    all = all.filter(function (x) { if (seen[x.r.url]) return false; seen[x.r.url] = 1; return true; });
    var types = Object.keys(RTYPE);
    var filter = { type: "", q: "" };
    var listEl = h("div", { class: "res-table" });
    function draw() {
      K.clear(listEl);
      var rows = all.filter(function (x) {
        if (filter.type && x.r.type !== filter.type) return false;
        if (filter.q && (x.r.title + " " + (x.r.note || "") + " " + x.m.title).toLowerCase().indexOf(filter.q) < 0) return false;
        return true;
      });
      listEl.appendChild(h("div", { class: "muted small", text: rows.length + " resources" }));
      rows.forEach(function (x) {
        listEl.appendChild(h("div", { class: "res-row" },
          h("span", { class: "res-type t-" + (x.r.type || "article"), text: RTYPE[x.r.type] || x.r.type }),
          h("div", null,
            h("a", { href: x.r.url, target: "_blank", rel: "noopener", text: x.r.title }),
            x.r.note ? h("div", { class: "res-note", text: x.r.note }) : null),
          h("a", { class: "res-mod", href: "#/m/" + x.id + (x.l ? "/" + x.l.id : ""), text: moduleNum(x.id) + " · " + x.m.title })));
      });
    }
    var controls = h("div", { class: "res-filters" },
      K.select({ label: "Type", options: [{ value: "", label: "All" }].concat(types.map(function (t) { return { value: t, label: RTYPE[t] }; })), onChange: function (v) { filter.type = v; draw(); } }),
      h("input", { type: "search", placeholder: "Filter resources…", oninput: function (e) { filter.q = e.target.value.toLowerCase(); draw(); } }));
    draw();
    return [h("section", { class: "hero slim" }, h("div", { class: "eyebrow", text: "Library" }), h("h1", { text: "Resource library" }),
      h("p", { class: "lead", text: "Every video, paper, repo and article referenced anywhere in the curriculum — de-duplicated and linked back to where it’s used." })), controls, listEl];
  }

  function viewExtra(name) {
    var ex = Course.extras[name];
    if (!ex) return [h("div", { class: "empty-state" }, h("h2", { text: "Not found" }))];
    var out = [h("section", { class: "hero slim" }, h("div", { class: "eyebrow", text: ex.eyebrow || "" }), h("h1", { text: ex.title }), ex.lead ? mdEl("div", "lead", ex.lead) : null)];
    if (ex.render) out = out.concat(ex.render({ h: h, K: K, md: mdEl, ORDER: ORDER, mod: mod, moduleNum: moduleNum, store: store }));
    else if (ex.md) out.push(mdEl("div", "prose", ex.md));
    return out;
  }

  function viewGuide() {
    return [h("section", { class: "hero slim" }, h("div", { class: "eyebrow", text: "Guide" }), h("h1", { text: "Using & editing this curriculum" })),
      mdEl("div", "prose", CFG.howToLearn),
      mdEl("div", "prose", MD(function () {/*
## Editing the curriculum (it’s built to change)

Everything is plain files — no build step. Open `index.html` by double-clicking.

| You want to… | Edit this |
|---|---|
| Reorder modules / phases, change week ranges | `content/course.js` |
| Change a module’s lessons, text, challenge, resources | `content/modules/<id>.js` |
| Add a new module | copy `content/modules/_template.js` → new id, then add the id to a phase in `course.js` |
| Add an interactive chart | add `Viz.register(...)` in any `assets/js/viz/*.js`, then use `::viz name {json}` in a lesson |
| Change the job-skills map / glossary | `content/skills.js`, `content/glossary.js` |
| Styling | `assets/css/style.css` |

Lesson text is Markdown written inside `MD(function(){/* … *\/})`, so you can use backticks, `$math$`, `$$display math$$` and code fences freely. Callouts: `> [!TIP]`, `> [!MATH]`, `> [!WARNING]`, `> [!INTUITION]`, `> [!REAL]`, `> [!PREREQ]`, `> [!CHECK]`. Task lists `- [ ]` become persistent checkboxes. Write a literal dollar sign as `\$`.

Progress is stored in your browser’s localStorage (key `aic-progress-v1`). Use the buttons below to back it up or move it to another browser.
      */})),
      h("div", { class: "guide-actions" },
        K.button("Export progress (JSON)", function () {
          var blob = new Blob([JSON.stringify(state, null, 1)], { type: "application/json" });
          var a = h("a", { href: URL.createObjectURL(blob), download: "ai-course-progress.json" }); a.click();
        }),
        K.button("Import progress", function () {
          var inp = h("input", { type: "file", accept: "application/json" });
          inp.onchange = function () {
            var f = inp.files[0]; if (!f) return;
            f.text().then(function (t) { try { state = JSON.parse(t); localStorage.setItem(STORE_KEY, t); route(); refreshProgress(); } catch (e) { alert("Invalid file"); } });
          };
          inp.click();
        }),
        K.button("Reset all progress", function () { if (confirm("Reset all progress?")) { state = {}; localStorage.removeItem(STORE_KEY); route(); refreshProgress(); } }, "danger"))];
  }

  // ---------------- sidebar & nav ----------------
  var TABS = [
    { href: "#/", label: "Curriculum" }, { href: "#/math", label: "Math track" }, { href: "#/projects", label: "Projects" },
    { href: "#/x/skills", label: "Job skills map" }, { href: "#/resources", label: "Library" }, { href: "#/x/glossary", label: "Glossary" }, { href: "#/guide", label: "Guide" },
  ];
  function renderNav() {
    var nav = document.getElementById("topnav");
    K.clear(nav);
    var cur = location.hash || "#/";
    TABS.forEach(function (t) {
      var active = t.href === "#/" ? (cur === "#/" || cur === "" || cur.indexOf("#/m/") === 0 || cur.indexOf("#/#") === 0) : cur.indexOf(t.href) === 0;
      nav.appendChild(h("a", { href: t.href, class: active ? "active" : "", text: t.label }));
    });
  }
  function renderSidebar() {
    K.clear(sidebar);
    var cur = (location.hash.match(/^#\/m\/([^/]+)/) || [])[1];
    var p = overallProgress();
    sidebar.appendChild(h("div", { class: "side-prog" }, h("div", { class: "side-prog-l" }, h("span", { text: "Progress" }), h("b", { text: Math.round(p * 100) + "%" })), h("div", { class: "bar" }, h("i", { style: { width: p * 100 + "%" } }))));
    CFG.phases.forEach(function (ph, i) {
      var g = h("div", { class: "side-phase" }, h("a", { class: "side-phase-t", href: "#/#phase-" + ph.id }, h("span", { text: i + " · " + ph.title })));
      ph.modules.forEach(function (id) {
        var m = mod(id);
        g.appendChild(h("a", { class: "side-mod" + (id === cur ? " active" : "") + (m ? "" : " missing"), href: "#/m/" + id },
          ring(moduleProgress(m), 16), h("span", { class: "side-num", text: moduleNum(id) }), h("span", { class: "side-t", text: m ? (m.short || m.title) : id })));
      });
      sidebar.appendChild(g);
    });
  }
  function refreshProgress() {
    // cheap: re-render sidebar & rings on home
    renderSidebar();
    document.querySelectorAll(".mod-card").forEach(function (card) {
      var id = card.id.replace("card-", ""), r = card.querySelector(".mod-ring");
      if (r) { K.clear(r); r.appendChild(ring(moduleProgress(mod(id)), 34)); }
    });
    var hp = document.querySelector(".hero-prog");
    if (hp) { var p = overallProgress(); hp.querySelector("b").textContent = Math.round(p * 100) + "%"; hp.querySelector(".bar i").style.width = p * 100 + "%"; }
  }

  // ---------------- search ----------------
  var index = [];
  function buildIndex() {
    ORDER.forEach(function (id) {
      var m = mod(id);
      if (!m) return;
      index.push({ t: m.title, s: m.tagline || "", body: (m.title + " " + (m.tagline || "") + " " + (m.goal || "")).toLowerCase(), href: "#/m/" + id, kind: "module", num: moduleNum(id) });
      m.lessons.forEach(function (l) {
        index.push({ t: l.title, s: m.title, body: (l.title + " " + (l.md || "")).toLowerCase(), href: "#/m/" + id + "/" + l.id, kind: l.kind, num: moduleNum(id), raw: l.md || "" });
      });
    });
    var gl = Course.extras.glossary;
    if (gl && gl.terms) gl.terms.forEach(function (t) { index.push({ t: t.term, s: "Glossary", body: (t.term + " " + t.def).toLowerCase(), href: "#/x/glossary", kind: "glossary", num: "G" }); });
  }
  function setupSearch() {
    var inp = document.getElementById("search"), box = document.getElementById("search-results");
    var sel = 0, results = [];
    function run() {
      var q = inp.value.trim().toLowerCase();
      if (q.length < 2) { box.hidden = true; return; }
      var terms = q.split(/\s+/);
      results = index.map(function (e) {
        var score = 0;
        for (var i = 0; i < terms.length; i++) {
          var t = terms[i];
          if (e.body.indexOf(t) < 0) return null;
          if (e.t.toLowerCase().indexOf(t) >= 0) score += 10;
          score += Math.min(5, e.body.split(t).length - 1);
        }
        if (e.kind === "module") score += 3;
        return { e: e, score: score };
      }).filter(Boolean).sort(function (a, b) { return b.score - a.score; }).slice(0, 12);
      K.clear(box);
      if (!results.length) box.appendChild(h("div", { class: "sr-empty", text: "No matches" }));
      results.forEach(function (r, i) {
        var snippet = "";
        if (r.e.raw) {
          var pos = r.e.body.indexOf(terms[0]);
          snippet = r.e.raw.slice(Math.max(0, pos - 40), pos + 80).replace(/[#*`$>\[\]\\]/g, "").replace(/\s+/g, " ");
        }
        box.appendChild(h("a", { class: "sr" + (i === sel ? " sel" : ""), href: r.e.href, onclick: function () { box.hidden = true; inp.blur(); } },
          h("span", { class: "sr-num", text: r.e.num }),
          h("div", null, h("div", { class: "sr-t", text: r.e.t }), h("div", { class: "sr-s", text: r.e.s + (snippet ? " · …" + snippet + "…" : "") }))));
      });
      box.hidden = false;
    }
    inp.addEventListener("input", function () { sel = 0; run(); });
    inp.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { sel = Math.min(sel + 1, results.length - 1); run(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { sel = Math.max(sel - 1, 0); run(); e.preventDefault(); }
      else if (e.key === "Enter" && results[sel]) { location.hash = results[sel].e.href; box.hidden = true; inp.blur(); }
      else if (e.key === "Escape") { box.hidden = true; inp.blur(); }
    });
    document.addEventListener("click", function (e) { if (!e.target.closest(".search-wrap")) box.hidden = true; });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") { e.preventDefault(); inp.focus(); }
    });
  }

  // ---------------- routing ----------------
  function route() {
    var hash = location.hash || "#/";
    var parts;
    K.clear(main);
    var nodes;
    if ((parts = hash.match(/^#\/m\/([^/]+)(?:\/([^/]+))?/))) nodes = viewModule(parts[1], parts[2]);
    else if (hash === "#/math") nodes = viewMath();
    else if (hash === "#/projects") nodes = viewProjects();
    else if (hash === "#/resources") nodes = viewResources();
    else if (hash === "#/guide") nodes = viewGuide();
    else if ((parts = hash.match(/^#\/x\/(\w+)/))) nodes = viewExtra(parts[1]);
    else nodes = viewHome();
    nodes.forEach(function (n) { if (n) main.appendChild(n); });
    renderNav();
    renderSidebar();
    document.body.classList.remove("side-open");
    var anchor = hash.match(/^#\/#(.+)$/);
    if (anchor) {
      var el = document.getElementById(anchor[1]);
      if (el) setTimeout(function () { el.scrollIntoView({ behavior: "smooth" }); }, 30);
    } else if (!parts || !parts[2]) window.scrollTo(0, 0);
    window.Viz.mountAll(main);
  }
  // In-page phase anchors (#phase-x) from the roadmap: keep on home view.
  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest("a[href^='#phase-']");
    if (a) { e.preventDefault(); location.hash = "#/#" + a.getAttribute("href").slice(1); }
  });

  function init() {
    buildIndex();
    setupSearch();
    window.addEventListener("hashchange", route);
    route();
    document.getElementById("theme-toggle").addEventListener("click", function () {
      var dark = !document.documentElement.classList.contains("dark");
      document.documentElement.classList.toggle("dark", dark);
      try { localStorage.setItem("aic-theme", dark ? "dark" : "light"); } catch (e) {}
      syncHljs();
      window.dispatchEvent(new Event("themechange"));
    });
    document.getElementById("menu-toggle").addEventListener("click", function () { document.body.classList.toggle("side-open"); });
    syncHljs();
  }
  function syncHljs() {
    var dark = document.documentElement.classList.contains("dark");
    document.getElementById("hljs-light").disabled = dark;
    document.getElementById("hljs-dark").disabled = !dark;
  }
})();
