/* _kit.js — tiny toolkit for interactive explainers (no dependencies).
 *
 * Register a visualization:
 *   Viz.register("my-viz", {
 *     title: "Shown in the card header",
 *     desc:  "One-line explanation under the title (optional)",
 *     render: function (card, params, K) { ... }   // card = {controls, stage, readout, root}
 *   });
 * Embed it in any lesson markdown on its own line:
 *   ::viz my-viz {"optional":"json params"}
 *
 * Colors come from CSS classes (.s1…s6 stroke, .f1…f6 fill) so dark mode just works.
 */
(function () {
  "use strict";
  var SVGNS = "http://www.w3.org/2000/svg";

  function h(tag, attrs) {
    var el = document.createElement(tag);
    setAttrs(el, attrs);
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function s(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    setAttrs(el, attrs);
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function setAttrs(el, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === undefined || v === null || v === false) return;
      if (k === "text") el.textContent = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.slice(0, 2) === "on" && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else el.setAttribute(k, v);
    });
  }
  function append(el, c) {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) return c.forEach(function (x) { append(el, x); });
    el.appendChild(typeof c === "object" ? c : document.createTextNode(String(c)));
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

  // ---------- controls ----------
  function slider(o) {
    var val = h("span", { class: "ctl-val" });
    var input = h("input", { type: "range", min: o.min, max: o.max, step: o.step || 1, value: o.value });
    function show() { val.textContent = o.fmt ? o.fmt(+input.value) : input.value; }
    input.addEventListener("input", function () { show(); o.onInput && o.onInput(+input.value); });
    show();
    var wrap = h("label", { class: "ctl ctl-slider" }, h("span", { class: "ctl-label", text: o.label }), input, val);
    wrap.input = input;
    wrap.set = function (v) { input.value = v; show(); };
    wrap.get = function () { return +input.value; };
    return wrap;
  }
  function select(o) {
    var sel = h("select");
    o.options.forEach(function (op) {
      var v = typeof op === "object" ? op.value : op, t = typeof op === "object" ? op.label : op;
      sel.appendChild(h("option", { value: v, text: t }));
    });
    if (o.value !== undefined) sel.value = o.value;
    sel.addEventListener("change", function () { o.onChange && o.onChange(sel.value); });
    var wrap = h("label", { class: "ctl ctl-select" }, h("span", { class: "ctl-label", text: o.label }), sel);
    wrap.input = sel;
    wrap.get = function () { return sel.value; };
    return wrap;
  }
  function button(label, onClick, cls) {
    return h("button", { class: "ctl-btn " + (cls || ""), type: "button", onclick: onClick, text: label });
  }
  function toggle(o) {
    var cb = h("input", { type: "checkbox" });
    cb.checked = !!o.value;
    cb.addEventListener("change", function () { o.onChange && o.onChange(cb.checked); });
    var wrap = h("label", { class: "ctl ctl-toggle" }, cb, h("span", { class: "ctl-label", text: o.label }));
    wrap.get = function () { return cb.checked; };
    wrap.input = cb;
    return wrap;
  }
  function number(o) {
    var input = h("input", { type: "number", value: o.value, min: o.min, max: o.max, step: o.step || "any" });
    input.addEventListener("input", function () { o.onInput && o.onInput(+input.value); });
    var wrap = h("label", { class: "ctl ctl-number" }, h("span", { class: "ctl-label", text: o.label }), input);
    wrap.get = function () { return +input.value; };
    wrap.set = function (v) { input.value = v; };
    return wrap;
  }

  // ---------- scales & axes ----------
  function scale(d0, d1, r0, r1, log) {
    if (log) {
      var l0 = Math.log10(d0), l1 = Math.log10(d1);
      var f = function (v) { return r0 + (Math.log10(Math.max(v, 1e-30)) - l0) / (l1 - l0) * (r1 - r0); };
      f.invert = function (p) { return Math.pow(10, l0 + (p - r0) / (r1 - r0) * (l1 - l0)); };
      return f;
    }
    var g = function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); };
    g.invert = function (p) { return d0 + (p - r0) / (r1 - r0) * (d1 - d0); };
    return g;
  }
  function niceTicks(a, b, n) {
    n = n || 5;
    var span = b - a, step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function logTicks(a, b) {
    var out = [];
    for (var e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) {
      var v = Math.pow(10, e);
      if (v >= a * 0.999 && v <= b * 1.001) out.push(v);
    }
    return out;
  }

  /* chart(o) -> {svg, g, sx, sy, W, H}
   * o: {w,h,pad:{l,r,t,b}, x:[a,b], y:[a,b], xLog, yLog, xLabel, yLabel, xFmt, yFmt, xTicks, yTicks, grid} */
  function chart(o) {
    var W = o.w || 640, H = o.h || 320;
    var p = Object.assign({ l: 56, r: 16, t: 14, b: 42 }, o.pad || {});
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg", preserveAspectRatio: "xMidYMid meet" });
    var sx = scale(o.x[0], o.x[1], p.l, W - p.r, o.xLog);
    var sy = scale(o.y[0], o.y[1], H - p.b, p.t, o.yLog);
    var axes = s("g", { class: "axes" });
    var xt = o.xTicks || (o.xLog ? logTicks(o.x[0], o.x[1]) : niceTicks(o.x[0], o.x[1], o.xN || 6));
    var yt = o.yTicks || (o.yLog ? logTicks(o.y[0], o.y[1]) : niceTicks(o.y[0], o.y[1], o.yN || 5));
    var xf = o.xFmt || fmtShort, yf = o.yFmt || fmtShort;
    xt.forEach(function (v) {
      var x = sx(v);
      if (o.grid !== false) axes.appendChild(s("line", { x1: x, x2: x, y1: p.t, y2: H - p.b, class: "grid" }));
      axes.appendChild(s("text", { x: x, y: H - p.b + 16, class: "tick", "text-anchor": "middle", text: xf(v) }));
    });
    yt.forEach(function (v) {
      var y = sy(v);
      if (o.grid !== false) axes.appendChild(s("line", { x1: p.l, x2: W - p.r, y1: y, y2: y, class: "grid" }));
      axes.appendChild(s("text", { x: p.l - 6, y: y + 4, class: "tick", "text-anchor": "end", text: yf(v) }));
    });
    axes.appendChild(s("line", { x1: p.l, x2: W - p.r, y1: H - p.b, y2: H - p.b, class: "axis" }));
    axes.appendChild(s("line", { x1: p.l, x2: p.l, y1: p.t, y2: H - p.b, class: "axis" }));
    if (o.xLabel) axes.appendChild(s("text", { x: (p.l + W - p.r) / 2, y: H - 6, class: "axis-label", "text-anchor": "middle", text: o.xLabel }));
    if (o.yLabel) axes.appendChild(s("text", { x: 14, y: (p.t + H - p.b) / 2, class: "axis-label", "text-anchor": "middle", transform: "rotate(-90 14 " + (p.t + H - p.b) / 2 + ")", text: o.yLabel }));
    svg.appendChild(axes);
    var g = s("g");
    svg.appendChild(g);
    return { svg: svg, g: g, sx: sx, sy: sy, W: W, H: H, pad: p };
  }
  function path(pts, sx, sy) {
    return pts.map(function (pt, i) { return (i ? "L" : "M") + sx(pt[0]).toFixed(2) + " " + sy(pt[1]).toFixed(2); }).join("");
  }

  // ---------- formatting ----------
  function fmtShort(v) {
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e12) return +(v / 1e12).toPrecision(3) + "T";
    if (a >= 1e9) return +(v / 1e9).toPrecision(3) + "G";
    if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
    if (a >= 1e3) return +(v / 1e3).toPrecision(3) + "k";
    if (a >= 1) return +v.toPrecision(3) + "";
    return +v.toPrecision(2) + "";
  }
  function fmtBytes(b) {
    var u = ["B", "KB", "MB", "GB", "TB"], i = 0;
    while (Math.abs(b) >= 1000 && i < u.length - 1) { b /= 1000; i++; }
    return (+b.toPrecision(3)) + " " + u[i];
  }
  function fmtNum(v, d) { return v.toLocaleString(undefined, { maximumFractionDigits: d === undefined ? 1 : d }); }

  // ---------- math utils ----------
  function softmax(xs, T) {
    T = T || 1;
    var m = Math.max.apply(null, xs), e = xs.map(function (x) { return Math.exp((x - m) / T); });
    var z = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (x) { return x / z; });
  }
  function rng(seed) { // deterministic mulberry32
    var t = seed >>> 0 || 1;
    return function () {
      t += 0x6d2b79f5; var r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(r) { var u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }

  // ---------- registry & mounting ----------
  var registry = {};
  var Viz = {
    registry: registry,
    register: function (name, def) { registry[name] = def; },
    mount: function (el) {
      if (el.dataset.mounted) return;
      var name = el.dataset.viz, def = registry[name];
      el.dataset.mounted = "1";
      if (!def) {
        el.innerHTML = '<div class="viz-missing">Visualization “' + name + '” not found.</div>';
        return;
      }
      var params = {};
      try { params = JSON.parse(el.dataset.params || "{}"); } catch (e) { console.warn("bad viz params", el.dataset.params); }
      var title = params.title || def.title || name;
      var head = h("div", { class: "viz-head" },
        h("span", { class: "viz-badge", text: "Interactive" }),
        h("span", { class: "viz-title", text: title }));
      var desc = (params.desc || def.desc) ? h("div", { class: "viz-desc", html: params.desc || def.desc }) : null;
      var controls = h("div", { class: "viz-controls" });
      var stage = h("div", { class: "viz-stage" });
      var readout = h("div", { class: "viz-readout" });
      var root = h("div", { class: "viz-card" }, head, desc, controls, stage, readout);
      el.appendChild(root);
      try {
        def.render({ root: root, controls: controls, stage: stage, readout: readout }, params, K);
      } catch (e) {
        console.error(e);
        stage.innerHTML = '<div class="viz-missing">Error rendering “' + name + '”: ' + e.message + "</div>";
      }
      if (!controls.childNodes.length) controls.remove();
      if (!readout.childNodes.length && !readout.textContent) readout.classList.add("empty");
    },
    mountAll: function (scope) {
      (scope || document).querySelectorAll(".viz[data-viz]:not([data-mounted])").forEach(function (el) {
        // only mount if visible (inside open <details> chain)
        var d = el.closest("details:not([open])");
        if (!d) Viz.mount(el);
      });
    },
  };

  var K = {
    h: h, s: s, clear: clear, slider: slider, select: select, button: button, toggle: toggle, number: number,
    scale: scale, chart: chart, path: path, niceTicks: niceTicks, logTicks: logTicks,
    fmtShort: fmtShort, fmtBytes: fmtBytes, fmtNum: fmtNum,
    softmax: softmax, rng: rng, randn: randn, clamp: clamp, lerp: lerp,
    // inline KaTeX for readouts
    tex: function (t, display) {
      try { return window.katex.renderToString(t, { displayMode: !!display, throwOnError: false }); } catch (e) { return t; }
    },
  };
  Viz.K = K;
  window.Viz = Viz;
})();
