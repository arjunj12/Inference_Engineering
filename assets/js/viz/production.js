/* production.js — queueing, autoscaling, serving cost, load balancing with prefix caches. */
(function () {
  "use strict";
  var V = window.Viz;
  var uid = 0;

  function fmtT(s) {
    var a = Math.abs(s);
    if (!isFinite(s)) return "∞";
    if (a === 0) return "0";
    if (a < 1e-3) return +(s * 1e6).toPrecision(3) + " µs";
    if (a < 1) return +(s * 1e3).toPrecision(3) + " ms";
    if (a < 120) return +s.toPrecision(3) + " s";
    return +(s / 60).toPrecision(3) + " min";
  }
  function money(v) {
    if (!isFinite(v)) return "—";
    if (v >= 1e6) return "$" + (v / 1e6).toFixed(2) + "M";
    if (v >= 1e4) return "$" + Math.round(v).toLocaleString();
    if (v >= 100) return "$" + v.toFixed(0);
    if (v >= 1) return "$" + v.toFixed(2);
    return "$" + v.toPrecision(2);
  }
  function stats(items) {
    return "<div class='stat-grid'>" + items.map(function (it) {
      return "<div class='stat'><small>" + it[0] + "</small><b" + (it[2] ? " style='color:" + it[2] + "'" : "") + ">" + it[1] + "</b></div>";
    }).join("") + "</div>";
  }
  function txt(K, x, y, t, cls, anchor, extra) {
    var a = { x: x, y: y, class: cls || "lbl-sm", text: t };
    if (anchor) a["text-anchor"] = anchor;
    if (extra) Object.keys(extra).forEach(function (k) { a[k] = extra[k]; });
    return K.s("text", a);
  }
  // Clip a chart's plot group to its plotting area so curves that shoot to infinity don't escape.
  function clipTo(K, ch) {
    var id = "pz-clip-" + (++uid), defs = K.s("defs"), cp = K.s("clipPath", { id: id });
    cp.appendChild(K.s("rect", { x: ch.pad.l, y: ch.pad.t, width: ch.W - ch.pad.l - ch.pad.r, height: ch.H - ch.pad.t - ch.pad.b }));
    defs.appendChild(cp);
    ch.svg.insertBefore(defs, ch.svg.firstChild);
    var g = K.s("g", { "clip-path": "url(#" + id + ")" });
    ch.g.appendChild(g);
    return g;
  }
  function pct(a, q) { if (!a.length) return 0; var s = a.slice().sort(function (x, y) { return x - y; }); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; }
  function mean(a) { return a.length ? a.reduce(function (x, y) { return x + y; }, 0) / a.length : 0; }
  // Run fn every animation frame while the card is attached and on screen.
  function animLoop(c, fn) {
    var visible = true, last = null, raf = 0;
    if (window.IntersectionObserver) {
      new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (visible && !raf) kick(); }).observe(c.root);
    }
    function frame(ts) {
      raf = 0;
      if (!c.root.isConnected) return;
      if (!visible || document.hidden) { last = null; setTimeout(kick, 300); return; }
      var dt = last === null ? 0 : Math.min(0.1, (ts - last) / 1000);
      last = ts;
      fn(dt);
      kick();
    }
    function kick() { if (!raf && c.root.isConnected) raf = requestAnimationFrame(frame); }
    kick();
  }

  // ---------------------------------------------------------------- queueing
  // M/M/c: Erlang C probability of waiting.
  function erlangC(lam, mu, c) {
    var a = lam / mu, rho = a / c;
    if (rho >= 1) return 1;
    var B = 1;
    for (var k = 1; k <= c; k++) B = a * B / (k + a * B);
    return B / (1 - rho * (1 - B));
  }
  function mmc(lam, mu, c) {
    var rho = lam / (c * mu);
    if (rho >= 1) return { rho: rho, C: 1, W: Infinity, Wq: Infinity, p99: Infinity };
    var C = erlangC(lam, mu, c), th = c * mu - lam;
    var Wq = C / th, W = Wq + 1 / mu;
    function tail(t) { // P(sojourn > t)
      var s = Math.abs(th - mu) < 1e-9 * mu ? Math.exp(-mu * t) * (1 + mu * t) : (mu * Math.exp(-th * t) - th * Math.exp(-mu * t)) / (mu - th);
      return (1 - C) * Math.exp(-mu * t) + C * s;
    }
    var lo = 0, hi = 1 / mu;
    while (tail(hi) > 0.01) hi *= 2;
    for (var i = 0; i < 60; i++) { var m = (lo + hi) / 2; if (tail(m) > 0.01) lo = m; else hi = m; }
    return { rho: rho, C: C, W: W, Wq: Wq, p99: hi };
  }

  V.register("queueing", {
    title: "Queueing: why latency explodes near 100% utilisation",
    desc: "Requests arrive at random (rate λ) and each server finishes them at rate μ (M/M/c). Latency stays flat at low load, then shoots up as utilisation ρ = λ/(cμ) → 1 — that's why you never run GPUs at 100%. The strip below is a live simulation; Little's law L = λW holds for it too.",
    render: function (c, p, K) {
      var lam = +p.lambda || 7, mu = +p.mu || 10, ns = +p.servers || 1, speed = 1, paused = false;
      var sl = K.slider({ label: "arrival rate λ", min: 0.5, max: 60, step: 0.5, value: lam, fmt: function (v) { return v.toFixed(1) + " req/s"; }, onInput: function (v) { lam = v; changed(); } });
      c.controls.appendChild(sl);
      c.controls.appendChild(K.slider({ label: "service rate μ (per server)", min: 1, max: 20, step: 0.5, value: mu, fmt: function (v) { return v.toFixed(1) + " req/s (" + fmtT(1 / v) + " each)"; }, onInput: function (v) { mu = v; changed(); } }));
      c.controls.appendChild(K.slider({ label: "servers c", min: 1, max: 8, value: ns, onInput: function (v) { ns = v; changed(); } }));
      c.controls.appendChild(K.select({ label: "sim speed", value: "1", options: [{ value: "0.4", label: "slow" }, { value: "1", label: "normal" }, { value: "4", label: "fast (4×)" }, { value: "20", label: "very fast (20×)" }], onChange: function (v) { speed = +v; } }));
      var pb = K.button("⏸ Pause", function () { paused = !paused; pb.textContent = paused ? "▶ Resume" : "⏸ Pause"; });
      c.controls.appendChild(pb);
      c.controls.appendChild(K.button("Reset sim", function () { resetSim(); }));

      var chartBox = K.h("div"), simBox = K.h("div"), simOut = K.h("div", { class: "muted", style: "font-size:13px;margin-top:4px" });
      K.clear(c.stage).appendChild(chartBox);
      c.stage.appendChild(simBox);
      c.stage.appendChild(simOut);

      function drawChart() {
        var cur = mmc(lam, mu, ns), S = 1 / mu;
        var ymax = Math.max(8 * S, Math.min(60 * S, isFinite(cur.p99) ? cur.p99 * 1.25 : 0));
        var ch = K.chart({ w: 720, h: 270, x: [0, 1], y: [0, ymax * 1000], xLabel: "utilisation ρ = λ / (c·μ)", yLabel: "latency (ms)", xFmt: function (v) { return Math.round(v * 100) + "%"; }, pad: { l: 60, t: 16 } });
        var g = clipTo(K, ch), sy = function (v) { return ch.sy(Math.min(v, ymax * 2) * 1000); };
        function curve(cc, key) {
          var d = "";
          for (var r = 0; r <= 0.9985; r += 0.0025) { var q = mmc(r * cc * mu, mu, cc); d += (d ? "L" : "M") + ch.sx(r).toFixed(1) + " " + sy(q[key]).toFixed(1); }
          return d;
        }
        g.appendChild(K.s("rect", { x: ch.sx(0.8), y: ch.pad.t, width: ch.sx(1) - ch.sx(0.8), height: ch.H - ch.pad.t - ch.pad.b, style: "fill:var(--red);opacity:0.07" }));
        g.appendChild(txt(K, ch.sx(0.9), ch.pad.t + 14, "danger zone", "lbl-sm", "middle", { style: "fill:var(--red)" }));
        g.appendChild(K.s("line", { x1: ch.pad.l, x2: ch.W - ch.pad.r, y1: sy(S), y2: sy(S), class: "ln-thin smuted dash" }));
        g.appendChild(txt(K, ch.pad.l + 6, sy(S) - 5, "service time 1/μ = " + fmtT(S), "lbl-sm"));
        if (ns > 1) {
          g.appendChild(K.s("path", { d: curve(1, "W"), class: "ln-thin smuted" }));
          var rl = Math.min(0.97, 1 - S / (0.8 * ymax));
          g.appendChild(txt(K, ch.sx(rl) - 6, sy(mmc(rl * mu, mu, 1).W), "one server (c = 1)", "lbl-sm", "end", { style: "fill:var(--muted)" }));
        }
        g.appendChild(K.s("path", { d: curve(ns, "p99"), class: "ln s2 dash" }));
        g.appendChild(K.s("path", { d: curve(ns, "W"), class: "ln s1" }));
        if (cur.rho < 1) {
          [["p99", "f2"], ["W", "f1"]].forEach(function (kc) {
            var y = sy(cur[kc[0]]);
            g.appendChild(K.s("line", { x1: ch.sx(cur.rho), x2: ch.sx(cur.rho), y1: ch.H - ch.pad.b, y2: y, class: "ln-thin sfg dash" }));
            g.appendChild(K.s("circle", { cx: ch.sx(cur.rho), cy: y, r: 6, class: kc[1], style: "stroke:var(--bg);stroke-width:2" }));
          });
        } else {
          g.appendChild(txt(K, ch.W - ch.pad.r - 8, ch.pad.t + 34, "ρ = " + cur.rho.toFixed(2) + " ≥ 1: unstable — the queue grows forever", "lbl", "end", { style: "fill:var(--red)" }));
        }
        var lx = ch.pad.l + 14, ly = ch.pad.t + 40;
        ch.g.appendChild(K.s("rect", { x: lx - 6, y: ly - 14, width: 206, height: 44, rx: 4, class: "fbg", opacity: 0.9 }));
        ch.g.appendChild(K.s("line", { x1: lx, x2: lx + 22, y1: ly - 4, y2: ly - 4, class: "ln s1" }));
        ch.g.appendChild(txt(K, lx + 28, ly, "mean latency W (c = " + ns + ")", "lbl-sm"));
        ch.g.appendChild(K.s("line", { x1: lx, x2: lx + 22, y1: ly + 16, y2: ly + 16, class: "ln s2 dash" }));
        ch.g.appendChild(txt(K, lx + 28, ly + 20, "p99 latency", "lbl-sm"));
        K.clear(chartBox).appendChild(ch.svg);

        var ok = cur.rho < 1;
        c.readout.innerHTML = stats([
          ["utilisation ρ", (cur.rho * 100).toFixed(0) + "%", cur.rho >= 0.9 ? "var(--red)" : cur.rho >= 0.75 ? "var(--amber)" : "var(--green)"],
          ["P(wait) — Erlang C", ok ? (cur.C * 100).toFixed(0) + "%" : "100%"],
          ["mean latency W", fmtT(cur.W)],
          ["mean queue wait Wq", fmtT(cur.Wq)],
          ["p99 latency", fmtT(cur.p99)],
          ["in system L = λW", ok ? (lam * cur.W).toFixed(2) : "∞"],
          ["waiting Lq = λWq", ok ? (lam * cur.Wq).toFixed(2) : "∞"],
        ]) + "<div style='margin-top:6px'>" + K.tex("W = \\frac{1}{\\mu} + \\frac{C(c,\\lambda/\\mu)}{c\\mu-\\lambda}, \\qquad L = \\lambda W") +
          " — for one server this is W = 1/(μ−λ): at ρ = 50% latency is 2× the service time, at 90% it's 10×, at 99% 100×. " +
          (ns > 1 ? "Pooling " + ns + " servers behind one queue (grey line = a single server at the same ρ) keeps latency low to a much higher utilisation — one reason big shared deployments are cheaper. " : "Try c > 1: pooling servers behind one queue tolerates higher utilisation. ") +
          "Real LLM servers batch, so μ itself grows with load up to a limit — but the shape (a knee, then a wall) is the same.</div>";
      }

      // ---- live discrete-event simulation (sim time in seconds; real time scaled so one service ≈ 0.8 s at "normal")
      var r = K.rng(7), T, nextA, queue, servers, area, nIn, arrivals, done, sumW, t0, dropped;
      function expo(rate) { return -Math.log(1 - r()) / rate; }
      function resetSim() {
        T = 0; t0 = 0; nextA = expo(lam); queue = []; servers = []; area = 0; nIn = 0; arrivals = 0; done = 0; sumW = 0; dropped = 0;
        for (var i = 0; i < ns; i++) servers.push(null);
      }
      function advance(to) {
        var guard = 0;
        while (guard++ < 20000) {
          var nd = Infinity, si = -1;
          servers.forEach(function (s, i) { if (s && s.end < nd) { nd = s.end; si = i; } });
          var ne = Math.min(nextA, nd);
          if (ne > to) break;
          area += nIn * (ne - T); T = ne;
          if (ne === nextA) {
            arrivals++; nextA = T + expo(lam);
            var job = { a: T, id: arrivals };
            var free = servers.indexOf(null);
            if (free >= 0) { servers[free] = { job: job, s: T, end: T + expo(mu) }; nIn++; }
            else if (queue.length < 2000) { queue.push(job); nIn++; }
            else dropped++;
          } else {
            var sv = servers[si]; done++; sumW += T - sv.job.a; nIn--;
            servers[si] = queue.length ? (function (j) { return { job: j, s: T, end: T + expo(mu) }; })(queue.shift()) : null;
          }
        }
        area += nIn * (to - T); T = to;
      }
      function changed() {
        drawChart();
        // restart the sim so its averages reflect the new parameters
        resetSim();
      }
      var W = 720, H = 132, mid = 70;
      var simSvg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
      simBox.appendChild(simSvg);
      var lastTxt = 0;
      function drawSim() {
        H = ns <= 2 ? 84 : 132; mid = 18 + (H - 18) / 2;
        simSvg.setAttribute("viewBox", "0 0 " + W + " " + H);
        K.clear(simSvg);
        simSvg.appendChild(txt(K, 8, 14, "live simulation", "lbl"));
        var qx1 = 470, rowH = Math.min(26, (H - 24) / ns), sy0 = mid - rowH * ns / 2 + 2.5;
        simSvg.appendChild(txt(K, 8, mid + 4, "arrivals →", "lbl-sm"));
        // queue: dots right-aligned toward the servers
        var maxDots = 40, shown = Math.min(queue.length, maxDots);
        for (var i = 0; i < shown; i++) {
          simSvg.appendChild(K.s("circle", { cx: qx1 - 8 - i * 9.5, cy: mid, r: 4, class: queue.length > 12 ? "f2" : "f1" }));
        }
        if (queue.length > maxDots) simSvg.appendChild(txt(K, qx1 - 8 - maxDots * 9.5 - 4, mid + 4, "+" + (queue.length - maxDots), "lbl-sm", "end", { style: "fill:var(--red)" }));
        simSvg.appendChild(txt(K, qx1 - 6, mid + 20, "queue: " + queue.length, "lbl-sm mono", "end"));
        for (var k = 0; k < ns; k++) {
          var y = sy0 + k * rowH, s = servers[k], bh = rowH - 5;
          simSvg.appendChild(K.s("rect", { x: qx1 + 20, y: y, width: 150, height: bh, rx: 4, class: "box" }));
          if (s) {
            var fr = K.clamp((T - s.s) / Math.max(1e-9, s.end - s.s), 0, 1);
            simSvg.appendChild(K.s("rect", { x: qx1 + 20, y: y, width: 150 * fr, height: bh, rx: 4, class: "f3", opacity: 0.6 }));
            simSvg.appendChild(K.s("circle", { cx: qx1 + 32, cy: y + bh / 2, r: 4, class: "f1" }));
          }
          if (rowH >= 14) simSvg.appendChild(txt(K, qx1 + 44, y + bh / 2 + 4, "server " + (k + 1) + (s ? " busy" : " idle"), "lbl-sm"));
        }
        simSvg.appendChild(txt(K, qx1 + 184, mid + 4, "→ done: " + done, "lbl-sm"));
      }
      function simText() {
        var Tel = T - t0;
        if (Tel <= 0 || done < 5) { simOut.textContent = "warming up…"; return; }
        var Lh = area / Tel, lh = arrivals / Tel, Wh = sumW / done;
        simOut.innerHTML = "Simulated " + Tel.toFixed(0) + " s, " + done + " requests: measured L̂ = " + Lh.toFixed(2) + " in system, λ̂ = " + lh.toFixed(2) +
          " req/s, Ŵ = " + fmtT(Wh) + " → λ̂·Ŵ = <b>" + (lh * Wh).toFixed(2) + "</b> ≈ L̂ (Little's law)." + (dropped ? " <span style='color:var(--red)'>Queue capped at 2000 — " + dropped + " dropped.</span>" : "");
      }
      resetSim();
      drawChart();
      drawSim();
      animLoop(c, function (dt) {
        if (paused) return;
        advance(T + dt * speed * 1.25 / mu);
        drawSim();
        lastTxt += dt;
        if (lastTxt > 0.4) { lastTxt = 0; simText(); }
      });
    },
  });

  // ---------------------------------------------------------------- autoscaling-sim
  V.register("autoscaling-sim", {
    title: "Autoscaling a day of traffic",
    desc: "24 hours of diurnal demand (in-flight requests). The autoscaler watches a trailing average, asks for ⌈demand / target⌉ replicas within [min, max], and new replicas only serve after the cold start. Each ready replica can hold <i>capacity</i> requests within the latency SLO; setting target below capacity is your headroom. Red = minutes where demand exceeds capacity (SLO violations). Cost counts every replica you pay for, booting ones included.",
    render: function (c, p, K) {
      var peak = +p.peak || 400, target = +p.target || 15, capR = +p.capacity || 20, mn = p.min !== undefined ? +p.min : 2, mx = +p.max || 40, cold = p.coldStart !== undefined ? +p.coldStart : 6;
      var delay = 10, win = 3, spike = p.spike !== undefined ? !!p.spike : true, price = 3, gpr = 1, seed = 11;
      function add(el) { c.controls.appendChild(el); return el; }
      add(K.slider({ label: "peak demand", min: 50, max: 1000, step: 10, value: peak, fmt: function (v) { return v + " in-flight"; }, onInput: function (v) { peak = v; run(); } }));
      add(K.slider({ label: "replica capacity (max concurrency within SLO)", min: 2, max: 64, value: capR, onInput: function (v) { capR = v; run(); } }));
      add(K.slider({ label: "autoscaler target / replica", min: 1, max: 64, value: target, onInput: function (v) { target = v; run(); } }));
      add(K.slider({ label: "min replicas", min: 0, max: 20, value: mn, onInput: function (v) { mn = v; run(); } }));
      add(K.slider({ label: "max replicas", min: 1, max: 100, value: mx, onInput: function (v) { mx = v; run(); } }));
      add(K.slider({ label: "cold start", min: 0, max: 20, step: 0.5, value: cold, fmt: function (v) { return v + " min"; }, onInput: function (v) { cold = v; run(); } }));
      add(K.slider({ label: "scale-down delay", min: 0, max: 60, value: delay, fmt: function (v) { return v + " min"; }, onInput: function (v) { delay = v; run(); } }));
      add(K.toggle({ label: "traffic spike at 18:00", value: spike, onChange: function (v) { spike = v; run(); } }));
      add(K.number({ label: "$ / GPU-hour", value: price, min: 0, step: 0.1, onInput: function (v) { price = v || 0; run(); } }));
      add(K.number({ label: "GPUs / replica", value: gpr, min: 1, max: 16, step: 1, onInput: function (v) { gpr = Math.max(1, v || 1); run(); } }));
      add(K.button("New day (noise)", function () { seed++; run(); }));

      function demandCurve() {
        var r = K.rng(seed), d = [], n = 0;
        for (var m = 0; m < 1440; m++) {
          var h = m / 60;
          var base = 0.12 + 0.88 * (1 + Math.cos(2 * Math.PI * (h - 15) / 24)) / 2;
          base *= 1 + 0.08 * Math.sin(2 * Math.PI * (h - 9) / 6); // lunchtime / evening wiggle
          n = 0.97 * n + 0.012 * K.randn(r);
                    var v = peak * base * (1 + n + 0.02 * K.randn(r)) / 1.08;
          if (spike && m >= 1080 && m < 1110) v += peak * 0.8 * Math.min(1, (m - 1080) / 2) * (m >= 1100 ? (1110 - m) / 10 : 1);
          d.push(Math.max(0, v));
        }
        return d;
      }
      function sim(d) {
        var ready = [], boot = [], tot = [], cap = [], desiredHist = [], live = [], gh = 0, viol = [], clipped = 0;
        var start = Math.min(mx, Math.max(mn, Math.ceil(d[0] / target)));
        for (var i = 0; i < start; i++) live.push(-1e9); // ready at time
        for (var m = 0; m < 1440; m++) {
          var s = 0, w = 0;
          for (var k = Math.max(0, m - win + 1); k <= m; k++) { s += d[k]; w++; }
          var obs = s / w;
          if (Math.ceil(obs / target) > mx) clipped++;
          var want = Math.min(mx, Math.max(mn, Math.ceil(obs / target)));
          desiredHist.push(want);
          var keep = want;
          for (k = Math.max(0, m - delay); k <= m; k++) keep = Math.max(keep, desiredHist[k]);
          if (want > live.length) { for (i = live.length; i < want; i++) live.push(m + cold); }
          else if (keep < live.length) {
            live.sort(function (a, b) { return b - a; }); // drop the still-booting (latest) ones first
            live.splice(0, live.length - keep);
          }
          var rd = 0; live.forEach(function (t) { if (t <= m) rd++; });
          ready.push(rd); boot.push(live.length - rd); tot.push(live.length);
          cap.push(rd * capR);
                    viol.push(d[m] > rd * capR + 1e-9);
          gh += live.length * gpr / 60;
        }
        return { ready: ready, boot: boot, tot: tot, cap: cap, viol: viol, gh: gh, clipped: clipped };
      }
      var box = K.h("div");
      K.clear(c.stage).appendChild(box);
      function run() {
        var d = demandCurve(), R = sim(d);
        var dmax = Math.max.apply(null, d), ymax = Math.max(dmax, Math.max.apply(null, R.cap)) * 1.08;
        var hx = function (h) { return (h < 10 ? "0" : "") + h + ":00"; };
        var xt = [0, 3, 6, 9, 12, 15, 18, 21, 24].map(function (h) { return h * 60; });
        var ch = K.chart({ w: 720, h: 250, x: [0, 1440], y: [0, ymax], xTicks: xt, xFmt: function (v) { return hx(v / 60); }, yLabel: "in-flight requests", pad: { l: 58, b: 26 } });
        var g = clipTo(K, ch), sx = ch.sx, sy = ch.sy;
        // SLO violation bands + red fill between capacity and demand
        var runStart = -1;
        for (var m = 0; m <= 1440; m++) {
          var v = m < 1440 && R.viol[m];
          if (v && runStart < 0) runStart = m;
          if (!v && runStart >= 0) {
            g.appendChild(K.s("rect", { x: sx(runStart), y: ch.pad.t, width: Math.max(1.5, sx(m) - sx(runStart)), height: ch.H - ch.pad.t - ch.pad.b, style: "fill:var(--red);opacity:0.1" }));
            var top = "", bot = "";
            for (var k = runStart; k < m; k++) { top += (top ? "L" : "M") + sx(k) + " " + sy(d[k]) + "L" + sx(k + 1) + " " + sy(d[k]); bot = "L" + sx(k + 1) + " " + sy(R.cap[k]) + "L" + sx(k) + " " + sy(R.cap[k]) + bot; }
            g.appendChild(K.s("path", { d: top + bot + "z", style: "fill:var(--red);opacity:0.55" }));
            runStart = -1;
          }
        }
        // capacity (step) as a filled area
        var cp = "M" + sx(0) + " " + sy(0);
        for (m = 0; m < 1440; m++) cp += "L" + sx(m) + " " + sy(R.cap[m]) + "L" + sx(m + 1) + " " + sy(R.cap[m]);
        cp += "L" + sx(1440) + " " + sy(0) + "z";
        g.appendChild(K.s("path", { d: cp, class: "f1", opacity: 0.15 }));
        g.appendChild(K.s("path", { d: cp.replace(/^M[^L]*/, "M" + sx(0) + " " + sy(R.cap[0])).replace(/L[^L]*z$/, ""), class: "ln s1" }));
        var dp = "";
        for (m = 0; m < 1440; m++) dp += (m ? "L" : "M") + sx(m + 0.5).toFixed(1) + " " + sy(d[m]).toFixed(1);
        g.appendChild(K.s("path", { d: dp, class: "ln-thin s2", "stroke-width": 1.6 }));
        var lx = ch.pad.l + 12, ly = ch.pad.t + 14;
        ch.g.appendChild(K.s("rect", { x: lx - 6, y: ly - 12, width: 420, height: 22, rx: 4, class: "fbg", opacity: 0.9 }));
        [["s2", "demand"], ["s1", "capacity = ready × replica capacity"], ["", "SLO violation"]].forEach(function (it, i) {
          var x = lx + [0, 80, 310][i];
          if (it[0]) ch.g.appendChild(K.s("line", { x1: x, x2: x + 16, y1: ly - 4, y2: ly - 4, class: "ln " + it[0] }));
          else ch.g.appendChild(K.s("rect", { x: x, y: ly - 9, width: 16, height: 10, style: "fill:var(--red);opacity:0.55" }));
          ch.g.appendChild(txt(K, x + 20, ly, it[1], "lbl-sm"));
        });

        // replica chart
        var rmax = Math.max(mx, 1) + 1;
        var c2 = K.chart({ w: 720, h: 140, x: [0, 1440], y: [0, rmax], xTicks: xt, xFmt: function (v) { return hx(v / 60); }, yLabel: "replicas", yN: 3, pad: { l: 58, t: 8, b: 26 } });
        var g2 = clipTo(K, c2);
        var bp = "", rp = "";
        for (m = 0; m < 1440; m++) {
          if (R.boot[m]) g2.appendChild(K.s("rect", { x: c2.sx(m), y: c2.sy(R.tot[m]), width: c2.sx(m + 1) - c2.sx(m) + 0.3, height: c2.sy(R.ready[m]) - c2.sy(R.tot[m]), class: "f4", opacity: 0.5 }));
          rp += (m ? "L" : "M") + c2.sx(m) + " " + c2.sy(R.ready[m]) + "L" + c2.sx(m + 1) + " " + c2.sy(R.ready[m]);
        }
        [mn, mx].forEach(function (v, i) {
          c2.g.appendChild(K.s("line", { x1: c2.pad.l, x2: c2.W - c2.pad.r, y1: c2.sy(v), y2: c2.sy(v), class: "ln-thin smuted dash" }));
          c2.g.appendChild(txt(K, c2.W - c2.pad.r - 4, c2.sy(v) - 3, i ? "max " + v : "min " + v, "lbl-sm", "end", { style: "fill:var(--muted)" }));
        });
        g2.appendChild(K.s("path", { d: rp, class: "ln s1" }));
        c2.g.appendChild(K.s("rect", { x: c2.pad.l + 10, y: 12, width: 12, height: 8, class: "f4", opacity: 0.5 }));
        c2.g.appendChild(txt(K, c2.pad.l + 26, 20, "booting (paid, not serving)", "lbl-sm"));
        c2.g.appendChild(K.s("line", { x1: c2.pad.l + 200, x2: c2.pad.l + 216, y1: 16, y2: 16, class: "ln s1" }));
        c2.g.appendChild(txt(K, c2.pad.l + 220, 20, "ready replicas", "lbl-sm"));
        K.clear(box).appendChild(ch.svg);
        box.appendChild(c2.svg);

        var vmin = R.viol.filter(Boolean).length;
        var staticRep = Math.ceil(dmax / capR), staticGH = staticRep * 24 * gpr;
        var peakRep = Math.max.apply(null, R.tot);
        c.readout.innerHTML = stats([
          ["GPU-hours / day", R.gh.toFixed(0)],
          ["cost / day", money(R.gh * price)],
          ["SLO violations", vmin + " min (" + (vmin / 14.4).toFixed(1) + "%)", vmin > 14 ? "var(--red)" : vmin ? "var(--amber)" : "var(--green)"],
          ["peak replicas", String(peakRep) + (R.clipped ? " (hit max)" : "")],
          ["static for peak: " + staticRep + " replicas", staticGH.toFixed(0) + " GPU-h · " + money(staticGH * price)],
          ["autoscaling saves", staticGH ? Math.round(100 * (1 - R.gh / staticGH)) + "%" : "—", R.gh < staticGH ? "var(--green)" : "var(--red)"],
        ]) + "<div style='margin-top:6px'>Per month that's " + money(R.gh * price * 30) + " vs " + money(staticGH * price * 30) + " provisioned for peak. " +
          (vmin && cold >= 2 ? "With a " + cold + "-minute cold start (pull a multi-GB image, load weights, warm up CUDA graphs) a sudden ramp like the 18:00 spike is served late: the new replicas arrive after the burst has already hit. Faster loading, or a target further below capacity (headroom), shrinks the red. " : vmin ? "Even with instant starts, the trailing-average metric reacts late to a sudden burst. " : "No violations: capacity (target " + target + " of " + capR + " per replica) keeps ahead of demand. Try a longer cold start, a target closer to capacity, or the spike. ") +
          (mn === 0 ? "min = 0 (scale to zero) is cheapest overnight, but the first request after idle waits a full cold start. " : "") +
          (R.clipped ? "<b style='color:var(--red)'>For " + R.clipped + " min the autoscaler wanted more than max = " + mx + " replicas</b> — max is a cost guard-rail, but it also caps capacity. " : "") +
          "Scale-down delay trades a few idle GPU-hours for not thrashing on noise.</div>";
      }
      run();
    },
  });

  // ---------------------------------------------------------------- cost-calc
  // Rough, illustrative on-demand cloud prices (2024–25) and aggregate output throughput per replica
  // at a batch size that still gives acceptable latency. Workload-dependent — every number is editable.
  var GPUS = [
    { id: "H100", name: "H100 80GB", price: 3.0, range: "≈ $2.5–4/h", w: { "8B": { g: 1, tps: 6000, note: "BF16" }, "70B": { g: 4, tps: 5000, note: "FP8, TP4" } } },
    { id: "A100", name: "A100 80GB", price: 1.8, range: "≈ $1.5–2/h", w: { "8B": { g: 1, tps: 3000, note: "BF16" }, "70B": { g: 4, tps: 2000, note: "BF16, TP4" } } },
    { id: "L4", name: "L4 24GB", price: 0.8, range: "≈ $0.7–1/h", w: { "8B": { g: 1, tps: 900, note: "FP8" }, "70B": { g: 8, tps: 450, note: "FP8, TP8" } } },
    { id: "T4", name: "T4 16GB", price: 0.35, range: "≈ $0.3–0.5/h", w: { "8B": { g: 1, tps: 250, note: "4-bit" }, "70B": null } },
  ];
  var WORK = { "8B": { label: "~8B model (Llama-3.1-8B)", api: 0.2 }, "70B": { label: "~70B model (Llama-3.1-70B)", api: 0.9 } };
  function perM(price, g, tps, u) { return price * g / (tps * u * 3600) * 1e6; }

  V.register("cost-calc", {
    title: "What does a million tokens cost on your own GPUs?",
    desc: "Cost per token = what you pay per hour ÷ tokens you actually produce per hour. Pick a GPU and workload (every number is editable) and slide utilisation — the fraction of the hour your replica is doing useful, well-batched work. Compare with an API's per-token price.",
    render: function (c, p, K) {
      var wk = WORK[p.model] ? p.model : "8B", gid = "H100", util = p.util !== undefined ? +p.util / (p.util > 1 ? 100 : 1) : 0.5, api = +p.api || WORK[wk].api;
      GPUS.forEach(function (G) { if (String(p.gpu || "").toUpperCase() === G.id) gid = G.id; });
      if (!GPUS.filter(function (G) { return G.id === gid; })[0].w[wk]) gid = "H100";
      var price, g, tps;
      function G() { return GPUS.filter(function (x) { return x.id === gid; })[0]; }
      function preset() { var x = G(), w = x.w[wk]; price = x.price; g = w.g; tps = w.tps; }
      preset();
      var wSel = K.select({ label: "workload", value: wk, options: Object.keys(WORK).map(function (k) { return { value: k, label: WORK[k].label }; }), onChange: function (v) {
        wk = v; if (!G().w[wk]) { gid = "H100"; gSel.input.value = gid; }
        api = WORK[wk].api; aIn.set(api); preset(); sync(); draw(); } });
      var gSel = K.select({ label: "GPU", value: gid, options: GPUS.map(function (x) { return { value: x.id, label: x.name + " (" + x.range + ")" }; }), onChange: function (v) {
        if (!GPUS.filter(function (x) { return x.id === v; })[0].w[wk]) { gSel.input.value = gid; note = v + " can't hold this model at a useful speed"; draw(); return; }
        gid = v; preset(); sync(); draw(); } });
      var pIn = K.number({ label: "$ / GPU-hour", value: price, min: 0, step: 0.05, onInput: function (v) { price = v || 0; draw(); } });
      var nIn = K.number({ label: "GPUs / replica", value: g, min: 1, max: 16, step: 1, onInput: function (v) { g = Math.max(1, v || 1); draw(); } });
      var tIn = K.number({ label: "throughput (output tok/s per replica)", value: tps, min: 1, step: 50, onInput: function (v) { tps = Math.max(1, v || 1); draw(); } });
      var uSl = K.slider({ label: "utilisation", min: 2, max: 100, value: Math.round(util * 100), fmt: function (v) { return v + "%"; }, onInput: function (v) { util = v / 100; draw(); } });
      var aIn = K.number({ label: "API price ($ / 1M output tok)", value: api, min: 0, step: 0.01, onInput: function (v) { api = v || 0; draw(); } });
      [wSel, gSel, uSl, pIn, nIn, tIn, aIn].forEach(function (el) { c.controls.appendChild(el); });
      var note = "";
      function sync() { pIn.set(price); nIn.set(g); tIn.set(tps); note = ""; }

      function draw() {
        var c1 = perM(price, g, tps, 1), cur = perM(price, g, tps, util), be = api > 0 ? c1 / api : Infinity;
        var lo = Math.min(c1, api || c1) * 0.5, hi = Math.max(c1 / 0.02, api * 2);
        var ch = K.chart({ w: 720, h: 260, x: [0.02, 1], y: [lo, hi], yLog: true, xLabel: "utilisation (share of the hour spent generating at full batch)", yLabel: "$ per 1M output tokens",
          xFmt: function (v) { return Math.round(v * 100) + "%"; }, yFmt: function (v) { return money(v); }, xTicks: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], pad: { l: 64 } });
        var gc = clipTo(K, ch), pts = [];
        for (var u = 0.02; u <= 1.0001; u += 0.005) pts.push([u, perM(price, g, tps, u)]);
        if (api > 0) {
          if (be < 1) gc.appendChild(K.s("rect", { x: ch.sx(Math.max(0.02, be)), y: ch.pad.t, width: ch.sx(1) - ch.sx(Math.max(0.02, be)), height: ch.H - ch.pad.t - ch.pad.b, style: "fill:var(--green);opacity:0.08" }));
          gc.appendChild(K.s("line", { x1: ch.pad.l, x2: ch.W - ch.pad.r, y1: ch.sy(api), y2: ch.sy(api), class: "ln s3 dash" }));
          gc.appendChild(txt(K, ch.W - ch.pad.r - 6, ch.sy(api) - 6, "API " + money(api) + " / 1M", "lbl-sm", "end", { style: "fill:var(--green);font-weight:600" }));
          if (be <= 1 && be >= 0.02) {
            gc.appendChild(K.s("line", { x1: ch.sx(be), x2: ch.sx(be), y1: ch.sy(api), y2: ch.H - ch.pad.b, class: "ln-thin s3 dash" }));
            gc.appendChild(txt(K, ch.sx(be) + 5, ch.H - ch.pad.b - 8, "break-even " + Math.round(be * 100) + "%", "lbl-sm", null, { style: "fill:var(--green)" }));
          }
        }
        gc.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln s1" }));
        gc.appendChild(K.s("circle", { cx: ch.sx(util), cy: ch.sy(cur), r: 6, class: "f1", style: "stroke:var(--bg);stroke-width:2" }));
        gc.appendChild(txt(K, ch.sx(util) + 9, ch.sy(cur) - 8, G().id + (g > 1 ? "×" + g : "") + ": " + money(cur), "lbl", util > 0.8 ? "end" : null));

        // bar comparison at the current utilisation
        var rows = GPUS.filter(function (x) { return x.w[wk]; }).map(function (x) {
          var mine = x.id === gid, w = x.w[wk];
          return { label: x.id + (w.g > 1 ? " ×" + w.g : "") + " · " + w.note, v: mine ? cur : perM(x.price, w.g, w.tps, util), mine: mine };
        });
        rows.push({ label: "API price", v: api, api: true });
        var bw = 720, bh = 26, l = 170, bH = rows.length * bh + 34, vmax = Math.max.apply(null, rows.map(function (r) { return r.v; })) * 1.18 || 1;
        var svg = K.s("svg", { viewBox: "0 0 " + bw + " " + bH, class: "viz-svg" });
        svg.appendChild(txt(K, 8, 14, "At " + Math.round(util * 100) + "% utilisation (" + WORK[wk].label + "):", "lbl"));
        rows.forEach(function (r, i) {
          var y = 24 + i * bh, w = (bw - l - 90) * r.v / vmax;
          svg.appendChild(txt(K, l - 8, y + 15, r.label, r.mine ? "lbl" : "lbl-sm", "end"));
          svg.appendChild(K.s("rect", { x: l, y: y + 3, width: Math.max(1, w), height: bh - 8, rx: 3, class: r.api ? "f3" : r.mine ? "f1" : "fmuted", opacity: r.mine || r.api ? 0.9 : 0.45 }));
          svg.appendChild(txt(K, l + w + 6, y + 16, money(r.v), "lbl-sm mono"));
        });
        var box = K.clear(c.stage);
        box.appendChild(ch.svg);
        box.appendChild(svg);

        var hrs = price * g, tokMonth = tps * util * 3600 * 730;
        c.readout.innerHTML = stats([
          ["replica cost", money(hrs) + "/h · " + money(hrs * 730) + "/month"],
          ["tokens / month at " + Math.round(util * 100) + "%", K.fmtShort(tokMonth)],
          ["your cost / 1M output tok", money(cur), api > 0 ? (cur <= api ? "var(--green)" : "var(--red)") : null],
          ["at 100% (theoretical floor)", money(c1)],
          ["break-even utilisation vs API", be <= 1 ? Math.round(be * 100) + "%" : "never (>100%)", be <= 1 ? null : "var(--red)"],
        ]) + (note ? "<div style='color:var(--red);margin-top:4px'>" + note + ".</div>" : "") +
          "<div style='margin-top:6px'>" + K.tex("\\frac{\\$}{1\\text{M tok}} = \\frac{" + price.toFixed(2) + "\\ \\$/\\text{h} \\times " + g + "}{" + tps + "\\ \\text{tok/s} \\times " + Math.round(util * 100) + "\\% \\times 3600} \\times 10^6 = \\text{\\$}" + (+cur.toPrecision(3)) + "") +
          " — utilisation is the lever: an idle GPU still bills by the hour. Self-hosting wins at steady high volume (or when you need privacy/customisation); bursty low volume is cheaper on a per-token API. " +
          "This counts output tokens only; long prompts add prefill work, and real bills include idle replicas, engineers and redundancy.</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- load-balancing
  var POLICIES = [
    { id: "random", label: "random" },
    { id: "rr", label: "round-robin" },
    { id: "least", label: "least-loaded" },
    { id: "hash", label: "prefix-hash (sticky)" },
    { id: "cache", label: "cache-aware" },
  ];
  function prefColor(i) { return "hsl(" + Math.round((i * 137.508) % 360) + ",62%,55%)"; }

  V.register("load-balancing", {
    title: "Load balancing LLM replicas: load vs prefix-cache hits",
    desc: "Requests belong to a few conversations/system prompts (prefixes, Zipf-popular). Each request needs ~150 ms of decoding plus a prefill; each replica keeps a small LRU prefix cache, and a hit skips the 2,000-token shared prefix (200 ms of prefill). The same request trace is replayed under five routing policies. Cache-aware routing sends a request to a replica that already holds its prefix unless that replica is overloaded (load cap).",
    render: function (c, p, K) {
      var N = +p.replicas || 4, pol = "cache", rate = 10, P = 16, zipf = 1.2, cacheCap = 4, loadCap = 2, seed = 1;
      POLICIES.forEach(function (q) { if (q.id === p.policy) pol = q.id; });
      function add(el) { c.controls.appendChild(el); return el; }
      add(K.slider({ label: "replicas", min: 2, max: 8, value: N, onInput: function (v) { N = v; run(); } }));
      add(K.slider({ label: "arrival rate", min: 2, max: 40, value: rate, fmt: function (v) { return v + " req/s"; }, onInput: function (v) { rate = v; run(); } }));
      add(K.slider({ label: "distinct prefixes", min: 2, max: 64, value: P, onInput: function (v) { P = v; run(); } }));
      add(K.slider({ label: "popularity skew (Zipf s)", min: 0, max: 2, step: 0.1, value: zipf, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { zipf = v; run(); } }));
      add(K.slider({ label: "cache size / replica", min: 1, max: 16, value: cacheCap, fmt: function (v) { return v + " prefixes"; }, onInput: function (v) { cacheCap = v; run(); } }));
      add(K.slider({ label: "load cap (cache-aware)", min: 0, max: 10, value: loadCap, fmt: function (v) { return "+" + v; }, onInput: function (v) { loadCap = v; run(); } }));
      add(K.select({ label: "inspect", value: pol, options: POLICIES.map(function (q) { return { value: q.id, label: q.label }; }), onChange: function (v) { pol = v; show(); } }));
      add(K.button("Re-run (new trace)", function () { seed++; run(); }));

      var NREQ = 4000, WARM = 300, BASE = 0.15, PRE = 2000, SUF = 100, TPS = 10000; // BASE = decode time, TPS = prefill tok/s
      var res = null;
      function trace() {
        var r = K.rng(seed * 97 + 3), w = [], tot = 0, t = 0, out = [];
        for (var i = 0; i < P; i++) { w.push(1 / Math.pow(i + 1, zipf)); tot += w[i]; }
        for (i = 0; i < NREQ; i++) {
          t += -Math.log(1 - r()) / rate;
          var u = r() * tot, k = 0;
          while (k < P - 1 && u > w[k]) { u -= w[k]; k++; }
          out.push({ t: t, pf: k });
        }
        return out;
      }
      function simulate(tr, policy) {
        var r = K.rng(seed * 31 + 7), busy = [], cache = [], ends = [], cnt = [], hits = [], lat = [], hitN = 0, rrI = 0;
        for (var i = 0; i < N; i++) { busy.push(0); cache.push([]); ends.push([]); cnt.push(0); hits.push(0); }
        function load(i, t) { var e = ends[i]; while (e.length && e[0] <= t) e.shift(); return e.length; }
        tr.forEach(function (q, n) {
          var loads = [], minL = Infinity, j;
          for (j = 0; j < N; j++) { loads.push(load(j, q.t)); minL = Math.min(minL, loads[j]); }
          function leastOf(ids) { var best = [], m = Infinity; ids.forEach(function (x) { if (loads[x] < m) { m = loads[x]; best = [x]; } else if (loads[x] === m) best.push(x); }); return best[Math.floor(r() * best.length)]; }
          var all = []; for (j = 0; j < N; j++) all.push(j);
          var pick;
          if (policy === "random") pick = Math.floor(r() * N);
          else if (policy === "rr") pick = rrI++ % N;
          else if (policy === "least") pick = leastOf(all);
          else if (policy === "hash") pick = (q.pf * 2654435761 >>> 0) % N;
          else {
            var holders = all.filter(function (x) { return cache[x].indexOf(q.pf) >= 0; });
            var h = holders.length ? leastOf(holders) : -1;
            pick = h >= 0 && loads[h] <= minL + loadCap ? h : leastOf(all);
          }
          var cc = cache[pick], ix = cc.indexOf(q.pf), hit = ix >= 0;
          if (hit) cc.splice(ix, 1);
          cc.unshift(q.pf);
          if (cc.length > cacheCap) cc.pop();
          var svc = BASE + ((hit ? 0 : PRE) + SUF) / TPS;
          var st = Math.max(q.t, busy[pick]), en = st + svc;
          busy[pick] = en; ends[pick].push(en);
          if (n >= WARM) { lat.push(en - q.t); cnt[pick]++; if (hit) { hits[pick]++; hitN++; } }
        });
        var tot = NREQ - WARM;
        return { mean: mean(lat), p99: pct(lat, 0.99), hit: hitN / tot, cnt: cnt, hits: hits, cache: cache };
      }
      function run() {
        var tr = trace();
        res = {};
        POLICIES.forEach(function (q) { res[q.id] = simulate(tr, q.id); });
        show();
      }
      function show() {
        var W = 720, colW = 232, rowH = 24, top = 36, H = top + POLICIES.length * rowH + 10;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var metrics = [
          { key: "mean", label: "mean latency", fmt: fmtT, cls: "f1", better: "lower" },
          { key: "p99", label: "p99 latency", fmt: fmtT, cls: "f2", better: "lower" },
          { key: "hit", label: "prefix-cache hit rate", fmt: function (v) { return Math.round(v * 100) + "%"; }, cls: "f3", better: "higher" },
        ];
        var lab = 118;
        POLICIES.forEach(function (q, i) {
          var y = top + i * rowH;
          if (q.id === pol) svg.appendChild(K.s("rect", { x: 2, y: y - 2, width: W - 4, height: rowH, rx: 4, class: "fbg3" }));
          svg.appendChild(txt(K, lab - 8, y + 14, q.label, q.id === pol ? "lbl" : "lbl-sm", "end"));
        });
        metrics.forEach(function (m, k) {
          var x0 = lab + k * (W - lab) / 3 + 4, bw = (W - lab) / 3 - 60;
          var vals = POLICIES.map(function (q) { return res[q.id][m.key]; });
          var vmax = m.key === "hit" ? 1 : Math.max.apply(null, vals) || 1;
          var best = m.better === "lower" ? Math.min.apply(null, vals) : Math.max.apply(null, vals);
          svg.appendChild(txt(K, x0, 20, m.label + " (" + m.better + " is better)", "lbl-sm", null, { style: "font-weight:600" }));
          POLICIES.forEach(function (q, i) {
            var v = res[q.id][m.key], y = top + i * rowH, w = Math.max(1, bw * v / vmax);
            svg.appendChild(K.s("rect", { x: x0, y: y + 2, width: w, height: rowH - 8, rx: 3, class: m.cls, opacity: q.id === pol ? 0.95 : 0.5 }));
            svg.appendChild(txt(K, x0 + w + 4, y + 14, m.fmt(v) + (v === best ? " ★" : ""), "lbl-sm mono"));
          });
        });
        // per-replica detail for the inspected policy
        var R = res[pol], tot = R.cnt.reduce(function (a, b) { return a + b; }, 0) || 1, rh = 26, H2 = 40 + N * rh;
        var s2 = K.s("svg", { viewBox: "0 0 " + W + " " + H2, class: "viz-svg" });
        var polLabel = POLICIES.filter(function (q) { return q.id === pol; })[0].label;
        s2.appendChild(txt(K, 8, 16, "Inside the replicas — " + polLabel, "lbl"));
        s2.appendChild(txt(K, 110, 34, "share of requests (dashed = fair 1/N)", "lbl-sm"));
        s2.appendChild(txt(K, 380, 34, "hit rate", "lbl-sm"));
        s2.appendChild(txt(K, 470, 34, "prefixes in its cache at the end (colour = prefix, #0 most popular)", "lbl-sm"));
        var shMax = Math.max(1.5 / N, Math.max.apply(null, R.cnt) / tot);
        for (var i = 0; i < N; i++) {
          var y = 42 + i * rh, sh = R.cnt[i] / tot, w = 240 * sh / shMax;
          s2.appendChild(txt(K, 100, y + 14, "replica " + i, "lbl-sm", "end"));
          s2.appendChild(K.s("rect", { x: 110, y: y + 3, width: Math.max(1, w), height: rh - 10, rx: 3, class: sh > 1.4 / N ? "f2" : "f1", opacity: 0.8 }));
          s2.appendChild(txt(K, 110 + w + 4, y + 15, Math.round(sh * 100) + "%", "lbl-sm mono"));
          s2.appendChild(K.s("line", { x1: 110 + 240 * (1 / N) / shMax, x2: 110 + 240 * (1 / N) / shMax, y1: y + 1, y2: y + rh - 5, class: "ln-thin sfg dash" }));
          s2.appendChild(txt(K, 380, y + 15, R.cnt[i] ? Math.round(100 * R.hits[i] / R.cnt[i]) + "%" : "—", "lbl-sm mono"));
          R.cache[i].forEach(function (pf, j) {
            if (j >= 16) return;
            var x = 470 + j * 15;
            s2.appendChild(K.s("rect", { x: x, y: y + 3, width: 13, height: rh - 10, rx: 3, style: "fill:" + prefColor(pf) }));
            if (P <= 20 && cacheCap <= 16) s2.appendChild(txt(K, x + 6.5, y + 14, String(pf), "lbl-sm t-white", "middle", { style: "font-size:9px" }));
          });
        }
        var box = K.clear(c.stage);
        box.appendChild(svg);
        box.appendChild(s2);

        var util = rate * (BASE + (PRE + SUF) / TPS) / N, utilHit = rate * (BASE + SUF / TPS) / N;
        c.readout.innerHTML = stats([
          ["inspected policy", polLabel],
          ["mean latency", fmtT(R.mean)],
          ["p99 latency", fmtT(R.p99)],
          ["cache hit rate", Math.round(R.hit * 100) + "%"],
          ["load if every request misses", Math.round(util * 100) + "%", util >= 1 ? "var(--red)" : null],
          ["load if every request hits", Math.round(utilHit * 100) + "%"],
        ]) + "<div style='margin-top:6px'>A miss costs " + fmtT(BASE + (PRE + SUF) / TPS) + " of GPU time, a hit " + fmtT(BASE + SUF / TPS) +
          " — so cache hits are capacity. Random / round-robin / least-loaded spread load evenly but scatter each prefix across every replica, so small caches thrash. " +
          "Sticky prefix-hashing maximises hits but piles the popular prefixes onto a few replicas (orange bars) → hot spots and a fat p99. " +
          "Cache-aware routing takes the hit unless the holder is more than <i>load cap</i> requests busier than the least-loaded replica — the approach used by LLM-aware routers (e.g. SGLang's router, llm-d, Dynamo). " +
          "Each replica here is a single FIFO server (no batching) to keep the effect visible.</div>";
      }
      run();
    },
  });
})();
