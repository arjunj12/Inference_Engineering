/* rl.js — interactive explainers for the reinforcement-learning track. */
(function () {
  "use strict";
  var V = window.Viz;

  // ---------------------------------------------------------------- shared helpers
  function num(v, d) { return typeof v === "number" && isFinite(v) ? v : d; }
  function svgPt(svg, ev) {
    var pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  function stats(items) {
    return '<div class="stat-grid">' + items.map(function (it) {
      return '<div class="stat"><small>' + it[0] + "</small><b" + (it[2] ? ' style="color:' + it[2] + '"' : "") + ">" + it[1] + "</b></div>";
    }).join("") + "</div>";
  }
  // Play/pause loop that stops itself when the card leaves the DOM.
  function player(c, K, tick, ms) {
    var timer = null;
    var btn = K.button("▶ Play", function () { if (timer) stop(); else start(); }, "primary");
    function start() {
      if (timer) return;
      timer = setInterval(function () {
        if (!c.root.isConnected) { stop(); return; }
        if (tick() === false) stop();
      }, ms || 60);
      btn.textContent = "⏸ Pause";
    }
    function stop() { clearInterval(timer); timer = null; btn.textContent = "▶ Play"; }
    return { btn: btn, start: start, stop: stop, running: function () { return !!timer; } };
  }
  function argmax(a, r) {
    var best = -Infinity, idx = [];
    for (var i = 0; i < a.length; i++) {
      if (a[i] > best + 1e-12) { best = a[i]; idx = [i]; } else if (Math.abs(a[i] - best) <= 1e-12) idx.push(i);
    }
    return idx.length === 1 || !r ? idx[0] : idx[Math.floor(r() * idx.length)];
  }
  function sampleFrom(pr, u) {
    var acc = 0;
    for (var i = 0; i < pr.length; i++) { acc += pr[i]; if (u <= acc) return i; }
    return pr.length - 1;
  }
  var RED = "var(--red)", GREEN = "var(--c3)";

  // ---------------------------------------------------------------- gridworld (tabular Q-learning)
  V.register("gridworld", {
    title: "Q-learning in a gridworld: values and a policy emerge from trial and error",
    desc: "The agent (orange dot) starts bottom-left and must reach the goal (+1) without falling into the pit (−1); every step costs a little. Press Play or fast-forward. Cell colour = best Q-value there, arrows = greedy policy. Click an empty cell to toggle a wall.",
    render: function (c, p, K) {
      var N = 6, DX = [0, 1, 0, -1], DY = [-1, 0, 1, 0], ANAME = ["↑", "→", "↓", "←"];
      var START = [0, 5], GOAL = [5, 0], PIT = [4, 2];
      var walls = {};
      [[1, 1], [2, 1], [3, 1], [1, 3], [3, 3], [3, 4]].forEach(function (w) { walls[w[1] * N + w[0]] = 1; });
      var eps = num(p.epsilon, 0.2), alpha = num(p.alpha, 0.5), gamma = num(p.gamma, 0.95), stepCost = num(p.stepCost, -0.04), slip = 0;
      var speed = "1", MAXSTEPS = 100;
      var r, Q, visited, pos, path, ret, steps, returns, lastUpd;
      function sIdx(x, y) { return y * N + x; }
      function isGoal(x, y) { return x === GOAL[0] && y === GOAL[1]; }
      function isPit(x, y) { return x === PIT[0] && y === PIT[1]; }
      function reset() {
        r = K.rng(42); Q = []; visited = [];
        for (var i = 0; i < N * N; i++) { Q.push([0, 0, 0, 0]); visited.push(0); }
        returns = []; lastUpd = null; newEpisode();
      }
      function newEpisode() { pos = START.slice(); path = [pos.slice()]; ret = 0; steps = 0; }
      function step() {
        var s = sIdx(pos[0], pos[1]);
        var a = r() < eps ? Math.floor(r() * 4) : argmax(Q[s], r);
        var act = r() < slip ? Math.floor(r() * 4) : a;
        var nx = pos[0] + DX[act], ny = pos[1] + DY[act];
        if (nx < 0 || ny < 0 || nx >= N || ny >= N || walls[sIdx(nx, ny)]) { nx = pos[0]; ny = pos[1]; }
        var rew = stepCost, term = false;
        if (isGoal(nx, ny)) { rew = 1; term = true; } else if (isPit(nx, ny)) { rew = -1; term = true; }
        var s2 = sIdx(nx, ny);
        var target = rew + (term ? 0 : gamma * Math.max.apply(null, Q[s2]));
        var old = Q[s][a];
        Q[s][a] += alpha * (target - Q[s][a]);
        visited[s] = 1;
        lastUpd = { s: s, a: a, old: old, nw: Q[s][a], rew: rew, target: target, term: term };
        pos = [nx, ny]; path.push(pos.slice()); ret += rew; steps++;
        if (term || steps >= MAXSTEPS) { returns.push(ret); newEpisode(); return true; }
        return false;
      }
      function runEpisode() { var guard = 0; while (!step() && guard++ < MAXSTEPS + 2) { /* run to end */ } }
      function bestPathLen() {
        var dist = {}, q = [START], key = function (x, y) { return y * N + x; };
        dist[key(START[0], START[1])] = 0;
        while (q.length) {
          var cur = q.shift(), d = dist[key(cur[0], cur[1])];
          if (isGoal(cur[0], cur[1])) return d;
          for (var a = 0; a < 4; a++) {
            var nx = cur[0] + DX[a], ny = cur[1] + DY[a], k = key(nx, ny);
            if (nx < 0 || ny < 0 || nx >= N || ny >= N || walls[k] || isPit(nx, ny) || dist[k] !== undefined) continue;
            dist[k] = d + 1; q.push([nx, ny]);
          }
        }
        return null;
      }

      var pl = player(c, K, function () {
        if (speed === "1") step();
        else if (speed === "20") { for (var i = 0; i < 20; i++) step(); }
        else runEpisode();
        draw();
      }, 70);
      c.controls.appendChild(pl.btn);
      c.controls.appendChild(K.select({ label: "speed", value: speed, options: [{ value: "1", label: "1 step / tick" }, { value: "20", label: "20 steps / tick" }, { value: "ep", label: "1 episode / tick" }], onChange: function (v) { speed = v; } }));
      c.controls.appendChild(K.button("Step", function () { step(); draw(); }));
      c.controls.appendChild(K.button("+1 episode", function () { runEpisode(); draw(); }));
      c.controls.appendChild(K.button("⏩ +100 episodes", function () { for (var i = 0; i < 100; i++) runEpisode(); draw(); }));
      c.controls.appendChild(K.button("Reset", function () { pl.stop(); reset(); draw(); }));
      c.controls.appendChild(K.slider({ label: "ε (explore)", min: 0, max: 1, step: 0.01, value: eps, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { eps = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "α (learning rate)", min: 0.01, max: 1, step: 0.01, value: alpha, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { alpha = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "γ (discount)", min: 0.5, max: 1, step: 0.01, value: gamma, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { gamma = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "step reward", min: -0.2, max: 0, step: 0.01, value: stepCost, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { stepCost = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "slip prob.", min: 0, max: 0.5, step: 0.05, value: slip, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { slip = v; draw(); } }));

      function arrow(g, cx, cy, a, len, op) {
        var dx = DX[a], dy = DY[a];
        var x1 = cx - dx * len / 2, y1 = cy - dy * len / 2, x2 = cx + dx * len / 2, y2 = cy + dy * len / 2;
        g.appendChild(K.s("line", { x1: x1, y1: y1, x2: x2, y2: y2, class: "sfg", "stroke-width": 2, opacity: op }));
        var pts = [x2 + dx * 3, y2 + dy * 3, x2 - dx * 4 - dy * 5, y2 - dy * 4 + dx * 5, x2 - dx * 4 + dy * 5, y2 - dy * 4 - dx * 5];
        g.appendChild(K.s("polygon", { points: pts.join(" "), class: "ffg", opacity: op }));
      }

      function draw() {
        var cs = 50, P = 6, S = N * cs + 2 * P;
        var svg = K.s("svg", { viewBox: "0 0 " + S + " " + S, class: "viz-svg", style: { maxWidth: "400px", margin: "0 auto" } });
        for (var y = 0; y < N; y++) for (var x = 0; x < N; x++) {
          (function (x, y) {
            var s = sIdx(x, y), X = P + x * cs, Y = P + y * cs;
            var g = K.s("g", { style: { cursor: "pointer" } });
            g.appendChild(K.s("rect", { x: X, y: Y, width: cs, height: cs, class: "box" }));
            if (walls[s]) {
              g.appendChild(K.s("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, class: "ffg", opacity: 0.72 }));
            } else if (isGoal(x, y) || isPit(x, y)) {
              g.appendChild(K.s("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, style: { fill: isGoal(x, y) ? GREEN : RED } }));
              g.appendChild(K.s("text", { x: X + cs / 2, y: Y + cs / 2 + 5, "text-anchor": "middle", class: "lbl t-white", text: isGoal(x, y) ? "+1" : "−1" }));
            } else {
              var mq = Math.max.apply(null, Q[s]);
              if (visited[s] && Math.abs(mq) > 1e-6) {
                g.appendChild(K.s("rect", { x: X + 1, y: Y + 1, width: cs - 2, height: cs - 2, style: { fill: mq > 0 ? GREEN : RED }, opacity: (0.08 + 0.62 * Math.min(1, Math.abs(mq))).toFixed(3) }));
              }
              if (visited[s]) {
                arrow(g, X + cs / 2, Y + cs / 2 + 4, argmax(Q[s]), 18, 0.75);
                g.appendChild(K.s("text", { x: X + 3, y: Y + 11, class: "lbl-sm mono", text: mq.toFixed(2) }));
              }
              if (x === START[0] && y === START[1]) g.appendChild(K.s("text", { x: X + cs - 3, y: Y + cs - 4, "text-anchor": "end", class: "lbl-sm", text: "start" }));
              g.addEventListener("click", function () {
                if (pos[0] === x && pos[1] === y) return;
                if (x === START[0] && y === START[1]) return;
                if (walls[s]) delete walls[s]; else walls[s] = 1;
                draw();
              });
            }
            svg.appendChild(g);
          })(x, y);
        }
        // current episode path + agent
        if (path.length > 1) {
          var d = path.map(function (q, i) { return (i ? "L" : "M") + (P + q[0] * cs + cs / 2) + " " + (P + q[1] * cs + cs / 2); }).join("");
          svg.appendChild(K.s("path", { d: d, class: "ln-thin s2", "stroke-width": 2.5, opacity: 0.6, style: { pointerEvents: "none" } }));
        }
        svg.appendChild(K.s("circle", { cx: P + pos[0] * cs + cs / 2, cy: P + pos[1] * cs + cs / 2, r: 10, class: "f2 sfg", "stroke-width": 1.5, style: { pointerEvents: "none" } }));

        // reward-per-episode chart
        var n = returns.length, lo = -2;
        returns.forEach(function (v) { if (v < lo) lo = v; });
        lo = Math.max(-10, Math.floor(lo));
        var ch = K.chart({ w: 420, h: S, x: [0, Math.max(20, n)], y: [lo, 1.2], xLabel: "episode", yLabel: "return (sum of rewards)", pad: { l: 50, r: 12, t: 16, b: 40 } });
        var stride = Math.max(1, Math.ceil(n / 400));
        for (var i = 0; i < n; i += stride) ch.g.appendChild(K.s("circle", { cx: ch.sx(i + 1), cy: ch.sy(Math.max(lo, returns[i])), r: 2, class: "f1", opacity: 0.35 }));
        var ma = [], acc = 0, W = 20;
        for (var j = 0; j < n; j++) {
          acc += returns[j]; if (j >= W) acc -= returns[j - W];
          if (j % stride === 0 || j === n - 1) ma.push([j + 1, Math.max(lo, acc / Math.min(W, j + 1))]);
        }
        if (ma.length > 1) ch.g.appendChild(K.s("path", { d: K.path(ma, ch.sx, ch.sy), class: "ln s1" }));
        var bl = bestPathLen(), best = bl === null ? null : 1 + (bl - 1) * stepCost;
        if (best !== null) {
          ch.g.appendChild(K.s("line", { x1: ch.sx(0), x2: ch.sx(Math.max(20, n)), y1: ch.sy(best), y2: ch.sy(best), class: "ln-thin s3 dash" }));
          ch.g.appendChild(K.s("text", { x: ch.sx(Math.max(20, n)) - 4, y: ch.sy(best) - 5, "text-anchor": "end", class: "lbl-sm", text: "best possible ≈ " + best.toFixed(2) }));
        }
        ch.g.appendChild(K.s("text", { x: 420 - 16, y: ch.sy(lo) - 8, "text-anchor": "end", class: "lbl-sm", text: "dots = each episode · line = 20-episode average" }));

        var row = K.h("div", { class: "viz-row" }, K.h("div", {}, svg), K.h("div", {}, ch.svg));
        K.clear(c.stage).appendChild(row);

        var last20 = returns.slice(-20), avg = last20.length ? last20.reduce(function (a, b) { return a + b; }, 0) / last20.length : 0;
        var qs = Q[sIdx(pos[0], pos[1])].map(function (v, a) { return ANAME[a] + " " + v.toFixed(2); }).join(" · ");
        var upd = lastUpd ? "Last update: " + K.tex("Q(s,a) \\leftarrow " + lastUpd.old.toFixed(2) + " + " + alpha.toFixed(2) + "\\,(" + lastUpd.target.toFixed(2) + " - " + lastUpd.old.toFixed(2) + ") = " + lastUpd.nw.toFixed(2)) +
          " where the target " + K.tex("r + \\gamma \\max_{a'} Q(s',a')") + " = " + lastUpd.target.toFixed(2) + (lastUpd.term ? " (terminal: no future)" : "") + "." : "Press <b>Step</b> to see a single Q-update, or Play / fast-forward to watch learning.";
        c.readout.innerHTML = stats([
          ["episodes finished", n], ["steps this episode", steps], ["last return", n ? returns[n - 1].toFixed(2) : "—"],
          ["avg return (last 20)", n ? avg.toFixed(2) : "—"], ["Q at agent cell", "<span style='font-size:12px'>" + qs + "</span>"],
        ]) + "<div style='margin-top:6px'>" + upd + "</div>" +
          "<div class='muted' style='margin-top:4px'>Values spread backwards from the goal one episode at a time. With ε &gt; 0 the agent keeps exploring, so its <i>actual</i> returns stay below the greedy policy's; γ &lt; 1 and the step penalty both make shorter paths worth more. Q-learning is <i>off-policy</i>: it learns the greedy path even while acting randomly — which is why it happily hugs the pit.</div>";
      }
      reset(); draw();
    },
  });

  // ---------------------------------------------------------------- multi-armed bandit
  V.register("bandit", {
    title: "Multi-armed bandit: explore vs exploit (greedy, ε-greedy, UCB)",
    desc: "Each arm pays a noisy reward around a hidden mean. Three strategies play the same slot machines side by side. Regret = reward lost versus always pulling the best arm. Press Play and watch whose regret flattens.",
    render: function (c, p, K) {
      var nArms = K.clamp(num(p.arms, 8), 2, 12), eps = num(p.epsilon, 0.1), cU = num(p.c, 2), seed = 1, MAXT = 3000;
      var mus, best, agents;
      var DEFS = [
        { key: "greedy", label: "greedy", col: "var(--c5)", cls: "s5", f: "f5" },
        { key: "eps", label: "ε-greedy", col: "var(--c2)", cls: "s2", f: "f2" },
        { key: "ucb", label: "UCB", col: "var(--c1)", cls: "s1", f: "f1" },
      ];
      function newBandit() {
        var r = K.rng(seed * 131 + nArms);
        mus = []; for (var i = 0; i < nArms; i++) mus.push(+K.randn(r).toFixed(2));
        best = argmax(mus); resetRun();
      }
      function resetRun() {
        agents = DEFS.map(function (d, i) {
          var z = mus.map(function () { return 0; });
          return { d: d, Q: z.slice(), N: z.slice(), t: 0, regret: 0, hist: [[0, 0]], r: K.rng(1000 + seed * 7 + i * 13) };
        });
      }
      function choose(ag) {
        var k = ag.d.key;
        if (k === "eps" && ag.r() < eps) return Math.floor(ag.r() * nArms);
        if (k === "ucb") {
          for (var i = 0; i < nArms; i++) if (!ag.N[i]) return i;
          var lt = Math.log(ag.t + 1);
          return argmax(ag.Q.map(function (q, j) { return q + cU * Math.sqrt(lt / ag.N[j]); }), ag.r);
        }
        return argmax(ag.Q, ag.r);
      }
      function stepAll(n) {
        for (var s = 0; s < n; s++) agents.forEach(function (ag) {
          if (ag.t >= MAXT) return;
          var a = choose(ag), rew = mus[a] + K.randn(ag.r);
          ag.N[a]++; ag.Q[a] += (rew - ag.Q[a]) / ag.N[a]; ag.t++;
          ag.regret += mus[best] - mus[a];
          ag.hist.push([ag.t, ag.regret]);
        });
        return agents[0].t < MAXT;
      }
      var pl = player(c, K, function () { var more = stepAll(10); draw(); return more; }, 50);
      c.controls.appendChild(pl.btn);
      c.controls.appendChild(K.button("+1 pull", function () { stepAll(1); draw(); }));
      c.controls.appendChild(K.button("+100", function () { stepAll(100); draw(); }));
      c.controls.appendChild(K.button("Restart run", function () { pl.stop(); resetRun(); draw(); }));
      c.controls.appendChild(K.button("New machines", function () { pl.stop(); seed++; newBandit(); draw(); }));
      c.controls.appendChild(K.slider({ label: "arms", min: 5, max: 10, value: nArms, onInput: function (v) { pl.stop(); nArms = v; newBandit(); draw(); } }));
      c.controls.appendChild(K.slider({ label: "ε", min: 0, max: 0.5, step: 0.01, value: eps, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { eps = v; } }));
      c.controls.appendChild(K.slider({ label: "UCB c", min: 0, max: 4, step: 0.1, value: cU, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { cU = v; } }));

      function draw() {
        var T = agents[0].t, maxR = 5;
        agents.forEach(function (ag) { maxR = Math.max(maxR, ag.regret * 1.1); });
        var ch = K.chart({ w: 400, h: 300, x: [0, Math.max(100, T)], y: [0, maxR], xLabel: "pulls (time t)", yLabel: "cumulative regret", pad: { l: 48, r: 60 } });
        var labY = agents.map(function (ag, k) { return { k: k, y: ch.sy(ag.regret) + 4 }; }).sort(function (a, b) { return a.y - b.y; });
        for (var li = 1; li < labY.length; li++) labY[li].y = Math.max(labY[li].y, labY[li - 1].y + 12);
        var labAt = []; labY.forEach(function (o) { labAt[o.k] = o.y; });
        agents.forEach(function (ag, k) {
          var h = ag.hist, stride = Math.max(1, Math.floor(h.length / 400)), pts = [];
          for (var i = 0; i < h.length; i += stride) pts.push(h[i]);
          pts.push(h[h.length - 1]);
          ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: "ln " + ag.d.cls }));
          ch.g.appendChild(K.s("text", { x: ch.sx(h[h.length - 1][0]) + 4, y: labAt[k], class: "lbl-sm", style: { fill: ag.d.col }, text: ag.d.label }));
        });
        // arms: true mean (grey) + each strategy's estimate
        var W = 400, H = 300, x0 = 40, bw = (W - x0 - 10) / nArms;
        var ylo = -3, yhi = 3;
        mus.forEach(function (m) { ylo = Math.min(ylo, Math.floor(m - 0.5)); yhi = Math.max(yhi, Math.ceil(m + 0.5)); });
        var sy = K.scale(ylo, yhi, H - 40, 16);
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        K.niceTicks(ylo, yhi, 6).forEach(function (v) {
          svg.appendChild(K.s("line", { x1: x0, x2: W - 10, y1: sy(v), y2: sy(v), class: v === 0 ? "axis" : "grid" }));
          svg.appendChild(K.s("text", { x: x0 - 6, y: sy(v) + 4, "text-anchor": "end", class: "tick", text: v }));
        });
        svg.appendChild(K.s("text", { x: 12, y: (H - 40 + 16) / 2, class: "axis-label", "text-anchor": "middle", transform: "rotate(-90 12 " + (H - 24) / 2 + ")", text: "mean reward" }));
        mus.forEach(function (m, i) {
          var X = x0 + i * bw;
          svg.appendChild(K.s("rect", { x: X + 3, y: Math.min(sy(0), sy(m)), width: bw - 6, height: Math.abs(sy(m) - sy(0)), class: "fmuted", opacity: 0.35 }));
          svg.appendChild(K.s("line", { x1: X + 3, x2: X + bw - 3, y1: sy(m), y2: sy(m), class: "sfg", "stroke-width": 2 }));
          agents.forEach(function (ag, k) {
            if (!ag.N[i]) return;
            var q = K.clamp(ag.Q[i], ylo, yhi), cx = X + bw * (0.28 + 0.22 * k);
            svg.appendChild(K.s("circle", { cx: cx, cy: sy(q), r: 3 + Math.min(4, Math.sqrt(ag.N[i] / Math.max(1, T)) * 6), class: ag.d.f, opacity: 0.9 }));
          });
          svg.appendChild(K.s("text", { x: X + bw / 2, y: H - 24, "text-anchor": "middle", class: "lbl", text: (i === best ? "★" : "") + (i + 1) }));
        });
        svg.appendChild(K.s("text", { x: x0 + 4, y: H - 6, class: "lbl-sm", text: "arm (★ = best) · bar = true mean · dots = estimates (size ∝ pulls)" }));
        K.clear(c.stage).appendChild(K.h("div", { class: "viz-row" }, K.h("div", {}, ch.svg), K.h("div", {}, svg)));
        c.readout.innerHTML = stats(agents.map(function (ag) {
          return [ag.d.label + " · best-arm pulls", (T ? Math.round(ag.N[best] / T * 100) : 0) + "% · regret " + ag.regret.toFixed(1), ag.d.col];
        })) + "<div style='margin-top:6px'>Regret after t pulls: " + K.tex("\\sum_{s\\le t} (\\mu^* - \\mu_{a_s})") +
          ". UCB picks " + K.tex("\\arg\\max_a \\; \\hat Q_a + c\\sqrt{\\ln t / N_a}") + " — an optimism bonus that shrinks as an arm is tried. Greedy often locks onto a lucky early arm (regret grows <i>linearly</i> forever); ε-greedy keeps paying ε for exploration forever; UCB's regret grows only like " + K.tex("\\log t") +
          ".</div><div class='muted' style='margin-top:4px'>LLM link: sampling several answers per prompt and keeping the good ones is the same explore/exploit trade-off; temperature plays the role of ε.</div>";
      }
      newBandit(); draw();
    },
  });

  // ---------------------------------------------------------------- discounting
  V.register("discount", {
    title: "Discount factor γ: how much the future is worth today",
    desc: "A reward k steps in the future is multiplied by γᵏ. Slide γ and watch how far ahead the agent effectively “cares”. Pick a reward pattern to see the discounted return G.",
    render: function (c, p, K) {
      var gamma = num(p.gamma, 0.9), pattern = "const", KMAX = 60;
      var PAT = {
        "const": { label: "+1 every step", f: function () { return 1; } },
        "late": { label: "single +10 at step 30", f: function (k) { return k === 30 ? 10 : 0; } },
        "cost": { label: "−1 per step, +40 at step 25", f: function (k) { return k < 25 ? -1 : k === 25 ? 40 : 0; } },
        "llm": { label: "LLM: 0 per token, +1 at end (token 50)", f: function (k) { return k === 50 ? 1 : 0; } },
      };
      var gs = K.slider({ label: "γ", min: 0, max: 0.999, step: 0.001, value: gamma, fmt: function (v) { return v.toFixed(3); }, onInput: function (v) { gamma = v; draw(); } });
      c.controls.appendChild(gs);
      c.controls.appendChild(K.select({ label: "rewards", value: pattern, options: Object.keys(PAT).map(function (k) { return { value: k, label: PAT[k].label }; }), onChange: function (v) { pattern = v; draw(); } }));
      [0.5, 0.9, 0.99, 1].forEach(function (g) {
        c.controls.appendChild(K.button("γ=" + g, function () { gamma = Math.min(g, 0.999); gs.set(gamma); draw(); }));
      });
      function draw() {
        var f = PAT[pattern].f, rew = [], w = [], G = 0, Gu = 0;
        for (var k = 0; k <= KMAX; k++) { rew.push(f(k)); w.push(Math.pow(gamma, k)); G += w[k] * rew[k]; Gu += rew[k]; }
        var maxR = Math.max.apply(null, rew.map(Math.abs)) || 1;
        var ch = K.chart({ w: 660, h: 300, x: [-0.5, KMAX + 0.5], y: [-0.05, 1.16], yTicks: [0, 0.25, 0.5, 0.75, 1], xLabel: "steps into the future k", yLabel: "weight γᵏ", pad: { l: 52, r: 16 } });
        var bw = ch.sx(1) - ch.sx(0);
        for (var j = 0; j <= KMAX; j++) {
          ch.g.appendChild(K.s("rect", { x: ch.sx(j) - bw * 0.4, y: ch.sy(w[j]), width: bw * 0.8, height: ch.sy(0) - ch.sy(w[j]), class: "f1", opacity: 0.55 }));
          if (rew[j]) {
            ch.g.appendChild(K.s("circle", { cx: ch.sx(j), cy: ch.sy(1.11), r: 2 + 3.5 * Math.abs(rew[j]) / maxR, style: { fill: rew[j] > 0 ? GREEN : RED } }));
          }
        }
        var hz = 1 / (1 - gamma);
        if (hz <= KMAX) {
          ch.g.appendChild(K.s("line", { x1: ch.sx(hz), x2: ch.sx(hz), y1: ch.sy(0), y2: ch.sy(1.05), class: "ln s2 dash" }));
          ch.g.appendChild(K.s("text", { x: ch.sx(hz) + 5, y: ch.sy(0.98), class: "lbl-sm", text: "effective horizon 1/(1−γ) = " + hz.toFixed(1) }));
        }
        if (gamma > 0 && gamma < 1) {
          var half = Math.log(0.5) / Math.log(gamma);
          if (half <= KMAX) ch.g.appendChild(K.s("circle", { cx: ch.sx(half), cy: ch.sy(0.5), r: 4, class: "f2" }));
        }

        K.clear(c.stage).appendChild(ch.svg);
        var half2 = gamma > 0 && gamma < 1 ? (Math.log(0.5) / Math.log(gamma)).toFixed(1) : "—";
        c.readout.innerHTML = K.tex("G = \\sum_{k} \\gamma^k r_k") + " = <b>" + G.toFixed(2) + "</b> (undiscounted sum = " + Gu.toFixed(0) + "). &nbsp;Half-life: a reward loses half its value after <b>" + half2 + "</b> steps; effective horizon " + K.tex("\\tfrac{1}{1-\\gamma}") + " = <b>" + K.fmtShort(hz) + "</b> steps" + (hz > KMAX ? " (beyond the " + KMAX + " shown)" : "") + ". Bars = weight γᵏ; top row of dots = rewards r<sub>k</sub> (green +, red −, size ∝ |r|)." +
          "<div class='muted' style='margin-top:4px'>γ = 0: only the next reward matters (myopic). γ → 1: far-future rewards count almost fully, but learning gets noisier. In RLHF / GRPO for LLMs the reward usually arrives only once, at the end of the answer, and γ = 1 is standard — the “episode” is one response.</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- REINFORCE policy gradient
  V.register("policy-gradient", {
    title: "Policy gradient (REINFORCE): nudge up whatever got rewarded — and why a baseline helps",
    desc: "A softmax policy picks one of 3 actions; each pays a noisy reward (true means shown). Two learners run in parallel on the same problem: one uses raw reward R, one uses R − b (b = running average reward). Raise the reward offset to see the no-baseline learner struggle.",
    render: function (c, p, K) {
      var BASE = [1.0, 1.5, 0.5], NAMES = ["A", "B", "C"], COLS = ["var(--c1)", "var(--c3)", "var(--c2)"], CLS = ["s1", "s3", "s2"], FCL = ["f1", "f3", "f2"];
      var lr = num(p.lr, 0.02), offset = num(p.offset, 5), sigma = num(p.noise, 1), mode = "both", seed = 1, MAXT = 2000;
      var agents;
      function reset() {
        agents = [{ name: "no baseline", useB: false }, { name: "with baseline", useB: true }].map(function (a, i) {
          return { name: a.name, useB: a.useB, th: [0, 0, 0], b: 0, n: 0, r: K.rng(seed * 17 + 5), hist: [[0, [1 / 3, 1 / 3, 1 / 3]]], gw: [], t: 0 };
        });
      }
      function stepAll(n) {
        for (var s = 0; s < n; s++) agents.forEach(function (ag) {
          if (ag.t >= MAXT) return;
          var pi = K.softmax(ag.th), a = sampleFrom(pi, ag.r());
          var R = BASE[a] + offset + sigma * K.randn(ag.r);
          var b = ag.useB ? (ag.n ? ag.b : R) : 0;
          var adv = R - b, g = pi.map(function (pk, k) { return adv * ((k === a ? 1 : 0) - pk); });
          for (var k = 0; k < 3; k++) ag.th[k] += lr * g[k];
          ag.n++; ag.b += (R - ag.b) / Math.min(ag.n, 50);
          ag.gw.push(g); if (ag.gw.length > 100) ag.gw.shift();
          ag.t++; ag.hist.push([ag.t, K.softmax(ag.th)]);
        });
        return agents[0].t < MAXT;
      }
      function gvar(ag) {
        if (ag.gw.length < 5) return null;
        var m = [0, 0, 0];
        ag.gw.forEach(function (g) { for (var k = 0; k < 3; k++) m[k] += g[k] / ag.gw.length; });
        return ag.gw.reduce(function (acc, g) { var d = 0; for (var k = 0; k < 3; k++) d += (g[k] - m[k]) * (g[k] - m[k]); return acc + d; }, 0) / ag.gw.length;
      }
      var pl = player(c, K, function () { var more = stepAll(4); draw(); return more; }, 40);
      c.controls.appendChild(pl.btn);
      c.controls.appendChild(K.button("+1 sample", function () { stepAll(1); draw(); }));
      c.controls.appendChild(K.button("+200", function () { stepAll(200); draw(); }));
      c.controls.appendChild(K.button("Reset", function () { pl.stop(); reset(); draw(); }));
      c.controls.appendChild(K.button("New seed", function () { pl.stop(); seed++; reset(); draw(); }));
      c.controls.appendChild(K.select({ label: "show", value: mode, options: [{ value: "both", label: "compare both" }, { value: "nob", label: "no baseline only" }, { value: "b", label: "with baseline only" }], onChange: function (v) { mode = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "learning rate", min: 0.002, max: 0.1, step: 0.002, value: lr, fmt: function (v) { return v.toFixed(3); }, onInput: function (v) { lr = v; } }));
      c.controls.appendChild(K.slider({ label: "reward offset", min: 0, max: 10, step: 0.5, value: offset, fmt: function (v) { return "+" + v.toFixed(1); }, onInput: function (v) { offset = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "reward noise σ", min: 0, max: 3, step: 0.1, value: sigma, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { sigma = v; } }));

      function shown() { return agents.filter(function (ag) { return mode === "both" || (mode === "b") === ag.useB; }); }
      function draw() {
        var list = shown();
        // bars
        var W = 300, H = 300, svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var sy = K.scale(0, 1, H - 52, 20), gw = (W - 50) / 3;
        [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
          svg.appendChild(K.s("line", { x1: 40, x2: W - 6, y1: sy(v), y2: sy(v), class: v ? "grid" : "axis" }));
          svg.appendChild(K.s("text", { x: 34, y: sy(v) + 4, "text-anchor": "end", class: "tick", text: Math.round(v * 100) + "%" }));
        });
        for (var k = 0; k < 3; k++) {
          var X = 44 + k * gw, bw = (gw - 14) / list.length;
          list.forEach(function (ag, j) {
            var pr = K.softmax(ag.th)[k];
            svg.appendChild(K.s("rect", { x: X + 4 + j * bw, y: sy(pr), width: bw - 3, height: sy(0) - sy(pr), class: FCL[k], opacity: ag.useB ? 0.9 : 0.4 }));
            svg.appendChild(K.s("text", { x: X + 4 + j * bw + bw / 2 - 1.5, y: sy(pr) - 4, "text-anchor": "middle", class: "lbl-sm mono", text: Math.round(pr * 100) }));
          });
          svg.appendChild(K.s("text", { x: X + gw / 2, y: H - 34, "text-anchor": "middle", class: "lbl", text: "action " + NAMES[k] }));
          svg.appendChild(K.s("text", { x: X + gw / 2, y: H - 20, "text-anchor": "middle", class: "lbl-sm", text: "E[R] = " + (BASE[k] + offset).toFixed(1) }));
        }
        svg.appendChild(K.s("text", { x: 44, y: H - 4, class: "lbl-sm", text: mode === "both" ? "pale = no baseline · solid = with baseline" : "policy probabilities π(a)" }));
        // probability lines over time
        var T = agents[0].t;
        var ch = K.chart({ w: 400, h: 300, x: [0, Math.max(100, T)], y: [0, 1], xLabel: "samples (updates)", yLabel: "π(action)", yFmt: function (v) { return Math.round(v * 100) + "%"; }, pad: { l: 50, r: 14 } });
        list.forEach(function (ag) {
          var stride = Math.max(1, Math.floor(ag.hist.length / 300));
          for (var k2 = 0; k2 < 3; k2++) {
            var pts = [];
            for (var i = 0; i < ag.hist.length; i += stride) pts.push([ag.hist[i][0], ag.hist[i][1][k2]]);
            pts.push([ag.hist[ag.hist.length - 1][0], ag.hist[ag.hist.length - 1][1][k2]]);
            ch.g.appendChild(K.s("path", { d: K.path(pts, ch.sx, ch.sy), class: (ag.useB ? "ln " : "ln-thin dash ") + CLS[k2] }));
          }
        });
        ch.g.appendChild(K.s("text", { x: 58, y: 28, class: "lbl-sm", text: mode === "both" ? "solid = with baseline · dashed = no baseline" : "" }));
        K.clear(c.stage).appendChild(K.h("div", { class: "viz-row" }, K.h("div", { style: { flex: "0.75" } }, svg), K.h("div", {}, ch.svg)));

        var st = [];
        agents.forEach(function (ag) {
          var pi = K.softmax(ag.th), er = pi.reduce(function (a, v, i) { return a + v * (BASE[i] + offset); }, 0), v = gvar(ag);
          st.push([ag.name + " · π(B)", Math.round(pi[1] * 100) + "% · E[R] " + er.toFixed(2)]);
          st.push([ag.name + " · grad variance", v === null ? "—" : v.toFixed(3)]);
        });
        var v0 = gvar(agents[0]), v1 = gvar(agents[1]);
        c.readout.innerHTML = stats(st) + "<div style='margin-top:6px'>Update per sample: " + K.tex("\\theta \\mathrel{+}= \\eta\\,(R - b)\\,\\nabla_\\theta \\log \\pi_\\theta(a)") + ", and for a softmax " + K.tex("\\nabla_{\\theta_k} \\log \\pi(a) = \\mathbb{1}[k=a] - \\pi_k") +
          ". Both estimates point the same way <i>on average</i>, but with b = 0 every action gets pushed up (rewards are all positive) and only the <i>size</i> of the push differs — pure noise. " +
          (v0 && v1 ? "Right now the baseline cuts gradient variance by <b>" + (v0 / Math.max(v1, 1e-9)).toFixed(1) + "×</b>." : "Run a few samples to measure variance.") +
          "</div><div class='muted' style='margin-top:4px'>This is exactly why PPO uses a value-function baseline (advantage A = R − V) and GRPO uses the group mean reward as b.</div>";
      }
      reset(); draw();
    },
  });

  // ---------------------------------------------------------------- PPO clipped objective
  V.register("ppo-clip", {
    title: "PPO's clipped objective: take a step, but not too far",
    desc: "r = π_new(a|s) / π_old(a|s) is how much more (or less) likely the action became. PPO maximises min(r·A, clip(r, 1−ε, 1+ε)·A). Slide ε and r; shaded red = the objective is flat there, so the gradient is zero and the update stops pushing.",
    render: function (c, p, K) {
      var eps = num(p.eps, 0.2), r = 1.3, Amag = 1;
      c.controls.appendChild(K.slider({ label: "clip ε", min: 0.05, max: 0.5, step: 0.01, value: eps, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { eps = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "ratio r", min: 0.1, max: 2.4, step: 0.01, value: r, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { r = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "|advantage|", min: 0.2, max: 2, step: 0.1, value: Amag, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { Amag = v; draw(); } }));
      function L(x, A) { var cl = K.clamp(x, 1 - eps, 1 + eps); return Math.min(x * A, cl * A); }
      function panel(A) {
        var ymax = 2.5 * Amag;
        var ch = K.chart({ w: 330, h: 290, x: [0, 2.5], y: A > 0 ? [-0.1 * Amag, ymax] : [-ymax, 0.1 * Amag], xLabel: "probability ratio r", yLabel: "objective L", pad: { l: 46, r: 10, t: 26 } });
        var y0 = ch.pad.t, y1 = ch.H - ch.pad.b;
        // trust region
        ch.g.appendChild(K.s("rect", { x: ch.sx(1 - eps), y: y0, width: ch.sx(1 + eps) - ch.sx(1 - eps), height: y1 - y0, class: "f1", opacity: 0.08 }));
        // clipped (zero-gradient) region
        var cx0 = A > 0 ? ch.sx(1 + eps) : ch.sx(0), cx1 = A > 0 ? ch.sx(2.5) : ch.sx(1 - eps);
        ch.g.appendChild(K.s("rect", { x: cx0, y: y0, width: cx1 - cx0, height: y1 - y0, style: { fill: RED }, opacity: 0.09 }));
        ch.g.appendChild(K.s("text", { x: (cx0 + cx1) / 2, y: y1 - 10, "text-anchor": "middle", class: "lbl-sm", text: "clipped → gradient 0" }));
        ch.g.appendChild(K.s("line", { x1: ch.sx(1), x2: ch.sx(1), y1: y0, y2: y1, class: "ln-thin smuted dash" }));
        var un = [], cl = [];
        for (var x = 0; x <= 2.5001; x += 0.01) { un.push([x, x * A]); cl.push([x, L(x, A)]); }
        ch.g.appendChild(K.s("path", { d: K.path(un, ch.sx, ch.sy), class: "ln-thin smuted dash" }));
        ch.g.appendChild(K.s("path", { d: K.path(cl, ch.sx, ch.sy), class: "ln " + (A > 0 ? "s3" : "s5") }));
        if (A > 0) ch.g.appendChild(K.s("text", { x: ch.sx(2.45), y: ch.sy(2.45 * A) + 14, "text-anchor": "end", class: "lbl-sm", text: "r·A (unclipped)" }));
        else ch.g.appendChild(K.s("text", { x: ch.sx(0.3) + 8, y: ch.sy(0.3 * A) + 4, class: "lbl-sm", text: "r·A (unclipped)" }));
        ch.g.appendChild(K.s("circle", { cx: ch.sx(r), cy: ch.sy(L(r, A)), r: 6, class: "f2" }));
        ch.svg.appendChild(K.s("text", { x: 50, y: 16, class: "lbl", text: A > 0 ? "A > 0: action was better than expected" : "A < 0: action was worse than expected" }));
        return ch.svg;
      }
      function explain(A) {
        var lo = 1 - eps, hi = 1 + eps;
        if (A > 0) {
          if (r > hi) return "<b>Clipped.</b> The good action is already " + r.toFixed(2) + "× more likely than under the old policy (> 1+ε). Objective is flat → no further push this round.";
          if (r < lo) return "<b>Active, full gradient.</b> The good action became <i>less</i> likely (r &lt; 1−ε): min picks r·A, so PPO pushes it back up at full strength.";
          return "<b>Active.</b> Inside the trust region: gradient = A, increase π(a).";
        }
        if (r < lo) return "<b>Clipped.</b> The bad action is already down to " + r.toFixed(2) + "× its old probability (< 1−ε). Flat → stop pushing it down.";
        if (r > hi) return "<b>Active, full penalty.</b> The bad action became <i>more</i> likely (r &gt; 1+ε): min keeps the pessimistic r·A, so PPO still pulls it down.";
        return "<b>Active.</b> Inside the trust region: gradient = A (negative), decrease π(a).";
      }
      function draw() {
        K.clear(c.stage).appendChild(K.h("div", { class: "viz-row" }, K.h("div", {}, panel(Amag)), K.h("div", {}, panel(-Amag))));
        c.readout.innerHTML = K.tex("L^{\\text{CLIP}} = \\mathbb{E}\\left[\\min\\big(r\\,A,\\ \\text{clip}(r, 1-\\epsilon, 1+\\epsilon)\\,A\\big)\\right],\\quad r = \\frac{\\pi_\\theta(a|s)}{\\pi_{\\text{old}}(a|s)}") +
          "<div style='margin-top:6px'>At r = <b>" + r.toFixed(2) + "</b>, ε = " + eps.toFixed(2) + " (trust region " + (1 - eps).toFixed(2) + "–" + (1 + eps).toFixed(2) + "):</div>" +
          "<div>• A &gt; 0 → L = <b>" + L(r, Amag).toFixed(2) + "</b>. " + explain(1) + "</div>" +
          "<div>• A &lt; 0 → L = <b>" + L(r, -Amag).toFixed(2) + "</b>. " + explain(-1) + "</div>" +
          "<div class='muted' style='margin-top:4px'>The min makes the objective a <i>pessimistic</i> bound: you never get credit for moving further than ε, but you always pay for moving the wrong way. Typical ε = 0.2. GRPO uses exactly this per-token clipped loss, with group-normalized advantages instead of a critic.</div>";
      }
      draw();
    },
  });

  // ---------------------------------------------------------------- GRPO group advantages
  V.register("grpo-group", {
    title: "GRPO: sample a group of answers, score them, compare each to the group",
    desc: "For one prompt the policy samples G answers. Each gets a reward (correct answer + proper format). The advantage is how much better each answer is than its siblings: A = (r − mean) / std. No value network needed. Resample to see different groups.",
    render: function (c, p, K) {
      var G = K.clamp(num(p.G, 8), 2, 16), pCorrect = num(p.pCorrect, 0.4), pFormat = 0.8, beta = 0.04, useStd = true, seed = 11, rows = [];
      var RIGHT = ["60 × 2.5 = 150", "2.5 h × 60 km/h = 150 km", "60 + 60 + 30 = 150", "distance = speed × time = 150"];
      var WRONG = [["60 × 2.5 = 125", 125], ["60 / 2.5 = 24", 24], ["2 × 60 = 120", 120], ["60 × 2.5 = 1500", 1500], ["60 + 2.5 = 62.5", 62.5]];
      c.controls.appendChild(K.slider({ label: "group size G", min: 2, max: 16, value: G, onInput: function (v) { G = v; sample(); } }));
      c.controls.appendChild(K.slider({ label: "p(correct)", min: 0, max: 1, step: 0.05, value: pCorrect, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { pCorrect = v; sample(); } }));
      c.controls.appendChild(K.slider({ label: "p(format ok)", min: 0, max: 1, step: 0.05, value: pFormat, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { pFormat = v; sample(); } }));
      c.controls.appendChild(K.slider({ label: "KL β", min: 0, max: 0.2, step: 0.01, value: beta, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { beta = v; draw(); } }));
      c.controls.appendChild(K.toggle({ label: "divide by std", value: useStd, onChange: function (v) { useStd = v; draw(); } }));
      c.controls.appendChild(K.button("🎲 Resample group", function () { seed++; sample(); }, "primary"));
      function sample() {
        var r = K.rng(seed * 7919 + G);
        rows = [];
        for (var i = 0; i < G; i++) {
          var ok = r() < pCorrect, fmt = r() < pFormat, txt, ans;
          if (ok) { txt = RIGHT[Math.floor(r() * RIGHT.length)]; ans = 150; } else { var w = WRONG[Math.floor(r() * WRONG.length)]; txt = w[0]; ans = w[1]; }
          var full = txt + (fmt ? " <answer>" + ans + "</answer>" : " so the answer is " + ans);
          rows.push({ txt: full, ok: ok, fmt: fmt, rew: (ok ? 1 : 0) + (fmt ? 0.2 : 0) });
        }
        draw();
      }
      function draw() {
        var n = rows.length, mean = rows.reduce(function (a, x) { return a + x.rew; }, 0) / n;
        var std = Math.sqrt(rows.reduce(function (a, x) { return a + (x.rew - mean) * (x.rew - mean); }, 0) / n);
        rows.forEach(function (x) { x.adv = (x.rew - mean) / (useStd ? std + 1e-4 : 1); });
        var maxA = Math.max(1, Math.max.apply(null, rows.map(function (x) { return Math.abs(x.adv); })));
        var rh = Math.min(22, 330 / n), top = 46, W = 720, H = top + n * rh + 26;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        svg.appendChild(K.s("text", { x: 8, y: 16, class: "lbl", text: "Prompt: “A train travels at 60 km/h for 2.5 hours. How far does it go? Put the final number in <answer> tags.”" }));
        var cx = 600, half = 75;
        [["#", 8], ["sampled answer", 30], ["correct", 356], ["format", 400], ["reward", 438], ["advantage Aᵢ", cx - 40]].forEach(function (hd) {
          svg.appendChild(K.s("text", { x: hd[1], y: top - 10, class: "lbl-sm", text: hd[0] }));
        });
        svg.appendChild(K.s("line", { x1: cx, x2: cx, y1: top - 4, y2: top + n * rh, class: "axis" }));
        rows.forEach(function (x, i) {
          var y = top + i * rh, yc = y + rh / 2 + 4, fs = rh < 18 ? "10px" : "11.5px";
          if (i % 2) svg.appendChild(K.s("rect", { x: 4, y: y, width: W - 8, height: rh, class: "fbg3", opacity: 0.7 }));
          svg.appendChild(K.s("text", { x: 8, y: yc, class: "lbl-sm mono", text: i + 1 }));
          var t = x.txt.length > 45 ? x.txt.slice(0, 44) + "…" : x.txt;
          svg.appendChild(K.s("text", { x: 30, y: yc, class: "mono", style: { fontSize: fs }, text: t }));
          svg.appendChild(K.s("text", { x: 374, y: yc, "text-anchor": "middle", style: { fill: x.ok ? GREEN : RED, fontWeight: 700 }, text: x.ok ? "✓ 1" : "✗ 0" }));
          svg.appendChild(K.s("text", { x: 416, y: yc, "text-anchor": "middle", style: { fill: x.fmt ? GREEN : RED }, text: x.fmt ? "+0.2" : "0" }));
          svg.appendChild(K.s("text", { x: 456, y: yc, "text-anchor": "middle", class: "lbl mono", text: x.rew.toFixed(1) }));
          var bw = x.adv / maxA * half;
          svg.appendChild(K.s("rect", { x: bw >= 0 ? cx : cx + bw, y: y + 3, width: Math.max(0.5, Math.abs(bw)), height: rh - 6, rx: 2, style: { fill: x.adv > 1e-6 ? GREEN : x.adv < -1e-6 ? RED : "var(--faint)" }, opacity: 0.85 }));
          svg.appendChild(K.s("text", { x: bw >= 0 ? cx + bw + 4 : cx + bw - 4, y: yc, "text-anchor": bw >= 0 ? "start" : "end", class: "lbl-sm mono", text: (x.adv >= 0 ? "+" : "") + x.adv.toFixed(2) }));
        });
        svg.appendChild(K.s("text", { x: cx - 6, y: H - 8, "text-anchor": "end", class: "lbl-sm", text: "← push down (less likely)" }));
        svg.appendChild(K.s("text", { x: cx + 6, y: H - 8, class: "lbl-sm", text: "push up (more likely) →" }));
        K.clear(c.stage).appendChild(svg);
        var nOk = rows.filter(function (x) { return x.ok; }).length, same = std < 1e-9;
        c.readout.innerHTML = stats([["group mean reward", mean.toFixed(3)], ["group std", std.toFixed(3)], ["correct in group", nOk + " / " + n], ["Σ advantages", rows.reduce(function (a, x) { return a + x.adv; }, 0).toFixed(3)]]) +
          "<div style='margin-top:6px'>" + K.tex("A_i = \\dfrac{r_i - \\operatorname{mean}(r_1..r_G)}{\\operatorname{std}(r_1..r_G)}") + " &nbsp;— every token of answer i gets the same advantage " + K.tex("A_i") + " inside the PPO-style clipped loss. " +
          (same ? "<b style='color:var(--red)'>All rewards are equal → every advantage is 0 → this prompt gives no learning signal.</b> (Too easy or too hard prompts are wasted compute; DAPO filters them out.) " : "Better-than-sibling answers are reinforced, worse ones suppressed; advantages always sum to ~0. ") +
          "</div><div style='margin-top:4px'>Full objective: " + K.tex("J = \\mathbb{E}\\big[\\min(r\\,A,\\ \\text{clip}(r)\\,A)\\big] - \\beta\\, D_{KL}(\\pi_\\theta \\,\\|\\, \\pi_{ref})") + " with β = <b>" + beta.toFixed(2) + "</b>" +
          (beta === 0 ? " — no KL leash (as in DAPO / many recent reasoning recipes): faster drift from the reference model, relying on clipping for stability." : " — the KL term keeps the policy near the reference (SFT) model; DeepSeekMath's GRPO used β = 0.04.") +
          "</div><div class='muted' style='margin-top:4px'>" + (useStd ? "Std-normalization up-weights prompts where the group barely disagrees; Dr. GRPO argues for dropping it (untick “divide by std”)." : "Without std-normalization, advantages keep the reward scale (Dr. GRPO style).") + "</div>";
      }
      sample();
    },
  });

  // ---------------------------------------------------------------- RL systems: sync vs async timeline
  V.register("rl-systems-timeline", {
    title: "RL training systems: synchronous vs asynchronous rollouts",
    desc: "LLM RL alternates generation (inference on rollout GPUs) and training. Synchronous: everyone waits for the slowest answer, then for training, then for the weight sync. Asynchronous: rollouts never stop, the trainer eats a buffer of slightly stale samples. Increase the length variance to grow the long tail.",
    render: function (c, p, K) {
      var med = 1000, sig = 0.8, trainS = 30, syncS = 6, maxStale = 2, seed = 1;
      var NR = 4, M = 16, STEPS = 4, TOK = 0.03, MAXTOK = 16384;
      c.controls.appendChild(K.slider({ label: "median response (tokens)", min: 200, max: 4000, step: 100, value: med, onInput: function (v) { med = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "length variance σ", min: 0, max: 1.5, step: 0.05, value: sig, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { sig = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "train step (s)", min: 5, max: 120, step: 5, value: trainS, onInput: function (v) { trainS = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "weight sync (s)", min: 1, max: 30, step: 1, value: syncS, onInput: function (v) { syncS = v; draw(); } }));
      c.controls.appendChild(K.slider({ label: "async max staleness", min: 1, max: 4, step: 1, value: maxStale, onInput: function (v) { maxStale = v; draw(); } }));
      c.controls.appendChild(K.button("🎲 Resample lengths", function () { seed++; draw(); }));
      function lenGen(r) { return function () { return K.clamp(Math.round(med * Math.exp(sig * K.randn(r))), 8, MAXTOK); }; }

      function simSync() {
        var L = lenGen(K.rng(seed * 101)), t = 0, rows = [], tr = [], busy = 0, tailT = 0;
        for (var g = 0; g < NR; g++) rows.push([]);
        for (var s = 0; s < STEPS; s++) {
          var ends = [];
          for (var g2 = 0; g2 < NR; g2++) {
            var ls = []; for (var i = 0; i < M; i++) ls.push(L());
            ls.sort(function (a, b) { return a - b; });
            ends.push({ mid: ls[Math.floor(M / 2)] * TOK, max: ls[M - 1] * TOK });
          }
          var gmax = Math.max.apply(null, ends.map(function (e) { return e.max; }));
          ends.forEach(function (e, g3) {
            rows[g3].push({ t0: t, t1: t + e.mid, k: "gen" }, { t0: t + e.mid, t1: t + e.max, k: "tail" }, { t0: t + e.max, t1: t + gmax, k: "idle" });
            busy += e.max; tailT += e.max - e.mid;
            rows[g3].push({ t0: t + gmax, t1: t + gmax + trainS, k: "idle" }, { t0: t + gmax + trainS, t1: t + gmax + trainS + syncS, k: "sync" });
          });
          tr.push({ t0: t, t1: t + gmax, k: "idle" }, { t0: t + gmax, t1: t + gmax + trainS, k: "train", v: s + 1 }, { t0: t + gmax + trainS, t1: t + gmax + trainS + syncS, k: "sync" });
          t += gmax + trainS + syncS;
        }
        return { rows: rows, tr: tr, H: t, steps: STEPS, rUtil: busy / (NR * t), tUtil: STEPS * trainS / t, tail: tailT / (NR * t) };
      }
      function simAsync(H) {
        var L = lenGen(K.rng(seed * 101 + 7)), dt = H / 3000, v = 0, buf = [], rows = [], tr = [], stale = [], dropped = 0, steps = 0, genT = 0;
        var gpus = [];
        for (var g = 0; g < NR; g++) {
          var slots = []; for (var i = 0; i < M; i++) slots.push({ rem: L() * TOK, ver: 0 });
          gpus.push({ slots: slots, pauseUntil: 0 }); rows.push([]);
        }
        var trState = { k: "idle", until: 0 };
        function seg(arr, t, k, ver) {
          var last = arr[arr.length - 1];
          if (last && last.k === k && last.v === ver && Math.abs(last.t1 - t) < dt * 1.5) last.t1 = t + dt;
          else arr.push({ t0: t, t1: t + dt, k: k, v: ver });
        }
        for (var t = 0; t < H; t += dt) {
          gpus.forEach(function (gp, gi) {
            if (t < gp.pauseUntil) { seg(rows[gi], t, "sync"); return; }
            seg(rows[gi], t, "gen", v); genT += dt;
            gp.slots.forEach(function (sl) {
              sl.rem -= dt;
              if (sl.rem <= 0) { buf.push(sl.ver); sl.rem = L() * TOK; sl.ver = v; }
            });
          });
          if (trState.k === "train" && t >= trState.until) {
            v++; steps++;
            trState = { k: "sync", until: t + syncS };
            gpus.forEach(function (gp) { gp.pauseUntil = t + Math.max(0.5, syncS * 0.2); });
          } else if (trState.k === "sync" && t >= trState.until) trState = { k: "idle", until: 0 };
          if (trState.k === "idle") {
            var keep = buf.filter(function (ver) { return v - ver <= maxStale; });
            dropped += buf.length - keep.length; buf = keep;
            if (buf.length >= NR * M) {
              var batch = buf.splice(0, NR * M), sMax = 0, sSum = 0;
              batch.forEach(function (ver) { sMax = Math.max(sMax, v - ver); sSum += v - ver; });
              stale.push({ max: sMax, mean: sSum / batch.length });
              trState = { k: "train", until: t + trainS, v: v + 1 };
            }
          }
          seg(tr, t, trState.k, trState.k === "train" ? trState.v : undefined);
        }
        var trainT = tr.reduce(function (a, sgm) { return a + (sgm.k === "train" ? sgm.t1 - sgm.t0 : 0); }, 0);
        var ms = stale.length ? stale.reduce(function (a, x) { return a + x.mean; }, 0) / stale.length : 0;
        var mx = stale.reduce(function (a, x) { return Math.max(a, x.max); }, 0);
        return { rows: rows, tr: tr, steps: steps, rUtil: genT / (NR * H), tUtil: trainT / H, meanStale: ms, maxStale: mx, dropped: dropped };
      }
      function draw() {
        var S = simSync(), A = simAsync(S.H);
        var W = 700, x0 = 110, x1 = W - 12, rh = 15, gap = 3;
        var sx = K.scale(0, S.H, x0, x1);
        var H = 2 * (26 + (NR + 1) * (rh + gap)) + 70;
        var svg = K.s("svg", { viewBox: "0 0 " + W + " " + H, class: "viz-svg" });
        var STY = {
          gen: { cls: "f1", op: 0.9 }, tail: { cls: "f1", op: 0.35 }, idle: { cls: "fbg3", op: 1 }, train: { cls: "f3", op: 0.9 }, sync: { cls: "f4", op: 0.85 },
        };
        function lane(y, segs, label) {
          svg.appendChild(K.s("text", { x: x0 - 6, y: y + rh - 3, "text-anchor": "end", class: "lbl-sm", text: label }));
          svg.appendChild(K.s("rect", { x: x0, y: y, width: x1 - x0, height: rh, class: "fbg3" }));
          segs.forEach(function (sg) {
            if (sg.k === "idle" || sg.t1 <= sg.t0) return;
            var st = STY[sg.k], op = st.op;
            if (sg.k === "gen" && sg.v !== undefined) op = sg.v % 2 ? 0.55 : 0.9;
            var x = sx(Math.min(sg.t0, S.H)), w = sx(Math.min(sg.t1, S.H)) - x;
            if (w <= 0) return;
            svg.appendChild(K.s("rect", { x: x, y: y, width: w, height: rh, class: st.cls, opacity: op }));
            if (sg.k === "train" && w > 26) svg.appendChild(K.s("text", { x: x + w / 2, y: y + rh - 4, "text-anchor": "middle", class: "t-white", style: { fontSize: "10px" }, text: "v" + sg.v }));
          });
        }
        var y = 8;
        [["Synchronous (on-policy)", S], ["Asynchronous (off-policy, ≤" + maxStale + " stale)", A]].forEach(function (blk) {
          svg.appendChild(K.s("text", { x: 4, y: y + 12, class: "lbl", text: blk[0] }));
          y += 20;
          for (var g = 0; g < NR; g++) { lane(y, blk[1].rows[g], "rollout GPU " + (g + 1)); y += rh + gap; }
          lane(y, blk[1].tr, "trainer"); y += rh + gap + 12;
        });
        K.niceTicks(0, S.H, 8).forEach(function (tv) {
          svg.appendChild(K.s("line", { x1: sx(tv), x2: sx(tv), y1: y - 6, y2: y - 2, class: "axis" }));
          svg.appendChild(K.s("text", { x: sx(tv), y: y + 10, "text-anchor": "middle", class: "tick", text: K.fmtShort(tv) }));
        });
        svg.appendChild(K.s("text", { x: (x0 + x1) / 2, y: y + 24, "text-anchor": "middle", class: "axis-label", text: "wall-clock time (s)" }));
        var lx = x0, ly = y + 38;
        [["gen", "generating"], ["tail", "long tail (few seqs left)"], ["idle", "idle"], ["train", "training (vN = new weights)"], ["sync", "weight sync"]].forEach(function (lg) {
          var st = STY[lg[0]];
          svg.appendChild(K.s("rect", { x: lx, y: ly - 9, width: 12, height: 10, class: st.cls, opacity: st.op, stroke: "var(--line2)" }));
          svg.appendChild(K.s("text", { x: lx + 16, y: ly, class: "lbl-sm", text: lg[1] }));
          lx += 28 + lg[1].length * 5.4;
        });
        svg.setAttribute("viewBox", "0 0 " + W + " " + (ly + 8));
        K.clear(c.stage).appendChild(svg);
        c.readout.innerHTML = stats([
          ["sync: rollout GPU util", Math.round(S.rUtil * 100) + "%"], ["sync: trainer util", Math.round(S.tUtil * 100) + "%"], ["sync: train steps", S.steps],
          ["async: rollout GPU util", Math.round(A.rUtil * 100) + "%", GREEN], ["async: trainer util", Math.round(A.tUtil * 100) + "%", GREEN], ["async: train steps (same time)", A.steps, GREEN],
          ["async: mean / max staleness", A.meanStale.toFixed(2) + " / " + A.maxStale + " versions"], ["async: samples dropped (too stale)", A.dropped],
        ]) + "<div style='margin-top:6px'>Sync mode loses <b>" + Math.round((1 - S.rUtil) * 100) + "%</b> of rollout-GPU time to waiting (" + Math.round(S.tail * 100) + "% more is spent in the under-filled long tail). Response lengths are lognormal: the slowest of " + NR * M + " answers sets the pace. " +
          "Async overlaps generation with training; the price is <b>staleness</b> — samples were produced by a policy 1–" + Math.max(1, A.maxStale) + " versions old, so the loss needs importance-ratio corrections (PPO's r already handles small gaps) and a cap on staleness.</div>" +
          "<div class='muted' style='margin-top:4px'>Simplified model: 4 rollout GPUs × 16 sequences, " + TOK * 1000 + " ms per decode step, trainer = one row. Real systems (veRL, OpenRLHF, AReaL, PipelineRL, slime) add partial rollouts, in-flight weight updates and length-aware scheduling. Note async batches fill first with <i>short</i> answers — a subtle length bias.</div>";
      }
      draw();
    },
  });
})();
