/* Family Games kit: the shared shell every family game is built on.
 *
 * A game calls FamilyKit.create({...}) once and gets: a HUD (score, best,
 * pause, sound), title / pause / game-over cards, a pause-aware game loop, a
 * DPR-correct canvas, pointer input in CSS pixels, synthesized sound effects,
 * haptics, particles and floating text, and the arcade score bridge.
 *
 * Score bridge: on game over the kit posts
 *   {source: "arcade-game", type: "score", slug, score, run_id}
 * to the parent (the arcade play page), which saves it under the signed-in
 * family member. Each run has its own run_id, so every finished game counts
 * once and only once. The kit never writes a "score"-named localStorage key
 * the arcade could mistake for a run.
 */
(function () {
  "use strict";

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeOutBack = (t) => 1 + 2.70158 * Math.pow(t - 1, 3) + 1.70158 * Math.pow(t - 1, 2);
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
  const reducedMotion = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  function store(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      window.localStorage.setItem(key, String(value));
    } catch (_) {
      /* storage blocked (private mode): the game still works */
    }
    return null;
  }

  function el(tag, cls, html) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html !== undefined) node.innerHTML = html;
    return node;
  }

  function esc(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function fmt(n) {
    return Math.round(n).toLocaleString("en-US");
  }

  /* ---------- Sound: tiny WebAudio synth, no files ---------- */
  function createSound(slug) {
    let ctx = null;
    let master = null;
    let muted = store("fk:muted") === "1";
    function unlock() {
      if (!ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        ctx = new AC();
        master = ctx.createGain();
        master.gain.value = muted ? 0 : 0.6;
        master.connect(ctx.destination);
      }
      if (ctx.state === "suspended") ctx.resume().catch(() => {});
    }
    function tone(freq, dur, opts) {
      if (!ctx || muted) return;
      const o = opts || {};
      const t0 = ctx.currentTime + (o.delay || 0);
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = o.type || "sine";
      osc.frequency.setValueAtTime(freq, t0);
      if (o.slide) osc.frequency.exponentialRampToValueAtTime(Math.max(20, o.slide), t0 + dur);
      const vol = o.vol === undefined ? 0.3 : o.vol;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(vol, t0 + Math.min(0.015, dur / 4));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(g);
      g.connect(master);
      osc.start(t0);
      osc.stop(t0 + dur + 0.02);
    }
    function noise(dur, opts) {
      if (!ctx || muted) return;
      const o = opts || {};
      const t0 = ctx.currentTime + (o.delay || 0);
      const len = Math.max(1, Math.floor(ctx.sampleRate * dur));
      const buf = ctx.createBuffer(1, len, ctx.sampleRate);
      const data = buf.getChannelData(0);
      for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const filter = ctx.createBiquadFilter();
      filter.type = o.filter || "lowpass";
      filter.frequency.value = o.freq || 1800;
      const g = ctx.createGain();
      g.gain.value = o.vol === undefined ? 0.25 : o.vol;
      src.connect(filter);
      filter.connect(g);
      g.connect(master);
      src.start(t0);
    }
    const presets = {
      tap: () => tone(660, 0.06, { type: "triangle", vol: 0.18 }),
      pop: (p) => tone(520 * (p || 1), 0.12, { type: "sine", slide: 900 * (p || 1), vol: 0.3 }),
      coin: () => { tone(988, 0.08, { type: "square", vol: 0.12 }); tone(1319, 0.18, { type: "square", vol: 0.12, delay: 0.07 }); },
      bounce: (p) => tone(240 * (p || 1), 0.16, { type: "sine", slide: 520 * (p || 1), vol: 0.28 }),
      whoosh: () => noise(0.25, { filter: "bandpass", freq: 900, vol: 0.18 }),
      thud: () => { tone(110, 0.18, { type: "sine", slide: 50, vol: 0.4 }); noise(0.08, { freq: 600, vol: 0.15 }); },
      bad: () => { tone(220, 0.18, { type: "sawtooth", slide: 140, vol: 0.14 }); },
      combo: (n) => {
        const base = 523.25 * Math.pow(2, Math.min(12, (n || 1) * 2) / 12);
        tone(base, 0.1, { type: "triangle", vol: 0.22 });
        tone(base * 1.5, 0.16, { type: "triangle", vol: 0.18, delay: 0.06 });
      },
      win: () => [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => tone(f, 0.22, { type: "triangle", vol: 0.22, delay: i * 0.09 })),
      lose: () => [392, 349.23, 293.66, 220].forEach((f, i) => tone(f, 0.26, { type: "triangle", vol: 0.2, delay: i * 0.12 })),
      start: () => [392, 523.25, 783.99].forEach((f, i) => tone(f, 0.14, { type: "triangle", vol: 0.2, delay: i * 0.07 })),
    };
    return {
      unlock,
      tone,
      noise,
      play(name, arg) {
        const fn = presets[name];
        if (fn) fn(arg);
      },
      get muted() {
        return muted;
      },
      setMuted(on) {
        muted = !!on;
        store("fk:muted", muted ? "1" : "0");
        if (master) master.gain.value = muted ? 0 : 0.6;
      },
      _slug: slug,
    };
  }

  /* ---------- Particles + floating text ---------- */
  function createFx() {
    let fontFamily = "";
    const parts = [];
    const texts = [];
    return {
      burst(x, y, color, count, opts) {
        const o = opts || {};
        const n = reducedMotion ? Math.ceil((count || 12) / 3) : count || 12;
        for (let i = 0; i < n; i++) {
          const a = Math.random() * Math.PI * 2;
          const s = (o.speed || 220) * (0.35 + Math.random() * 0.65);
          parts.push({
            x, y,
            vx: Math.cos(a) * s,
            vy: Math.sin(a) * s - (o.lift || 60),
            life: 0,
            max: (o.life || 0.7) * (0.6 + Math.random() * 0.6),
            r: (o.size || 5) * (0.6 + Math.random() * 0.8),
            color: Array.isArray(color) ? color[i % color.length] : color,
            g: o.gravity === undefined ? 520 : o.gravity,
            shape: o.shape || "circle",
            spin: (Math.random() - 0.5) * 10,
            rot: Math.random() * Math.PI,
          });
        }
      },
      text(x, y, str, opts) {
        const o = opts || {};
        texts.push({ x, y, str, life: 0, max: o.life || 0.9, color: o.color || "#fff", size: o.size || 24 });
      },
      update(dt) {
        for (let i = parts.length - 1; i >= 0; i--) {
          const p = parts[i];
          p.life += dt;
          if (p.life >= p.max) { parts.splice(i, 1); continue; }
          p.vy += p.g * dt;
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.rot += p.spin * dt;
        }
        for (let i = texts.length - 1; i >= 0; i--) {
          const t = texts[i];
          t.life += dt;
          if (t.life >= t.max) texts.splice(i, 1);
        }
      },
      draw(ctx) {
        for (const p of parts) {
          const k = 1 - p.life / p.max;
          ctx.globalAlpha = clamp(k * 1.4, 0, 1);
          ctx.fillStyle = p.color;
          if (p.shape === "star") {
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot);
            starPath(ctx, 0, 0, p.r * 1.4, p.r * 0.6, 5);
            ctx.fill();
            ctx.restore();
          } else if (p.shape === "square") {
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot);
            ctx.fillRect(-p.r, -p.r * 0.6, p.r * 2, p.r * 1.2);
            ctx.restore();
          } else {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r * (0.5 + 0.5 * k), 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.globalAlpha = 1;
        for (const t of texts) {
          const k = t.life / t.max;
          const scale = k < 0.2 ? easeOutBack(k / 0.2) : 1;
          ctx.globalAlpha = k > 0.7 ? 1 - (k - 0.7) / 0.3 : 1;
          if (!fontFamily) fontFamily = getComputedStyle(document.body).fontFamily || "sans-serif";
          ctx.font = `900 ${Math.round(t.size * scale)}px ${fontFamily}`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.lineWidth = Math.max(3, t.size / 6);
          ctx.strokeStyle = "rgba(30, 15, 70, 0.55)";
          const y = t.y - k * 40;
          ctx.strokeText(t.str, t.x, y);
          ctx.fillStyle = t.color;
          ctx.fillText(t.str, t.x, y);
        }
        ctx.globalAlpha = 1;
      },
      clear() {
        parts.length = 0;
        texts.length = 0;
      },
      get count() {
        return parts.length + texts.length;
      },
    };
  }

  function starPath(ctx, x, y, outer, inner, points) {
    ctx.beginPath();
    for (let i = 0; i < points * 2; i++) {
      const r = i % 2 === 0 ? outer : inner;
      const a = (Math.PI * i) / points - Math.PI / 2;
      const px = x + Math.cos(a) * r;
      const py = y + Math.sin(a) * r;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  /* ---------- The kit ---------- */
  function create(cfg) {
    const slug = cfg.slug;
    const higherIsBetter = cfg.higherIsBetter !== false;
    const bestKey = `fk:${slug}:pb`;
    const sound = createSound(slug);
    const fx = createFx();
    const embedded = window.parent && window.parent !== window;

    if (cfg.accent) document.documentElement.style.setProperty("--fk-accent", cfg.accent);
    if (cfg.accent2) document.documentElement.style.setProperty("--fk-accent-2", cfg.accent2);
    document.body.classList.add("fk");
    document.title = cfg.title;

    const stage = document.getElementById("fk-stage") || document.body.appendChild(el("div"));
    stage.id = "fk-stage";
    const hud = el("div");
    hud.id = "fk-hud";
    hud.innerHTML =
      `<span class="fk-pill" data-fk="score"><small>${esc(cfg.scoreLabel || "Score")}</small><span>0</span></span>` +
      `<span class="fk-pill fk-extra" data-fk="extra"></span>` +
      `<span class="fk-spacer"></span>` +
      `<span class="fk-pill" data-fk="best"><small>Best</small><span>0</span></span>` +
      `<button class="fk-icon-btn" type="button" data-fk="sound" aria-label="Sound on or off"></button>` +
      `<button class="fk-icon-btn" type="button" data-fk="pause" aria-label="Pause">⏸</button>`;
    document.body.appendChild(hud);
    const overlay = el("div");
    overlay.id = "fk-overlay";
    overlay.setAttribute("role", "dialog");
    document.body.appendChild(overlay);
    const toastEl = el("div");
    toastEl.id = "fk-toast";
    document.body.appendChild(toastEl);

    const scoreEl = hud.querySelector('[data-fk="score"] span');
    const scorePill = hud.querySelector('[data-fk="score"]');
    const bestEl = hud.querySelector('[data-fk="best"] span');
    const extraEl = hud.querySelector('[data-fk="extra"]');
    const soundBtn = hud.querySelector('[data-fk="sound"]');
    const pauseBtn = hud.querySelector('[data-fk="pause"]');

    let state = "title"; // title | playing | paused | over
    let score = 0;
    let runId = "";
    let runCount = 0;
    let lastFrame = 0;
    let rafId = 0;
    let updateFn = null;
    let renderFn = null;
    let shakeT = 0;
    let shakeMag = 0;

    function best() {
      const raw = Number(store(bestKey));
      return Number.isFinite(raw) && store(bestKey) !== null ? raw : null;
    }
    function renderBest() {
      const b = best();
      bestEl.textContent = b === null ? "-" : fmt(b);
      // No best yet: hide the pill rather than show a lone dash.
      bestEl.parentElement.style.display = b === null ? "none" : "";
    }
    function renderSound() {
      soundBtn.textContent = sound.muted ? "🔇" : "🔊";
    }
    renderBest();
    renderSound();
    pauseBtn.style.visibility = "hidden";

    function setScore(n, opts) {
      const next = Math.max(0, Math.round(n));
      const changed = next !== score;
      score = next;
      scoreEl.textContent = fmt(score);
      if (changed && !(opts && opts.quiet) && !reducedMotion) {
        scorePill.classList.remove("fk-score-bump");
        void scorePill.offsetWidth;
        scorePill.classList.add("fk-score-bump");
      }
    }
    function addScore(n) {
      setScore(score + n);
    }
    function setExtra(html) {
      extraEl.innerHTML = html || "";
    }

    function haptic(ms) {
      if (sound.muted) return;
      try {
        if (navigator.vibrate) navigator.vibrate(ms || 12);
      } catch (_) {
        /* not allowed in this frame */
      }
    }
    function toast(text, color) {
      toastEl.textContent = text;
      toastEl.style.color = color || "#fff";
      toastEl.classList.remove("fk-on");
      void toastEl.offsetWidth;
      toastEl.classList.add("fk-on");
    }
    function shake(mag, dur) {
      if (reducedMotion) return;
      shakeMag = Math.max(shakeMag, mag || 6);
      shakeT = Math.max(shakeT, dur || 0.25);
    }

    function post(msg) {
      if (!embedded) return;
      try {
        window.parent.postMessage({ source: "arcade-game", slug, ...msg }, "*");
      } catch (_) {
        /* parent gone */
      }
    }

    function hideOverlay() {
      overlay.classList.remove("fk-show");
      overlay.innerHTML = "";
    }
    function showCard(html, bind) {
      overlay.innerHTML = `<div class="fk-card">${html}</div>`;
      overlay.classList.add("fk-show");
      if (bind) bind(overlay);
    }
    function onTap(node, fn) {
      // pointerup, not click: a click can be swallowed when the finger moved a
      // little, which feels broken on a phone.
      let downAt = null;
      node.addEventListener("pointerdown", (e) => {
        downAt = { x: e.clientX, y: e.clientY };
      });
      node.addEventListener("pointerup", (e) => {
        if (!downAt) return;
        const moved = Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y);
        downAt = null;
        if (moved < 24) {
          e.preventDefault();
          fn(e);
        }
      });
      node.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          fn(e);
        }
      });
    }

    function howToHtml() {
      return (cfg.howTo || [])
        .map((line) => {
          const [icon, text] = Array.isArray(line) ? line : ["•", line];
          return `<li><span class="fk-i">${icon}</span><span>${esc(text)}</span></li>`;
        })
        .join("");
    }

    function showTitle() {
      state = "title";
      pauseBtn.style.visibility = "hidden";
      const b = best();
      showCard(
        `<div class="fk-emoji">${cfg.emoji || "🎮"}</div><h1>${esc(cfg.title)}</h1>` +
          (cfg.tagline ? `<p class="fk-tag">${esc(cfg.tagline)}</p>` : "") +
          `<ul class="fk-how">${howToHtml()}</ul>` +
          `<button class="fk-btn" type="button" data-fk="go">▶ Play</button>` +
          (b !== null ? `<div class="fk-best-line">Your best: ${fmt(b)}</div>` : ""),
        (root) => {
          const go = root.querySelector('[data-fk="go"]');
          onTap(go, startRun);
        },
      );
    }

    function startRun() {
      sound.unlock();
      sound.play("start");
      hideOverlay();
      fx.clear();
      runCount += 1;
      runId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}-${runCount}`;
      setScore(0, { quiet: true });
      setExtra("");
      state = "playing";
      pauseBtn.style.visibility = "visible";
      if (cfg.onStart) cfg.onStart();
      post({ type: "start", run_id: runId });
      lastFrame = performance.now();
    }

    function pause() {
      if (state !== "playing") return;
      state = "paused";
      if (cfg.onPause) cfg.onPause();
      showCard(
        `<div class="fk-emoji">⏸️</div><h2>Paused</h2><p class="fk-tag">Take a breather.</p>` +
          `<div class="fk-stats"><div><b>${fmt(score)}</b><span>${esc(cfg.scoreLabel || "Score")}</span></div></div>` +
          `<button class="fk-btn" type="button" data-fk="resume">▶ Keep playing</button>` +
          `<button class="fk-btn fk-ghost" type="button" data-fk="quit">End this game</button>`,
        (root) => {
          onTap(root.querySelector('[data-fk="resume"]'), resume);
          onTap(root.querySelector('[data-fk="quit"]'), () => {
            hideOverlay();
            state = "playing";
            gameOver({ reason: "Ended early" });
          });
        },
      );
    }
    function resume() {
      if (state !== "paused") return;
      sound.unlock();
      hideOverlay();
      state = "playing";
      lastFrame = performance.now();
      if (cfg.onResume) cfg.onResume();
    }

    /**
     * Finish the run: save the personal best, send the score to the family
     * board and show the result card. `info.stats` is a list of
     * [label, value] pairs; `info.title` / `info.emoji` override the header.
     */
    function gameOver(info) {
      if (state !== "playing") return;
      const o = info || {};
      state = "over";
      pauseBtn.style.visibility = "hidden";
      const prev = best();
      const isBest = prev === null ? score > 0 : higherIsBetter ? score > prev : score < prev;
      if (isBest || prev === null) {
        store(bestKey, score);
        renderBest();
      }
      const finalScore = score;
      // Every finished run is announced; only a scoring run is saved.
      post({ type: "over", score: finalScore, run_id: runId });
      if (finalScore > 0 || cfg.postZero) post({ type: "score", score: finalScore, run_id: runId });
      sound.play(o.win ? "win" : "lose");
      const stats = (o.stats || [])
        .map(([label, value]) => `<div><b>${esc(typeof value === "number" ? fmt(value) : value)}</b><span>${esc(label)}</span></div>`)
        .join("");
      const savedText = embedded && finalScore > 0 ? "Saving to the family scoreboard…" : "";
      showCard(
        `<div class="fk-emoji">${o.emoji || (o.win ? "🏆" : cfg.emoji || "🎮")}</div>` +
          `<h2>${esc(o.title || (o.win ? "You did it!" : "Game over"))}</h2>` +
          (o.reason ? `<p class="fk-tag">${esc(o.reason)}</p>` : "") +
          `<div class="fk-big-score">${fmt(finalScore)}</div><div class="fk-sub">${esc(cfg.scoreLabel || "Score")}</div>` +
          (isBest && prev !== null ? `<div class="fk-newbest">⭐ New personal best!</div>` : "") +
          (stats ? `<div class="fk-stats">${stats}</div>` : "") +
          `<div class="fk-saved" data-fk="saved">${savedText}</div>` +
          `<button class="fk-btn" type="button" data-fk="again">↻ Play again</button>` +
          `<button class="fk-btn fk-ghost" type="button" data-fk="menu">How to play</button>`,
        (root) => {
          const again = root.querySelector('[data-fk="again"]');
          // A short guard so a frantic tap that ended the game does not
          // immediately start the next one.
          again.disabled = true;
          setTimeout(() => {
            again.disabled = false;
          }, 650);
          onTap(again, () => {
            if (!again.disabled) startRun();
          });
          onTap(root.querySelector('[data-fk="menu"]'), showTitle);
        },
      );
    }

    // The arcade page answers with the saved rank; show it on the card.
    window.addEventListener("message", (event) => {
      const msg = event.data;
      if (!msg || msg.source !== "arcade-host" || msg.slug !== slug) return;
      const saved = overlay.querySelector('[data-fk="saved"]');
      if (!saved) return;
      if (msg.type === "saved") {
        saved.textContent = msg.rank ? `Saved! #${msg.rank} on the family board 🎉` : "Saved to the family scoreboard ✓";
      } else if (msg.type === "error") {
        saved.style.color = "#c0392b";
        saved.textContent = "Couldn't save this score.";
      }
    });

    onTap(soundBtn, () => {
      sound.unlock();
      sound.setMuted(!sound.muted);
      renderSound();
      if (!sound.muted) sound.play("tap");
    });
    onTap(pauseBtn, pause);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) pause();
    });
    window.addEventListener("blur", () => pause());
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" || e.key === "p") {
        if (state === "playing") pause();
        else if (state === "paused") resume();
      }
    });
    // No long-press menus, pinch zoom or double-tap zoom inside the game.
    document.addEventListener("contextmenu", (e) => e.preventDefault());
    document.addEventListener("gesturestart", (e) => e.preventDefault());
    document.addEventListener("dblclick", (e) => e.preventDefault());
    document.addEventListener("pointerdown", () => sound.unlock(), { passive: true });

    /* Canvas with correct device-pixel scaling. Drawing uses CSS pixels. */
    function canvas(onResize) {
      const c = el("canvas");
      stage.appendChild(c);
      const ctx = c.getContext("2d");
      const view = { canvas: c, ctx, w: 0, h: 0, dpr: 1 };
      function resize() {
        const rect = stage.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
        view.w = Math.max(1, rect.width);
        view.h = Math.max(1, rect.height);
        view.dpr = dpr;
        c.width = Math.round(view.w * dpr);
        c.height = Math.round(view.h * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (onResize) onResize(view);
      }
      view.resize = resize;
      window.addEventListener("resize", resize);
      if (window.visualViewport) window.visualViewport.addEventListener("resize", resize);
      resize();
      return view;
    }

    /* Pointer input in stage CSS pixels, one finger tracked at a time. */
    function pointer(target, handlers) {
      let activeId = null;
      const node = target || stage;
      const pos = (e) => {
        const r = node.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top, id: e.pointerId, t: performance.now() };
      };
      node.addEventListener("pointerdown", (e) => {
        if (state !== "playing" || activeId !== null) return;
        activeId = e.pointerId;
        try {
          node.setPointerCapture(e.pointerId);
        } catch (_) {
          /* synthetic pointer */
        }
        e.preventDefault();
        if (handlers.down) handlers.down(pos(e), e);
      });
      node.addEventListener("pointermove", (e) => {
        if (e.pointerId !== activeId || state !== "playing") return;
        e.preventDefault();
        if (handlers.move) handlers.move(pos(e), e);
      });
      const end = (e) => {
        if (e.pointerId !== activeId) return;
        activeId = null;
        // Always release, even when the game paused mid-touch, so a finger
        // lifted during the pause card never stays "down".
        if (handlers.up) handlers.up(pos(e), e);
      };
      node.addEventListener("pointerup", end);
      node.addEventListener("pointercancel", end);
    }

    /* Game loop: update(dt) only while playing; render(dt) every frame so
     * the title and game-over screens keep their animated background. */
    function run(update, render) {
      updateFn = update;
      renderFn = render;
      if (rafId) return;
      lastFrame = performance.now();
      const frame = (now) => {
        rafId = requestAnimationFrame(frame);
        const dt = Math.min(1 / 30, Math.max(0, (now - lastFrame) / 1000));
        lastFrame = now;
        if (state === "playing" && updateFn) updateFn(dt);
        fx.update(state === "paused" ? 0 : dt);
        if (shakeT > 0) {
          shakeT = Math.max(0, shakeT - dt);
          if (shakeT === 0) shakeMag = 0;
        }
        if (renderFn) renderFn(dt);
      };
      rafId = requestAnimationFrame(frame);
    }

    /** Apply the current screen shake to a context (call inside save/restore). */
    function applyShake(ctx) {
      if (shakeT <= 0) return;
      const m = shakeMag * (shakeT / 0.25);
      ctx.translate((Math.random() - 0.5) * m, (Math.random() - 0.5) * m);
    }

    post({ type: "ready", bridge: 1 });
    if (cfg.autoTitle !== false) setTimeout(showTitle, 0);

    const api = {
      canvas,
      pointer,
      run,
      start: startRun,
      gameOver,
      pause,
      resume,
      showTitle,
      setScore,
      addScore,
      setExtra,
      toast,
      haptic,
      shake,
      applyShake,
      sound,
      fx,
      get score() {
        return score;
      },
      get state() {
        return state;
      },
      get best() {
        return best();
      },
      get playing() {
        return state === "playing";
      },
    };
    // Test hook for scripts/family_games_check.mjs; harmless in play.
    window.__familyKit = api;
    return api;
  }

  window.FamilyKit = {
    create,
    clamp,
    lerp,
    easeOutBack,
    easeOutCubic,
    roundRect,
    starPath,
    reducedMotion,
    rand: (lo, hi) => lo + Math.random() * (hi - lo),
    randInt: (lo, hi) => Math.floor(lo + Math.random() * (hi - lo + 1)),
    pick: (arr) => arr[Math.floor(Math.random() * arr.length)],
    shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    },
    fmt,
  };
})();
