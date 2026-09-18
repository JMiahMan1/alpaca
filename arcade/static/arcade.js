/* Alpaca Arcade play-page logic: score submit + auto-capture + star ratings. */
(function () {
  const slug = window.ARCADE_SLUG;
  const $ = (id) => document.getElementById(id);
  const CALLSIGN_KEY = "arcade_callsign";

  function watchBoard(refresh) {
    let pending = false;
    const update = async () => {
      if (document.hidden || pending) return;
      pending = true;
      try {
        await refresh();
      } catch (_) {
      } finally {
        pending = false;
      }
    };
    setInterval(update, 15000);
    window.addEventListener("focus", update);
    document.addEventListener("visibilitychange", update);
  }

  if (!slug) {
    watchBoard(async () => {
      const player = document.querySelector(".player-wrap");
      if (player) {
        const res = await fetch(window.location.pathname, { cache: "no-store" });
        if (!res.ok) return;
        const page = new DOMParser().parseFromString(await res.text(), "text/html");
        const updated = page.querySelector(".player-wrap");
        if (updated) player.replaceWith(updated);
        return;
      }
      await Promise.all(Array.from(document.querySelectorAll("[data-game-slug]"), async (card) => {
        try {
          const res = await fetch(`/api/games/${encodeURIComponent(card.dataset.gameSlug)}`, { cache: "no-store" });
          const data = await res.json();
          if (!data.success) return;
          const top = (data.game.scores || [])[0];
          card.querySelector(".game-top").textContent = top ? `${top.initials} — ${top.score}` : "No scores yet";
        } catch (_) {
        }
      }));
    });
    return;
  }

  // Remember the player's callsign across machines; it IS the arcade account.
  const savedCallsign = (localStorage.getItem(CALLSIGN_KEY) || "").toUpperCase();
  if (savedCallsign && $("score-initials") && !$("score-initials").value) {
    $("score-initials").value = savedCallsign;
  }
  function rememberCallsign(v) {
    const clean = String(v || "").replace(/[^A-Za-z0-9]/g, "").slice(0, 3).toUpperCase();
    if (clean) localStorage.setItem(CALLSIGN_KEY, clean);
    return clean;
  }

  // Fullscreen play: the game owns the screen, arcade-level keys pinned
  // at the bottom + an exit. Uses the Fullscreen API (no iframe reload).
  // Auto-enters on touch-first devices when play starts; ⛶ toggles.
  const screenEl = document.querySelector(".screen");
  // Give the game keyboard focus once loaded so Space/arrows work immediately.
  const frame = $("game-frame");
  if (frame) {
    frame.addEventListener("load", () => {
      try {
        frame.focus();
        frame.contentWindow.focus();
      } catch (_) {
        /* cross-origin focus is best-effort */
      }
    });
  }
  const COARSE = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  const controlsBtn = $("btn-controls");
  let showControls = localStorage.getItem("arcade_show_controls") === "1";
  function paintControls() {
    if (!screenEl || !controlsBtn) return;
    screenEl.classList.toggle("show-controls", showControls);
    controlsBtn.textContent = showControls ? "Hide Controls" : "Show Controls";
    controlsBtn.setAttribute("aria-pressed", String(showControls));
  }
  if (controlsBtn) controlsBtn.addEventListener("click", () => {
    showControls = !showControls;
    localStorage.setItem("arcade_show_controls", showControls ? "1" : "0");
    paintControls();
  });
  paintControls();
  function isFull() {
    return !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      (screenEl && screenEl.classList.contains("fallback-full"))
    );
  }
  // iPhone Safari has no element Fullscreen API: pin the cabinet to the
  // viewport with a class styled identically to native fullscreen.
  function setFallback(on) {
    if (!screenEl) return;
    screenEl.classList.toggle("fallback-full", on);
    notifyLauncherFull(on);
  }
  function enterFull() {
    if (!screenEl || isFull()) return;
    // Touch devices: use CSS fallback fullscreen to avoid the
    // browser's native "To exit full screen" overlay that covers
    // the on-screen controls.
    if (COARSE || !screenEl.requestFullscreen && !screenEl.webkitRequestFullscreen) {
      setFallback(true);
      return;
    }
    const p = screenEl.requestFullscreen ? screenEl.requestFullscreen() : screenEl.webkitRequestFullscreen();
    if (p && p.catch) p.catch(() => setFallback(true));
  }
  function exitFull() {
    if (screenEl && screenEl.classList.contains("fallback-full")) {
      setFallback(false);
      return;
    }
    if (document.exitFullscreen && document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (document.webkitExitFullscreen && document.webkitFullscreenElement) document.webkitExitFullscreen();
  }
  // Tell the embedded launcher when we go full (it hides its own toolbar
  // for max game area) and back (it restores it).
  document.addEventListener("fullscreenchange", () => notifyLauncherFull(isFull()));
  document.addEventListener("webkitfullscreenchange", () => notifyLauncherFull(isFull()));
  function notifyLauncherFull(on) {
    const live = $("live-frame");
    if (!live || live.style.display === "none") return;
    try {
      live.contentWindow.postMessage({ source: "arcade-key", fullscreen: on }, "*");
    } catch (_) {
      /* cross-origin post is best-effort */
    }
  }
  if ($("btn-fullscreen")) $("btn-fullscreen").addEventListener("click", () => (isFull() ? exitFull() : enterFull()));
  if ($("btn-exit-full")) $("btn-exit-full").addEventListener("click", exitFull);
  // Code-game sound: the embedded launcher autoplays muted and unmutes
  // itself, but a strict browser may keep it silent until a gesture-backed
  // toggle arrives. Post the launcher's 'arcade-audio' channel (same
  // postMessage family as arcade-key): unmute on iframe load, toggle on
  // the Sound button. Mute is not autoplay-gated, so this needs no gesture.
  let soundOn = true;
  function postLiveAudio(on) {
    const fr = $("live-frame");
    if (!fr || fr.style.display === "none") return;
    try {
      fr.contentWindow.postMessage({ source: "arcade-audio", on: !!on }, "*");
    } catch (_) {
      /* cross-origin post is best-effort */
    }
  }
  const soundBtn = $("btn-sound-live");
  if (soundBtn) soundBtn.addEventListener("click", () => {
    soundOn = !soundOn;
    soundBtn.textContent = soundOn ? "🔊 Sound" : "🔇 Muted";
    soundBtn.title = soundOn ? "Mute game sound" : "Unmute game sound";
    postLiveAudio(soundOn);
  });
  // ⌨ Type into the game: focus a hidden proxy input so the device
  // keyboard opens, then forward keystrokes into the game (high-score
  // name entry etc). Stays in fullscreen — never navigates away.
  // Whitelist: single letters/digits pass through, named keys map to
  // xdotool names; everything else (notably shell metacharacters) is
  // dropped before it can reach the launcher command line.
  function mapXkey(key) {
    if (/^[a-zA-Z0-9]$/.test(key)) return key;
    const named = {
      " ": "space",
      Enter: "Return",
      Escape: "Escape",
      Tab: "Tab",
      Backspace: "BackSpace",
      Delete: "Delete",
      ArrowUp: "Up",
      ArrowDown: "Down",
      ArrowLeft: "Left",
      ArrowRight: "Right",
    };
    return Object.prototype.hasOwnProperty.call(named, key) ? named[key] : null;
  }
  function sendGameKey(key, code, xkey, down) {
    if (xkey === null || xkey === undefined) return;
    const t = arcadeKeyTarget();
    if (!t) return;
    if (t.kind === "live") {
      try {
        t.live.contentWindow.postMessage({ source: "arcade-key", xkey, down }, "*");
      } catch (_) {
        /* cross-origin post is best-effort */
      }
      return;
    }
    try {
      t.target.dispatchEvent(
        new KeyboardEvent(down ? "keydown" : "keyup", {
          key,
          code,
          bubbles: true,
          cancelable: true,
        }),
      );
    } catch (_) {
      /* cross-origin keys are best-effort */
    }
    if (down) {
      try {
        const gf = $("game-frame");
        gf.focus();
        gf.contentWindow.focus();
      } catch (_) {
        /* cross-origin focus is best-effort */
      }
    }
  }
  const kbdProxy = $("kbd-proxy");
  if ($("btn-keyboard") && kbdProxy) {
    $("btn-keyboard").addEventListener("click", () => {
      // Toggle: tap again (or Exit) to dismiss the keyboard.
      if (document.activeElement === kbdProxy) {
        kbdProxy.blur();
        return;
      }
      try {
        kbdProxy.focus({ preventScroll: true });
      } catch (_) {
        kbdProxy.focus();
      }
    });
    kbdProxy.addEventListener("focus", () => {
      $("btn-keyboard").setAttribute("aria-pressed", "true");
      $("btn-keyboard").classList.add("on");
    });
    kbdProxy.addEventListener("blur", () => {
      $("btn-keyboard").setAttribute("aria-pressed", "false");
      $("btn-keyboard").classList.remove("on");
      kbdProxy.value = "";
    });
    kbdProxy.addEventListener("keydown", (ev) => {
      sendGameKey(ev.key, ev.code, mapXkey(ev.key), true);
      // Keep the proxy empty so every keystroke arrives as a fresh event;
      // (Backspace still reaches the game via the keydown above.)
      if (ev.key !== "Backspace") kbdProxy.value = "";
    });
    kbdProxy.addEventListener("keyup", (ev) => {
      sendGameKey(ev.key, ev.code, mapXkey(ev.key), false);
    });
    // Mobile soft keyboards often emit text without key events: forward
    // any materialized characters as taps and clear the field.
    kbdProxy.addEventListener("input", () => {
      const text = kbdProxy.value;
      kbdProxy.value = "";
      for (const ch of text) {
        sendGameKey(ch, "", mapXkey(ch), true);
        sendGameKey(ch, "", mapXkey(ch), false);
      }
    });
  }
  // Overlay key bar: code games forward X11 key names to the embedded
  // launcher (xdotool on :99); playable games get synthetic KeyboardEvents
  // in the same-origin game iframe. Press-and-hold semantics: pointerdown
  // sends keydown, release sends keyup — taps work for menus, holds work
  // for movement (a click-only tap is too short for games that poll held
  // keys, which is why arrows felt dead while Space worked).
  function arcadeKeyTarget() {
    const live = $("live-frame");
    if (live && live.style.display !== "none") return { kind: "live", live };
    const gf = $("game-frame");
    if (!gf) return null;
    try {
      const doc = gf.contentDocument;
      if (!doc) return null;
      return {
        kind: "playable",
        target:
          doc.activeElement && doc.activeElement !== doc.body
            ? doc.activeElement
            : doc.querySelector("canvas") || doc.body,
      };
    } catch (_) {
      return null;
    }
  }
  function sendArcadeKey(btn, down) {
    sendGameKey(btn.dataset.key, btn.dataset.code, btn.dataset.xkey, down);
  }
  document.querySelectorAll("#play-keys button").forEach((b) => {
    b.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      try {
        b.setPointerCapture(ev.pointerId);
      } catch (_) {
        /* older browsers */
      }
      sendArcadeKey(b, true);
    });
    for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) {
      b.addEventListener(ev, () => sendArcadeKey(b, false));
    }
    b.addEventListener("keydown", (ev) => {
      if (ev.key === " " || ev.key === "Enter") sendArcadeKey(b, true);
    });
    b.addEventListener("keyup", (ev) => {
      if (ev.key === " " || ev.key === "Enter") sendArcadeKey(b, false);
    });
  });
  if (COARSE) {
    // Auto-fullscreen on touch devices: playable games once loaded, code
    // games once the sandbox goes live. Uses CSS fallback (no native
    // fullscreen API) to avoid the browser's "To exit full screen" overlay
    // that covers the on-screen controls.
    if (frame) frame.addEventListener("load", () => setTimeout(enterFull, 400));
    window.__arcadeAutoFull = true;
  }

  // Achievement unlock celebration: toast + lightweight confetti burst.
  function celebrate(unlocks) {
    if (!unlocks || !unlocks.length) return;
    const toast = document.createElement("div");
    toast.className = "unlock-toast";
    toast.innerHTML = `<div class="unlock-title">🏆 ACHIEVEMENT UNLOCKED</div>` + unlocks
      .map((u) => `<div class="unlock-item">${u.icon} <strong>${u.name}</strong> — ${u.tagline}</div>`)
      .join("");
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
    setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => toast.remove(), 600);
    }, 6000);
    const canvas = document.createElement("canvas");
    canvas.className = "confetti-canvas";
    document.body.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const colors = ["#ff2fb3", "#22d3ee", "#ffd23f", "#a3e635", "#ffffff"];
    const bits = Array.from({ length: 160 }, () => ({
      x: window.innerWidth / 2 + (Math.random() - 0.5) * 240,
      y: window.innerHeight * 0.3,
      vx: (Math.random() - 0.5) * 12,
      vy: Math.random() * -9 - 3,
      s: Math.random() * 7 + 3,
      c: colors[(Math.random() * colors.length) | 0],
      r: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.3,
    }));
    let frames = 0;
    (function tick() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const b of bits) {
        b.x += b.vx;
        b.y += b.vy;
        b.vy += 0.35;
        b.r += b.vr;
        ctx.save();
        ctx.translate(b.x, b.y);
        ctx.rotate(b.r);
        ctx.fillStyle = b.c;
        ctx.fillRect(-b.s / 2, -b.s / 2, b.s, b.s * 0.6);
        ctx.restore();
      }
      if (++frames < 180) requestAnimationFrame(tick);
      else canvas.remove();
    })();
  }

  async function refreshScores() {
    const res = await fetch(`/api/games/${slug}`, { cache: "no-store" });
    const data = await res.json();
    if (!data.success) return;
    const list = $("score-list");
    list.innerHTML = "";
    (data.game.scores || []).forEach((s) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `/player/${s.initials}`;
      a.textContent = s.initials;
      const span = document.createElement("span");
      span.appendChild(a);
      const strong = document.createElement("strong");
      strong.textContent = s.score;
      li.appendChild(span);
      li.appendChild(strong);
      list.appendChild(li);
    });
    if (!(data.game.scores || []).length) list.innerHTML = `<li class="muted">No scores yet — be the first!</li>`;
  }

  watchBoard(refreshScores);

  $("btn-submit-score").addEventListener("click", async () => {
    const msg = $("score-msg");
    const callsign = rememberCallsign($("score-initials").value);
    const body = { initials: callsign, score: Number($("score-value").value) };
    if (!Number.isFinite(body.score) || body.score < 0) {
      msg.textContent = "Enter a valid score first (or use 🎮 Get my score).";
      return;
    }
    const res = await fetch(`/api/games/${slug}/scores`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.success) {
      const who = data.player_url ? ` <a href="${data.player_url}">📊 my stats</a>` : "";
      let extra = "";
      if (data.personal_best) extra += " 🚀 New personal record!";
      if (data.pioneer) extra += " 🚩 First score ever on this machine!";
      msg.innerHTML = (data.made_board ? `🏆 #${data.rank} on the board!` : "Saved, but outside the top 5.") + extra + who;
      refreshScores();
      celebrate(data.new_unlocks);
    } else {
      msg.textContent = data.error || "Submit failed.";
    }
  });

  // Live sandbox play for code-kind games: same UI sandbox the Tests page
  // uses (Xvfb + x11vnc + noVNC). The backend launches the container and
  // hands back a launcher URL, which streams into the cabinet screen.
  // (Absent on playable games, which already run in an iframe.)
  if ($("btn-play-live")) $("btn-play-live").addEventListener("click", async () => {
    const btn = $("btn-play-live");
    const status = $("live-status");
    const frame = $("live-frame");
    const hero = $("code-hero");
    btn.disabled = true;
    if ($("btn-stop-live")) $("btn-stop-live").disabled = true;
    btn.textContent = "⏳ Starting sandbox…";
    status.textContent = "Spinning up a display sandbox (up to ~2 min on first launch)…";
    // Fresh auto-fullscreen state per launch attempt (a relaunch after
    // exiting fullscreen must be able to re-enter on connect).
    window.__arcadeFullDone = false;
    if (window.__arcadeFullTimer) clearTimeout(window.__arcadeFullTimer);
    try {
      await stopLiveSession();
      frame.removeAttribute("src");
      const res = await fetch(`/api/games/${slug}/launch`, { method: "POST" });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Launch failed.");
      window.__arcadeCid = data.container_id || null;
      if (hero) hero.style.display = "none";
      frame.src = data.launcher_url;
      frame.style.display = "block";
      // The launcher autoplays muted and unmutes itself; re-assert unmuted
      // on iframe load in case the message raced the launcher script or
      // the frame reloaded. Also resets the Sound toggle to on.
      frame.addEventListener("load", () => {
        soundOn = true;
        if (soundBtn) {
          soundBtn.textContent = "🔊 Sound";
          soundBtn.title = "Mute game sound";
        }
        postLiveAudio(true);
      }, { once: true });
      status.textContent = "🟢 Live! Click inside to focus, then play with keyboard/mouse.";
      btn.textContent = "↻ Restart sandbox";
      btn.disabled = false;
      if ($("btn-stop-live")) $("btn-stop-live").disabled = false;
      startScorePoll(status);
      // Touch devices: auto-fullscreen via CSS fallback once the live VNC
      // stream is actually connected — never on a fixed timer. A blind
      // timer reshapes the stream iframe mid-handshake on slow links; the
      // launcher reports its inner noVNC status (see listener below) and
      // that event drives enterFull. 45 s backstop preserves the old
      // behavior if the signal ever misses.
      if (COARSE && !window.__arcadeFullDone) {
        window.__arcadeFullTimer = setTimeout(() => {
          if (!window.__arcadeFullDone) {
            window.__arcadeFullDone = true;
            enterFull();
          }
        }, 45000);
      }
      // Stream telemetry from the embedded launcher (same-origin parent,
      // cross-origin child posts state). Surfaces silent phone-side stream
      // failures that emulation cannot reproduce.
      if (!window.__arcadeVncListener) {
        window.__arcadeVncListener = true;
        window.addEventListener("message", (ev) => {
          const m = ev && ev.data;
          if (!m || m.source !== "arcade-vnc") return;
          const s = $("live-status");
          if (!s) return;
          const detail = m.detail ? ` — ${m.detail}` : "";
          s.textContent = `🟢 Live! Stream: ${m.state}${detail}`;
          // Event-driven auto-fullscreen: enter only once the inner noVNC
          // client reports connected. The backstop timer above covers a
          // missed signal — clear it once we're in.
          if (
            COARSE &&
            !window.__arcadeFullDone &&
            m.state === "inner-status" &&
            /connected/i.test(m.detail || "")
          ) {
            window.__arcadeFullDone = true;
            if (window.__arcadeFullTimer) clearTimeout(window.__arcadeFullTimer);
            enterFull();
          }
        });
      }
    } catch (e) {
      status.textContent = `Launch failed: ${e.message}`;
      if (window.__arcadeCid) {
        if ($("btn-stop-live")) $("btn-stop-live").disabled = false;
        startScorePoll(status);
      }
      btn.textContent = "▶ Play";
      btn.disabled = false;
      // No stream coming — cancel the fullscreen backstop so it can't
      // fire over the failure message.
      if (window.__arcadeFullTimer) clearTimeout(window.__arcadeFullTimer);
    }
  });

  let scorePollToken = 0;
  let scorePollTimer;
  let stoppingSession = null;

  function stopScorePoll() {
    ++scorePollToken;
    clearTimeout(scorePollTimer);
  }

  function applyScoreSync(data, status) {
    if (!data) return;
    if (!data.success) {
      if (status) status.textContent = `Score sync: ${data.error || "failed"}`;
      return;
    }
    if (data.status === "no score yet" || data.duplicate) return;
    const initials = $("score-initials");
    const value = $("score-value");
    if (initials && !initials.value) initials.value = data.initials || "";
    if (value) value.value = data.score ?? "";
    const msg = $("score-msg");
    if (msg) msg.textContent = `Auto-captured: ${data.score} points (${data.rank ? "#" + data.rank + " on the board" : "saved"})`;
    refreshScores().catch(() => {});
    celebrate(data.new_unlocks);
  }

  async function syncScore(cid) {
    const res = await fetch(`/api/games/${slug}/sync_score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ container_id: cid }),
    });
    return res.json();
  }

  function startScorePoll(status) {
    stopScorePoll();
    const token = scorePollToken;
    const cid = window.__arcadeCid;
    async function tick() {
      if (token !== scorePollToken || !cid || cid !== window.__arcadeCid) return;
      try {
        const data = await syncScore(cid);
        if (token === scorePollToken && cid === window.__arcadeCid) applyScoreSync(data, status);
      } catch (_) {
      } finally {
        if (token === scorePollToken && cid === window.__arcadeCid) scorePollTimer = setTimeout(tick, 3000);
      }
    }
    tick();
  }

  function stopLiveSession() {
    if (stoppingSession) return stoppingSession;
    const cid = window.__arcadeCid;
    stopScorePoll();
    if (!cid) return Promise.resolve();
    stoppingSession = (async () => {
      try {
        try {
          applyScoreSync(await syncScore(cid), $("live-status"));
        } catch (_) {
        }
        const res = await fetch(`/api/games/${slug}/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ container_id: cid }),
        });
        const data = await res.json();
        applyScoreSync(data.score_sync, $("live-status"));
        if (!res.ok || data.error || data.success === false) throw new Error(data.error || "Stop failed.");
        if (window.__arcadeCid === cid) window.__arcadeCid = null;
      } finally {
        stoppingSession = null;
      }
    })();
    return stoppingSession;
  }

  if ($("btn-stop-live")) $("btn-stop-live").addEventListener("click", async () => {
    const btn = $("btn-stop-live");
    const play = $("btn-play-live");
    btn.disabled = true;
    play.disabled = true;
    try {
      await stopLiveSession();
      if (window.__arcadeFullTimer) clearTimeout(window.__arcadeFullTimer);
      exitFull();
      $("live-frame").removeAttribute("src");
      $("live-frame").style.display = "none";
      if ($("code-hero")) $("code-hero").style.display = "";
      play.textContent = "Play";
    } catch (e) {
      $("live-status").textContent = `Stop failed: ${e.message}`;
      btn.disabled = false;
      startScorePoll($("live-status"));
    } finally {
      play.disabled = false;
    }
  });

  window.addEventListener("pagehide", () => {
    stopScorePoll();
    const cid = window.__arcadeCid;
    if (!cid) return;
    const url = `/api/games/${slug}/stop`;
    const body = JSON.stringify({ container_id: cid });
    let queued = false;
    try {
      queued = navigator.sendBeacon(url, new Blob([body], { type: "application/json" }));
    } catch (_) {
    }
    if (!queued) {
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        keepalive: true,
      }).catch(() => {});
    }
    window.__arcadeCid = null;
  });

  window.addEventListener("pageshow", (ev) => {
    if (!ev.persisted || window.__arcadeCid) return;
    if ($("live-frame")) {
      $("live-frame").removeAttribute("src");
      $("live-frame").style.display = "none";
    }
    if ($("btn-play-live")) $("btn-play-live").disabled = false;
    if ($("btn-stop-live")) $("btn-stop-live").disabled = true;
  });

  // Score auto-capture: the game is served same-origin, so we can read its
  // localStorage. Scan for numeric candidates and let the player pick one.
  // Falls back to manual entry when nothing plausible is found.
  // (Absent on code-kind games, which have no iframe to read from.)
  if ($("btn-fetch-score")) $("btn-fetch-score").addEventListener("click", () => {
    const box = $("score-candidates");
    const msg = $("score-msg");
    box.innerHTML = "";
    let found = [];
    try {
      const ls = document.getElementById("game-frame").contentWindow.localStorage;
      for (let i = 0; i < ls.length; i++) {
        const key = ls.key(i);
        const raw = ls.getItem(key);
        if (/score|high|best|top|points|record/i.test(key)) {
          found.push({ key, raw });
        }
      }
    } catch (e) {
      msg.textContent = "Could not read the game storage — enter your score manually.";
      return;
    }
    const candidates = [];
    for (const { key, raw } of found) {
      const num = Number(raw);
      if (Number.isFinite(num) && raw.trim() !== "") {
        candidates.push({ label: `${key}: ${num}`, score: Math.floor(num) });
        continue;
      }
      try {
        const parsed = JSON.parse(raw);
        const arr = Array.isArray(parsed) ? parsed : parsed.scores || parsed.highscores || parsed.highScores || [];
        (Array.isArray(arr) ? arr.slice(0, 5) : []).forEach((entry) => {
          const s = Number(entry && (entry.score ?? entry.points ?? entry.value));
          if (Number.isFinite(s)) candidates.push({ label: `${key}: ${s}`, score: Math.floor(s) });
        });
        ["highScore", "highscore", "best", "bestScore", "topScore"].forEach((k) => {
          const s = Number(parsed && parsed[k]);
          if (Number.isFinite(s)) candidates.push({ label: `${key}.${k}: ${s}`, score: Math.floor(s) });
        });
      } catch (e) {
        /* not JSON — skip */
      }
    }
    if (!candidates.length) {
      msg.textContent = found.length
        ? "Found game data but no clear score — enter it manually."
        : "No score found in the game — enter it manually.";
      return;
    }
    msg.textContent = "Pick your score:";
    candidates.slice(0, 8).forEach((c) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = c.label;
      b.addEventListener("click", () => {
        $("score-value").value = c.score;
        msg.textContent = "Score filled in — add initials and Submit!";
      });
      box.appendChild(b);
    });
  });

  // Star ratings.
  const widget = $("rating-widget");
  function paintRating(avg, count) {
    if (!widget) return;
    const full = Math.round(avg);
    widget.querySelector(".rating-stars").textContent = "★★★★★".slice(0, full) + "☆☆☆☆☆".slice(0, 5 - full);
    widget.querySelector(".rating-count").textContent = count ? `${avg} (${count} vote${count === 1 ? "" : "s"})` : "not rated yet";
  }
  if (widget) paintRating(Number(widget.dataset.avg || 0), Number(widget.dataset.count || 0));
  document.querySelectorAll(".rate-buttons button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const res = await fetch(`/api/games/${slug}/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stars: Number(btn.dataset.stars), initials: rememberCallsign($("score-initials").value) || undefined }),
      });
      const data = await res.json();
      const msg = $("rate-msg");
      if (data.success) {
        paintRating(data.rating.average, data.rating.count);
        msg.textContent = "Thanks for rating!";
        document.querySelectorAll(".rate-buttons button").forEach((b) => b.classList.toggle("rated", b === btn));
      } else {
        msg.textContent = data.error || "Rating failed.";
      }
    });
  });
})();
