/* Alpaca Arcade play-page logic: score submit + auto-capture + star ratings. */
(function () {
  const slug = window.ARCADE_SLUG;
  const $ = (id) => document.getElementById(id);
  const CALLSIGN_KEY = "arcade_callsign";

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
    const res = await fetch(`/api/games/${slug}`);
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
    btn.textContent = "⏳ Starting sandbox…";
    status.textContent = "Spinning up a display sandbox (up to ~2 min on first launch)…";
    // Relaunching without stopping leaks a container (the embedded
    // launcher hides its own Stop button). Kill the previous session first.
    if (window.__arcadeCid) {
      const old = window.__arcadeCid;
      window.__arcadeCid = null;
      try {
        await fetch(`/api/games/${slug}/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ container_id: old }),
        });
      } catch (e) { /* best effort — relaunch anyway */ }
      frame.removeAttribute("src");
    }
    try {
      const res = await fetch(`/api/games/${slug}/launch`, { method: "POST" });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Launch failed.");
      window.__arcadeCid = data.container_id || null;
      if (hero) hero.style.display = "none";
      frame.src = data.launcher_url;
      frame.style.display = "block";
      status.textContent = "🟢 Live! Click inside to focus, then play with keyboard/mouse.";
      btn.textContent = "↻ Restart sandbox";
      btn.disabled = false;
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
        });
      }
    } catch (e) {
      status.textContent = `Launch failed: ${e.message}`;
      btn.textContent = "▶ Play";
      btn.disabled = false;
    }
  });

  // Leaving the page with a live sandbox leaks the container. Fire-and-
  // forget stop on pagehide (sendBeacon survives navigation/close).
  window.addEventListener("pagehide", () => {
    if (!window.__arcadeCid) return;
    try {
      navigator.sendBeacon(
        `/api/games/${slug}/stop`,
        new Blob([JSON.stringify({ container_id: window.__arcadeCid })], { type: "application/json" }),
      );
    } catch (e) { /* nothing to do on unload */ }
    window.__arcadeCid = null;
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
