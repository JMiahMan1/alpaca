/* Alpaca Arcade play-page logic: score submit + auto-capture + star ratings. */
(function () {
  const slug = window.ARCADE_SLUG;
  const $ = (id) => document.getElementById(id);

  async function refreshScores() {
    const res = await fetch(`/api/games/${slug}`);
    const data = await res.json();
    if (!data.success) return;
    const list = $("score-list");
    list.innerHTML = "";
    (data.game.scores || []).forEach((s) => {
      const li = document.createElement("li");
      li.innerHTML = `<span></span><strong></strong>`;
      li.children[0].textContent = s.initials;
      li.children[1].textContent = s.score;
      list.appendChild(li);
    });
    if (!(data.game.scores || []).length) list.innerHTML = `<li class="muted">No scores yet — be the first!</li>`;
  }

  $("btn-submit-score").addEventListener("click", async () => {
    const msg = $("score-msg");
    const body = { initials: $("score-initials").value, score: Number($("score-value").value) };
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
      msg.textContent = data.made_board ? `🏆 #${data.rank} on the board!` : "Saved, but outside the top 5.";
      refreshScores();
    } else {
      msg.textContent = data.error || "Submit failed.";
    }
  });

  // Score auto-capture: the game is served same-origin, so we can read its
  // localStorage. Scan for numeric candidates and let the player pick one.
  // Falls back to manual entry when nothing plausible is found.
  $("btn-fetch-score").addEventListener("click", () => {
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
    const full = Math.round(avg);
    widget.querySelector(".rating-stars").textContent = "★★★★★".slice(0, full) + "☆☆☆☆☆".slice(0, 5 - full);
    widget.querySelector(".rating-count").textContent = count ? `${avg} (${count} vote${count === 1 ? "" : "s"})` : "not rated yet";
  }
  paintRating(Number(widget.dataset.avg || 0), Number(widget.dataset.count || 0));
  document.querySelectorAll(".rate-buttons button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const res = await fetch(`/api/games/${slug}/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stars: Number(btn.dataset.stars) }),
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
