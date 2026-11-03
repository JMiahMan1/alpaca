#!/usr/bin/env node
/*
 * Phone check for a Family Game: builds it with the kit inlined, plays it in
 * Playwright at several phone/tablet sizes inside a stand-in arcade host, and
 * fails on any console error, page error, missing ready/score bridge message
 * or a run that never starts.
 *
 *   PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
 *     node scripts/family_games_check.mjs <name> [out_dir] [--seconds 8] [--devices iphone,pixel]
 *
 * Screenshots land in out_dir (default .tmp/family-check/<name>/): one title,
 * one mid-play and one game-over shot per device, plus summary.json.
 */
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium, devices } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..");
const args = process.argv.slice(2);
const name = args[0];
if (!name) {
  console.error("usage: family_games_check.mjs <name> [out_dir] [--seconds N] [--devices a,b]");
  process.exit(2);
}
const flag = (key, fallback) => {
  const i = args.indexOf(key);
  return i >= 0 ? args[i + 1] : fallback;
};
const outDir = path.resolve(args[1] && !args[1].startsWith("--") ? args[1] : path.join(repo, ".tmp", "family-check", name));
const seconds = Number(flag("--seconds", "8"));
const DEVICES = {
  iphone: { label: "iphone-13", ...devices["iPhone 13"] },
  se: { label: "iphone-se", ...devices["iPhone SE"] },
  pixel: { label: "pixel-7", ...devices["Pixel 7"] },
  tablet: { label: "ipad-landscape", ...devices["iPad Mini landscape"] },
};
const wanted = flag("--devices", "iphone,se,pixel,tablet").split(",").filter((d) => DEVICES[d]);
fs.mkdirSync(outDir, { recursive: true });

const built = path.join(outDir, "game.html");
execFileSync("python3", [path.join(repo, "arcade", "family.py"), "build", name, built], { stdio: "inherit" });
const html = fs.readFileSync(built, "utf8");

// Stand-in for the arcade play page: an iframe that records bridge messages.
const host = `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>html,body{margin:0;height:100%;background:#000}iframe{border:0;width:100%;height:100%;display:block}</style></head>
<body><iframe id="g" src="/game.html#embed=1&safe=47,0,34,0&name=Jeremiah" allow="autoplay"></iframe><script>
window.__msgs=[];addEventListener('message',e=>{const m=e.data;if(!m||m.source!=='arcade-game')return;window.__msgs.push(m);
if(m.type==='score')document.getElementById('g').contentWindow.postMessage({source:'arcade-host',type:'saved',slug:m.slug,rank:1},'*');});
</script></body></html>`;
const server = http.createServer((req, res) => {
  res.setHeader("Cache-Control", "no-store");
  if (req.url.startsWith("/game.html")) return res.end(html);
  res.setHeader("Content-Type", "text/html");
  res.end(host);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch();
const summary = { name, devices: {}, ok: true };
for (const key of wanted) {
  const dev = DEVICES[key];
  const ctx = await browser.newContext({ ...dev, label: undefined });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  await page.goto(`${base}/`);
  const frameEl = await page.waitForSelector("#g");
  const frame = await frameEl.contentFrame();
  frame.on?.("pageerror", (e) => errors.push(`frame pageerror: ${e.message}`));
  await frame.waitForSelector("#fk-overlay.fk-show [data-fk=go]", { timeout: 8000 });
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(outDir, `${dev.label}-1-title.png`) });

  // Tap Play like a finger would.
  const go = await frame.$("[data-fk=go]");
  const box = await go.boundingBox();
  await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(300);
  const started = await frame.evaluate(() => window.__familyKit && window.__familyKit.state);

  // Fuzz: taps and swipes across the play area for a few seconds.
  const vp = page.viewportSize();
  const end = Date.now() + seconds * 1000;
  let shot = false;
  while (Date.now() < end) {
    const state = await frame.evaluate(() => window.__familyKit.state);
    if (state !== "playing") break;
    const x = 20 + Math.random() * (vp.width - 40);
    const y = 90 + Math.random() * (vp.height - 130);
    if (Math.random() < 0.5) {
      await page.touchscreen.tap(x, y);
    } else {
      const dx = (Math.random() - 0.5) * 160;
      const dy = (Math.random() - 0.5) * 160;
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x + dx, y + dy, { steps: 6 });
      await page.mouse.up();
    }
    await page.waitForTimeout(120);
    if (!shot && Date.now() > end - (seconds * 1000) / 2) {
      await page.screenshot({ path: path.join(outDir, `${dev.label}-2-play.png`) });
      shot = true;
    }
  }
  if (!shot) await page.screenshot({ path: path.join(outDir, `${dev.label}-2-play.png`) });

  // Make sure there is something to report, then end the run.
  await frame.evaluate(() => {
    const kit = window.__familyKit;
    if (kit.state === "paused") kit.resume();
    if (kit.state === "playing") {
      if (kit.score === 0) kit.addScore(1);
      kit.gameOver({ reason: "check" });
    }
  });
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(outDir, `${dev.label}-3-over.png`) });
  const msgs = await page.evaluate(() => window.__msgs);
  // Played as the family app runs it (embed=1, a notch and a home bar): the
  // HUD's exit, sound and pause buttons must all be on screen and tappable.
  const hudProblems = await frame.evaluate(() => {
    const out = [];
    for (const key of ["exit-hud", "sound", "pause"]) {
      const el = document.querySelector(`[data-fk="${key}"]`);
      if (!el) { out.push(`HUD ${key} button missing`); continue; }
      const r = el.getBoundingClientRect();
      if (r.width && (r.right > innerWidth + 0.5 || r.left < -0.5)) out.push(`HUD ${key} off screen (${Math.round(r.left)}..${Math.round(r.right)} of ${innerWidth})`);
      if (r.width && (r.width < 43.5 || r.height < 43.5)) out.push(`HUD ${key} too small ${Math.round(r.width)}x${Math.round(r.height)}`);
      if (r.width && r.top < 46) out.push(`HUD ${key} under the notch (top ${Math.round(r.top)})`);
    }
    return out;
  });
  const overflow = await frame.evaluate(() => {
    const d = document.documentElement;
    return { scrollW: d.scrollWidth, scrollH: d.scrollHeight, w: innerWidth, h: innerHeight };
  });
  const problems = [...errors, ...hudProblems];
  if (started !== "playing") problems.push(`Play did not start a run (state=${started})`);
  if (!msgs.some((m) => m.type === "ready")) problems.push("no ready message");
  const over = msgs.filter((m) => m.type === "over");
  if (!over.length) problems.push("no game-over message");
  // A run that ended on its own at 0 points has nothing to save; any run
  // that scored must have posted its score.
  const scored = over.some((m) => m.score > 0);
  if (scored && !msgs.some((m) => m.type === "score" && Number.isInteger(m.score) && m.score > 0 && m.run_id)) {
    problems.push("a scoring run posted no score message");
  }
  if (overflow.scrollW > overflow.w + 1 || overflow.scrollH > overflow.h + 1) problems.push(`page overflows: ${JSON.stringify(overflow)}`);
  summary.devices[dev.label] = { problems, messages: msgs.map((m) => m.type) };
  if (problems.length) summary.ok = false;
  await ctx.close();
}
await browser.close();
server.close();
fs.writeFileSync(path.join(outDir, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
process.exit(summary.ok ? 0 : 1);
