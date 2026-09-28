# Alpaca LLM Benchmark Suite - Test Catalog

This document describes every benchmark category and task defined in `benchmark_tests.json`.
The suite is **fully data-driven**: adding a category or test to the JSON file automatically
exposes it in the web dashboard (Test Browser), the leaderboard, and the runner.

## How grading works

- Each test is graded by `LLMModelBenchmark._verify_functional_response` in `llm_benchmark_suite.py`.
- Knowledge / multiple-choice / numeric tests carry an `expected` field and are graded objectively.
- Code / creative / task tests are graded by per-test keyword rules (see the source for each `test_id`).
- Unknown test ids fall back to a minimal content gate (>= 30 chars, >= 8 words) so empty/garbage cannot score.

### Unified 0-100 score (all tests)

Every test now carries a single comparable `score` (0-100) in addition to the pass/fail flag:

- **Code / UI tests** are actually executed in the locked-down `alpaca-sandbox` container
  (Python, Node, C++, Java, SQL, Bash, Go, Rust; HTML/WebGL/canvas apps are rendered in
  headless Chromium and screenshotted). A clean run scores 60, plus up to 40 more when its
  `expected_output` matches the captured output. A UI that renders a screenshot scores 100.
  A crash/timeout scores 0. The prompt for code categories is suffixed with a directive
  (`CODE_DIRECTIVE`) requiring complete, runnable, self-contained programs, so "runnable
  code" is verified, not assumed. CLI games receive a scripted stdin stream so `input()`
  prompts do not crash the sandbox run.
- **code_review tests** list a buggy snippet plus `expected_issues`; the score is the fraction
  of expected issues the response actually names.
- **web tests** are graded on production (non-empty, non-refusal) and are viewable live via the
  dashboard's "Serve & View" button (hosted on a local port) or rendered inline in an iframe.
- **knowledge / open / creative** tests use the functional pass/fail as the 0-100 score.

Per-group and overall scores are reported as a percentage with a letter grade and star rating:

| Band | Letter | Stars |
|------|--------|-------|
| 90-100 | A | ★★★★★ |
| 80-89  | B | ★★★★☆ |
| 70-79  | C | ★★★☆☆ |
| 60-69  | D | ★★☆☆☆ |
| <60    | F | ★☆☆☆☆ |

- `category_<group>` blocks each carry `score`, `letter`, `stars`, `tests_run`, `tests_passed`.
- A model's `overall_score` / `overall_letter` / `overall_stars` is the mean of the per-group
  scores (each group weighted equally), making models directly comparable regardless of how many
  tests a category holds. `group_scores` is the ordered per-group list for the summary table.

### Running selected groups

`POST /api/run` accepts an optional `groups` array of group ids (the top-level categories in
`benchmark_tests.json`). When supplied, only those groups run; when omitted, all groups run.
The run config modal exposes a group multi-select populated from `GET /api/benchmark/groups`.
Progress totals (`get_total_tests_per_model`) honor the same filter.

### Re-running only outdated benchmarks

`POST /api/run` also accepts `"outdated_only": true`. When set, the backend computes which of
the selected models' recorded results are stale (their stored test definition hash or prompt
no longer matches `benchmark_tests.json` — see `_compute_test_hash`) and runs **only** those
test ids. If nothing is outdated the API returns `{"status": "No outdated benchmarks", ...}`
and no run starts. The dashboard's **⚠ Run Outdated** button on the General page triggers this
endpoint. Since explicitly passing `test_ids` always bypasses resume-skip, outdated re-runs
are never skipped.

## Code quality + AI-watermark scoring (all tests)

Every response that produced text is additionally scored and stored on the result:

- `code_quality` (0-100): fenced code, definitions, comments/docstrings, length, placeholder/truncation
  penalties, and a real Python `ast` syntax check where applicable (`syntax_valid`).

- `watermark` (0-100, higher = more AI 'signature'): flags em/en dashes, box-drawing glyphs,
  boilerplate phrases ('Certainly', 'Here is', 'As an AI', 'Feel free to', ...), and excessive emoji.

These let the leaderboard rank models not only on correctness but on how clean / 'human' the
output is. All generated source in this repo is kept free of em/en dashes and box-drawing chars.

## One-shot, error-free expectation

Each task is designed to be solvable in a single pass. A strong model's output should run without
errors; the `code_quality.syntax_valid` flag surfaces Python that fails to parse. For non-multimodal
models the prompts allow text/code workarounds (e.g. describe the image task) - models should try
their best and work around capability gaps creatively rather than refuse.

## Game conventions

Games are the one category type with explicit structural rules:

- **UI-first**: most games are `pygame` (gamedev / retrogames / youtuber) or web UI
  (`gamedev_alt`: HTML5 canvas, three.js, raw WebGL). Only **three** CLI games exist —
  `guess_game` (number guessing), `text_adventure` (story-driven choose-your-destiny), and
  `game_checkers_cli` (high-logic terminal TUI). No arcade game has a CLI counterpart.
- **Scoring contract**: every arcade/board game prompt requires (1) points awarded for
  gameplay actions, (2) a name/initials entry when a run finishes, (3) a persistent top-5
  high-score board saved to a local JSON file (or `localStorage` for web games) that survives
  restarts, and (4) the score resetting to 0 on every new game (never carry the previous
  session's score forward). `_has_persistent_scoreboard` enforces all four conditions during
  grading, so a game that scores but omits name entry, persistence, or reset fails.
- **CLI presentation**: the CLI games must look polished — `game_checkers_cli` is explicitly
  a TUI (terminal user interface), not a bare text dump.
- **Web games**: must be single-file HTML using a local `three.min.js` (never a CDN), render
  into an 800x600 canvas, and auto-start without a user gesture so the headless-Chromium
  screenshot captures a non-blank frame.

## Categories

**Total: 46 categories, 292 tests.**

| Category | Tests | Tasks |
|----------|-------|-------|
| `agentic` | 3 | Agentic: Multi-hour incident forensics (real-world needle-in-haystack); Agentic: Large legacy codebase migration plan (complexity + compaction); Agentic: Long-running autonomous service simulation (pygame, resilience) |
| `android` | 4 | Android: MainActivity + RecyclerView (Kotlin); Android: ViewModel + LiveData counter (Kotlin); Android: Retrofit service interface (Kotlin); Android: Room DB + DAO (Kotlin) |
| `appdev` | 8 | App: Flask TODO CRUD API; App: Order workflow state machine; App: LRU cache with eviction; App: Common Log Format parser; App: Token-bucket rate limiter; Fake OS Desktop (Web); Kanban Task Board (Web); Personal Expense Tracker (Web) |
| `bash` | 4 | Bash: tar backup with rotation; Bash: CSV column sums with awk; Bash: HTTP health-check loop; Bash: top CPU process monitor |
| `basic` | 4 | BASIC: number guessing game (yabasic); BASIC: Fibonacci loop (yabasic); BASIC: grade calculator (yabasic); BASIC: countdown loop (yabasic) |
| `biblical` | 3 | Biblical: OT covenants; Biblical: ANE traditions in Genesis; Biblical: 2nd Temple NT context |
| `code_review` | 5 | Off-by-one in loop; Null dereference; SQL injection; Race condition; Resource leak |
| `coding` | 9 | LVGL: button and label screen (C); LVGL: multi-widget dashboard (C); ESPHome: DHT22 climate sensor (YAML); ESPHome: multi-component automation (YAML); Python: debug logic error; Code: refactor for efficiency; Game: Number Guessing Game; Game: Text Adventure Game; Game: Checkers (terminal TUI) |
| `cpp` | 5 | C++: sum a std::vector<int>; C++: RAII with std::unique_ptr; C++: function template max(a,b); C++: BankAccount class; C++: two threads + mutex counter |
| `creative` | 4 | Creative: sci-fi story opening; Creative: generate analogy; Creative: SVG theme motif tile + icon sprite sheet; Creative: illuminated manuscript of John 1 (late-medieval HTML leaf) |
| `database` | 5 | SQL: join customers/orders, HAVING count>3; SQL: add index on orders(customer_id); SQL: atomic $100 transfer w/ rollback; SQL: CREATE TABLE users; SQL: monthly revenue per product |
| `debugging` | 5 | Debug: off-by-one array copy; Debug: NullPointerException on chained call; Debug: unsynchronized shared counter; Debug: infinite loop (no increment); Debug: SQL injection via concatenation |
| `frontier_diagnostics` | 15 | Frontier: Python boundary debugging; Frontier: single-pass anagram refactor; Frontier: Python API contract; Frontier: self-contained canvas signal lab; Frontier: rational work-rate math; Frontier: minimum label swaps; Frontier: API conflict knowledge; Frontier: strict JSON feature patch; Frontier: exact tool-call JSON; Frontier: exact three-line transform; Frontier: noisy release-train needle; Frontier: idempotent tool ledger; Frontier: cross-note header constraint; Frontier: async cache robustness review; Frontier: atomic publish review |
| `gamedev` | 5 | Game: Pong (playable, pygame UI); Game: Snake (playable, pygame UI); Game: Breakout (playable, pygame UI); Game: Space Defender (Vanilla HTML5 Canvas, zero-framework UI); Game: Rotating 3D cube (pygame + PyOpenGL) |
| `gamedev_alt` | 11 | Game: 3D Pong (three.js WebGL); Game: 3D voxel terrain flyover (raw WebGL, no framework); Game: Snake (HTML5 canvas UI); Game: Breakout (HTML5 canvas UI); Game: 3D Asteroids (three.js WebGL); Game: Checkers (modern UI); Slime Mold Maze Agents (Web); Procedural Dungeon Fog of War (Web); Memory Match Pairs (Web); Top-Down Driving Track (Web); Game: sprite + music generated by the image and audio services |
| `gpqa_diamond` | 8 | GPQA-Diamond: SN2 steric hindrance; GPQA-Diamond: infinite well ground state probability; GPQA-Diamond: Okazaki fragment polymerase; GPQA-Diamond: Henderson-Hasselbalch ratio; GPQA-Diamond: photon energy-momentum; GPQA-Diamond: nitro group directing effect; GPQA-Diamond: Okazaki fragment joining enzyme; GPQA-Diamond: nitrogen ground-state config |
| `hle` | 8 | HLE: Heisenberg uncertainty conjugate variable; HLE: least common multiple 1..10; HLE: 10th prime number; HLE: sum of integers 1 to 100; HLE: chemical formula of water; HLE: halting problem 1936 paper; HLE: first crewed Moon landing year; HLE: incompleteness theorems author |
| `home_automation` | 2 | HA: control smart device; HA: report device status |
| `iac` | 4 | IaC: Terraform VPC + subnet + security group; IaC: Terraform EC2 instance; IaC: GitHub Actions CI workflow (YAML); IaC: Pulumi YAML S3 bucket + IAM user |
| `ifeval` | 8 | IFEval: include word 'penguin'; IFEval: valid JSON object only; IFEval: end with 'done'; IFEval: mention country 'Brazil'; IFEval: begin with 'Greetings'; IFEval: state capital of Japan; IFEval: include word 'necessary'; IFEval: respond 'affirmative' only |
| `instruction` | 3 | JSON: extract structured data; Summarization: 3 bullet points; Strict Adherence Summary (Luke) |
| `java` | 5 | Java: Stream filter evens; Java: parseInt with NumberFormatException; Java: HashMap word frequencies; Java: Shape interface + Circle; Java: JDBC SELECT with try-with-resources |
| `knowledge` | 17 | Knowledge: MMLU: solve for x; Knowledge: MMLU: chemical symbol for gold; Knowledge: MMLU: the Red Planet; Knowledge: MMLU: ATP organelle; Knowledge: MMLU: WWII end year; Knowledge: GPQA: exothermic reaction; Knowledge: GPQA: Pauli exclusion principle; Knowledge: GSM8K: trees planted; Knowledge: GSM8K: total apples; Knowledge: TruthfulQA: objects falling; Knowledge: TruthfulQA: 10% brain myth; Knowledge: HellaSwag: eggs at home; Knowledge: HellaSwag: opened the fridge; Knowledge: WinoGrande: trophy and suitcase; Knowledge: WinoGrande: cat scratched dog; Knowledge: ARC: renewable energy; Knowledge: ARC: force keeping planets in orbit |
| `languages` | 7 | Lang: Go net/http JSON endpoint; Lang: Rust stdin line counter; Lang: Node.js time endpoint; Lang: static HTML form + inline script; Lang: Python file line/word counter; Lang: TypeScript DOM click handler; Lang: Kotlin Ktor ping route (framework) |
| `life` | 9 | Life: calming bedtime story; Life: 3 original dad jokes; Life: what a timing belt does; Life: RAM vs storage for a parent; Life: balcony container garden; Life: backyard chicken husbandry; Life: seasonal home maintenance; Life: kids chore chart + rewards; Life: 50/30/20 budget explainer |
| `linux_admin` | 5 | Linux: find files >100MB sorted; Linux: recursive 755 dirs / 644 files; Linux: journalctl error logs (last hour, nginx); Linux: top 10 largest dirs under /; Linux: harden sshd_config |
| `linux_driver` | 4 | Linux driver: char device (C); Linux driver: platform driver (C); Linux driver: ioctl handler (C); Linux driver: misc device (C) |
| `logic` | 7 | Logic: Knights and Knaves; Logic: wolf/goat/cabbage crossing; Logic: modus tollens; Logic: categorical syllogism; Logic: 8 balls, 1 heavier, 2 weighings; Bridge Crossing Torch Puzzle; Twins Birthday Paradox Trick |
| `math_hard` | 8 | Math-Hard: single-elimination matches; Math-Hard: divisors of 360; Math-Hard: 2 to the 10th power; Math-Hard: combinations C(10,3); Math-Hard: hexagon interior angle sum; Math-Hard: solve 3^x = 81; Math-Hard: 2x2 determinant; Math-Hard: infinite geometric series |
| `metacog` | 2 | Meta: resist overthinking (concise); Meta: infinite-loop detection |
| `mmlu_pro` | 11 | MMLU-Pro: probability two dice sum to 7; MMLU-Pro: derivative of sin(x^2); MMLU-Pro: special relativity length contraction; MMLU-Pro: lowest boiling noble gas; MMLU-Pro: photosynthesis organelle; MMLU-Pro: US Declaration of Independence author; MMLU-Pro: Miranda rights amendment; MMLU-Pro: author of Critique of Pure Reason; MMLU-Pro: binary search complexity; MMLU-Pro: Keynesian recession policy; MMLU-Pro: Schwarzschild radius scaling |
| `multimodal` | 3 | Image: identify the ON light switch; HTML: render a contact form; Node: compute the 10th Fibonacci number |
| `networking` | 4 | Net: TCP echo server (Python socket); Net: HTTP client status + body length (Python); Net: DNS hostname resolver (Python); Net: threaded TCP port scanner (Python) |
| `office` | 7 | Office: professional delay email; Office: spreadsheet formulas (SUM/AVG/VLOOKUP); Office: 5-slide pitch deck outline; Office: Pillow image scaling; Office: SVG logo for a coffee shop; Office: rewrite/proofread a paragraph; Office: text-to-speech via Python |
| `pascal` | 4 | Pascal: array of records (fpc); Pascal: recursive factorial (fpc); Pascal: bubble sort (fpc); Pascal: line/word counter (fpc) |
| `performance` | 2 | Performance: Medium Load (800 tokens); Performance: Long Load (1000 tokens) |
| `reasoning` | 2 | Logic: identify rule; Math: train meeting problem |
| `retrogames` | 24 | Retro: Space Invaders (playable, pygame UI); Retro: 3D Asteroids (pygame + PyOpenGL); Retro: Vertical space shooter (playable, pygame UI); Retro: 3D Subway-Surfers runner (pygame + PyOpenGL); Retro: Temple-Run runner (playable, pygame UI); Retro: Donkey Kong (playable, pygame UI); Retro: Super Mario platformer (playable, pygame UI); Retro: 2D Asteroids vector arcade (playable, pygame UI); Retro: Crossy Road hopper (playable, pygame UI); Retro: Flappy Bird (playable, pygame UI); Retro: Ecco-the-Dolphin swim (playable, pygame UI); Retro: Pac-Man (playable, pygame UI); Retro: Tetris (playable, pygame UI); Retro: 3D Minecraft voxel chunk (pygame + PyOpenGL); Retro: 3D first-person shooter scene (pygame + PyOpenGL); Retro: Block Blast board (playable, pygame UI); Retro: Sokoban puzzle (playable, pygame UI); Retro: Arkanoid breakout (playable, pygame UI); Retro: City-sim tick (playable, pygame UI); Retro: Space Invaders (node); Retro: Space Invaders (threejs); Retro: Space Invaders (golang); Retro: Space Invaders (rust); Retro: Space Invaders (cpp) |
| `rpm` | 4 | RPM: minimal spec file; RPM: full lifecycle build spec; RPM: -devel subpackage spec; RPM: %doc/%config/%attr spec |
| `threedprint` | 8 | 3DP: G-code 20mm cube outline; 3DP: G-code heat and wait; 3DP: OpenSCAD cube with center hole; 3DP: OpenSCAD spur gear; 3DP: Python binary STL generator; 3DP: send job to OctoPrint REST API; 3DP: submit job to a laser cutter API; 3DP: slicer config (layer/infill/support) |
| `tvdev` | 5 | TV/App: Android Activity + Button; TV/App: Android TV leanback browse; TV/App: Roku BrightScript SceneGraph; TV/App: Samsung Tizen web app; TV/App: LG webOS TV app |
| `typescript` | 4 | TS: shapes + interface (tsc); TS: memoized Fibonacci (tsc); TS: string utility functions (tsc); TS: typed JSON parsing (tsc) |
| `uiux` | 15 | UI/UX: WCAG login-form critique; UI/UX: responsive blog layout; UI/UX: WCAG contrast computation; UI/UX: signup screen wireframe; UI/UX: password-reset user flow; UI/UX: fluid design system & CSS tokens; UI/UX: WCAG 2.2 accessible modal dialog; UI/UX: responsive metrics dashboard grid; UI/UX: interactive form validation & password meter; UI/UX: stacked toast notification system; UI/UX: mobile bottom sheet & desktop flyout; UI/UX: live theme customizer & palette switcher; UI/UX: interactive accessible data table UX; UI/UX: multi-step onboarding wizard; UI/UX: animated segmented tab control |
| `usb` | 4 | USB: claim interface + bulk read (C ioctl); USB: libusb descriptor dump (C); USB: HID report descriptor parser (C); USB: read /dev/hidraw reports (C) |
| `webdev` | 5 | Web: fetch JSON and render into DOM; Web: event delegation on a list; Web: validate email + 8-char password; Web: persist a theme in localStorage; Web: toggle a hidden class on click |
| `youtuber` | 3 | Falling Sand Physics (Web); Game: Conway's Game of Life (pygame, seedable + patterns); Game: Boids flocking simulation (pygame, emergent behavior) |

## Tests that drive the other services

Four tests exist to measure the *pipeline*, not the model: they call sd-server, the
audio server, or both, and are graded on criteria rather than on a grader reading prose.

| Test id | Category | Type | What the score means |
|---|---|---|---|
| `creative_svg_theme_pack` | `creative` | functional | A **seamless** motif tile (square viewBox with matching `width`/`height`, `fill="none"`, `stroke-width` 1–1.5, `stroke-opacity` 0.06–0.22, colours as `{{accent}}` placeholders, no glyphs) **and** a 6-`<symbol>` sprite sheet. Both are required; a tile that is not square seams at every edge when CSS repeats it, and a baked hex is wrong on every theme but one. |
| `creative_illuminated_manuscript_john1` | `creative` | ui | A late-medieval illuminated leaf of John 1 (KJV): all 18 verses present, a page-named container with **its own** border, a real inline `<svg>` with a positive viewBox, and a working page flip. Graded by `manuscript_rubric.py`, then `grade_code(ui=True)`. |
| `composite_sprite_bgm_game` | `gamedev_alt` | composite | The full four-stage tool path — the LLM authors the game, sd-server renders the sprite, the audio server renders the loop, both placeholders become `data:` URIs — then headless `grade_code`. Four criteria: `llm_game_authored`, `sprite_generated`, `bgm_generated`, `game_runs_headless`; `score = round(100 * passed / 4)`. |
| `office_tts` | `office` | functional | Scores the **code**, not the audio. It writes Python against a TTS library; the real Kokoro service is exercised by `tests/test_podcast_routes.py` and `tests/test_audio_server_unit.py`, not by a benchmark. |

**Why the first two are hand-authored SVG and not diffusion:** every SD flyer preset's
negative prompt bans "garbled text, distorted letters, bad typography", and the repo ships
**zero font files**, so any `<text>` in a generated asset falls back to whatever the render
host happens to have. Deterministic SVG is also machine-checkable, which is what makes
"seamless" and "no baked colours" gradable at all.

**The composite test is the only one that writes an artifact.** Its assembled HTML lands in
`data/artifacts/<model>__<test_id>.html`, which is exactly what `arcade_publish.find_artifact_file`
globs — so a passing run is playable in the arcade without a manual publish step. It lives in
`gamedev_alt` (a `GAME_CATEGORIES` member) because a `composite` record's `type` is neither
`ui` nor a game category, so bulk publish would otherwise skip it.

## Running, resuming, exporting, and deleting

Benchmarks are driven from `web/app.py` (`/api/run`) and `llm_benchmark_suite.py`
(`LLMModelBenchmark`). All task content is read from `benchmark_tests.json` - there is
no hardcoded task list in code.

### Run
- `POST /api/run` with `{ "models": [...], "use_proxy": true, "test_ids": [...], "resume": false }`.
- `test_ids` runs a subset; omit to run everything.
- Categories run **easier-first** (instruction, creative, reasoning, home_automation,
  metacog, life, biblical, uiux, office, then code/game/app/web/sys/db/language/TV
  categories, then heavy knowledge corpora) so failures surface early.

### Resume / crash safety
- Each model's results are saved to a per-model file (`data/llm_benchmarks/models/general_<model>.json`)
  as soon as that model finishes, so a crash mid-batch never loses completed models.
- Pass `resume: true` to skip any test the model already passed in its per-model file and
  reuse the prior result (marked with a fresh `last_run` timestamp). Resume only applies when
  no `test_ids` are supplied, so explicit test selections (including `outdated_only`) always
  re-run.

### Export
- `GET /api/benchmarks/export?format=json` - full JSON of every test across every model.
- `GET /api/benchmarks/export?format=csv` - flat per-test CSV (model, category, test_id,
  success, code_quality, syntax_valid, watermark, tokens/sec, ttft, tokens, last_run).
- The dashboard "Export Report" button downloads a Markdown summary (overall score, per
  category breakdown, code quality + watermark per test, prompt/response logs); "CSV"
  downloads the flat export.

### Delete
- Per-model: `DELETE /api/benchmarks/model/<model>` removes that model's benchmark file and
  artifacts and prunes it from the merged snapshot.
- Removing a local model via the dashboard prompts whether to also delete its benchmark
  history (`remove_benchmarks`); choosing Cancel keeps the history.

## Per-test result schema
Each stored test includes: `success`, `error`, `response`, `score` (unified 0-100),
`code_quality` ({score, language, syntax_valid, notes}), `watermark` ({score, flags}),
plus timing/token fields. For code/UI tests it also includes `code_ran` (bool), `code_score`,
`code_output`, `code_error`, and (for rendered UIs) `screenshot` (base64 PNG).
The leaderboard aggregates `category_*` stats and `computeGeneralRow` produces an overall
score (80% success + 20% speed) with avg TPS, TTFT, and tokens. The new unified `overall_score`
/ `overall_letter` / `overall_stars` and `group_scores` are computed in
`LLMModelBenchmark._compute_overall` and surfaced in the dashboard's results summary.
