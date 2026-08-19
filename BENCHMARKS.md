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

**Total: 34 categories, 213 tests.**

| Category | Tests | Tasks |
|----------|-------|-------|
| `agentic` | 3 | Agentic: incident forensics (needle-in-haystack secret); Agentic: legacy codebase migration plan; Agentic: long-running autonomous service (pygame, resilience) |
| `appdev` | 5 | App: Flask TODO CRUD API; App: Order workflow state machine; App: LRU cache with eviction; App: Common Log Format parser; App: Token-bucket rate limiter |
| `biblical` | 3 | Biblical: OT covenants; Biblical: ANE traditions in Genesis; Biblical: 2nd Temple NT context |
| `code_review` | 5 | Code Review: off-by-one loop; Code Review: null dereference; Code Review: SQL injection; Code Review: race condition; Code Review: resource leak |
| `coding` | 5 | Python: debug logic error; Code: refactor for efficiency; Game: Number Guessing Game; Game: Text Adventure Game; Game: Checkers (terminal TUI) |
| `cpp` | 5 | C++: sum a std::vector<int>; C++: RAII with std::unique_ptr; C++: function template max(a,b); C++: BankAccount class; C++: two threads + mutex counter |
| `creative` | 2 | Creative: sci-fi story opening; Creative: generate analogy |
| `database` | 5 | SQL: join customers/orders, HAVING count>3; SQL: add index on orders(customer_id); SQL: atomic $100 transfer w/ rollback; SQL: CREATE TABLE users; SQL: monthly revenue per product |
| `debugging` | 5 | Debug: off-by-one array copy; Debug: NullPointerException on chained call; Debug: unsynchronized shared counter; Debug: infinite loop (no increment); Debug: SQL injection via concatenation |
| `gamedev` | 5 | Game: Pong ball/paddle collision + scoring; Game: Snake step + self-collision; Game: RPG inventory with stack cap; Game: Turn-based combat resolution; Game: Match-3 run detection |
| `gamedev_alt` | 6 | Game: 3D Pong (three.js WebGL); Game: 3D voxel terrain flyover (raw WebGL); Game: Snake (HTML5 canvas UI); Game: Breakout (HTML5 canvas UI); Game: 3D Asteroids (three.js WebGL); Game: Checkers (modern UI) |
| `gpqa_diamond` | 8 | GPQA-Diamond: SN2 steric hindrance; GPQA-Diamond: infinite well ground state probability; GPQA-Diamond: Okazaki fragment polymerase; GPQA-Diamond: Henderson-Hasselbalch ratio; GPQA-Diamond: photon energy-momentum; GPQA-Diamond: nitro group directing effect; GPQA-Diamond: Okazaki fragment joining enzyme; GPQA-Diamond: nitrogen ground-state config |
| `hle` | 8 | HLE: Heisenberg uncertainty conjugate variable; HLE: least common multiple 1..10; HLE: 10th prime number; HLE: sum of integers 1 to 100; HLE: chemical formula of water; HLE: halting problem 1936 paper; HLE: first crewed Moon landing year; HLE: incompleteness theorems author |
| `home_automation` | 2 | HA: control smart device; HA: report device status |
| `ifeval` | 8 | IFEval: include word 'penguin'; IFEval: valid JSON object only; IFEval: end with 'done'; IFEval: mention country 'Brazil'; IFEval: begin with 'Greetings'; IFEval: state capital of Japan; IFEval: include word 'necessary'; IFEval: respond 'affirmative' only |
| `instruction` | 2 | JSON: extract structured data; Summarization: 3 bullet points |
| `java` | 5 | Java: Stream filter evens; Java: parseInt with NumberFormatException; Java: HashMap word frequencies; Java: Shape interface + Circle; Java: JDBC SELECT with try-with-resources |
| `knowledge` | 17 | Knowledge: MMLU  -  solve for x; Knowledge: MMLU  -  chemical symbol for gold; Knowledge: MMLU  -  the Red Planet; Knowledge: MMLU  -  ATP organelle; Knowledge: MMLU  -  WWII end year; Knowledge: GPQA  -  exothermic reaction; Knowledge: GPQA  -  Pauli exclusion principle; Knowledge: GSM8K  -  trees planted; Knowledge: GSM8K  -  total apples; Knowledge: TruthfulQA  -  objects falling; Knowledge: TruthfulQA  -  10% brain myth; Knowledge: HellaSwag  -  eggs at home; Knowledge: HellaSwag  -  opened the fridge; Knowledge: WinoGrande  -  trophy and suitcase; Knowledge: WinoGrande  -  cat scratched dog; Knowledge: ARC  -  renewable energy; Knowledge: ARC  -  force keeping planets in orbit |
| `languages` | 7 | Lang: Go net/http JSON endpoint; Lang: Rust stdin line counter; Lang: Node.js time endpoint; Lang: static HTML form + inline script; Lang: Python file line/word counter; Lang: TypeScript DOM click handler; Lang: Kotlin Ktor ping route (framework) |
| `life` | 9 | Life: calming bedtime story; Life: 3 original dad jokes; Life: what a timing belt does; Life: RAM vs storage for a parent; Life: balcony container garden; Life: backyard chicken husbandry; Life: seasonal home maintenance; Life: kids chore chart + rewards; Life: 50/30/20 budget explainer |
| `linux_admin` | 5 | Linux: find files >100MB sorted; Linux: recursive 755 dirs / 644 files; Linux: journalctl error logs (last hour, nginx); Linux: top 10 largest dirs under /; Linux: harden sshd_config |
| `logic` | 5 | Logic: Knights and Knaves; Logic: wolf/goat/cabbage crossing; Logic: modus tollens; Logic: categorical syllogism; Logic: 8 balls, 1 heavier, 2 weighings |
| `math_hard` | 8 | Math-Hard: single-elimination matches; Math-Hard: divisors of 360; Math-Hard: 2 to the 10th power; Math-Hard: combinations C(10,3); Math-Hard: hexagon interior angle sum; Math-Hard: solve 3^x = 81; Math-Hard: 2x2 determinant; Math-Hard: infinite geometric series |
| `metacog` | 2 | Meta: resist overthinking (concise); Meta: infinite-loop detection |
| `mmlu_pro` | 11 | MMLU-Pro: probability two dice sum to 7; MMLU-Pro: derivative of sin(x^2); MMLU-Pro: special relativity length contraction; MMLU-Pro: lowest boiling noble gas; MMLU-Pro: photosynthesis organelle; MMLU-Pro: US Declaration of Independence author; MMLU-Pro: Miranda rights amendment; MMLU-Pro: author of Critique of Pure Reason; MMLU-Pro: binary search complexity; MMLU-Pro: Keynesian recession policy; MMLU-Pro: Schwarzschild radius scaling |
| `multimodal` | 3 | Image: identify the ON light switch; HTML: render a contact form; Node: compute the 10th Fibonacci number |
| `office` | 7 | Office: professional delay email; Office: spreadsheet formulas (SUM/AVG/VLOOKUP); Office: 5-slide pitch deck outline; Office: Pillow image scaling; Office: SVG logo for a coffee shop; Office: rewrite/proofread a paragraph; Office: text-to-speech via Python |
| `reasoning` | 2 | Logic: identify rule; Math: train meeting problem |
| `retrogames` | 19 | Retro: Space Invaders wave + shooting; Retro: Maelstrom (Asteroids) split; Retro: vertical space shooter waves; Retro: Subway Surfers lane runner; Retro: Temple Run runner state; Retro: Donkey Kong climb + barrels; Retro: Super Mario (NES) stomp + gravity; Retro: creative arcade game skeleton; Retro: Crossy Road / Frogger hopper; Retro: Flappy Bird step; Retro: Ecco the Dolphin swim step; Retro: Pac-Man maze step; Retro: Tetris piece step; Game: Minecraft-style voxel chunk; Game: first-person shooter hitscan; Game: Block Blast board; Game: Sokoban (80s) solver check; Game: Breakout (80s) step; Game: SimCity (90s) tick |
| `threedprint` | 8 | 3DP: G-code 20mm cube outline; 3DP: G-code heat and wait; 3DP: OpenSCAD cube with center hole; 3DP: OpenSCAD spur gear; 3DP: Python binary STL generator; 3DP: send job to OctoPrint REST API; 3DP: submit job to a laser cutter API; 3DP: slicer config (layer/infill/support) |
| `tvdev` | 5 | TV/App: Android Activity + Button; TV/App: Android TV leanback browse; TV/App: Roku BrightScript SceneGraph; TV/App: Samsung Tizen web app; TV/App: LG webOS TV app |
| `uiux` | 15 | UI/UX: WCAG login-form critique; UI/UX: responsive blog layout; UI/UX: WCAG contrast computation; UI/UX: signup screen wireframe; UI/UX: password-reset user flow; UI/UX: fluid design system & CSS tokens; UI/UX: WCAG 2.2 accessible modal dialog; UI/UX: responsive metrics dashboard grid; UI/UX: interactive form validation & password meter; UI/UX: stacked toast notification system; UI/UX: mobile bottom sheet & desktop flyout; UI/UX: live theme customizer & palette switcher; UI/UX: interactive accessible data table UX; UI/UX: multi-step onboarding wizard; UI/UX: animated segmented tab control |
| `webdev` | 5 | Web: fetch JSON and render into DOM; Web: event delegation on a list; Web: validate email + 8-char password; Web: persist a theme in localStorage; Web: toggle a hidden class on click |
| `youtuber` | 3 | Game: Falling Sand (cellular-automata particle sandbox, pygame); Game: Conway's Game of Life (pygame, seedable + patterns); Game: Boids flocking simulation (pygame, emergent behavior) |

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
