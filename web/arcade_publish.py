"""Publish benchmark games to the standalone arcade service (port 5001).

The arcade keeps its own copies under ``data/arcade/games/<slug>/`` so a
published game survives benchmark-data or model deletion in Alpaca until
explicitly unpublished. Publishing never overwrites player data:
``scores.json`` / ``ratings.json`` are created once and left alone on
republish (e.g. a better benchmark score for the same game).
"""

import contextlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARCADE_DIR = Path(os.getenv("ARCADE_DIR", "data/arcade"))
GAMES_DIR = ARCADE_DIR / "games"
AUTO_PUBLISH_SCORE = float(os.getenv("ARCADE_AUTO_PUBLISH_SCORE", "80"))

# Benchmark categories whose tests produce playable games. Used by the
# deliberate bulk reconcile to decide which results are eligible.
GAME_CATEGORIES = frozenset({"gamedev", "gamedev_alt", "retrogames", "youtuber"})


def slugify(model: str, test_id: str) -> str:
    """Stable URL slug for a model+test game, e.g. ``qwen3-35b_space-invaders``."""

    def clean(part: str) -> str:
        part = (part or "").lower()
        part = re.sub(r"[^a-z0-9]+", "-", part).strip("-")
        return part or "game"

    return f"{clean(model)}_{clean(test_id)}"


def find_artifact_file(model: str, test_id: str) -> Path | None:
    """Locate the saved game HTML under data/artifacts for a model+test.

    Only ever matches THIS model's own file (newest first) — it never falls
    back to another model's artifact, which would publish someone else's game.
    """
    artifacts = Path("data/artifacts")
    if not artifacts.is_dir():
        return None
    sanitized = re.sub(r"[/:.]", "_", model or "")
    candidates = sorted(
        artifacts.glob(f"{sanitized}__{test_id}.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def is_published(slug: str) -> bool:
    game_dir = GAMES_DIR / slug
    if not game_dir.is_dir() or not (game_dir / "meta.json").exists():
        return False
    return (game_dir / "game.html").exists() or (game_dir / "game.py").exists()


def _iter_result_records(test_id: str | None = None):
    """Yield (record) dicts from all suites.

    Scans general per-model files (data/llm_benchmarks/models/general_*.json)
    plus multistep per-model files, keeping the same record shape the
    /api/tests/responses endpoint serves. Pass ``test_id`` to match a single
    test, or ``None`` to walk every test (used by the bulk reconcile).

    Each record carries the benchmark provenance metadata (category, label,
    type/kind, run_id, test_hash, validation, score scale) so publishing can
    preserve it in the arcade's meta.json.
    """
    for pattern in ("data/llm_benchmarks/models/general_*.json", "data/multistep_benchmarks/models/multistep_*.json"):
        for fp in sorted(Path(".").glob(pattern)):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            doc_run_id = data.get("run_id")
            doc_model = data.get("model")
            for run in data.get("results") or []:
                if not isinstance(run, dict):
                    continue
                run_id = run.get("run_id") or doc_run_id
                run_model = run.get("model") or doc_model
                task_lists = []
                for value in run.values():
                    if isinstance(value, dict) and isinstance(value.get("tests"), list):
                        task_lists.append(value["tests"])
                    elif isinstance(value, list):
                        task_lists.append(value)
                for tasks in task_lists:
                    for t in tasks:
                        if not isinstance(t, dict):
                            continue
                        tid = t.get("test_id") or t.get("id")
                        if test_id is not None and tid != test_id:
                            continue
                        resp = t.get("response") or ""
                        if not str(resp).strip():
                            continue
                        try:
                            score = float(t.get("score") or 0)
                        except (TypeError, ValueError):
                            score = 0.0
                        yield {
                            "model": t.get("model") or run_model or doc_model or fp.stem,
                            "test_id": tid,
                            "category": t.get("test_category") or t.get("category") or "",
                            "label": t.get("test_label") or t.get("label") or "",
                            "type": t.get("type") or t.get("task_type"),
                            "kind": t.get("kind"),
                            "run_id": t.get("run_id") or run_id or "",
                            "test_hash": t.get("test_hash"),
                            "validation": t.get("validation"),
                            "max_score": t.get("max_score"),
                            "higher_is_better": t.get("higher_is_better"),
                            "metric": t.get("metric"),
                            "response": str(resp),
                            "screenshot": t.get("screenshot"),
                            "score": score,
                            "prompt": t.get("prompt") or t.get("prompt_steps") or "",
                            "run_date": t.get("run_date")
                            or t.get("benchmark_date")
                            or t.get("last_run")
                            or t.get("timestamp")
                            or t.get("date")
                            or "",
                            "source_file": str(fp),
                        }


def _record_date_key(rec: dict) -> tuple:
    """Sortable key preferring a real run date, falling back to file mtime."""
    date = str(rec.get("run_date") or "")
    mtime = 0.0
    source_file = rec.get("source_file")
    if source_file:
        with contextlib.suppress(OSError):
            mtime = Path(source_file).stat().st_mtime
    return (bool(date), date, mtime)


def find_model_response(model: str, test_id: str) -> dict | None:
    """Newest stored response for a model+test across result files (or None)."""
    best = None
    for rec in _iter_result_records(test_id):
        if rec["model"] != model:
            continue
        if best is None or _record_date_key(rec) > _record_date_key(best):
            best = rec
    return best


_FENCE_LANGS = {
    "html": "html",
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
}


def _response_lang(resp: str) -> str:
    """Language evidenced by the model's own response (first fenced block tag).

    Falls back to "python" only when the response carries no language marker
    at all — the extraction library's own default for unfenced code.
    """
    m = re.search(r"```(\w+)", resp or "")
    if m:
        return _FENCE_LANGS.get(m.group(1).lower(), m.group(1).lower())
    return "python"


def _js_shell(code: str, title: str) -> str:
    """Wrap browser-runnable JavaScript in a minimal game page.

    Lets JS benchmark responses play in the hero area exactly like HTML
    games — no sandbox needed. The shell is intentionally generic (no
    assumed canvas/DOM shape): scripts that create their own elements
    just work; ``</script`` inside the code is escaped so the page can't
    break out of its own script block (identical string semantics).
    """
    safe = code.replace("</script", "<\\/script")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>html,body{{margin:0;padding:0;height:100%;background:#000;overflow:hidden}}canvas{{display:block;margin:0 auto;max-width:100vw;max-height:100vh}}#arcade-err{{position:fixed;left:8px;bottom:8px;max-width:90vw;font:12px monospace;color:#f66;display:none}}</style>
</head>
<body>
<div id="arcade-err"></div>
<script>
window.addEventListener('error',function(e){{var d=document.getElementById('arcade-err');d.style.display='block';d.textContent='Error: '+(e.message||e.error);}});
{safe}
</script>
</body>
</html>
"""


def _catalog_test(test_id: str) -> dict:
    """The static test definition from benchmark_tests.json (or {})."""
    try:
        catalog = json.loads((Path(__file__).resolve().parent.parent / "benchmark_tests.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(catalog, dict):
        return {}
    for tests in catalog.values():
        if not isinstance(tests, list):
            continue
        for t in tests:
            if isinstance(t, dict) and (t.get("id") == test_id):
                return t
    return {}


def _catalog_prompt(test_id: str) -> str:
    """The static test prompt from benchmark_tests.json (fallback when the
    stored run record carries no prompt of its own)."""
    p = _catalog_test(test_id).get("prompt") or ""
    return p if isinstance(p, str) else ""


def _file_date(path: str | Path | None) -> str:
    """YYYY-MM-DD of a file's mtime (last-resort run-date fallback)."""
    if not path:
        return ""
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return ""


def publish_game(
    *,
    model: str,
    test_id: str,
    benchmark_score: float | None = None,
    max_score: float | None = None,
    prompt: str = "",
    run_date: str = "",
    source_file: str | Path | None = None,
    title: str | None = None,
    category: str = "",
    label: str = "",
    type: str | None = None,
    run_id: str = "",
    test_hash: str = "",
    validation: dict | None = None,
    higher_is_better: bool = True,
    metric: str | None = None,
    auto: bool = False,
) -> dict:
    """Copy a game into the arcade. Returns ``{"slug": ..., "url": ..., "republished": ...}``.

    Raises FileNotFoundError when no game source can be found and ValueError
    when model/test_id are missing.

    Source resolution, in order:
    1. an explicit ``source_file`` (copied as-is, playable HTML assumed);
    2. a saved ``data/artifacts/<model>__<test>.html`` file (playable);
    3. the model's stored benchmark response: embedded HTML becomes a
       playable ``game.html``; otherwise the extracted program (e.g. pygame
       Python) is frozen as ``game.py`` plus the run screenshot, served as a
       code-kind game page with download.
    """
    if not model or not test_id:
        raise ValueError("model and test_id are required to publish a game")
    slug = slugify(model, test_id)
    kind = "playable"
    resp_lang = "html"
    screenshot_b64 = None
    src = Path(source_file) if source_file else find_artifact_file(model, test_id)
    code_text = None
    rec = None
    if src is not None and not src.exists():
        src = None
    if src is None:
        rec = find_model_response(model, test_id)
        if rec is None:
            raise FileNotFoundError(f"no saved game found for {model} / {test_id}")
        resp = rec["response"]
        if re.search(r"<!doctype|<html|<script|<canvas", resp, re.I):
            from sandbox_exec import extract_clean_code

            code_text = ("__html__", extract_clean_code(resp, "web"))
            resp_lang = "html"
        else:
            from sandbox_exec import extract_clean_code

            resp_lang = _response_lang(resp)
            code = extract_clean_code(resp, resp_lang)
            if resp_lang in ("javascript", "typescript"):
                # Browser-runnable: wrap in a minimal page so it plays in
                # the hero area like any HTML game (no sandbox needed).
                code_text = (
                    "__html__",
                    _js_shell(code, title or test_id.replace("_", " ").replace("-", " ").title()),
                )
            else:
                code_text = ("__py__", code)
        screenshot_b64 = rec.get("screenshot")
        if benchmark_score is None:
            benchmark_score = rec.get("score")
        if max_score is None:
            record_max_score = rec.get("max_score")
            if record_max_score is not None:
                with contextlib.suppress(TypeError, ValueError):
                    max_score = float(record_max_score)

    game_dir = GAMES_DIR / slug
    game_dir.mkdir(parents=True, exist_ok=True)
    republished = (game_dir / "meta.json").exists()
    # Preserve the play counter (and it alone) across republish — it lives in
    # meta.json, which this overwrites, while player data lives in the
    # scores.json / ratings.json files that are left untouched below.
    previous_plays = 0
    if republished:
        with contextlib.suppress(OSError, ValueError):
            previous_plays = int((json.loads((game_dir / "meta.json").read_text()) or {}).get("plays", 0) or 0)
    if code_text is None:
        assert src is not None
        shutil.copyfile(src, game_dir / "game.html")
        (game_dir / "game.py").unlink(missing_ok=True)
    elif code_text[0] == "__html__":
        (game_dir / "game.html").write_text(code_text[1], encoding="utf-8")
        (game_dir / "game.py").unlink(missing_ok=True)
    else:
        kind = "code"
        (game_dir / "game.py").write_text(code_text[1], encoding="utf-8")
        (game_dir / "game.html").unlink(missing_ok=True)
        if screenshot_b64:
            import base64

            with contextlib.suppress(Exception):
                data = screenshot_b64.split(",", 1)[-1]
                (game_dir / "screenshot.png").write_bytes(base64.b64decode(data))
        # else: keep any previously saved screenshot on republish.

    # Align scores.json with the arcade service's dict schema; leave an
    # existing file (player data) untouched on republish.
    scores_path = game_dir / "scores.json"
    if not scores_path.exists():
        scores_path.write_text(
            json.dumps({"scores": [], "bests": {"submits": 0, "first_by": None, "players": {}}, "captures": []})
        )
    ratings_path = game_dir / "ratings.json"
    if not ratings_path.exists():
        ratings_path.write_text(json.dumps({}))

    # Stored run records are minimal (often no prompt/date), so backfill from
    # the record, the source file's mtime, then the static test catalog.
    rec_prompt = (rec or {}).get("prompt") or ""
    resolved_prompt = prompt or rec_prompt or _catalog_prompt(test_id)
    rec_date = (rec or {}).get("run_date") or ""
    src_for_date = None
    if src is not None:
        src_for_date = src
    elif rec is not None:
        src_for_date = rec.get("source_file")
    resolved_date = run_date or rec_date or _file_date(src_for_date)

    catalog = _catalog_test(test_id)
    resolved_category = category or (rec or {}).get("category") or catalog.get("category") or ""
    resolved_label = label or (rec or {}).get("label") or catalog.get("label") or ""
    resolved_type = type or (rec or {}).get("type") or catalog.get("type")
    resolved_run_id = run_id or (rec or {}).get("run_id") or ""
    resolved_test_hash = test_hash or (rec or {}).get("test_hash") or ""
    resolved_validation = validation if validation is not None else (rec or {}).get("validation")
    rec_higher_is_better = (rec or {}).get("higher_is_better")
    resolved_higher_is_better = bool(higher_is_better) if rec_higher_is_better is None else bool(rec_higher_is_better)
    resolved_metric = metric or (rec or {}).get("metric") or "score"

    meta = {
        "slug": slug,
        "title": title or test_id.replace("_", " ").replace("-", " ").title(),
        "model": model,
        "test_id": test_id,
        "test_label": resolved_label,
        "category": resolved_category,
        "label": resolved_label,
        "type": resolved_type,
        "kind": kind,
        "lang": resp_lang,
        "has_screenshot": (game_dir / "screenshot.png").exists(),
        "benchmark_score": benchmark_score,
        "max_score": max_score,
        "higher_is_better": resolved_higher_is_better,
        "metric": resolved_metric,
        "run_id": resolved_run_id,
        "test_hash": resolved_test_hash,
        "validation": resolved_validation,
        "prompt": resolved_prompt,
        "run_date": resolved_date,
        "benchmark_date": resolved_date,
        "published_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "auto_published": bool(auto),
        "plays": previous_plays,
    }
    (game_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    # The arcade container runs as a non-root user while publishes often
    # happen as root (web container / sudo). Leave everything group- and
    # world-writable so score/rating/play-count writes never 500.
    for path in (
        game_dir,
        game_dir / "game.html",
        game_dir / "game.py",
        game_dir / "screenshot.png",
        game_dir / "meta.json",
    ):
        if not path.exists():
            continue
        with contextlib.suppress(OSError):
            os.chmod(path, 0o777 if path.is_dir() else 0o666)
    for name in ("scores.json", "ratings.json"):
        with contextlib.suppress(OSError):
            os.chmod(game_dir / name, 0o666)
    return {"slug": slug, "url": f"/play/{slug}", "republished": republished}


def unpublish_game(slug: str) -> bool:
    """Remove a published game (including its player scores). Returns True if removed."""
    game_dir = GAMES_DIR / slug
    if not game_dir.is_dir():
        return False
    shutil.rmtree(game_dir)
    return True


def published_slugs() -> set[str]:
    if not GAMES_DIR.is_dir():
        return set()
    return {p.name for p in GAMES_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists()}


def get_auto_publish_score() -> float:
    """Live auto-publish threshold (UI-owned via ARCADE_AUTO_PUBLISH_SCORE)."""
    try:
        return float(os.getenv("ARCADE_AUTO_PUBLISH_SCORE", "80"))
    except ValueError:
        return 80.0


def _count_player_scores(game_dir: Path) -> int:
    """Count player score entries, handling both the arcade's dict schema
    (``{"scores": [...]}``) and the legacy list schema."""
    scores_path = game_dir / "scores.json"
    if not scores_path.exists():
        return 0
    try:
        raw = json.loads(scores_path.read_text())
    except (OSError, ValueError):
        return 0
    if isinstance(raw, dict):
        entries = raw.get("scores") or []
        return len(entries) if isinstance(entries, list) else 0
    if isinstance(raw, list):
        return len(raw)
    return 0


def published_games() -> list:
    """Metadata for every published game, newest first (drives the dashboard Arcade section)."""
    games: list[dict[str, Any]] = []
    if not GAMES_DIR.is_dir():
        return games
    for game_dir in sorted(GAMES_DIR.iterdir()):
        meta_path = game_dir / "meta.json"
        if not game_dir.is_dir() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        plays = _count_player_scores(game_dir)
        games.append(
            {
                "slug": game_dir.name,
                "title": meta.get("title") or game_dir.name,
                "model": meta.get("model") or "",
                "test_id": meta.get("test_id") or "",
                "category": meta.get("category") or "",
                "label": meta.get("label") or meta.get("test_label") or "",
                "kind": meta.get("kind") or "",
                "max_score": meta.get("max_score"),
                "higher_is_better": meta.get("higher_is_better") is not False,
                "metric": meta.get("metric") or "score",
                "run_id": meta.get("run_id") or "",
                "benchmark_score": meta.get("benchmark_score"),
                "run_date": meta.get("benchmark_date") or meta.get("run_date") or "",
                "published_at": meta.get("published_at") or "",
                "auto_published": bool(meta.get("auto_published")),
                "plays": plays,
            }
        )
    games.sort(key=lambda g: g["published_at"] or "", reverse=True)
    return games


def iter_eligible_game_results():
    """Yield every benchmark result that describes a playable game.

    A result is eligible when its test type is ``ui`` or its category is a
    known game category. Used by the deliberate bulk reconcile; never runs
    automatically.
    """
    for rec in _iter_result_records(None):
        category = (rec.get("category") or "").lower()
        rtype = str(rec.get("type") or "").lower()
        if rtype == "ui" or category in GAME_CATEGORIES:
            yield rec


def publish_all_eligible() -> dict:
    """Deliberately publish every eligible benchmark game result (idempotent).

    Re-running is safe: already-published games are republished in place
    (player scores/ratings/play counts are preserved), and games with no
    usable source are skipped rather than failing the whole sweep.
    """
    published: list[str] = []
    republished: list[str] = []
    skipped: list[dict] = []
    failed: dict[str, str] = {}
    seen: set[str] = set()
    for rec in iter_eligible_game_results():
        model = rec.get("model") or ""
        test_id = rec.get("test_id") or ""
        if not model or not test_id:
            continue
        slug = slugify(model, test_id)
        if slug in seen:
            continue
        seen.add(slug)
        try:
            result = publish_game(
                model=model,
                test_id=test_id,
                benchmark_score=rec.get("score"),
                max_score=rec.get("max_score"),
                prompt=rec.get("prompt") or "",
                run_date=rec.get("run_date") or "",
                category=rec.get("category") or "",
                label=rec.get("label") or "",
                type=rec.get("type"),
                run_id=rec.get("run_id") or "",
                test_hash=rec.get("test_hash") or "",
                validation=rec.get("validation"),
                higher_is_better=(
                    bool(rec.get("higher_is_better")) if rec.get("higher_is_better") is not None else True
                ),
                metric=rec.get("metric") or "score",
            )
            (republished if result["republished"] else published).append(result["slug"])
        except FileNotFoundError:
            skipped.append({"model": model, "test_id": test_id, "reason": "no saved game source"})
        except (TypeError, ValueError):
            skipped.append({"model": model, "test_id": test_id, "reason": "invalid result"})
        except Exception as e:  # pragma: no cover - defensive bulk sweep
            failed[slug] = str(e)
    return {
        "published": published,
        "republished": republished,
        "skipped": skipped,
        "failed": failed,
        "counts": {
            "published": len(published),
            "republished": len(republished),
            "skipped": len(skipped),
            "failed": len(failed),
        },
    }
