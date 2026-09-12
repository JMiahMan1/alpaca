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

ARCADE_DIR = Path(os.getenv("ARCADE_DIR", "data/arcade"))
GAMES_DIR = ARCADE_DIR / "games"
AUTO_PUBLISH_SCORE = float(os.getenv("ARCADE_AUTO_PUBLISH_SCORE", "80"))


def slugify(model: str, test_id: str) -> str:
    """Stable URL slug for a model+test game, e.g. ``qwen3-35b_space-invaders``."""

    def clean(part: str) -> str:
        part = (part or "").lower()
        part = re.sub(r"[^a-z0-9]+", "-", part).strip("-")
        return part or "game"

    return f"{clean(model)}_{clean(test_id)}"


def find_artifact_file(model: str, test_id: str) -> Path | None:
    """Locate the saved game HTML under data/artifacts for a model+test."""
    artifacts = Path("data/artifacts")
    if not artifacts.is_dir():
        return None
    sanitized = re.sub(r"[/:.]", "_", model or "")
    candidates = sorted(artifacts.glob(f"{sanitized}__{test_id}.html"))
    if candidates:
        return candidates[0]
    # Fallback: any artifact ending with __<test_id>.html (newest first).
    fallback = sorted(artifacts.glob(f"*__{test_id}.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fallback[0] if fallback else None


def is_published(slug: str) -> bool:
    game_dir = GAMES_DIR / slug
    if not game_dir.is_dir() or not (game_dir / "meta.json").exists():
        return False
    return (game_dir / "game.html").exists() or (game_dir / "game.py").exists()


def _iter_result_records(test_id: str):
    """Yield (response, screenshot, score) records for a test from all suites.

    Scans general per-model files (data/llm_benchmarks/models/general_*.json)
    plus multistep per-model files, keeping the same record shape the
    /api/tests/responses endpoint serves.
    """
    for pattern in ("data/llm_benchmarks/models/general_*.json", "data/multistep_benchmarks/models/multistep_*.json"):
        for fp in sorted(Path(".").glob(pattern)):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            for run in data.get("results") or []:
                if not isinstance(run, dict):
                    continue
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
                        if (t.get("test_id") or t.get("id")) != test_id:
                            continue
                        resp = t.get("response") or ""
                        if not str(resp).strip():
                            continue
                        try:
                            score = float(t.get("score") or 0)
                        except (TypeError, ValueError):
                            score = 0.0
                        yield {
                            "response": str(resp),
                            "screenshot": t.get("screenshot"),
                            "score": score,
                            "model": data.get("model") or fp.stem,
                        }


def find_model_response(model: str, test_id: str) -> dict | None:
    """Longest stored response for a model+test across result files (or None)."""
    best = None
    for rec in _iter_result_records(test_id):
        if rec["model"] != model:
            continue
        if best is None or len(rec["response"]) > len(best["response"]):
            best = rec
    return best


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
    screenshot_b64 = None
    src = Path(source_file) if source_file else find_artifact_file(model, test_id)
    code_text = None
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
        else:
            from sandbox_exec import extract_clean_code

            code_text = ("__py__", extract_clean_code(resp, "python"))
        screenshot_b64 = rec.get("screenshot")
        if benchmark_score is None:
            benchmark_score = rec.get("score")

    game_dir = GAMES_DIR / slug
    game_dir.mkdir(parents=True, exist_ok=True)
    republished = (game_dir / "meta.json").exists()
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

    for name in ("scores.json", "ratings.json"):
        path = game_dir / name
        if not path.exists():
            path.write_text(json.dumps([] if name == "scores.json" else {}))

    meta = {
        "slug": slug,
        "title": title or test_id.replace("_", " ").replace("-", " ").title(),
        "model": model,
        "test_id": test_id,
        "kind": kind,
        "lang": "python" if kind == "code" else "html",
        "has_screenshot": (game_dir / "screenshot.png").exists(),
        "benchmark_score": benchmark_score,
        "max_score": max_score,
        "prompt": prompt or "",
        "run_date": run_date or "",
        "benchmark_date": run_date or "",
        "published_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "auto_published": bool(auto),
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


def published_games() -> list:
    """Metadata for every published game, newest first (drives the dashboard Arcade section)."""
    games = []
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
        try:
            plays = (
                sum(1 for _ in json.loads((game_dir / "scores.json").read_text()))
                if (game_dir / "scores.json").exists()
                else 0
            )
        except (OSError, ValueError):
            plays = 0
        games.append(
            {
                "slug": game_dir.name,
                "title": meta.get("title") or game_dir.name,
                "model": meta.get("model") or "",
                "test_id": meta.get("test_id") or "",
                "benchmark_score": meta.get("benchmark_score"),
                "run_date": meta.get("benchmark_date") or meta.get("run_date") or "",
                "published_at": meta.get("published_at") or "",
                "auto_published": bool(meta.get("auto_published")),
                "plays": plays,
            }
        )
    games.sort(key=lambda g: g["published_at"] or "", reverse=True)
    return games
