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
    return (game_dir / "game.html").exists() and (game_dir / "meta.json").exists()


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

    Raises FileNotFoundError when no game HTML can be found and ValueError
    when model/test_id are missing.
    """
    if not model or not test_id:
        raise ValueError("model and test_id are required to publish a game")
    slug = slugify(model, test_id)
    src = Path(source_file) if source_file else find_artifact_file(model, test_id)
    if src is None or not src.exists():
        raise FileNotFoundError(f"no saved game HTML found for {model} / {test_id}")

    game_dir = GAMES_DIR / slug
    game_dir.mkdir(parents=True, exist_ok=True)
    republished = (game_dir / "meta.json").exists()
    shutil.copyfile(src, game_dir / "game.html")

    for name in ("scores.json", "ratings.json"):
        path = game_dir / name
        if not path.exists():
            path.write_text(json.dumps([] if name == "scores.json" else {}))

    meta = {
        "slug": slug,
        "title": title or test_id.replace("_", " ").replace("-", " ").title(),
        "model": model,
        "test_id": test_id,
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
    for path in (game_dir, game_dir / "game.html", game_dir / "meta.json"):
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
