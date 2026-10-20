"""OODA curation for the family arcade.

Observe every eligible benchmark game, Orient by benchmark score, mobile
readiness and family engagement, Decide a small mobile-first shelf with room
for manual overrides, and Act by publishing only the chosen additions.

The loop is deliberately capped and additive: it never mass-publishes every
benchmark result, never hides a manually chosen game, and never deletes
scores/files. A person can still Hide or Unpublish from the dashboard.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from web.arcade_publish import (
    GAME_CATEGORIES,
    ARCADE_DIR,
    iter_eligible_game_results,
    load_curated,
    publish_game,
    published_games,
    published_slugs,
    slugify,
)

STATUS_FILE = ARCADE_DIR / "ooda_status.json"
MAX_GAMES = int(os.getenv("ARCADE_OODA_MAX_GAMES", "24"))
MIN_BENCHMARK = float(os.getenv("ARCADE_OODA_MIN_BENCHMARK", "70"))


def _title_key(row: dict[str, Any]) -> str:
    raw = str(row.get("label") or row.get("test_id") or row.get("title") or "")
    raw = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    return raw or str(row.get("slug") or "")


def _candidate_rows() -> list[dict[str, Any]]:
    """The same candidate shape the dashboard Add list uses."""
    try:
        from arcade.mobile import assess_html
    except ImportError:
        from mobile import assess_html

    curated = set(load_curated())
    published = published_slugs()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for rec in iter_eligible_game_results():
        model = str(rec.get("model") or "")
        test_id = str(rec.get("test_id") or "")
        if not model or not test_id:
            continue
        slug = slugify(model, test_id)
        if slug in seen:
            continue
        seen.add(slug)
        response = str(rec.get("response") or "")
        html = response.lower()
        playable = str(rec.get("type") or "").lower() == "ui" or "<html" in html or "<!doctype" in html
        mobile = assess_html(response) if playable else {"score": 0, "ready": False}
        rows.append(
            {
                "slug": slug,
                "model": model,
                "test_id": test_id,
                "title": rec.get("label") or rec.get("test_id") or slug,
                "label": rec.get("label") or "",
                "category": rec.get("category") or "",
                "type": rec.get("type"),
                "kind": "playable" if playable else (rec.get("kind") or "code"),
                "benchmark_score": float(rec.get("score") or 0),
                "max_score": rec.get("max_score"),
                "higher_is_better": rec.get("higher_is_better") is not False,
                "metric": rec.get("metric") or "score",
                "run_id": rec.get("run_id") or "",
                "run_date": rec.get("run_date") or "",
                "test_hash": rec.get("test_hash") or "",
                "validation": rec.get("validation"),
                "prompt": rec.get("prompt") or "",
                "mobile_score": int(mobile.get("score") or 0),
                "mobile_ready": bool(mobile.get("ready")),
                "published": slug in published,
                "curated": slug in curated,
            }
        )
    return rows


def observe() -> dict[str, Any]:
    rows = _candidate_rows()
    existing = {g["slug"]: g for g in published_games(include_hidden=True)}
    return {"candidates": rows, "existing": existing}


def orient(candidates: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Best mobile-ready game per title, with family engagement as a tie-breaker."""
    ranked: list[dict[str, Any]] = []
    for row in candidates:
        slug = row["slug"]
        game = existing.get(slug, {})
        plays = int(game.get("plays") or 0)
        rating = game.get("rating") or {}
        family = float(rating.get("average") or 0) if isinstance(rating, dict) else 0
        row = dict(row)
        row["family_rating"] = family
        row["plays"] = plays
        row["ooda_score"] = (
            float(row.get("benchmark_score") or 0)
            + (18.0 if row.get("mobile_ready") else 0.0)
            + min(10.0, int(row.get("mobile_score") or 0) * 0.10)
            + min(6.0, plays * 0.15)
            + family
        )
        ranked.append(row)
    ranked.sort(key=lambda r: (-r["ooda_score"], -float(r.get("benchmark_score") or 0), r["title"]))
    return ranked


def decide(ranked: list[dict[str, Any]], existing_slugs: set[str]) -> list[dict[str, Any]]:
    """Choose additions: best mobile-ready result per title, capped."""
    room = max(0, MAX_GAMES - len(existing_slugs))
    if room == 0:
        return []
    chosen: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for row in ranked:
        if row["slug"] in existing_slugs:
            continue
        if not row.get("mobile_ready"):
            continue
        if float(row.get("benchmark_score") or 0) < MIN_BENCHMARK:
            continue
        key = _title_key(row)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        chosen.append(row)
        if len(chosen) >= room:
            break
    return chosen


def act(observe_result: dict[str, Any] | None = None) -> dict[str, Any]:
    observed = observe_result or observe()
    existing = observed["existing"]
    ranked = orient(observed["candidates"], existing)
    chosen = decide(ranked, set(existing))
    added: list[str] = []
    errors: list[dict[str, str]] = []
    for row in chosen:
        try:
            result = publish_game(
                model=row["model"],
                test_id=row["test_id"],
                benchmark_score=row.get("benchmark_score"),
                max_score=row.get("max_score"),
                prompt=row.get("prompt") or "",
                run_date=row.get("run_date") or "",
                title=row.get("title"),
                category=row.get("category") or "",
                label=row.get("label") or "",
                type=row.get("type"),
                run_id=row.get("run_id") or "",
                test_hash=row.get("test_hash") or "",
                validation=row.get("validation"),
                higher_is_better=bool(row.get("higher_is_better", True)),
                metric=row.get("metric"),
            )
            added.append(str(result.get("slug") or row["slug"]))
        except Exception as e:  # one bad benchmark result must not stop the sweep
            errors.append({"slug": row["slug"], "error": str(e)})
    status = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned": len(observed["candidates"]),
        "existing": len(existing),
        "chosen": [r["slug"] for r in chosen],
        "added": added,
        "errors": errors,
        "cap": MAX_GAMES,
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    return status


def status() -> dict[str, Any]:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except (OSError, ValueError):
            pass
    return {"at": None, "scanned": 0, "existing": 0, "chosen": [], "added": [], "errors": [], "cap": MAX_GAMES}
