"""Which published arcade games are allowed on the public shelf.

Publishing a game copies it under ``data/arcade/games`` and keeps its player
data. Curation is a separate, explicit decision: a game directory can exist
(and keep scores/ratings) without being visible in the arcade. The selection
lives in ``data/arcade/curated.json`` so the arcade and the dashboard read the
same list.
"""

import json
import os
import re
from pathlib import Path

ARCADE_DIR = Path(os.getenv("ARCADE_DIR", "data/arcade"))
CURATED_FILE = ARCADE_DIR / "curated.json"
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _clean_slugs(slugs) -> list[str]:
    if not isinstance(slugs, (list, tuple, set)):
        return []
    return [slug for slug in dict.fromkeys(str(s) for s in slugs) if SLUG_RE.match(slug)]


def load_curated() -> list[str]:
    """The chosen game slugs. A missing or invalid file means nothing is chosen."""
    try:
        raw = json.loads(CURATED_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(raw, dict):
        raw = raw.get("slugs", [])
    return _clean_slugs(raw)


def save_curated(slugs) -> list[str]:
    clean = _clean_slugs(slugs)
    CURATED_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURATED_FILE.write_text(json.dumps({"slugs": clean}, indent=2) + "\n", encoding="utf-8")
    return clean


def is_curated(slug: str) -> bool:
    return bool(slug) and slug in set(load_curated())


def set_curated(slug: str, curated: bool) -> bool:
    """Add or remove one game from the public shelf. Returns its new state."""
    if not SLUG_RE.match(slug or ""):
        raise ValueError("invalid slug")
    current = load_curated()
    slugs = [s for s in current if s != slug]
    if curated:
        slugs.append(slug)
    save_curated(slugs)
    return bool(curated)
