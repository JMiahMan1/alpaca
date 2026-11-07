"""Saying Bible references out loud, and the story behind every writing prompt.

The announcer gives each prompt its story first (``STORY_CONTEXT``, by prompt id), so a
player who has never heard of Eutychus still knows what the joke is about. The verse
itself is shown on screen, not read. References are spoken the way a person reads them:
"1 Sam 17:40" is "First Samuel chapter 17".
"""

from __future__ import annotations

import re

try:
    from arcade.party_story_context import STORY_CONTEXT
except ImportError:  # the arcade image runs with arcade/ on sys.path
    try:
        from party_story_context import STORY_CONTEXT  # type: ignore[no-redef]
    except ImportError:
        STORY_CONTEXT = {}

BOOKS = (
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "Samuel", "Kings",
    "Chronicles", "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Psalm", "Proverbs", "Ecclesiastes",
    "Song of Solomon", "Song of Songs", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "Corinthians", "Galatians", "Ephesians", "Philippians",
    "Colossians", "Thessalonians", "Timothy", "Titus", "Philemon", "Hebrews", "James", "Peter", "Jude", "Revelation",
)
ABBREVIATIONS = {
    "Gen": "Genesis", "Ex": "Exodus", "Exod": "Exodus", "Lev": "Leviticus", "Num": "Numbers", "Deut": "Deuteronomy",
    "Josh": "Joshua", "Judg": "Judges", "Sam": "Samuel", "Kgs": "Kings", "Chr": "Chronicles", "Neh": "Nehemiah",
    "Esth": "Esther", "Ps": "Psalm", "Psa": "Psalm", "Prov": "Proverbs", "Eccl": "Ecclesiastes", "Eccles": "Ecclesiastes",
    "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations", "Ezek": "Ezekiel", "Dan": "Daniel", "Hos": "Hosea",
    "Obad": "Obadiah", "Jon": "Jonah", "Mic": "Micah", "Hab": "Habakkuk", "Zeph": "Zephaniah", "Hag": "Haggai",
    "Zech": "Zechariah", "Mal": "Malachi", "Matt": "Matthew", "Mt": "Matthew", "Mk": "Mark", "Lk": "Luke", "Jn": "John",
    "Rom": "Romans", "Cor": "Corinthians", "Gal": "Galatians", "Eph": "Ephesians", "Phil": "Philippians",
    "Col": "Colossians", "Thess": "Thessalonians", "Tim": "Timothy", "Phlm": "Philemon", "Heb": "Hebrews",
    "Jas": "James", "Pet": "Peter", "Rev": "Revelation",
}
ORDINALS = {"1": "First", "2": "Second", "3": "Third", "I": "First", "II": "Second", "III": "Third"}
_NAMES = sorted({*BOOKS, *ABBREVIATIONS}, key=len, reverse=True)
#: "1 Sam 17:40", "Genesis 8:16-18", "Ps 23", "Song of Solomon 2:4", and a bare "1:6-8" after a book.
REFERENCE = re.compile(
    r"\b(?:(?P<n>[123]|I{1,3})\s+)?(?P<book>" + "|".join(re.escape(b) for b in _NAMES) + r")\.?\s+(?P<ch>\d+)(?::[\d,\-\u2013\s]*\d)?"
)


def _book(n: str | None, book: str) -> str:
    name = ABBREVIATIONS.get(book, book)
    if name == "Psalms":
        name = "Psalm"
    return f"{ORDINALS[n]} {name}" if n else name


def spoken_reference(ref: str) -> str:
    """ "1 Sam 17:40" -> "First Samuel chapter 17"; "Psalm 23:1" -> "Psalm 23". Unknown text comes back as is."""
    m = REFERENCE.search(ref or "")
    if not m:
        return (ref or "").strip()
    book = _book(m.group("n"), m.group("book"))
    return f"{book} {m.group('ch')}" if book == "Psalm" else f"{book} chapter {m.group('ch')}"


def story_context(prompt: dict) -> str:
    """The story behind a writing prompt, with its chapter; every prompt has one.

    A prompt without its own (an AI-written one) at least names where it comes from.
    """
    text = (prompt.get("context") or STORY_CONTEXT.get(str(prompt.get("id") or "")) or "").strip()
    if text:
        return text
    note = str(prompt.get("scripture_note") or "")
    m = REFERENCE.search(note)
    return f"This one comes from {spoken_reference(m.group(0))}." if m else ""
