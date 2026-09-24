"""
tts_text.py

Text normalization for Kokoro narration. misaki (Kokoro's G2P) already reads
years, currency, and plain numbers well; this module covers what it misreads:

- Scripture references  "Mark 14:3-9"  -> "Mark chapter 14, verses 3 through 9"
- Roman numerals        "Henry VIII"   -> "Henry the Eighth", "Chapter IV" -> "Chapter 4"
- Year ranges           "1703-1791"    -> "1703 to 1791"
- Titles                "Rev. Smith"   -> "Reverend Smith"
- Parentheses           "(an aside)"   -> ", an aside,"  (Kokoro phrases commas better)
- A lexicon file of ordered custom replacements and IPA pronunciation
  overrides, emitted in misaki's inline syntax: [Bresee](/bɹəzˈi/).

Numbers are left as digits for misaki to voice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading

logger = logging.getLogger("tts_text")

# --------------------------------------------------------------------------- #
# Scripture                                                                    #
# --------------------------------------------------------------------------- #

_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Psalm", "Proverbs", "Ecclesiastes", "Song of Songs",
    "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
    "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
]
# Common abbreviations -> full name. Matching is case-insensitive; a trailing
# period on the abbreviation is optional.
_BOOK_ABBR = {
    "Gen": "Genesis", "Gn": "Genesis", "Ex": "Exodus", "Exod": "Exodus", "Lev": "Leviticus",
    "Num": "Numbers", "Deut": "Deuteronomy", "Dt": "Deuteronomy", "Josh": "Joshua",
    "Judg": "Judges", "1 Sam": "1 Samuel", "2 Sam": "2 Samuel", "1 Kgs": "1 Kings",
    "2 Kgs": "2 Kings", "1 Chr": "1 Chronicles", "2 Chr": "2 Chronicles", "Neh": "Nehemiah",
    "Esth": "Esther", "Ps": "Psalm", "Psa": "Psalm", "Prov": "Proverbs", "Eccl": "Ecclesiastes",
    "Eccles": "Ecclesiastes", "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations",
    "Ezek": "Ezekiel", "Dan": "Daniel", "Hos": "Hosea", "Obad": "Obadiah", "Mic": "Micah",
    "Nah": "Nahum", "Hab": "Habakkuk", "Zeph": "Zephaniah", "Hag": "Haggai",
    "Zech": "Zechariah", "Mal": "Malachi", "Matt": "Matthew", "Mt": "Matthew", "Mk": "Mark",
    "Lk": "Luke", "Jn": "John", "Rom": "Romans", "1 Cor": "1 Corinthians",
    "2 Cor": "2 Corinthians", "Gal": "Galatians", "Eph": "Ephesians", "Phil": "Philippians",
    "Col": "Colossians", "1 Thess": "1 Thessalonians", "2 Thess": "2 Thessalonians",
    "1 Tim": "1 Timothy", "2 Tim": "2 Timothy", "Phlm": "Philemon", "Heb": "Hebrews",
    "Jas": "James", "1 Pet": "1 Peter", "2 Pet": "2 Peter", "1 Jn": "1 John", "2 Jn": "2 John",
    "3 Jn": "3 John", "Rev": "Revelation",
}
_BOOK_LOOKUP = {b.lower(): b for b in _BOOKS} | {k.lower(): v for k, v in _BOOK_ABBR.items()}
_ORDINAL_BOOK = {"1": "First", "2": "Second", "3": "Third", "I": "First", "II": "Second", "III": "Third"}

_book_alt = "|".join(
    re.escape(k).replace(r"\ ", r"\s+") for k in sorted(_BOOK_LOOKUP, key=len, reverse=True)
)
# "1 Cor. 13:4-7", "John 3:16", "Mark 14:3-9, 12", "John 3:16-4:2"
_SCRIPTURE = re.compile(
    rf"\b(?P<book>{_book_alt})\.?\s+(?P<ch>\d{{1,3}}):(?P<vs>\d{{1,3}}[a-c]?"
    rf"(?:\s*[-–]\s*\d{{1,3}}(?::\d{{1,3}})?[a-c]?)?(?:\s*,\s*\d{{1,3}}(?:\s*[-–]\s*\d{{1,3}})?)*)"
    rf"(?P<ff>\s*ff?\b)?",
    re.IGNORECASE,
)


def _speak_book(raw: str) -> str:
    book = _BOOK_LOOKUP[re.sub(r"\s+", " ", raw).lower()]
    num, _, rest = book.partition(" ")
    # "1 Corinthians" reads as "First Corinthians".
    return f"{_ORDINAL_BOOK[num]} {rest}" if num in _ORDINAL_BOOK and rest else book


def _speak_scripture(m: re.Match) -> str:
    book, ch, vs = _speak_book(m.group("book")), m.group("ch"), m.group("vs")
    vs = re.sub(r"\s+", "", vs).replace("–", "-")
    # Cross-chapter range: "3:16-4:2"
    cross = re.fullmatch(r"(\d+)[a-c]?-(\d+):(\d+)[a-c]?", vs)
    if cross:
        return f"{book} chapter {ch}, verse {cross.group(1)} through chapter {cross.group(2)}, verse {cross.group(3)}"
    parts = [p.replace("-", " through ") for p in vs.split(",")]
    plural = len(parts) > 1 or "-" in vs
    verses = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    verses = re.sub(r"(\d)([a-c])\b", r"\1 \2", verses)
    ff = m.group("ff")
    tail = "" if not ff else (" and following" if ff.strip().lower() == "ff" else " and the following verse")
    return f"{book} chapter {ch}, {'verses' if plural else 'verse'} {verses}{tail}"


def normalize_scripture(text: str) -> str:
    return _SCRIPTURE.sub(_speak_scripture, text)


# --------------------------------------------------------------------------- #
# Roman numerals                                                               #
# --------------------------------------------------------------------------- #

_ROMAN_VALID = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
_ROMAN_TOKEN = re.compile(r"(?<![\w'’-])(?P<prev>[A-Za-z]+\.?\s+)?(?P<num>[IVXLCDM]+)\b(?![-'’.]\w)")
# Words after which a numeral is a plain count: "Chapter IV" -> "Chapter 4".
_CARDINAL_CUES = {
    "chapter", "part", "book", "volume", "vol", "section", "act", "scene", "unit", "article",
    "phase", "stage", "war", "level", "type", "class", "appendix", "psalm", "canto", "lecture",
}
# Title-case words that are never regnal names (sentence starters, pronoun-ish).
_NOT_NAMES = {
    "I", "A", "The", "Then", "When", "So", "And", "But", "If", "As", "Now", "Here", "There",
    "What", "Why", "How", "Where", "Did", "Do", "Can", "May", "Will", "Should", "Would",
}
_ORDINALS = [
    "", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth",
    "Tenth", "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth",
    "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth",
]
_TENS_ORD = {20: "Twent", 30: "Thirt"}


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = vals[a]
        total += -v if b != " " and vals[b] > v else v
    return total


def _ordinal(n: int) -> str:
    if n < len(_ORDINALS):
        return _ORDINALS[n]
    if n < 40:
        tens, ones = divmod(n, 10)
        base = _TENS_ORD[tens * 10]
        return f"{base}y-{_ORDINALS[ones].lower()}" if ones else f"{base}ieth"
    return str(n)


def _speak_roman(m: re.Match) -> str:
    prev, num = m.group("prev") or "", m.group("num")
    if not _ROMAN_VALID.fullmatch(num):
        return m.group(0)
    word = prev.strip().rstrip(".")
    value = _roman_to_int(num)
    title_case = bool(word) and word[0].isupper() and word[1:].islower()
    # "Chapter IV", "World War I". A lone "I" needs a Title-case cue so
    # "the book I read" stays a pronoun.
    if word.lower() in _CARDINAL_CUES and (len(num) >= 2 or num != "I" or title_case):
        return f"{prev}{value}"
    # Regnal / papal names: "Henry VIII", "Pius IX", "Charles V". Single letters
    # other than V are too ambiguous ("Malcolm X", "Plan B"-style labels), and
    # values past 39 are almost always acronyms ("DC", "XL").
    if title_case and word not in _NOT_NAMES and value <= 39 and (len(num) >= 2 or num == "V"):
        return f"{prev}the {_ordinal(value)}"
    return m.group(0)


def normalize_roman(text: str) -> str:
    return _ROMAN_TOKEN.sub(_speak_roman, text)


# --------------------------------------------------------------------------- #
# Small fixes misaki misreads                                                  #
# --------------------------------------------------------------------------- #

_YEAR_RANGE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\s*[-–—]\s*(1[0-9]{3}|20[0-9]{2})\b")
_TITLES = [
    (re.compile(r"\bRev\.\s+(?=[A-Z])"), "Reverend "),
    (re.compile(r"\bFr\.\s+(?=[A-Z])"), "Father "),
    (re.compile(r"\bSt\.\s+(?=[A-Z])"), "Saint "),
    (re.compile(r"\bGen\.\s+(?=[A-Z])"), "General "),  # "Gen. 1:1" is already expanded by then
]
# "(aside)" -> ", aside," but never touch misaki links "[word](/ipa/)".
_PAREN = re.compile(r"(?<!\])\s*\(([^()\[\]]{1,200})\)")


def _paren_to_commas(text: str) -> str:
    text = _PAREN.sub(lambda m: f", {m.group(1).strip()},", text)
    text = re.sub(r",\s*([,.;:!?])", r"\1", text)   # "aside, ." -> "aside."
    text = re.sub(r"([.;:!?]),", r"\1", text)        # "(See below.) Next" -> "See below. Next"
    return re.sub(r"(^|\n)\s*,\s*", r"\1", text)


# --------------------------------------------------------------------------- #
# Sentence splitting                                                           #
# --------------------------------------------------------------------------- #

# Tokens ending in "." that do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "rev", "prof", "gen", "vol", "ch",
    "no", "vs", "etc", "qtd", "ed", "eds", "trans", "p", "pp", "cf", "e.g", "i.e",
    "a.m", "p.m", "u.s", "u.k", "mt", "ft", "approx", "dept", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_SENTENCE_END = re.compile(r"""([.!?…]+["'”’)\]]*)\s+(?=["'“‘(\[]?[A-Z0-9])""")
_MAX_SENTENCE_CHARS = 380


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


def sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, respecting initials and abbreviations."""
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    out, start = [], 0
    for m in _SENTENCE_END.finditer(paragraph):
        head = paragraph[start:m.start()]
        last = head.rsplit(" ", 1)[-1].lower().strip("(\"'“‘")
        # Initials ("P. F. Bresee") and known abbreviations do not end sentences.
        if m.group(1).startswith(".") and ((len(last) == 1 and last.isalpha()) or last in _ABBREVIATIONS):
            continue
        out.append(paragraph[start:m.end(1)].strip())
        start = m.end()
    out.append(paragraph[start:].strip())
    # Very long sentences are split at clause boundaries to stay well inside
    # Kokoro's context, where its prosody is most stable.
    result = []
    for s in filter(None, out):
        while len(s) > _MAX_SENTENCE_CHARS:
            cut = max(s.rfind(sep, 0, _MAX_SENTENCE_CHARS) for sep in ("; ", ": ", ", ", " — "))
            if cut < _MAX_SENTENCE_CHARS // 3:
                break
            result.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        result.append(s)
    return result


# --------------------------------------------------------------------------- #
# Lexicon                                                                      #
# --------------------------------------------------------------------------- #


class Lexicon:
    """Ordered replacements + IPA pronunciations, reloaded when the file changes.

    File format (JSON):
      {
        "replacements": [
          {"match": "NMI", "replace": "Nazarene Missions International"},
          {"match": "\\\\bch\\\\. (\\\\d+)", "replace": "chapter \\\\1", "regex": true}
        ],
        "pronunciations": {"Bresee": "bɹəzˈi"}
      }
    Literal matches are whole-word and case-sensitive unless "ignore_case": true.
    """

    def __init__(self, path: str | None):
        self.path = path
        self._mtime: float | None = None
        self._lock = threading.Lock()
        self.replacements: list[tuple[re.Pattern, str]] = []
        self.pronunciations: re.Pattern | None = None
        self._ipa: dict[str, str] = {}

    def _maybe_reload(self) -> None:
        if not self.path:
            return
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        if mtime == self._mtime:
            return
        with self._lock:
            try:
                data = json.load(open(self.path, encoding="utf-8"))
            except Exception as e:  # keep the last good lexicon
                logger.error(f"[tts_text] lexicon {self.path} not loaded: {e}")
                self._mtime = mtime
                return
            reps = []
            for r in data.get("replacements", []):
                flags = re.IGNORECASE if r.get("ignore_case") else 0
                pat = r["match"] if r.get("regex") else rf"(?<!\w){re.escape(r['match'])}(?!\w)"
                reps.append((re.compile(pat, flags), r["replace"] if r.get("regex") else r["replace"].replace("\\", "\\\\")))
            ipa = {k: v.strip("/") for k, v in (data.get("pronunciations") or {}).items() if v}
            self.replacements = reps
            self._ipa = ipa
            words = sorted(ipa, key=len, reverse=True)
            # Skip words already inside a misaki link: "[Bresee](/.../)".
            self.pronunciations = (
                re.compile(r"(?<![\w\[])(" + "|".join(map(re.escape, words)) + r")(?![\w\]])") if words else None
            )
            self._mtime = mtime
            logger.info(f"[tts_text] lexicon loaded: {len(reps)} replacements, {len(ipa)} pronunciations")

    def apply_replacements(self, text: str) -> str:
        self._maybe_reload()
        for pat, rep in self.replacements:
            text = pat.sub(rep, text)
        return text

    def apply_pronunciations(self, text: str) -> str:
        self._maybe_reload()
        if not self.pronunciations:
            return text
        return self.pronunciations.sub(lambda m: f"[{m.group(1)}](/{self._ipa[m.group(1)]}/)", text)


_default_lexicon = Lexicon(os.getenv("TTS_LEXICON_PATH", os.path.join(os.path.dirname(__file__), "audio", "tts_lexicon.json")))


def normalize(text: str, lexicon: Lexicon | None = None) -> str:
    """Full normalization pipeline; order matters."""
    lex = lexicon or _default_lexicon
    text = lex.apply_replacements(text)
    text = normalize_scripture(text)      # before parens: "(John 3:16)" keeps its book context
    text = _YEAR_RANGE.sub(r"\1 to \2", text)
    for pat, rep in _TITLES:
        text = pat.sub(rep, text)
    text = normalize_roman(text)
    text = _paren_to_commas(text)
    return lex.apply_pronunciations(text)  # last, so links are not rewritten
