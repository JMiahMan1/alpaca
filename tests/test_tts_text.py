import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tts_text  # noqa: E402
from tts_text import Lexicon, normalize, normalize_roman, normalize_scripture, paragraphs, sentences  # noqa: E402


@pytest.mark.parametrize(
    "src, want",
    [
        ("Read Mark 14:3-9 today.", "Read Mark chapter 14, verses 3 through 9 today."),
        ("John 3:16 says", "John chapter 3, verse 16 says"),
        ("1 Cor. 13:4-7", "First Corinthians chapter 13, verses 4 through 7"),
        ("Rom 8:28, 31", "Romans chapter 8, verses 28 and 31"),
        ("Ps. 23:1–6", "Psalm chapter 23, verses 1 through 6"),
        ("John 3:16-4:2", "John chapter 3, verse 16 through chapter 4, verse 2"),
        ("Heb 11:1ff", "Hebrews chapter 11, verse 1 and following"),
        ("Rev. 21:1-4", "Revelation chapter 21, verses 1 through 4"),
        ("It was 3:16 in the afternoon.", "It was 3:16 in the afternoon."),
    ],
)
def test_scripture(src, want):
    assert normalize_scripture(src) == want


@pytest.mark.parametrize(
    "src, want",
    [
        ("Henry VIII", "Henry the Eighth"),
        ("Pope Pius IX spoke", "Pope Pius the Ninth spoke"),
        ("Charles V", "Charles the Fifth"),
        ("Chapter IV", "Chapter 4"),
        ("World War II and World War I", "World War 2 and World War 1"),
        ("Then I went", "Then I went"),
        ("the book I read", "the book I read"),
        ("Malcolm X", "Malcolm X"),
        ("Washington DC", "Washington DC"),
        ("Size XL", "Size XL"),
        ("CIVIL WAR", "CIVIL WAR"),
    ],
)
def test_roman(src, want):
    assert normalize_roman(src) == want


def _lex(tmp_path, data):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return Lexicon(str(p))


def test_pipeline_misc(tmp_path):
    lex = _lex(tmp_path, {})
    assert normalize("John Wesley (1703-1791) preached.", lex) == "John Wesley, 1703 to 1791, preached."
    assert normalize("Rev. Smith read Rev. 21:1.", lex) == "Reverend Smith read Revelation chapter 21, verse 1."
    assert normalize("(See below.) Next.", lex) == "See below. Next."
    assert normalize("St. Paul wrote.", lex) == "Saint Paul wrote."


def test_lexicon_replacements_and_pronunciations(tmp_path):
    lex = _lex(tmp_path, {
        "replacements": [
            {"match": "NMI", "replace": "Nazarene Missions International"},
            {"match": r"\bch\. (\d+)", "replace": r"chapter \1", "regex": True},
        ],
        "pronunciations": {"Bresee": "bɹəzˈi"},
    })
    out = normalize("NMI leaders read Bresee in ch. 6. NMIs stay.", lex)
    assert out == "Nazarene Missions International leaders read [Bresee](/bɹəzˈi/) in chapter 6. NMIs stay."
    # Already-linked words are not double-wrapped.
    assert normalize("[Bresee](/bɹəsˈi/) spoke", lex) == "[Bresee](/bɹəsˈi/) spoke"


def test_lexicon_hot_reload_and_bad_file(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"replacements": [{"match": "A1", "replace": "one"}]}))
    lex = Lexicon(str(p))
    assert lex.apply_replacements("A1") == "one"
    import os
    p.write_text("{not json")
    os.utime(p, (1, 2))
    assert lex.apply_replacements("A1") == "one"  # keeps last good lexicon


def test_shipped_lexicon_is_valid():
    lex = Lexicon(str(Path(tts_text.__file__).parent / "audio" / "tts_lexicon.json"))
    assert normalize("our NMI leaders, qtd. in Smith", lex) == "our Nazarene Missions International leaders, quoted in Smith"


@pytest.mark.parametrize(
    "src, want",
    [
        ("Phineas F. Bresee spoke. He left.", ["Phineas F. Bresee spoke.", "He left."]),
        ('H. D. Brown said, "Mr. Chairman, let them go." Dr. Bresee answered.',
         ['H. D. Brown said, "Mr. Chairman, let them go."', "Dr. Bresee answered."]),
        ("Read verses 4 through 7. John Wesley preached.", ["Read verses 4 through 7.", "John Wesley preached."]),
        ("It was 1908. Then came 1919!", ["It was 1908.", "Then came 1919!"]),
        ("The U.S. church grew. Was it 30,000? Yes.", ["The U.S. church grew.", "Was it 30,000?", "Yes."]),
    ],
)
def test_sentences(src, want):
    assert sentences(src) == want


def test_long_sentence_splits_at_clause():
    s = "word " * 60 + "and then, " + "more " * 30 + "end."
    parts = sentences(s)
    assert len(parts) == 2 and all(len(p) <= 380 for p in parts) and parts[0].endswith(",")


def test_paragraphs():
    assert paragraphs("One.\n\n  Two.\n\n\n") == ["One.", "Two."]
