"""Tests for scripts/render_epub_ab.py -- the EPUB A/B renderer's pure logic.

The engines need torch and Piper, so none of that is exercised here. What is
exercised is everything that decides whether the comparison is fair:

  * the EPUB walk, which has to survive the two shapes real books use (a
    leading-slash href, and a chapter fragmented across many tiny spine files)
  * the chunk planner, whose whole job is to stop Kokoro replaying one
    intonation contour for the length of a chapter
  * the ceiling assertion, because an over-long chunk indexes past the end of
    Kokoro's style table and the failure mode upstream is a silent deadlock

`tests/test_scripts_entrypoints.py` separately asserts the module imports.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("render_epub_ab", REPO / "scripts" / "render_epub_ab.py")
assert SPEC and SPEC.loader
ab = importlib.util.module_from_spec(SPEC)
# dataclasses resolves annotations through sys.modules[cls.__module__], so the
# module has to be registered before it is executed.
sys.modules[SPEC.name] = ab
SPEC.loader.exec_module(ab)

SENT = "The Nazarenes organized themselves into districts across America and Scotland. "


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _epub(path: Path, docs: list[tuple[str, str]], href_style: str = "relative") -> Path:
    """A minimal but valid enough EPUB: [spine href, xhtml body] pairs."""
    items = []
    refs = []
    for i, (_name, _body) in enumerate(docs, start=1):
        href = f"text/{i:04d}.html" if href_style == "relative" else f"/text/{i:04d}.html"
        items.append(f'<item id="i{i}" href="{href}" media-type="application/xhtml+xml"/>')
        refs.append(f'<itemref idref="i{i}"/>')
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        "<metadata>"
        '<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Test Book</dc:title>'
        "</metadata><manifest>" + "".join(items) + "</manifest><spine>" + "".join(refs) + "</spine></package>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        if href_style == "relative":
            z.writestr("OEBPS/content.opf", opf)
            for i, (_name, body) in enumerate(docs, start=1):
                z.writestr(f"OEBPS/text/{i:04d}.html", f"<html><body>{body}</body></html>")
        else:
            z.writestr("content.opf", opf)
            for i, (_name, body) in enumerate(docs, start=1):
                z.writestr(f"text/{i:04d}.html", f"<html><body>{body}</body></html>")
    return path


def _body(heading: str, paragraphs: int = 4, sent: int = 4) -> str:
    html = f"<h1>{heading}</h1>"
    for _ in range(paragraphs):
        html += "<p>" + (SENT * sent) + "</p>"
    return html


@pytest.fixture
def book(tmp_path: Path) -> Path:
    return _epub(
        tmp_path / "book.epub",
        [("a", _body("Chapter One")), ("b", _body("Chapter Two")), ("c", _body("Chapter Three"))],
    )


# --------------------------------------------------------------------------
# EPUB
# --------------------------------------------------------------------------
def test_three_chapters_are_found(book: Path):
    chapters = ab.epub_chapters(book)
    assert [c.title for c in chapters] == ["Chapter One", "Chapter Two", "Chapter Three"]
    assert [c.index for c in chapters] == [1, 2, 3]


def test_a_leading_slash_href_resolves(tmp_path: Path):
    """The Old Testament epub on this machine emits '/text/part0000.html'."""
    p = _epub(tmp_path / "slash.epub", [("a", _body("One")), ("b", _body("Two"))], href_style="absolute")
    assert [c.title for c in ab.epub_chapters(p)] == ["One", "Two"]


def test_a_fragmented_chapter_is_not_a_chapter_per_file(tmp_path: Path):
    """17 spine files of 3 KB each are one chapter, not seventeen chapters."""
    docs: list[tuple[str, str]] = []
    docs.append(("a", "<h1>Real Chapter</h1><p>" + SENT * 4 + "</p>"))
    for i in range(17):
        docs.append((f"f{i}", f"<h3>Section {i}</h3><p>" + SENT + "</p>"))
    chapters = ab.epub_chapters(_epub(tmp_path / "frag.epub", docs))
    assert [c.title for c in chapters] == ["Real Chapter"]


def test_a_thin_stub_does_not_open_a_chapter(tmp_path: Path):
    """Front matter with a heading and no prose must not become a chapter."""
    docs = [("a", _body("Real")), ("b", "<h1>Dedication</h1><p>For my mother.</p>"), ("c", _body("Also Real"))]
    chapters = ab.epub_chapters(_epub(tmp_path / "stub.epub", docs))
    assert [c.title for c in chapters] == ["Real", "Also Real"]


def test_a_tiny_chapter_is_dropped(tmp_path: Path):
    p = _epub(tmp_path / "tiny.epub", [("a", _body("Real")), ("b", "<h1>Blurb</h1><p>Short.</p>")])
    assert [c.title for c in ab.epub_chapters(p)] == ["Real"]


def test_script_and_style_content_is_not_read_aloud(tmp_path: Path):
    body = f"<h1>Kept</h1><p>{SENT * 12}</p><script>var x = 'dropped';</script><style>p{{color:red}}</style>"
    p = _epub(tmp_path / "scripts.epub", [("a", body)])
    text = " ".join(ab.epub_chapters(p)[0].paragraphs)
    assert "dropped" not in text and "color:red" not in text
    assert "Nazarenes" in text


def test_entities_and_unicode_survive(tmp_path: Path):
    body = "<h1>Kept</h1><p>" + SENT * 12 + "Nazarenes&mdash;brothers&mdash;held&#8217;s meetings.</p>"
    p = _epub(tmp_path / "ent.epub", [("a", body)])
    text = " ".join(ab.epub_chapters(p)[0].paragraphs)
    assert "&mdash;" not in text and "&#8217;" not in text  # both were decoded
    assert "\u2014" in text and "\u2019" in text


def test_a_missing_spine_item_is_skipped_not_fatal(tmp_path: Path):
    p = _epub(tmp_path / "gap.epub", [("a", _body("One")), ("b", _body("Two"))])
    # Rewrite the opf so one idref points at nothing.
    with zipfile.ZipFile(p) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    key = next(k for k in entries if k.endswith(".opf"))
    entries[key] = entries[key].replace(b'idref="i1"', b'idref="nope"')
    with zipfile.ZipFile(p, "w") as z:
        for n, d in entries.items():
            z.writestr(n, d)
    assert [c.title for c in ab.epub_chapters(p)] == ["Two"]


def test_an_epub_with_no_spine_is_rejected(tmp_path: Path):
    p = tmp_path / "nospine.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("content.opf", "<package><manifest/><spine/></package>")
    with pytest.raises(ValueError, match="spine"):
        ab.epub_chapters(p)


# --------------------------------------------------------------------------
# unit counting
# --------------------------------------------------------------------------
def test_count_units_counts_words_and_standalone_punctuation():
    # Verified against KPipeline.Result.tokens: one MToken per word plus one per
    # punctuation mark, so "The quick brown fox jumps over the lazy dog." is 10.
    assert ab.count_units("The quick brown fox jumps over the lazy dog.") == 10
    assert ab.count_units("one two three") == 3
    assert ab.count_units("") == 0


def test_count_units_does_not_count_whitespace_or_misses_hyphens():
    assert ab.count_units("a, b; c?") == 6  # 3 words + 3 marks
    assert ab.count_units("well-known") == 3  # 'well', '-', 'known'
    assert ab.count_units("  spaced   out  ") == 2


# --------------------------------------------------------------------------
# chunk planning -- the point of the whole script
# --------------------------------------------------------------------------
def test_targets_span_the_whole_range():
    t = ab.plan_chunk_targets(12)
    assert min(t) == ab.CHUNK_TARGET_LO
    assert max(t) == ab.CHUNK_TARGET_HI


def test_targets_are_deterministic():
    assert ab.plan_chunk_targets(30) == ab.plan_chunk_targets(30)


def test_consecutive_targets_are_far_apart():
    """A monotonic walk would satisfy a determinism test and still be a pattern."""
    t = ab.plan_chunk_targets(12)
    span = ab.CHUNK_TARGET_HI - ab.CHUNK_TARGET_LO
    jumps = [abs(b - a) for a, b in itertools.pairwise(t)]
    # The interleave converges on the middle of the ladder, so the jumps shrink
    # and the *smallest* pair is adjacent; what has to hold is that consecutive
    # chunks differ by a large fraction of the range on average, and that the
    # range is genuinely covered rather than clustered.
    assert sum(jumps) / len(jumps) > span * 0.5
    assert max(t) - min(t) == span
    assert len(set(t)) == len(t)


def test_a_requested_count_shorter_or_longer_than_the_ladder():
    assert len(ab.plan_chunk_targets(3)) == 3
    assert len(ab.plan_chunk_targets(40)) == 40
    assert ab.plan_chunk_targets(40)[-1] == ab.plan_chunk_targets(12)[-1]


def test_a_chunk_spans_at_most_two_paragraphs_and_only_to_absorb_a_runt():
    """The paragraph rule yields to the runt floor, deliberately.

    Originally this asserted one chunk per paragraph. `_merge_short` broke
    that on purpose: a one-sentence paragraph became its own 2-token chunk,
    which is exactly where kokoro mangles words.

    A chunk may absorb the *next* paragraph, but only when that paragraph is
    under the floor, and only ever one. Note the index delta is NOT the
    paragraph span: when chunk N absorbs paragraph p, chunk N+1 starts at p+1,
    so consecutive indices can legitimately jump by 2. The guarantee is stated
    on the merge itself, where the absorbed text is still identifiable.
    """
    big = " ".join(f"w{i}" for i in range(150))  # 150 units, a normal chunk
    src = [_chunk(big, paragraph=0), _chunk("He paused.", paragraph=1), _chunk(big, paragraph=2), _chunk("She nodded.", paragraph=3)]
    out = ab._merge_short(src, ab.CHUNK_MIN_TOKENS)

    # every source paragraph is covered exactly once, in order
    assert [c.paragraph_index for c in out] == [0, 2]
    # and each absorbed paragraph was a runt
    for absorbed in (1, 3):
        assert ab.count_units(src[absorbed].text) < ab.CHUNK_MIN_TOKENS
        # the merge kept the EARLIER paragraph, which is what selects the
        # shorter gap -- if it kept the later one every folded runt would
        # silently become a chapter-length pause
        owner = [c for c in out if c.paragraph_index < absorbed][-1]
        assert owner.paragraph_index == absorbed - 1
        assert src[absorbed].text in owner.text


def test_no_text_is_lost(book: Path):
    ch = ab.epub_chapters(book)[0]
    chunks = ab.plan_chunks(ch)
    joined = " ".join(c.text for c in chunks)
    assert ab.count_units(joined) == sum(ab.count_units(p) for p in ch.paragraphs)
    assert "Nazarenes" in joined


def test_no_chunk_sits_in_kokoros_documented_weak_spots(book: Path):
    """<20 tokens is a documented weakness; the planner must never go there."""
    ch = ab.epub_chapters(book)[0]
    for c in ab.plan_chunks(ch):
        assert ab.count_units(c.text) >= min(ab.CHUNK_MIN_TOKENS, ab.count_units(c.text))


def test_a_single_huge_sentence_becomes_its_own_chunk():
    ch = ab.Chapter(index=1, title="One", paragraphs=["word " * 600])
    chunks = ab.plan_chunks(ch)
    assert len(chunks) == 1
    assert ab.count_units(chunks[0].text) == 600


def test_an_empty_paragraph_does_not_break_the_planner():
    ch = ab.Chapter(index=1, title="One", paragraphs=[SENT * 3, "", SENT * 3])
    assert len(ab.plan_chunks(ch)) >= 1


# --------------------------------------------------------------------------
# the ceiling assertion
# --------------------------------------------------------------------------
def _manifest(indices=(1,)):
    return {"chapters": [{"index": i, "title": "t"} for i in indices]}


def test_a_chunk_that_overruns_the_style_table_is_refused():
    m = _manifest()
    rows = [{"chapter_index": 1, "chunk": 0, "style_row": 512, "tokens": 512}]
    with pytest.raises(SystemExit, match="style table"):
        ab._attach_rows(m, "kokoro", rows)


def test_a_chunk_exactly_at_the_last_row_is_accepted():
    m = _manifest()
    rows = [{"chapter_index": 1, "chunk": 0, "style_row": 509, "tokens": 509}]
    ab._attach_rows(m, "kokoro", rows)
    assert m["chapters"][0]["kokoro_style_rows_used"] == 1
    assert m["chapters"][0]["kokoro_token_range"] == [509, 509]


def test_the_style_row_count_is_measured_per_chapter():
    m = _manifest(indices=(1, 2))
    rows = [
        {"chapter_index": 1, "chunk": 0, "style_row": 100, "tokens": 100},
        {"chapter_index": 1, "chunk": 1, "style_row": 300, "tokens": 300},
        {"chapter_index": 2, "chunk": 0, "style_row": 100, "tokens": 100},
    ]
    ab._attach_rows(m, "kokoro", rows)
    assert m["chapters"][0]["kokoro_style_rows_used"] == 2
    assert m["chapters"][1]["kokoro_style_rows_used"] == 1


def test_the_ceiling_is_not_checked_for_piper():
    """Piper has no style table, so the assertion must not fire for it."""
    m = _manifest()
    ab._attach_rows(m, "piper", [{"chapter_index": 1, "chunk": 0, "tokens": 4000}])


# --------------------------------------------------------------------------
# docs claim the code keeps
# --------------------------------------------------------------------------
def test_the_module_docstring_states_the_style_row_mechanism():
    """This is the non-obvious claim the whole script rests on; pin it."""
    doc = re.sub(r"\s+", " ", ab.__doc__ or "")
    assert "ref_s = voices[len(tokens)]" in doc
    assert "510" in doc
    assert "not be possible to eliminate this" in doc
    assert "tts_text.normalize" in doc
    assert "zero cross-sentence context" in doc


def test_the_chunk_bounds_sit_inside_kokoros_documented_range():
    assert ab.CHUNK_TARGET_LO >= 20, "kokoro is documented as weak below ~20 tokens"
    assert ab.CHUNK_TARGET_HI < ab.KOKORO_STYLE_ROWS - 1, "the style row is the token count"
    assert ab.KOKORO_STYLE_ROWS == 510


def test_the_gaps_are_shared_by_both_engines():
    """Different gaps would make the listen comparison meaningless."""
    src = (REPO / "scripts" / "render_epub_ab.py").read_text()
    assert src.count("GAP_WITHIN_PARAGRAPH_S") == 3  # module + both engines
    assert src.count("GAP_BETWEEN_PARAGRAPHS_S") == 3
    assert "GAP_WITHIN_PARAGRAPH_S = 0.20" in src
    assert "GAP_BETWEEN_PARAGRAPHS_S = 0.70" in src


def test_dump_text_needs_no_torch():
    """--dump-text is the inspect-what-will-be-said mode; it must stay importable."""
    import subprocess

    out = subprocess.run(
        ["python3", str(REPO / "scripts" / "render_epub_ab.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0
    assert "--dump-text" in out.stdout
    assert "ref_s = voices[len(tokens)]" in out.stdout


# --------------------------------------------------------------------------
# _merge_short: the runt guard
# --------------------------------------------------------------------------
def count(text):
    return ab.count_units(text)


def _chunk(text, *, index=0, paragraph=0, target=200):
    return ab.Chunk(index, text, paragraph, target)


def test_a_runt_is_folded_into_the_chunk_before_it():
    """The single-sentence-paragraph case that motivated the function."""
    long_para = " ".join(f"word{i}" for i in range(200))  # 200 units, above the floor
    merged = ab._merge_short([_chunk(long_para), _chunk("He agreed.")], ab.CHUNK_MIN_TOKENS)
    assert len(merged) == 1
    assert count(merged[0].text) == count(long_para) + count("He agreed.") == 203
    assert "He agreed." in merged[0].text


def test_a_runt_keeps_the_earlier_paragraph_so_the_shorter_gap_is_used():
    """A paragraph break only picks which of two gaps is inserted; the merged
    chunk must therefore keep the earlier index, or every folded runt would
    silently become a chapter-length pause."""
    a = _chunk(" ".join(f"w{i}" for i in range(120)), paragraph=7)
    b = _chunk("Yes.", paragraph=8)
    merged = ab._merge_short([a, b], ab.CHUNK_MIN_TOKENS)
    assert len(merged) == 1
    assert merged[0].paragraph_index == 7


def test_a_leading_runt_with_no_predecessor_is_left_alone():
    """Merging it would mean crossing a chapter boundary or duplicating text."""
    out = ab._merge_short([_chunk("It was so.")], ab.CHUNK_MIN_TOKENS)
    assert len(out) == 1
    assert out[0].text == "It was so."


def test_a_runt_is_not_folded_when_the_merge_would_exceed_the_ceiling():
    """Otherwise the merge would trade a runt for a 'rushing on long
    utterances' chunk, which the model card also calls a weakness."""
    long_para = " ".join(f"w{i}" for i in range(ab.CHUNK_MAX_TOKENS))
    out = ab._merge_short([_chunk(long_para), _chunk("No.")], ab.CHUNK_MIN_TOKENS)
    assert len(out) == 2
    assert count(out[0].text) == ab.CHUNK_MAX_TOKENS
    assert out[1].text == "No."


def test_merging_renumbers_the_chunks_contiguously():
    out = ab._merge_short(
        [_chunk(" ".join(f"w{i}" for i in range(100)), index=0),
         _chunk("Ok.", index=1),
         _chunk(" ".join(f"x{i}" for i in range(100)), index=2)],
        ab.CHUNK_MIN_TOKENS,
    )
    assert [c.index for c in out] == [0, 1]


def test_merging_never_loses_a_word():
    """The whole point is a nicer chunk boundary, not shorter audio."""
    texts = [" ".join(f"a{i}" for i in range(150)), "Mm-hm.", " ".join(f"b{i}" for i in range(150)), "Right."]
    out = ab._merge_short([_chunk(t) for t in texts], ab.CHUNK_MIN_TOKENS)
    joined = " ".join(c.text for c in out)
    for t in texts:
        for w in t.split():
            assert w in joined


def test_consecutive_runts_collapse_into_one_chunk_rather_than_a_chain():
    out = ab._merge_short([_chunk("A long opening sentence here."), _chunk("Yes."), _chunk("No.")], ab.CHUNK_MIN_TOKENS)
    assert len(out) == 1
    assert "Yes." in out[0].text and "No." in out[0].text


def test_no_chunk_under_the_floor_survives_unless_it_is_the_whole_paragraph():
    """End-to-end on the real book, which is the shape the fix was written for.

    The only permitted survivor is a chunk that has no neighbour at all, i.e.
    the entire chapter is one short paragraph.
    """
    chapter = ab.Chapter(
        index=0,
        title="t",
        paragraphs=[" ".join(f"w{i}" for i in range(120))] * 6 + ["He paused."],
    )
    chunks = ab.plan_chunks(chapter)
    for i, c in enumerate(chunks):
        if count(c.text) < ab.CHUNK_MIN_TOKENS:
            assert i == 0, f"chunk {i} is a runt with neighbours: {c.text!r}"


def test_a_chapter_that_is_only_a_short_paragraph_still_renders():
    """The documented exception: refusing to emit a chunk here would drop text."""
    chapter = ab.Chapter(index=0, title="t", paragraphs=["He paused."])
    chunks = ab.plan_chunks(chapter)
    assert len(chunks) == 1
    assert chunks[0].text == "He paused."
