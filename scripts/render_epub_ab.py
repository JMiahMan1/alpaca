#!/usr/bin/env python3
"""Render the same chapters of an EPUB through Kokoro and Piper, for listening.

WHY THIS SCRIPT EXISTS
----------------------
Kokoro and Piper are both recommended for local narration and there is **no
published long-form comparison of the two** -- no paper, no MOS study, no
listening test. Every "Kokoro beats Piper" claim in circulation comes from
short-utterance voice-assistant testing, which is not the audiobook workload.
(One widely-copied "Kokoro 4.8/5 vs Piper 3.5/5" pair is fabricated: the page it
comes from shows Piper as "No votes yet".) So the question has to be answered by
listening, and the answer depends on the book.

THE STRUCTURAL FINDING THIS SCRIPT IS BUILT AROUND
--------------------------------------------------
The two engines fail in *opposite* ways, and only one of them is a property of
the model rather than of the chunker:

* **Piper synthesises one sentence at a time, natively.** `PiperVoice.synthesize`
  phonemises with a phonemiser that returns text "grouped by sentence" and loops,
  so it is structurally immune to attention collapse, repetition loops and
  length-extrapolation failure. It also has zero cross-sentence context: no arc,
  and the same sentence is intoned identically every time it recurs.

* **Kokoro is a non-autoregressive, all-frames-at-once model with a hard
  510-token ceiling.** It wants 100-200 tokens of context and genuinely uses it.
  A book is ~1.1M words, so a book is necessarily 12,000-15,000 forward passes,
  and the seams cannot be eliminated -- the author says so: *"It will not be
  possible to eliminate this, as this artificial separation is not intended."*
  Worse, and this is the part almost nobody names:

      ref_s = voices[len(tokens)]        # kokoro/pipeline.py, KModel.forward

  **The style vector is selected by input length.** Every voice ships 510 style
  rows and the row index IS the token count. Chunk a book into 15,000 segments
  of 150 tokens and every one of them selects style row 150: the same intonation
  contour, deterministically, once per chunk, for ten hours. That is a monotony
  generator built into the model and it is independent of how good the voice is.

  The fix is cheap and this script does it: **deliberately vary the chunk
  lengths** so the selected style row keeps moving. `plan_chunk_targets` walks
  the range with a golden-ratio step, so it is deterministic (no seed to pass,
  identical on every run, which is what makes the A/B repeatable) without being
  periodic.

  Note the production `/api/tts` route cannot do this: `audio_server.py`
  synthesises sentence by sentence, so every live request is in the model's
  documented short-utterance weakness (<20 tokens). An audiobook needs its own
  chunker, which is why this is a script and not a route parameter.

WHY THE TEXT GOES THROUGH `tts_text.normalize`
----------------------------------------------
Both engines use an external G2P and both are weak on numbers, dates, currency,
abbreviations and scripture references -- Piper+espeak renders "World War II" as
*"World War roman 2"* and Kokoro+misaki silently drops words. `tts_text.py` is
this repo's pre-normaliser for exactly that, and it is already the production
TTS path's. Both engines get the **same normalised text**: that is the
controlled variable, and it is the one place where this repo is strictly better
out of the box than either engine.

METHODOLOGY (the confounds this removes)
----------------------------------------
1. **Same text.** One chunk list, both engines. Piper sees the identical strings
   Kokoro does; it just phonemises them differently.
2. **Identical gaps.** Same silence between chunks and paragraphs.
3. **Level-matched.** Piper's `SynthesisConfig.normalize_audio` makes its output
   louder than Kokoro's, and louder sounds better. Both are scaled to the same
   peak so the comparison is about voice, not level.
4. **Native sample rates, not resampled.** Kokoro 24000 Hz, Piper 22050 Hz.
   There is no 24 kHz English Piper "high" tier -- the difference between Piper
   tiers is model size and training data, not rate -- and resampling to force a
   match would smear exactly the high-frequency detail under comparison.

WHAT IT WRITES
--------------
    <out>/text/chNN.txt              the normalised text, for inspection
    <out>/kokoro/chNN.wav            peak-matched to PEAK_MATCH
    <out>/piper/chNN.wav             peak-matched to PEAK_MATCH
    <out>/manifest.json              per-chunk rows, the measured token range,
                                     and the set of Kokoro style rows used

The manifest records the **measured** token count per chunk (from
`KPipeline.Result.tokens`, the same list the style row is indexed on) rather
than trusting the planner's estimate, so "the chunk lengths really did vary, and
really did stay in range" is a measurement, not a claim.

USAGE
-----
    python3 render_epub_ab.py book.epub --out /out --chapters 3
    python3 render_epub_ab.py book.epub --out /out --engines kokoro
    python3 render_epub_ab.py book.epub --out /out --dump-text

Must run where torch + kokoro + piper-tts + soundfile are installed; the audio
image (`Dockerfile.audio`) has all of them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Chunk sizing. Kokoro indexes its 510 style rows by token count, and its model
# card puts the useful range at 100-200 tokens, calls <10-20 a documented
# weakness, and warns about >400. Over ~509 the ONNX path raises IndexError
# *inside a thread-pool worker*, which deadlocks the stream with no output and no
# crash, so staying under the ceiling is not optional politeness.
CHUNK_MIN_TOKENS = 70
CHUNK_MAX_TOKENS = 400
CHUNK_TARGET_LO = 90
CHUNK_TARGET_HI = 360

GAP_WITHIN_PARAGRAPH_S = 0.20
GAP_BETWEEN_PARAGRAPHS_S = 0.70

# Both engines are peak-matched to this before writing, so the louder one does
# not win on level. Speech only, single voice, so there is no bed to bury.
PEAK_MATCH = 0.5

MIN_CHAPTER_CHARS = 800


# --------------------------------------------------------------------------
# EPUB
# --------------------------------------------------------------------------
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCKS = frozenset({"p", "div", "li", "blockquote", "td", "th", "section", "article", "dd", "dt"})
# Content that must never reach the synthesiser: reading a <style> rule or a
# variable name aloud is the classic EPUB-to-speech failure.
_SKIP = frozenset({"script", "style", "head", "nav", "svg"})


class _DocParser(HTMLParser):
    """Pull (kind, text) runs out of one XHTML document.

    `kind` is "h1".."h6" or "p". Inline markup is transparent and entities are
    decoded; <script>/<style>/<head> content is dropped rather than read aloud.
    """

    HEADINGS = _HEADINGS
    BLOCKS = _BLOCKS
    SKIP = _SKIP

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.runs: list[tuple[str, str]] = []
        self._buf: list[str] = []
        self._skip = 0
        self._kind = "p"

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in self.HEADINGS or tag in self.BLOCKS:
            self.flush()
            self._kind = f"h{tag[1:]}" if tag in self.HEADINGS else "p"

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in self.HEADINGS or tag in self.BLOCKS:
            self.flush()
            self._kind = "p"

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        if text:
            self.runs.append((self._kind, text))


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


def _resolve(names: set[str], opf_name: str, href: str) -> str:
    """EPUB hrefs are relative to the OPF, but some books emit a leading '/'."""
    base = opf_name.rsplit("/", 1)[0] if "/" in opf_name else ""
    cands = [f"{base}/{href}" if base else href, href.lstrip("/"), href]
    for cand in cands:
        cand = re.sub(r"//+", "/", cand)
        if cand in names:
            return cand
    raise KeyError(f"spine item {href!r} (opf {opf_name}) is not in the archive")


def epub_chapters(epub: Path) -> list[Chapter]:
    """Chapters in spine order, split on the first heading of each spine document.

    Some books fragment a chapter across many tiny spine documents (the Old
    Testament EPUB on this machine has 17 files under 3 KB each), so a heading
    per document would produce 17 "chapters" of nothing. A document that carries
    real prose therefore continues the open chapter, and only a document that
    starts a new heading *and* carries prose opens one.
    """
    with zipfile.ZipFile(epub) as z:
        names = set(z.namelist())
        opf_name = next(n for n in names if n.endswith(".opf"))
        opf = z.read(opf_name).decode("utf-8", "replace")
        items: dict[str, str] = {}
        for m in re.finditer(r"<item\b([^>]*?)/?>", opf):
            attrs = m.group(1)
            i = re.search(r'\bid="([^"]+)"', attrs)
            h = re.search(r'\bhref="([^"]+)"', attrs)
            if i and h:
                items[i.group(1)] = h.group(1)
        spine = re.findall(r'<itemref[^>]*idref="([^"]+)"', opf)
        if not spine:
            raise ValueError(f"{epub}: no <itemref> spine in {opf_name}")

        chapters: list[Chapter] = []
        current: Chapter | None = None
        for sid in spine:
            if sid not in items:
                continue
            parser = _DocParser()
            parser.feed(z.read(_resolve(names, opf_name, items[sid])).decode("utf-8", "replace"))
            parser.close()
            parser.flush()
            if not parser.runs:
                continue
            heading = next((t for k, t in parser.runs if k.startswith("h")), "")
            prose = [t for k, t in parser.runs if not k.startswith("h")]
            if current is not None and not any(len(p) > 200 for p in prose):
                current.paragraphs.extend(prose)
                continue
            if current is not None:
                chapters.append(current)
            current = Chapter(index=len(chapters) + 1, title=heading or f"Chapter {len(chapters) + 1}")
            current.paragraphs.extend(prose)
        if current is not None:
            chapters.append(current)
    return [c for c in chapters if c.chars > MIN_CHAPTER_CHARS]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
_UNIT_RE = re.compile(r"\w+|[^\w\s]")


def count_units(text: str) -> int:
    """Approximate Kokoro's token count: words plus standalone punctuation.

    Measured against `KPipeline.Result.tokens` on this book, Kokoro emits one
    MToken per word and one per punctuation mark, so this is within a few
    percent. It is only ever used to *place chunk boundaries*; the manifest
    records the real `len(Result.tokens)` for every chunk afterwards, so the
    claim that chunk lengths varied is measured rather than assumed. Using a
    word count rather than chars/4 also means `--dump-text` works without torch.
    """
    return len(_UNIT_RE.findall(text))


def plan_chunk_targets(n: int) -> list[int]:
    """`n` chunk sizes spanning the target range, alternating low and high.

    Alternating matters more than it looks. A golden-ratio or low-discrepancy
    walk is equidistributed over a long run but is locally *monotonic*, so eight
    consecutive chunks would descend steadily and the ear would hear a
    slow-drifting contour -- still a pattern, just a slower one. Interleaving the
    ends of the range makes consecutive chunks the most different pair available,
    which is the property that actually breaks "the same intonation, N times".

    The sequence is a deterministic function of the range, so there is no seed to
    pass and two runs of this script render the same book identically.
    """
    steps = 12
    span = CHUNK_TARGET_HI - CHUNK_TARGET_LO
    sizes = [CHUNK_TARGET_LO + round(i * span / (steps - 1)) for i in range(steps)]
    out: list[int] = []
    lo, hi = 0, steps - 1
    while lo <= hi:
        out.append(sizes[lo])
        lo += 1
        if lo <= hi:
            out.append(sizes[hi])
            hi -= 1
    return out[:n] if n < len(out) else out + [out[-1]] * (n - len(out))


@dataclass
class Chunk:
    index: int
    text: str
    paragraph_index: int
    target_units: int


def plan_chunks(chapter: Chapter) -> list[Chunk]:
    """Pack a chapter's sentences into chunks of deliberately varied length.

    Boundaries are always sentence boundaries -- from the repo's own
    `tts_text.sentences`, the same splitter the production TTS route uses -- and
    never cross a paragraph, because a paragraph break is where the longer gap
    goes and because a heading or a citation is a bad place to glue two
    unrelated sentences together.
    """
    from tts_text import sentences

    rough = max(1, sum(max(1, count_units(p) // 120) for p in chapter.paragraphs))
    targets = plan_chunk_targets(rough)

    chunks: list[Chunk] = []
    ti = 0
    for pi, para in enumerate(chapter.paragraphs):
        units = sentences(para) or [para]
        cur: list[str] = []
        cur_n = 0
        target = targets[ti % len(targets)]
        for sentence in units:
            n = count_units(sentence)
            # Flush *before* overshooting, so one long sentence becomes its own
            # chunk instead of being glued to a short one and pushing both past
            # the ceiling.
            if cur and cur_n + n > target:
                chunks.append(Chunk(len(chunks), " ".join(cur), pi, target))
                ti += 1
                target = targets[ti % len(targets)]
                cur, cur_n = [], 0
            cur.append(sentence)
            cur_n += n
        if cur:
            chunks.append(Chunk(len(chunks), " ".join(cur), pi, target))
            ti += 1
    return _merge_short(chunks, CHUNK_MIN_TOKENS)


def _merge_short(chunks: list[Chunk], floor: int) -> list[Chunk]:
    """Fold a runt chunk into its neighbour.

    A paragraph that is one short sentence ("He agreed.") becomes its own chunk,
    and a 1-2 token chunk is squarely in the regime kokoro's model card calls a
    weakness -- short utterances are where it mangles words, mispronounces and
    drops phonemes. Measured on this book before the fix: chunks of 1 and 2
    tokens, ~30 of them in three chapters.

    The paragraph rule yields to the floor on purpose. A paragraph break is a
    nicety (it only decides which of two gap lengths is inserted); a word
    rendered on its own is a defect, so the short side merges and the merged
    chunk keeps the *earlier* paragraph's index, i.e. the shorter gap.

    A runt with no neighbour -- a whole paragraph that is itself one short
    sentence -- is left alone, because merging it would mean crossing into a
    different chapter or duplicating text.
    """
    out: list[Chunk] = []
    for c in chunks:
        n = count_units(c.text)
        if out and n < floor:
            prev = out[-1]
            total = count_units(prev.text) + n
            if total <= CHUNK_MAX_TOKENS:
                out[-1] = Chunk(prev.index, f"{prev.text} {c.text}", prev.paragraph_index, prev.target_units)
                continue
        out.append(c)
    return [Chunk(i, c.text, c.paragraph_index, c.target_units) for i, c in enumerate(out)]


# --------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------
def _silence(seconds: float, sample_rate: int):
    import numpy as np

    # max(0, ...) because np.zeros rejects a negative length, and round() of a
    # negative float is negative.
    return np.zeros(max(0, round(seconds * sample_rate)), dtype=np.float32)


def _join(pieces: list, gaps: list[float], sample_rate: int):
    import numpy as np

    if not pieces:
        return np.zeros(0, dtype=np.float32)
    out: list = []
    for i, piece in enumerate(pieces):
        if i:
            out.append(_silence(gaps[i - 1] if i - 1 < len(gaps) else 0.0, sample_rate))
        out.append(np.asarray(piece, dtype=np.float32))
    return np.concatenate(out)


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    """Write 16-bit PCM at a matched peak. The match is deliberate: see PEAK_MATCH."""
    import numpy as np
    import soundfile as sf

    data = np.nan_to_num(np.asarray(samples, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1e-9:
        data = data * (PEAK_MATCH / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(data, -1.0, 1.0), sample_rate, subtype="PCM_16")


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------
class KokoroEngine:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, voice: str, speed: float) -> None:
        import torch
        from kokoro import KPipeline

        self.voice = voice
        self.speed = speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = KPipeline(lang_code="a")
        self.style_rows = int(self.pipe.load_voice(voice).shape[0])

    def render(self, chapter: Chapter, chunks: list[Chunk]) -> tuple[list, list, list[dict], float]:
        import numpy as np

        pieces: list = []
        gaps: list[float] = []
        rows: list[dict] = []
        t0 = time.perf_counter()
        for ch in chunks:
            # split_pattern=None is essential: kokoro splits on its own newline
            # pattern by default, which would re-chunk us back to roughly one
            # sentence per call and undo the entire point. Same call the
            # production route makes.
            results = list(self.pipe(ch.text, voice=self.voice, speed=self.speed, split_pattern=None))
            audio = (
                np.concatenate(
                    [np.asarray(r.audio, dtype=np.float32) for r in results if getattr(r, "audio", None) is not None]
                )
                if results
                else np.zeros(0, dtype=np.float32)
            )
            tokens = max((len(r.tokens) for r in results if getattr(r, "tokens", None) is not None), default=0)
            pieces.append(audio)
            gaps.append(
                GAP_BETWEEN_PARAGRAPHS_S
                if ch.index and chunks[ch.index - 1].paragraph_index != ch.paragraph_index
                else GAP_WITHIN_PARAGRAPH_S
            )
            rows.append(
                {
                    "chapter_index": chapter.index,
                    "chunk": ch.index,
                    "paragraph": ch.paragraph_index,
                    "chars": len(ch.text),
                    "planned_units": ch.target_units,
                    "tokens": tokens,
                    "style_row": tokens,
                    "seconds": round(audio.size / self.sample_rate, 2),
                }
            )
        return pieces, gaps, rows, time.perf_counter() - t0


class PiperEngine:
    name = "piper"

    def __init__(self, model_path: Path, config_path: Path | None, use_cuda: bool) -> None:
        from piper import PiperVoice

        self.model_path = model_path
        self.sample_rate = 22050
        self.voice = PiperVoice.load(
            str(model_path), str(config_path) if config_path else None, use_cuda=use_cuda
        )
        sr = getattr(getattr(self.voice, "config", None), "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)

    def render(self, chapter: Chapter, chunks: list[Chunk]) -> tuple[list, list, list[dict], float]:
        import numpy as np

        pieces: list = []
        gaps: list[float] = []
        rows: list[dict] = []
        t0 = time.perf_counter()
        for ch in chunks:
            # Piper phonemises per SENTENCE internally, so chunk size changes how
            # many calls it makes but not what it synthesises. Feeding it the
            # same chunks anyway is what keeps the two engines driven
            # identically; the returned chunk count is recorded as the measure
            # of Piper's own sentence segmentation.
            audios: list = []
            rates: set[int] = set()
            for audio_chunk in self.voice.synthesize(ch.text):
                audios.append(np.asarray(audio_chunk.audio_float_array, dtype=np.float32))
                rates.add(int(audio_chunk.sample_rate))
            if len(rates) > 1:
                raise RuntimeError(f"piper returned mixed sample rates {sorted(rates)} for chunk {ch.index}")
            if rates:
                self.sample_rate = rates.pop()
            pieces.append(_join(audios, [0.0] * max(0, len(audios) - 1), self.sample_rate))
            gaps.append(
                GAP_BETWEEN_PARAGRAPHS_S
                if ch.index and chunks[ch.index - 1].paragraph_index != ch.paragraph_index
                else GAP_WITHIN_PARAGRAPH_S
            )
            rows.append(
                {
                    "chapter_index": chapter.index,
                    "chunk": ch.index,
                    "paragraph": ch.paragraph_index,
                    "chars": len(ch.text),
                    "planned_units": ch.target_units,
                    "piper_sentence_chunks": len(audios),
                    "seconds": round(pieces[-1].size / self.sample_rate, 2),
                }
            )
        return pieces, gaps, rows, time.perf_counter() - t0


# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("epub", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chapters", type=int, default=3, help="how many chapters to render (0 = all)")
    ap.add_argument("--engines", default="kokoro,piper", help="comma list of: kokoro,piper")
    ap.add_argument("--kokoro-voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--piper-voice", default="en_US-lessac-high", help="Piper's best English voice")
    ap.add_argument("--piper-model", type=Path, help="a downloaded .onnx voice; fetched if absent")
    ap.add_argument("--piper-config", type=Path, help="voice .json, if it is not <model>.json")
    ap.add_argument("--piper-voices-dir", type=Path, default=Path("/voices"))
    ap.add_argument("--dump-text", action="store_true", help="write the normalised text and stop")
    ap.add_argument(
        "--measure-only",
        action="store_true",
        help="plan and report the chunking (and therefore the style rows) without synthesising; "
        "needs no torch, no GPU and no voice",
    )
    return ap.parse_args(argv)


def _summarise(name: str, rows: list[dict], elapsed: float, sample_rate: int, extra: dict) -> dict:
    audio_s = round(sum(r["seconds"] for r in rows), 2)
    return {
        "engine": name,
        "sample_rate": sample_rate,
        "audio_seconds": audio_s,
        "wall_seconds": round(elapsed, 2),
        "realtime_factor": round(audio_s / elapsed, 2) if elapsed > 0 else None,
        "chunks": len(rows),
        **extra,
    }


def main(argv: list[str] | None = None) -> int:
    import tts_text

    args = parse_args(argv)
    if not args.epub.is_file():
        raise SystemExit(f"no such epub: {args.epub}")
    wanted = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in wanted if e not in {"kokoro", "piper"}]
    if unknown:
        raise SystemExit(f"unknown engine(s): {unknown}")

    chapters = epub_chapters(args.epub)
    if not chapters:
        raise SystemExit(f"{args.epub}: no chapter carried more than {MIN_CHAPTER_CHARS} characters")
    if args.chapters:
        # The longest chapters, not the first ones: an EPUB's opening spine items
        # are usually a dedication and two forewords, which are the least
        # representative prose in the book and the wrong thing to form an
        # opinion from. Rendered in book order regardless of which were picked.
        picked = sorted(chapters, key=lambda c: -c.chars)[: args.chapters]
        chapters = sorted(picked, key=lambda c: c.index)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "text").mkdir(exist_ok=True)

    # Normalise once per chapter, before sentence splitting: the scripture, date
    # and roman-numeral rules want the whole passage in view.
    manifest: dict = {"epub": str(args.epub), "peak_match": PEAK_MATCH, "chapters": [], "engines": {}}
    planned: list[tuple[Chapter, list[Chunk]]] = []
    for ch in chapters:
        text = tts_text.normalize("\n\n".join(ch.paragraphs))
        (out / "text" / f"ch{ch.index:02d}.txt").write_text(text, encoding="utf-8")
        chunks = plan_chunks(ch)
        planned.append((ch, chunks))
        sizes = [c.target_units for c in chunks]
        print(
            f"  ch{ch.index:02d} {ch.title[:52]!r}: {ch.chars} chars -> {len(chunks)} chunks, "
            f"planned {min(sizes)}-{max(sizes)} units",
            flush=True,
        )
        manifest["chapters"].append(
            {
                "index": ch.index,
                "title": ch.title,
                "raw_chars": ch.chars,
                "normalised_chars": len(text),
                "chunks": len(chunks),
                "wav": f"ch{ch.index:02d}.wav",
            }
        )
    print(f"chapters: {len(chapters)} of {args.epub.name}", flush=True)

    _save_manifest(out, manifest)
    if args.dump_text:
        print(f"wrote {out}/text/ and manifest.json")
        return 0
    if args.measure_only:
        for ch, chunks in planned:
            sizes = [c.target_units for c in chunks]
            actual = [count_units(c.text) for c in chunks]
            for entry in manifest["chapters"]:
                if entry["index"] == ch.index:
                    entry.update(
                        {
                            "kokoro_style_rows_used": len(set(actual)),
                            "kokoro_token_range": [min(actual), max(actual)],
                            "chunks": len(chunks),
                        }
                    )
                    break
            print(
                f"  ch{ch.index:02d} {ch.title[:44]!r}: {len(chunks)} chunks, "
                f"tokens {min(actual)}-{max(actual)} over {len(set(actual))} distinct style rows",
                flush=True,
            )
        _save_manifest(out, manifest)
        print(f"wrote {out}/manifest.json (no audio rendered)")
        return 0

    if "kokoro" in wanted:
        eng = KokoroEngine(args.kokoro_voice, args.speed)
        all_rows: list[dict] = []
        wall = 0.0
        for ch, chunks in planned:
            pieces, gaps, rows, elapsed = eng.render(ch, chunks)
            wall += elapsed
            _write_wav(out / "kokoro" / f"ch{ch.index:02d}.wav", _join(pieces, gaps, eng.sample_rate), eng.sample_rate)
            all_rows.extend(rows)
            print(f"  kokoro ch{ch.index:02d} -> {ch.title[:52]!r} ({elapsed:.1f}s wall)", flush=True)
        toks = [r["tokens"] for r in all_rows if r["tokens"]]
        manifest["engines"]["kokoro"] = _summarise(
            "kokoro",
            all_rows,
            wall,
            eng.sample_rate,
            {
                "voice": args.kokoro_voice,
                "device": eng.device,
                "voice_style_rows": eng.style_rows,
                "measured_tokens_min": min(toks, default=None),
                "measured_tokens_max": max(toks, default=None),
                "distinct_style_rows": len({r["style_row"] for r in all_rows}),
            },
        )
        _attach_rows(manifest, "kokoro", all_rows)
        _save_manifest(out, manifest)

    if "piper" in wanted:
        model = args.piper_model
        if model is None or not Path(model).is_file():
            model = _fetch_piper_voice(args.piper_voice, args.piper_voices_dir)
        eng = PiperEngine(Path(model), args.piper_config, use_cuda=False)
        all_rows = []
        wall = 0.0
        for ch, chunks in planned:
            pieces, gaps, rows, elapsed = eng.render(ch, chunks)
            wall += elapsed
            _write_wav(out / "piper" / f"ch{ch.index:02d}.wav", _join(pieces, gaps, eng.sample_rate), eng.sample_rate)
            all_rows.extend(rows)
            print(
                f"  piper  ch{ch.index:02d} -> {ch.title[:52]!r} ({elapsed:.1f}s wall, "
                f"{sum(r['piper_sentence_chunks'] for r in rows)} internal sentence chunks)",
                flush=True,
            )
        manifest["engines"]["piper"] = _summarise(
            "piper",
            all_rows,
            wall,
            eng.sample_rate,
            {
                "model": Path(model).name,
                "internal_sentence_chunks": sum(r["piper_sentence_chunks"] for r in all_rows),
                "note": "piper phonemises per sentence internally; chunk size changes its call count, not its output",
            },
        )
        _attach_rows(manifest, "piper", all_rows)
        _save_manifest(out, manifest)

    for name, summary in manifest["engines"].items():
        print(f"  {name}: {json.dumps(summary)}", flush=True)
    print(f"wrote {out}/manifest.json")
    return 0


# Kokoro's style table is 510 rows and the row index IS the token count, so a
# chunk of 510+ tokens indexes off the end of it. Asserted after rendering, from
# the measured token counts, because that is the failure the plan predicts.
KOKORO_STYLE_ROWS = 510


def _save_manifest(out: Path, manifest: dict) -> None:
    """Persist after every stage. A run that dies in the second engine must not
    take the first engine's measurements with it -- they are the expensive half."""
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _attach_rows(manifest: dict, engine: str, rows: list[dict], engine_style_ceiling: int = KOKORO_STYLE_ROWS) -> None:
    """Hang each chunk's measured row under the chapter it belongs to.

    Chunk indices restart per chapter, so they are matched on the chapter's own
    first/last row rather than assumed to be global.
    """
    by_chapter: dict[int, list[dict]] = {}
    for row in rows:
        by_chapter.setdefault(row["chapter_index"], []).append(row)
    for entry in manifest["chapters"]:
        mine = by_chapter.get(entry["index"], [])
        entry[engine] = mine
        if engine == "kokoro":
            # The whole point of the varied chunking, measured rather than claimed:
            # how many of the 510 style rows this chapter actually reached, and
            # the token range it spanned. A uniform chunker scores 1 here.
            toks = [r["tokens"] for r in mine if r["tokens"]]
            entry["kokoro_style_rows_used"] = len({r["style_row"] for r in mine})
            entry["kokoro_token_range"] = [min(toks, default=None), max(toks, default=None)]
            if toks and max(toks) >= engine_style_ceiling:
                raise SystemExit(
                    f"chapter {entry['index']} produced a {max(toks)}-token chunk; kokoro's style table has "
                    f"{engine_style_ceiling} rows, so the row index would be out of range. Lower "
                    f"CHUNK_TARGET_HI rather than shipping a truncated render."
                )


def _fetch_piper_voice(voice: str, voices_dir: Path) -> Path:
    from piper.download_voices import download_voice

    voices_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading piper voice {voice} into {voices_dir} ...", flush=True)
    download_voice(voice, voices_dir)
    path = voices_dir / f"{voice}.onnx"
    if not path.is_file():
        raise SystemExit(f"download_voice({voice!r}, {voices_dir}) did not produce {path}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
