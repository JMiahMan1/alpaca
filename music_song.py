"""Whole songs from MusicGen, one section at a time.

MusicGen writes at most ~30 s per call (1500 tokens at 50 Hz), and a long
single call drifts: it has no idea a chorus should come back. So a song is
planned as sections (intro, verse, chorus, ...) and generated in order, each
section *continuing* the last few seconds of the one before it (MusicGen's
audio-prompt mode) so key, tempo and instruments carry over. A chorus is
written once and repeated, which is what makes it a chorus; verses are new
each time. Sections are joined with a short equal-power crossfade.

Everything here is pure (numpy only) so it is tested without a GPU; the
generation itself is passed in as a callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

#: What each section is asked to sound like, appended to the user's prompt.
SECTION_HINTS = {
    "intro": "soft intro, sparse arrangement, building up",
    "verse": "verse, steady groove, laid back, room for a melody",
    "pre-chorus": "pre-chorus, rising tension, building energy",
    "chorus": "chorus, full arrangement, energetic, catchy memorable hook",
    "bridge": "bridge, contrasting section, new chords, different texture",
    "solo": "instrumental solo, lead melody in front",
    "outro": "outro, winding down, gentle ending, fading out",
}

#: Relative weight of each section's length; scaled to the requested total.
SECTION_WEIGHT = {
    "intro": 0.6,
    "verse": 1.4,
    "pre-chorus": 0.7,
    "chorus": 1.2,
    "bridge": 1.0,
    "solo": 1.0,
    "outro": 0.6,
}

STRUCTURES = {
    "song": ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"],
    "short": ["intro", "verse", "chorus", "outro"],
    "ballad": ["intro", "verse", "verse", "chorus", "verse", "chorus", "outro"],
    "pop": ["intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "bridge", "chorus", "outro"],
    "loop": ["verse"],
}

#: MusicGen's ceiling per call is 30 s including the continuation prompt.
MAX_CALL_S = 30.0
MIN_SECTION_S = 4.0
#: How much of the previous section is fed back so the next one continues it.
CONTEXT_S = 4.0
CROSSFADE_S = 0.6


@dataclass
class Section:
    name: str
    seconds: float
    #: Index of an earlier section whose audio is reused (a repeated chorus).
    repeat_of: int | None = None
    prompt: str = ""


@dataclass
class SongPlan:
    sections: list[Section] = field(default_factory=list)

    @property
    def to_generate(self) -> int:
        return sum(1 for s in self.sections if s.repeat_of is None)


def resolve_structure(structure: object) -> list[str]:
    """A preset name or an explicit list of section names, validated."""
    if structure is None or structure == "":
        return list(STRUCTURES["song"])
    if isinstance(structure, str):
        if structure not in STRUCTURES:
            raise ValueError(f"unknown structure '{structure}' (choose from {', '.join(STRUCTURES)})")
        return list(STRUCTURES[structure])
    if isinstance(structure, list) and structure and all(isinstance(s, str) for s in structure):
        names = [s.strip().lower() for s in structure]
        unknown = sorted({n for n in names if n not in SECTION_HINTS})
        if unknown:
            raise ValueError(f"unknown section(s) {', '.join(unknown)} (choose from {', '.join(SECTION_HINTS)})")
        if len(names) > 16:
            raise ValueError("a song can have at most 16 sections")
        return names
    raise ValueError("structure is a preset name or a list of section names")


def plan_song(prompt: str, total_s: float, structure: object = None) -> SongPlan:
    """Lay out sections so their lengths add up to ``total_s``.

    Every repeat of a chorus reuses the first one, so it gets the first
    chorus's length; the other sections share the remaining time by weight.
    """
    names = resolve_structure(structure)
    if total_s < MIN_SECTION_S:
        raise ValueError(f"a song is at least {MIN_SECTION_S:g} seconds")
    first_of: dict[str, int] = {}
    weights = [SECTION_WEIGHT[n] for n in names]
    # Repeated choruses take no new time to generate but still play, so the
    # weights cover every section and the planned total is what is heard.
    scale = total_s / sum(weights)
    sections: list[Section] = []
    for i, name in enumerate(names):
        if name == "chorus" and name in first_of:
            src = sections[first_of[name]]
            sections.append(Section(name, src.seconds, repeat_of=first_of[name], prompt=src.prompt))
            continue
        seconds = min(max(weights[i] * scale, MIN_SECTION_S), MAX_CALL_S - CONTEXT_S)
        first_of.setdefault(name, i)
        sections.append(Section(name, round(seconds, 2), prompt=f"{prompt.strip()}, {SECTION_HINTS[name]}"))
    return SongPlan(sections)


def crossfade_concat(parts: list, sample_rate: int, fade_s: float = CROSSFADE_S):
    """Join mono float arrays with an equal-power crossfade at each seam."""
    import numpy as np

    parts = [np.asarray(p, dtype=np.float32) for p in parts if p is not None and len(p)]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    out = parts[0]
    n = int(fade_s * sample_rate)
    for part in parts[1:]:
        k = min(n, len(out), len(part))
        if k <= 0:
            out = np.concatenate([out, part])
            continue
        t = np.linspace(0.0, np.pi / 2, k, dtype=np.float32)
        seam = out[-k:] * np.cos(t) + part[:k] * np.sin(t)
        out = np.concatenate([out[:-k], seam, part[k:]])
    return out


#: generate(prompt, seconds, context_audio_or_None) -> new audio only (no context)
Generate = Callable[[str, float, object], object]


def render_song(
    plan: SongPlan,
    sample_rate: int,
    generate: Generate,
    on_progress: Callable[[int, int, str], None] | None = None,
):
    """Generate each new section in order and assemble the song."""
    import numpy as np

    audio: list = []
    done = 0
    previous = None
    for section in plan.sections:
        if section.repeat_of is not None:
            part = audio[section.repeat_of]
        else:
            if on_progress:
                on_progress(done, plan.to_generate, section.name)
            context = None
            if previous is not None and len(previous):
                context = previous[-int(CONTEXT_S * sample_rate):]
            part = np.asarray(generate(section.prompt, section.seconds, context), dtype=np.float32)
            done += 1
        audio.append(part)
        previous = part
    if on_progress:
        on_progress(done, plan.to_generate, "mixing")
    return crossfade_concat(audio, sample_rate)
