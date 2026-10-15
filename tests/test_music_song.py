"""Songs built section by section: planning, chorus reuse, continuation, seams."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import music_song as ms


def test_the_default_song_has_the_familiar_shape():
    plan = ms.plan_song("folk", 150)
    assert [s.name for s in plan.sections] == ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]


def test_every_chorus_after_the_first_reuses_it():
    plan = ms.plan_song("folk", 150)
    first = next(i for i, s in enumerate(plan.sections) if s.name == "chorus")
    repeats = [s for s in plan.sections if s.name == "chorus"][1:]
    assert repeats and all(s.repeat_of == first for s in repeats)
    assert plan.to_generate == 6  # 8 sections, two of them repeated choruses


def test_section_lengths_add_up_to_the_request_and_fit_one_call():
    plan = ms.plan_song("folk", 180)
    assert abs(sum(s.seconds for s in plan.sections) - 180) < 8
    assert all(ms.MIN_SECTION_S <= s.seconds <= ms.MAX_CALL_S - ms.CONTEXT_S for s in plan.sections)


def test_each_section_is_asked_for_by_name():
    plan = ms.plan_song("calm piano", 60, "short")
    assert plan.sections[0].prompt == f"calm piano, {ms.SECTION_HINTS['intro']}"


@pytest.mark.parametrize("structure", ["nonsense", ["verse", "rap"], [1, 2], []])
def test_unknown_structures_are_refused(structure):
    with pytest.raises(ValueError):
        ms.plan_song("x", 60, structure)


def test_a_custom_list_of_sections_is_used_as_given():
    plan = ms.plan_song("x", 60, ["Verse", "chorus", "solo", "chorus"])
    assert [s.name for s in plan.sections] == ["verse", "chorus", "solo", "chorus"]
    assert plan.sections[3].repeat_of == 1


def test_render_continues_each_new_section_from_the_last_and_repeats_choruses():
    sr = 100
    plan = ms.plan_song("x", 40, ["verse", "chorus", "verse", "chorus"])
    calls = []

    def generate(prompt, seconds, context):
        calls.append((prompt.split("x, ", 1)[1], None if context is None else len(context)))
        return np.full(int(seconds * sr), float(len(calls)), dtype=np.float32)

    progress = []
    out = ms.render_song(plan, sr, generate, lambda d, t, s: progress.append((d, t, s)))
    assert [c[0] for c in calls] == [ms.SECTION_HINTS["verse"], ms.SECTION_HINTS["chorus"], ms.SECTION_HINTS["verse"]]
    assert calls[0][1] is None and calls[1][1] == int(ms.CONTEXT_S * sr)
    assert progress[0] == (0, 3, "verse") and progress[-1] == (3, 3, "mixing")
    # The repeated chorus is the first chorus's audio (value 2.0), not a new take.
    assert np.isclose(out[-1], 2.0)


def test_crossfade_keeps_length_minus_the_overlaps_and_has_no_jump():
    sr = 1000
    a, b = np.ones(2000, dtype=np.float32), -np.ones(2000, dtype=np.float32)
    out = ms.crossfade_concat([a, b], sr, fade_s=0.5)
    assert len(out) == 3500
    assert np.max(np.abs(np.diff(out))) < 0.01
