"""Schema and content checks for the Parable Party extra prompt bank (``arcade/party_prompts.py``)."""

from __future__ import annotations

import re

import pytest

from arcade import party
from arcade.party import PROMPT_BANK, ROUND_META, _matches_truth
from arcade.party_prompts import EXTRA_PROMPTS, NEW_ROUND_META

EXPECTED_COUNTS = {
    "blank_o_matic": 30,
    "extra_extra": 25,
    "sub_the_scroll": 25,
    "truth_or_bluff": 25,
    "survey_says": 25,
    "parable_remix": 20,
    "bible_bios": 20,
    "hymn_remix": 20,
    "church_bulletin": 20,
}

ID_PREFIX = {
    "blank_o_matic": "bom_",
    "extra_extra": "ee_",
    "sub_the_scroll": "sts_",
    "truth_or_bluff": "tob_",
    "survey_says": "ss_",
    "parable_remix": "pr_",
    "bible_bios": "bb_",
    "hymn_remix": "hr_",
    "church_bulletin": "cb_",
}

NEW_KINDS = ("parable_remix", "bible_bios", "hymn_remix", "church_bulletin")
BASE_FIELDS = {"id", "prompt", "decoy", "bot_answers", "scripture_note", "announcer"}
EXTRA_FIELDS = {
    "extra_extra": {"scene", "subprompt"},
    "sub_the_scroll": {"scene", "subprompt", "speaker"},
    "truth_or_bluff": {"truth", "truth_keywords"},
}
SCRIPTURE_RE = re.compile(r"^[1-3]?\s?[A-Z][a-z]+(?: [A-Z][a-z]+)* \d+:\d+")
SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿]")
COMMON_WORDS = {"a", "an", "the", "and", "of", "it", "he", "she", "they", "to", "in", "on", "was", "is", "because"}

ALL_ITEMS = [(kind, item) for kind, items in EXTRA_PROMPTS.items() for item in items]
ITEM_IDS = [item["id"] for _, item in ALL_ITEMS]


def test_counts_per_kind():
    assert {k: len(v) for k, v in EXTRA_PROMPTS.items()} == EXPECTED_COUNTS


def _base_bank(key: str) -> list[str]:
    """``PROMPT_BANK`` values for ``key`` without this module's items (party.py merges them in)."""
    mine = {id(item) for _, item in ALL_ITEMS}
    return [item[key] for items in PROMPT_BANK.values() for item in items if id(item) not in mine and key in item]


def test_ids_unique_and_do_not_collide_with_prompt_bank():
    assert len(ITEM_IDS) == len(set(ITEM_IDS))
    assert not set(ITEM_IDS) & set(_base_bank("id"))
    merged = [item["id"] for items in PROMPT_BANK.values() for item in items]
    assert len(merged) == len(set(merged))


def test_new_round_meta():
    assert set(NEW_ROUND_META) == set(NEW_KINDS)
    required = set(ROUND_META["blank_o_matic"])
    for kind, meta in NEW_ROUND_META.items():
        assert set(meta) == required, kind
        assert meta["multiplier"] == 1
        for key in ("title", "badge", "icon", "instructions"):
            assert isinstance(meta[key], str) and meta[key].strip(), (kind, key)
        assert ROUND_META.get(kind, meta) == meta  # party.py merges NEW_ROUND_META verbatim


def test_every_kind_is_playable():
    for kind in EXTRA_PROMPTS:
        assert kind in ROUND_META or kind in NEW_ROUND_META, kind


@pytest.mark.parametrize(("kind", "item"), ALL_ITEMS, ids=ITEM_IDS)
def test_item_schema(kind, item):
    assert set(item) == BASE_FIELDS | EXTRA_FIELDS.get(kind, set())
    for key, value in item.items():
        if key in ("bot_answers", "truth_keywords"):
            assert all(isinstance(v, str) and v.strip() for v in value), key
        else:
            assert isinstance(value, str) and value.strip() == value and value, key

    assert item["id"].startswith(ID_PREFIX[kind])
    assert SNAKE_RE.match(item["id"])
    assert len(item["prompt"]) <= 200
    assert len(item["decoy"]) <= 110
    assert len(item["announcer"]) <= 180
    assert "_" not in item["announcer"]
    assert not EMOJI_RE.search(item["announcer"])

    bots = item["bot_answers"]
    assert len(bots) == 3
    assert len({b.strip().lower() for b in bots}) == 3
    assert all(len(b) <= 110 for b in bots)
    assert item["decoy"].strip().lower() not in {b.strip().lower() for b in bots}

    assert SCRIPTURE_RE.match(item["scripture_note"]), item["scripture_note"]
    assert " - " in item["scripture_note"]


@pytest.mark.parametrize(("kind", "item"), ALL_ITEMS, ids=ITEM_IDS)
def test_kind_specific_rules(kind, item):
    if kind == "blank_o_matic":
        assert "______" in item["prompt"]
    if kind == "extra_extra":
        assert item["announcer"].startswith("Extra! Extra!")
    if kind in ("extra_extra", "sub_the_scroll"):
        assert SNAKE_RE.match(item["scene"])
        assert item["subprompt"].startswith("SCENE: ")
        assert len(item["subprompt"]) <= 200
    if kind == "sub_the_scroll":
        assert item["speaker"].strip()
    if kind == "survey_says":
        assert item["prompt"].startswith("WE ASKED 100 ")
    if kind == "truth_or_bluff":
        keywords = item["truth_keywords"]
        assert 3 <= len(keywords) <= 5
        for kw in keywords:
            assert kw.lower() not in COMMON_WORDS, kw
        assert item["truth"].strip()


@pytest.mark.parametrize(
    "item", EXTRA_PROMPTS["truth_or_bluff"], ids=[i["id"] for i in EXTRA_PROMPTS["truth_or_bluff"]]
)
def test_truth_or_bluff_house_answers_are_not_flagged_as_the_truth(item):
    """The decoy and bot bluffs must not trip the engine's give-away-the-truth check."""
    tagged = {**item, "kind": "truth_or_bluff"}
    assert _matches_truth(item["truth"], tagged)
    for bluff in [item["decoy"], *item["bot_answers"]]:
        assert not _matches_truth(bluff, tagged), bluff


def test_scenes_unique_across_banks():
    scenes = [i["scene"] for k in ("extra_extra", "sub_the_scroll") for i in EXTRA_PROMPTS[k]]
    assert len(scenes) == len(set(scenes))
    assert not set(scenes) & set(_base_bank("scene"))


def test_prompts_are_new():
    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    existing = {norm(p) for p in _base_bank("prompt")}
    mine = [norm(item["prompt"]) for _, item in ALL_ITEMS]
    assert not set(mine) & existing
    assert len(mine) == len(set(mine))


# --- the story behind each prompt, and the verse behind each race answer -------------

def test_every_writing_prompt_has_its_story_with_a_chapter():
    from arcade import party_scripture as s
    from arcade.party_story_context import STORY_CONTEXT
    ids = {it["id"] for items in party.PROMPT_BANK.values() for it in items}
    assert set(STORY_CONTEXT) == ids  # none missing, none stale
    for pid, text in STORY_CONTEXT.items():
        assert 80 <= len(text) <= 240, pid
        assert "chapter" in text or "Psalm" in text, pid
        assert '"' not in text and not re.search(r"\d+:\d+", text), pid  # spoken: no quotes, no verse numbers
    for item in party.PROMPT_BANK["truth_or_bluff"]:  # the story never gives the truth away
        assert not party._matches_truth(s.story_context(item), item), item["id"]


def test_the_story_is_told_before_the_prompt_and_the_paper_boy_still_goes_first():
    from arcade import party_scripture as s
    for kind in ("blank_o_matic", "extra_extra", "truth_or_bluff"):
        item = party.PROMPT_BANK[kind][0]
        room = {"phase": "writing", "match_id": "m", "round": 2, "lines_told": {},
                "current_prompt": {**item, "kind": kind, "context": s.story_context(item)}}
        line = party.announce_line(room)
        story = s.story_context(item)
        assert story in line and line.index(story) < line.index(item["announcer"][-15:])  # story, then the prompt
        if kind == "extra_extra":
            assert line.startswith("Extra! Extra!")


def test_a_prompt_without_a_story_still_names_its_chapter():
    from arcade import party_scripture as s
    assert s.story_context({"id": "ai-1", "scripture_note": "1 Sam 17:40 - five smooth stones"}) == \
        "This one comes from First Samuel chapter 17."
    assert s.story_context({"id": "ai-2"}) == ""


def test_references_are_said_the_way_people_read_them():
    from arcade import party_scripture as s
    assert s.spoken_reference("1 Sam 17:40") == "First Samuel chapter 17"
    assert s.spoken_reference("Psalms 23:1-4") == "Psalm 23"
    assert s.spoken_reference("2 Cor 5:17") == "Second Corinthians chapter 5"


def test_the_verse_is_on_screen_but_never_gives_a_truth_away():
    for kind in ("blank_o_matic", "truth_or_bluff"):
        party._rooms.clear()
        party.create_or_join_room(code="VRS", player_id="v0", name="V0")
        room = party._rooms["VRS"]
        room["schedule"] = [kind] * 7
        with party._lock:
            party._start_round_unlocked(room, 2)  # round 1 draws a fresh schedule
        note = party.serialize_room(room, "v0")["prompt"]["scripture_note"]
        assert bool(note) == (kind != "truth_or_bluff")
    party._rooms.clear()


def test_race_answers_carry_their_verse_for_the_screen():
    from arcade.party_race_content import LIGHTNING, QUICK_DRAW
    assert all(it["note"] for it in (*QUICK_DRAW, *LIGHTNING))
