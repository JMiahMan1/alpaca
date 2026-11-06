"""Schema and content checks for the Parable Party extra prompt bank (``arcade/party_prompts.py``)."""

from __future__ import annotations

import re

import pytest

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
    assert " — " in item["scripture_note"]


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
