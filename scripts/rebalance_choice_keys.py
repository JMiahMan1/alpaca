#!/usr/bin/env python3
"""Spread the answer keys of multiple-choice benchmarks evenly across the options.

The keyed benchmarks drifted badly: of 36 letter-keyed tests, 14 answered C and
14 answered B while D answered once and F-J never. A model that ignores every
question and always replies "B" scores 39% on that set, so the suite was partly
measuring key position instead of knowledge.

This rewrites each multiple-choice prompt so the correct option lands on the
least-used letter its option count allows. The permutation of the remaining
options is seeded from the test id, so re-running produces the same file and
only a real edit to the tests changes it.

Option lines must be the contiguous ``X) text`` block at the end of the prompt -
the shape every keyed test in ``benchmark_tests.json`` uses.

Usage:
  python scripts/rebalance_choice_keys.py            # report the change, write nothing
  python scripts/rebalance_choice_keys.py --apply    # rewrite benchmark_tests.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS_JSON = REPO / "benchmark_tests.json"

LETTERS = "ABCDEFGHIJ"
OPTION_RE = re.compile(r"^([A-J])\)\s+(.+)$")
KEY_RE = re.compile(r"^[A-J]$")


def split_prompt(prompt: str) -> tuple[list[str], list[str]] | None:
    """Split a prompt into (lead lines, option texts), or None if it has no option block."""
    lines = prompt.splitlines()
    options: list[str] = []
    index = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        match = OPTION_RE.match(lines[i].strip())
        if not match:
            break
        options.append(match.group(2).strip())
        index = i
    options.reverse()
    if len(options) < 2:
        return None
    # The block must be labelled A), B), ... in order for the key to mean anything.
    if [LETTERS[i] for i in range(len(options))] != [
        OPTION_RE.match(lines[index + i].strip()).group(1) for i in range(len(options))
    ]:
        return None
    return lines[:index], options


def rebalance(tests_by_category: dict[str, list[dict]]) -> list[dict]:
    """Return one change record per rewritten test, mutating the tests in place."""
    keyed: list[tuple[str, dict, list[str], list[str]]] = []
    for category, tests in tests_by_category.items():
        for test in tests:
            key = str(test.get("expected") or "").strip()
            if not KEY_RE.fullmatch(key):
                continue
            parts = split_prompt(test.get("prompt", ""))
            if parts is None:
                continue
            lead, options = parts
            keyed.append((category, test, lead, options))

    # Give each test the least-used letter its option count allows. Greedy over
    # a stable ordering flattens the whole set rather than each group alone, so
    # the four-option tests do not pile their surplus onto the same letters the
    # ten-option tests already use.
    used: collections.Counter[int] = collections.Counter()
    changes = []
    for category, test, lead, options in sorted(keyed, key=lambda item: (-len(item[3]), item[1]["id"])):
        count = len(options)
        old_key = str(test["expected"]).strip()
        correct = options[LETTERS.index(old_key)]
        target_index = min(range(count), key=lambda i: (used[i], i))
        used[target_index] += 1

        # Sort before shuffling so the arrangement depends only on the option
        # texts, never on the order they currently sit in - running the script
        # twice then leaves the file unchanged.
        others = sorted(opt for i, opt in enumerate(options) if i != LETTERS.index(old_key))
        random.Random(f"{test['id']}:{count}").shuffle(others)
        reordered = [*others[:target_index], correct, *others[target_index:]]

        new_key = LETTERS[target_index]
        new_prompt = "\n".join(lead + [f"{LETTERS[i]}) {opt}" for i, opt in enumerate(reordered)])
        if new_prompt == test["prompt"] and new_key == old_key:
            continue
        test["prompt"] = new_prompt
        test["expected"] = new_key
        changes.append({"category": category, "id": test["id"], "from": old_key, "to": new_key, "options": count})
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write benchmark_tests.json")
    args = parser.parse_args()

    data = json.loads(TESTS_JSON.read_text(encoding="utf-8"))
    before = collections.Counter(
        str(t["expected"]).strip()
        for tests in data.values()
        for t in tests
        if KEY_RE.fullmatch(str(t.get("expected") or "").strip())
    )
    changes = rebalance(data)
    after = collections.Counter(
        str(t["expected"]).strip()
        for tests in data.values()
        for t in tests
        if KEY_RE.fullmatch(str(t.get("expected") or "").strip())
    )

    print(f"rebalanced {len(changes)} of {sum(before.values())} keyed tests")
    print("  before:", dict(sorted(before.items())))
    print("  after: ", dict(sorted(after.items())))
    worst_before = max(before.values()) / sum(before.values())
    worst_after = max(after.values()) / sum(after.values())
    print(f"  always-guess-one-letter score: {worst_before:.0%} -> {worst_after:.0%}")

    if not args.apply:
        print("\n(dry run - pass --apply to write)")
        return
    # Match the file's existing encoding so the diff is only the rebalanced tests.
    TESTS_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {TESTS_JSON}")


if __name__ == "__main__":
    main()
