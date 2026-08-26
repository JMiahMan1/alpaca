#!/usr/bin/env python3
"""Generate per-test `reasoning_estimate` fields in benchmark_tests.json.

For every benchmark test we calculate how many tokens a competent model needs
for its thinking/reasoning phase on THAT test, derived from timeless task-shape
rules (UI games reason about full program architecture, code categories plan
runnable programs, exam-style categories deliberate over answers, simple
functional tests barely reason at all). The suite then DOUBLES each estimate
when allocating output headroom (`_test_num_predict`: base + 2 x estimate), so
thinking plus the complete answer always fit without truncation - the failure
mode that used to strand reasoning models on large UI tests (all tokens burned
by the reasoning field, empty content, endless retries).

An explicit per-test "reasoning_estimate" value in benchmark_tests.json always
wins over the tier rules, giving manual control without code edits.

Usage: python scripts/gen_reasoning_estimates.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib

TESTS_PATH = pathlib.Path(__file__).resolve().parent.parent / "benchmark_tests.json"

# Task-shape tiers -> estimated reasoning tokens (before doubling).
TIER_UI_GAME = 4096  # Full playable games: entity systems, collision, loops, rendering.
TIER_CODE = 3072  # Runnable-program categories: architecture + edge cases.
TIER_DELIBERATE = 2048  # Exam-style / multi-step planning questions.
TIER_STANDARD = 1024  # Short factual / formatting / small-snippet answers.

# Category sets mirroring LLMModelBenchmark.CODE_CATEGORIES shape, kept here so
# the generated data file is self-describing (suite falls back to same rules).
UI_GAME_CATEGORIES = {"gamedev", "gamedev_alt", "retrogames", "youtuber"}
CODE_CATEGORIES = {
    "coding",
    "appdev",
    "webdev",
    "linux_driver",
    "iac",
    "android",
    "typescript",
    "rpm",
    "usb",
    "networking",
    "bash",
    "basic",
    "pascal",
}
DELIBERATE_CATEGORIES = {
    "gpqa_diamond",
    "hle",
    "math_hard",
    "mmlu_pro",
    "logic",
    "reasoning",
    "metacog",
    "agentic",
    "code_review",
}


def estimate_for(test: dict) -> int:
    """Timeless per-test reasoning token estimate."""
    explicit = int(test.get("reasoning_estimate") or 0)
    if explicit:
        return explicit
    category = str(test.get("category") or "")
    test_type = str(test.get("type") or "")
    if test_type == "ui" or category in UI_GAME_CATEGORIES:
        tier = TIER_UI_GAME
    elif category in CODE_CATEGORIES:
        tier = TIER_CODE
    elif category in DELIBERATE_CATEGORIES:
        tier = TIER_DELIBERATE
    else:
        tier = TIER_STANDARD
    # Never below the test's own on/off threshold hint: a test flagged as
    # heavy-reasoning (2048) should estimate at least that much thinking.
    return max(tier, int(test.get("reasoning_budget") or 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print planned changes only")
    args = parser.parse_args()

    data = json.loads(TESTS_PATH.read_text())
    changed = 0
    from collections import Counter

    tally: Counter[str] = Counter()
    for category, tests in data.items():
        for test in tests:
            old = test.get("reasoning_estimate")
            new = estimate_for({**test, "category": category})
            tally[str(new)] += 1
            if old != new:
                changed += 1
                if not args.dry_run:
                    test["reasoning_estimate"] = new
                print(f"  {category}/{test['id']}: {old} -> {new}")

    print(f"\ntests: {sum(len(v) for v in data.values())}, changed: {changed}")
    print("estimate distribution:", dict(sorted(tally.items(), key=lambda kv: -int(kv[0]))))
    if not args.dry_run and changed:
        TESTS_PATH.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {TESTS_PATH}")


if __name__ == "__main__":
    main()
