"""Unit tests for the data-integrity tools in `scripts/`.

`scripts/rebalance_choice_keys.py` and `scripts/gen_reasoning_estimates.py` both
mutate `benchmark_tests.json` in place, and both claim in their docstrings that
running them twice leaves the file unchanged. That idempotence is the whole
safety story: without it nobody can run either script without risking a diff
that silently invalidates stored benchmark results (rewriting a prompt changes
its `_compute_test_hash`, so `outdated_only` re-runs that test for every model).

Nothing about either guarantee was verified anywhere, so it is verified here.
Both modules are import-by-path (`scripts/` is not a package on sys.path).
"""

from __future__ import annotations

import collections
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS_JSON = REPO / "benchmark_tests.json"


def load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rebalance():
    return load_script("rebalance_choice_keys.py")


@pytest.fixture(scope="module")
def estimates():
    return load_script("gen_reasoning_estimates.py")


@pytest.fixture(scope="module")
def shipped_data() -> dict:
    """A fresh read of benchmark_tests.json, never mutated across tests."""
    return json.loads(TESTS_JSON.read_text(encoding="utf-8"))


def keyed_map(data: dict, rebalance) -> dict[str, str]:
    """test id -> the option text the current `expected` letter points at."""
    out: dict[str, str] = {}
    for tests in data.values():
        for test in tests:
            key = str(test.get("expected") or "").strip()
            if not rebalance.KEY_RE.fullmatch(key):
                continue
            parts = rebalance.split_prompt(test.get("prompt", ""))
            if parts is None:
                continue
            _, options = parts
            out[test["id"]] = options[rebalance.LETTERS.index(key)]
    return out


def skewed_data(rebalance) -> dict:
    """A synthetic suite whose keys are deliberately piled onto B and C.

    This is the pre-`rebalance_choice_keys` state the script was written to fix:
    28 of 36 keyed tests answered B or C, so a model that always replied "B"
    scored 39%. The shipped file is already balanced, so the interesting
    behaviour has to be exercised on a fresh skewed copy.
    """
    letters = rebalance.LETTERS
    rng = random.Random(0)
    data: dict[str, list[dict]] = {}
    for n in range(30):
        count = 4 if n % 3 else 10
        options = [f"option {n}.{i}" for i in range(count)]
        rng.shuffle(options)
        correct = options[0]
        target = min(range(count), key=lambda i: (1 if i < 3 else 0, i))
        rest = [opt for i, opt in enumerate(options) if opt != correct]
        reordered = [*rest[:target], correct, *rest[target:]]
        prompt = f"Question {n}.\n\n" + "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(reordered))
        data.setdefault(f"cat{n % 3}", []).append(
            {"id": f"q{n}", "prompt": prompt, "expected": letters[target]}
        )
    return data


# --- scripts/rebalance_choice_keys.py --------------------------------------


def test_split_prompt_reads_a_contiguous_option_block(rebalance):
    prompt = "Pick one.\n\nA) alpha\nB) beta\nC) gamma"
    lead, options = rebalance.split_prompt(prompt)
    assert options == ["alpha", "beta", "gamma"]
    assert lead == ["Pick one.", ""]


def test_split_prompt_ignores_a_block_that_is_not_labelled_in_order(rebalance):
    # A) ... C) ... D) ... with B missing: the key would be meaningless.
    assert rebalance.split_prompt("Pick one.\nA) alpha\nC) gamma\nD) delta") is None
    # A single option is not a multiple-choice question.
    assert rebalance.split_prompt("Pick one.\nA) alpha") is None
    # No options at all.
    assert rebalance.split_prompt("Answer in prose.") is None


def test_split_prompt_ignores_option_like_lines_that_are_not_at_the_end(rebalance):
    # "see A) the rubric" is prose, not an option block, so the trailing
    # option lines do not form the answer set.
    assert rebalance.split_prompt("Follow the A) rule then choose:\nA) alpha\nB) beta") is not None
    assert rebalance.split_prompt("Choose:\nA) alpha\nB) beta\nUse A) or B).") is None


def test_shipped_file_is_already_rebalanced(rebalance, shipped_data):
    """The committed benchmark_tests.json must be a fixed point.

    If it is not, `python scripts/rebalance_choice_keys.py --apply` would be
    writing an uncommitted diff, and every rewritten prompt changes that test's
    `_compute_test_hash` - which marks its stored results outdated and re-runs
    the test against every model.
    """
    assert rebalance.rebalance(shipped_data) == []


def test_rebalance_flattens_a_skewed_key_distribution(rebalance):
    data = skewed_data(rebalance)

    def worst(d) -> float:
        counts = collections.Counter(
            str(test["expected"]).strip()
            for tests in d.values()
            for test in tests
            if rebalance.KEY_RE.fullmatch(str(test.get("expected") or "").strip())
        )
        return max(counts.values()) / sum(counts.values())

    assert worst(data) > 0.25, "the synthetic fixture is not actually skewed"
    rebalance.rebalance(data)
    assert worst(data) <= 0.25


def test_rebalance_is_idempotent(rebalance):
    data = skewed_data(rebalance)
    first = rebalance.rebalance(data)
    after_first = json.dumps(data, sort_keys=True)
    second = rebalance.rebalance(data)
    after_second = json.dumps(data, sort_keys=True)

    assert first, "the skewed fixture produced no changes - the fixture is stale"
    # The second pass must find nothing left to change...
    assert second == []
    # ...and must not have touched the data at all.
    assert after_first == after_second


def test_rebalance_preserves_the_correct_answer_under_the_new_key(rebalance):
    data = skewed_data(rebalance)
    before = keyed_map(data, rebalance)
    changes = rebalance.rebalance(data)
    after = keyed_map(data, rebalance)

    assert changes
    assert set(before) == set(after)
    for test_id, text in before.items():
        assert after[test_id] == text, f"{test_id}: rebalancing moved the wrong answer"


def test_rebalance_keeps_the_option_set_intact(rebalance):
    data = skewed_data(rebalance)
    option_sets_before = {
        test["id"]: sorted(rebalance.split_prompt(test["prompt"])[1])
        for tests in data.values()
        for test in tests
    }
    rebalance.rebalance(data)
    letters = rebalance.LETTERS
    for tests in data.values():
        for test in tests:
            lead, options = rebalance.split_prompt(test["prompt"])
            assert sorted(options) == option_sets_before[test["id"]]
            # The block must still be relabelled A), B), ... in order, which is
            # exactly what split_prompt's second check enforces.
            block = "\n".join(lead[-0:] + [f"{letters[i]}) {opt}" for i, opt in enumerate(options)])
            assert test["prompt"].endswith(block)
            assert rebalance.split_prompt(test["prompt"])[1] == options


def test_rebalance_skips_unkeyed_and_optionless_tests(rebalance):
    data = {
        "cat": [
            {"id": "prose", "prompt": "Write a poem. No options here.", "expected": "yes"},
            {"id": "unkeyed_mc", "prompt": "Pick.\nA) a\nB) b", "expected": "banana"},
        ]
    }
    snapshot = json.dumps(data, sort_keys=True)
    assert rebalance.rebalance(data) == []
    assert json.dumps(data, sort_keys=True) == snapshot


# --- scripts/gen_reasoning_estimates.py ------------------------------------


def test_estimate_tiers_follow_task_shape(estimates):
    def est(test):
        return estimates.estimate_for(test)

    assert est({"type": "ui"}) == estimates.TIER_UI_GAME
    assert est({"category": "gamedev_alt", "type": "code"}) == estimates.TIER_UI_GAME
    assert est({"category": "coding"}) == estimates.TIER_CODE
    assert est({"category": "math_hard"}) == estimates.TIER_DELIBERATE
    assert est({"category": "creative"}) == estimates.TIER_STANDARD
    # An explicit value always wins - that is the manual override.
    assert est({"category": "coding", "reasoning_estimate": 999}) == 999
    # Never below the test's own thinking threshold, or a heavy-reasoning test
    # would be given a budget smaller than the flag that turns thinking on.
    assert est({"category": "creative", "reasoning_budget": 2048}) == 2048


def test_every_shipped_test_has_a_reasoning_estimate(estimates, shipped_data):
    missing = [
        f"{category}/{test['id']}"
        for category, tests in shipped_data.items()
        for test in tests
        if not test.get("reasoning_estimate")
    ]
    assert not missing, f"tests without a reasoning_estimate: {missing}"


def test_shipped_estimates_match_the_rules(estimates, shipped_data):
    """Every generated value must equal what the tier rules compute.

    This is the real idempotence claim: re-running the generator must be a
    no-op, which is only true if no test carries a stale hand-edited value.
    """
    drift = []
    for category, tests in shipped_data.items():
        for test in tests:
            expected = estimates.estimate_for({**test, "category": category})
            if int(test.get("reasoning_estimate") or 0) != expected:
                drift.append(f"{category}/{test['id']}: {test.get('reasoning_estimate')} != {expected}")
    assert not drift, "stale reasoning_estimate values: " + "; ".join(drift)


# --- the generator as a subprocess (exercises main() + the write path) -----


def test_gen_reasoning_estimates_dry_run_writes_nothing():
    before = TESTS_JSON.read_bytes()
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_reasoning_estimates.py"), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert TESTS_JSON.read_bytes() == before
    assert "changed: 0" in proc.stdout


def test_rebalance_dry_run_writes_nothing():
    before = TESTS_JSON.read_bytes()
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "rebalance_choice_keys.py")],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert TESTS_JSON.read_bytes() == before
    assert "dry run" in proc.stdout


# --- benchmark_tests.json schema -------------------------------------------


# --- importability of the hyphenated / top-level entry points --------------
#
# Several entry-point modules have hyphens in their names, so no test can
# `import` them: only a by-path load works. A syntax error or a bad import in
# one of them therefore goes unnoticed until the container tries to start it,
# and `pytest` itself never touches most of them.

IMPORTABLE_MODULES = [
    "analyzer.py",
    "context_awareness.py",
    "imageops.py",
    "sandbox_exec.py",
    "tts_text.py",
    "voice_clone.py",
    "multistep_benchmark.py",
    "settings_scan.py",
    "telemetry_monitor.py",
    "llm_benchmark_suite.py",
    "online_providers.py",
    "alpaca-proxy.py",
    "alpaca-puller.py",
    # Heavy ML imports (torch/kokoro/openvoice) are resolved lazily inside
    # audio_server.py, so the module imports on a machine with no GPU stack -
    # which is what lets it be imported at all outside its container.
    "audio_server.py",
]

# scripts/ entry points. These import nothing heavy at module scope -- the
# engines are imported inside the render functions -- so they import on a plain
# machine, which is the same property that lets them be linted and tested away
# from the GPU.
SCRIPT_MODULES = [
    "scripts/render_epub_ab.py",
]


@pytest.mark.parametrize("filename", IMPORTABLE_MODULES)
def test_entry_point_module_imports(filename):
    import importlib.util
    import sys

    path = REPO / filename
    assert path.exists(), f"{filename} is referenced but missing"
    name = "alpaca_entry_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("relpath", SCRIPT_MODULES)
def test_script_module_imports(relpath):
    import importlib.util
    import sys

    path = REPO / relpath
    assert path.exists(), f"{relpath} is referenced but missing"
    name = "alpaca_script_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)


def test_benchmark_tests_json_shape(shipped_data):
    assert isinstance(shipped_data, dict)
    assert len(shipped_data) > 20, "category count collapsed"
    ids = [test["id"] for tests in shipped_data.values() for test in tests]
    duplicates = [test_id for test_id, count in collections.Counter(ids).items() if count > 1]
    assert not duplicates, f"duplicate test ids: {duplicates}"
    for category, tests in shipped_data.items():
        assert isinstance(tests, list) and tests, f"category {category} is empty"
        for test in tests:
            assert "id" in test and "prompt" in test, f"{category}/{test.get('id')}"
            assert test["id"], f"{category} has a test with an empty id"
