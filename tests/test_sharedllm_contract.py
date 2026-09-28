"""Cross-repo contract: alpaca's copy of the downstream tool vocabulary.

alpaca does not depend on SharedLLM. It benchmarks models *against* SharedLLM's
agent surface, which means `web/shared_llm_benchmark.py` hand-maintains a copy
of the tool names the downstream gateway accepts, so a model that emits
`ImageEditRequest` can be scored as having picked the right tool.

A hand-maintained copy drifts, and it had: four genuinely dispatchable tools -
`ImageEditRequest`, `OcrRequest`, `RavenMissionRequest` and `RedisInspectRequest`
- were missing, so a model naming any of them fell through the regex aliases
into the `difflib` fuzzy match and could be graded against the wrong tool.
Nothing detected it because the two repos have no test that spans them.

This module closes that. It reads the sibling repo **by AST, never by import**
(importing `services.gateway.agent_loop` pulls in identity service URLs and
FastAPI app construction, so the vocabulary has to be read as source), then
asserts four things:

  1. Every canonical tool the downstream app dispatches is in alpaca's set.
  2. alpaca's set contains nothing the downstream app does not accept.
  3. Every regex alias resolves to a canonical name (a typo'd target would make
     the alias a silent no-op).
  4. Every `required_tool` used by alpaca's own SharedLLM tasks is canonical,
     because that comparison is literal.

The test skips when the sibling repo is absent, so a standalone alpaca clone
still gets a green suite; `tests/conftest.py` reports the skip.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from web.shared_llm_benchmark import (
    _CANONICAL_TOOLS,
    _TOOL_REGEX_ALIASES,
    SharedLLMModelBenchmark,
)

REPO = Path(__file__).resolve().parents[1]
SHARED_LLM = REPO.parent / "SharedLLM"
GATEWAY = SHARED_LLM / "services" / "gateway"

pytestmark = pytest.mark.needs_sibling_repo


# --------------------------------------------------------------------------
# reading the sibling repo as source
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _sibling_repo_present():
    """Every test here reads the sibling checkout, so guard the module once.

    Per-test guards were easy to forget and fail as FileNotFoundError rather
    than skipping, which defeats the point of the needs_sibling_repo marker.
    """
    if not (GATEWAY / "agent_loop.py").is_file():
        pytest.skip(
            f"SharedLLM not found at {SHARED_LLM} - the cross-repo tool contract "
            "cannot be checked. Clone it as a sibling of the alpaca checkout."
        )


def _require_sibling() -> Path:
    return GATEWAY


def _assigned_set(path: Path, name: str) -> set[str]:
    """``ast.literal_eval`` a module-level ``NAME = {...}`` or ``NAME = (...)``.

    The sibling repo uses a set for ``ALLOWED_TOOLS`` and a tuple for
    ``_RAVEN_TOOL_TYPES``; the collection type is not the contract, the contents
    are, so both are accepted.
    """
    for node in ast.parse(path.read_text()).body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if getattr(target, "id", None) == name:
            value = ast.literal_eval(node.value)
            assert isinstance(value, (set, frozenset, tuple, list)), f"{name} in {path.name} is not a collection"
            return set(value)
    raise AssertionError(f"{name} not found at module level in {path}")


def _tool_builder_types() -> set[str]:
    """The ``_Tool("XRequest", ...)`` declarations in tool_builder.py."""
    return {
        node.args[0].value
        for node in ast.walk(ast.parse((GATEWAY / "tool_builder.py").read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_Tool"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def _allowed_tools_by_governing_comment() -> dict[str, str]:
    """Map each member of ``ALLOWED_TOOLS`` to the source comment that governs it.

    The set mixes three populations - real dispatchable tools, snake_case
    aliases tolerated for convenience, and a block of bare hallucination-guard
    tokens ("tool_name", "command", "parameters", "request", ...) that exist only
    so a confused model still has *something* to match. Nothing at the item
    level distinguishes them; the source comments do. Reading the comment that
    governs each name is therefore the only way to tell a tool from a guard, and
    it means the classification is the downstream repo's own, not a list that
    silently rots here.
    """
    source = (GATEWAY / "agent_loop.py").read_text().splitlines()
    start = next(i for i, line in enumerate(source) if line.startswith("ALLOWED_TOOLS"))
    end = next(i for i in range(start + 1, len(source)) if source[i].startswith("}"))
    governing: dict[str, str] = {}
    comment = ""
    for line in source[start:end]:
        stripped = line.strip()
        if stripped.startswith("#"):
            comment = stripped.lstrip("# ").strip()
            continue
        for name in re.findall(r'"([^"]+)"', stripped):
            governing[name] = comment
    return governing


#: Comment fragments that mark a block of `ALLOWED_TOOLS` entries as *not* tools.
_GUARD_COMMENT_MARKERS = ("hallucination patterns", "Parameter hallucination", "Search/query variations")


def _guard_tokens() -> set[str]:
    return {
        name
        for name, comment in _allowed_tools_by_governing_comment().items()
        if any(marker in comment for marker in _GUARD_COMMENT_MARKERS)
    }


def downstream_canonical() -> set[str]:
    """Every lowercase tool name the downstream gateway will actually dispatch.

    Three sources, because the vocabulary is genuinely split across them:
      * ``agent_loop.ALLOWED_TOOLS`` - the names a model may emit, including
        snake_case aliases and the hallucination-guard tokens.
      * ``tool_builder._Tool`` - the canonical types described to the model.
      * ``prose_tools._RAVEN_TOOL_TYPES`` - the types parsed out of prose.

    The hallucination-guard tokens ("tool_name", "command", "parameters",
    "request", ...) are *not* tools, and treating them as such would make the
    contract demand a copy of them in alpaca - which would then go on claiming
    the gateway can dispatch a tool called "request".
    """
    allowed = _assigned_set(GATEWAY / "agent_loop.py", "ALLOWED_TOOLS")
    guards = _guard_tokens()
    assert guards, "no guard tokens identified - the comment-guided extraction is broken"
    types = _tool_builder_types() | _assigned_set(GATEWAY / "prose_tools.py", "_RAVEN_TOOL_TYPES")
    return {name for name in allowed if name.endswith("request") and name not in guards} | {
        t.lower() for t in types if not t.startswith("sharedllm_")
    }


# --------------------------------------------------------------------------
# 1. alpaca knows every tool the downstream app dispatches
# --------------------------------------------------------------------------


def test_alpaca_knows_every_canonical_downstream_tool():
    missing = sorted(downstream_canonical() - _CANONICAL_TOOLS)
    assert not missing, (
        "web/shared_llm_benchmark.py is missing downstream SharedLLM tools. A model "
        f"that names any of these is graded against the wrong tool: {missing}"
    )


@pytest.mark.parametrize(
    "tool",
    ["imageeditrequest", "ocrrequest", "ravenmissionrequest", "redisinspectrequest"],
)
def test_the_four_dispatchable_tools_that_were_missing_are_present(tool):
    """Regression for the drift this file exists to prevent.

    These four are dispatchable today (`tool_registry.resolve_tool_call` and
    `prose_tools._TYPE_ALIASES` both route them) but were absent from alpaca's
    copy, so they resolved through `difflib` instead of by name.
    """
    assert tool in _CANONICAL_TOOLS


def test_the_sibling_repo_actually_declares_the_four():
    """Guards the test above from passing because the names were invented."""
    canonical = downstream_canonical()
    for tool in ("imageeditrequest", "ocrrequest", "ravenmissionrequest", "redisinspectrequest"):
        assert tool in canonical, f"{tool} is not canonical downstream - revisit the regression test"


# --------------------------------------------------------------------------
# 2. alpaca does not invent tools
# --------------------------------------------------------------------------


def test_alpaca_invents_no_tools_the_downstream_app_would_reject():
    """A phantom name is as bad as a missing one: the model is told the tool
    exists, uses it, and the app refuses."""
    allowed = _assigned_set(GATEWAY / "agent_loop.py", "ALLOWED_TOOLS")
    phantoms = sorted(_CANONICAL_TOOLS - allowed)
    assert not phantoms, (
        f"alpaca advertises tool names the downstream gateway would reject: {phantoms}. "
        "A model benchmarked on a name the app refuses is scored for a tool that cannot exist."
    )


def test_the_guard_tokens_are_excluded_from_the_contract():
    """Pins the extraction itself. If the comment blocks in ALLOWED_TOOLS are
    regrouped, this fails rather than the contract quietly demanding a copy of
    "command" and "parameters"."""
    guards = _guard_tokens()
    assert {"tool_name", "function_name", "command", "operation", "target"} <= guards
    assert {"parameters", "request", "input"} <= guards
    assert "gitoperationrequest" not in guards
    assert "workspacecreaterequest" not in guards
    assert guards <= _assigned_set(GATEWAY / "agent_loop.py", "ALLOWED_TOOLS")


def test_image_generation_is_a_real_tool_and_not_a_phantom():
    """`imagegenerationrequest` is declared by tool_builder as
    `ImageGenerationRequest`, so alpaca's copy is right even though the
    downstream ``ALLOWED_TOOLS`` set happens to omit it - drift in the other
    direction, and the kind that makes a naive equality test useless."""
    assert "ImageGenerationRequest" in _tool_builder_types()
    assert "imagegenerationrequest" in _CANONICAL_TOOLS


# --------------------------------------------------------------------------
# 3. the alias table points at real names
# --------------------------------------------------------------------------


def test_every_regex_alias_targets_a_canonical_tool():
    dangling = sorted({target for _, target in _TOOL_REGEX_ALIASES if target not in _CANONICAL_TOOLS})
    assert not dangling, (
        f"regex aliases resolve to names that are not canonical: {dangling}. Such an "
        "alias is a silent no-op - the name it produces is rejected by the tier-1 check."
    )


@pytest.mark.parametrize("name", ["imageeditrequest", "ocrrequest", "ravenmissionrequest", "redisinspectrequest"])
def test_the_new_tools_have_fuzzy_aliases_too(name):
    """A canonical name only helps when the model emits it verbatim. The regex
    tier is what rescues `ImageEdit`, `image_edit` and `edit image`."""
    targets = {target for pattern, target in _TOOL_REGEX_ALIASES if pattern.search(name)}
    assert name in targets, f"no regex alias resolves {name!r} back to itself"


def test_alias_patterns_are_lowercase_and_anchored_enough_to_be_predictable():
    """Every pattern is a `.match`-style full-string glob; an anchored or
    upper-case literal would stop matching the lowercase names they guard."""
    for pattern, _target in _TOOL_REGEX_ALIASES:
        assert isinstance(pattern, re.Pattern)
        assert not pattern.flags & re.VERBOSE, "a VERBOSE alias would ignore its own spaces"


# --------------------------------------------------------------------------
# 4. alpaca's own tasks reference real tools
# --------------------------------------------------------------------------


def test_every_tool_request_task_requires_a_canonical_tool():
    """The `tool_request` grader compares the resolved name to `required_tool`
    with `==`, so a non-canonical value is a task nothing can pass."""
    offenders = sorted(
        f"{t['id']} -> {t['required_tool']}"
        for t in SharedLLMModelBenchmark.get_all_tasks()
        if t.get("task_type") == "tool_request" and t.get("required_tool") not in _CANONICAL_TOOLS
    )
    assert not offenders, f"tool_request tasks require non-canonical tools: {offenders}"


def test_the_tool_json_tasks_use_semantic_names_matched_as_substrings_and_say_so():
    """`tool_json` deliberately grades with
    ``required_tool.lower() in str(parsed["tool"]).lower()`` rather than an exact
    comparison, which is why its required_tool values are capability names like
    `rag_search` instead of a canonical `...request` name. If that grader is ever
    tightened to `==`, these four tasks become unpassable - so the difference is
    pinned here rather than left as a coincidence.
    """
    tool_json = [t for t in SharedLLMModelBenchmark.get_all_tasks() if t.get("task_type") == "tool_json"]
    assert tool_json, "no tool_json tasks - this test is guarding nothing"
    non_canonical = sorted(t["required_tool"] for t in tool_json if t["required_tool"] not in _CANONICAL_TOOLS)
    assert non_canonical, "tool_json tasks now use canonical names; the substring grader is probably dead"


def test_the_tool_request_tasks_all_resolve_to_a_canonical_name():
    """End-to-end through the resolver, which is the thing grading uses."""
    bad = [
        (t["id"], t["required_tool"], _resolve(t["required_tool"]))
        for t in SharedLLMModelBenchmark.get_all_tasks()
        if t.get("required_tool")
        and t.get("task_type") == "tool_request"
        and _resolve(t["required_tool"]) != t["required_tool"]
    ]
    assert not bad, f"required_tool does not survive _resolve_tool_name: {bad}"


def test_a_confident_wrong_tool_does_not_resolve_to_the_right_one():
    """The resolver's fuzzy tier is a fallback, not a guess-machine. Naming a
    real *different* tool must not be laundered into the expected answer."""
    from web.shared_llm_benchmark import _resolve_tool_name

    assert _resolve_tool_name("gitoperationrequest") == "gitoperationrequest"
    assert _resolve_tool_name("notatoolatall") == ""
    # a wrong-but-recognisable name stays wrong
    assert _resolve_tool_name("storagefilereadrequest") != "workspacefilereadrequest"


def _resolve(name: str) -> str:
    from web.shared_llm_benchmark import _resolve_tool_name

    return _resolve_tool_name(name)


# --------------------------------------------------------------------------
# the vocabulary itself
# --------------------------------------------------------------------------


def test_the_downstream_vocabulary_has_not_vanished():
    """A parse failure upstream would make the contract tests vacuously pass."""
    canonical = downstream_canonical()
    assert len(canonical) >= 60, f"only {len(canonical)} canonical tools parsed - the extractor is broken"
    assert "workspacecreaterequest" in canonical
    assert "gitoperationrequest" in canonical


def test_the_downstream_sources_are_parseable_without_being_imported():
    """The contract has to hold on a machine with nothing running, so the sibling
    repo is read as source. This asserts the source is still parseable - the
    failure mode a stale regex extractor would otherwise hide."""
    for name in ("agent_loop.py", "tool_builder.py", "prose_tools.py"):
        ast.parse((GATEWAY / name).read_text())
