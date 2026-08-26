"""Tests for the multi-step agentic harness (multistep_benchmark.py).

Covers Glider 2026 workflow registration, HTML extraction helpers,
egg-manifest parsing, weighted check evaluation, scoring math, and
the web routes (/api/tests/multistep, /api/run/multistep).
"""

import asyncio
import json
from unittest.mock import patch

import pytest

import multistep_benchmark as msb
from multistep_benchmark import MultiStepBenchmark, extract_manifest_state


@pytest.fixture()
def bench():
    return msb.MultiStepBenchmark()


@pytest.fixture()
def glider(bench):
    workflows = {w["id"]: w for w in bench.get_all_workflows()}
    return workflows["glider_2026_house"]


GOOD_DOC = """
<!DOCTYPE html>
<html>
<head><title>Glider 2026</title></head>
<body>
<canvas id="game"></canvas>
<script>
const GLIDER_ROOMS = [{id:'den',name:'The Den',exits:['kitchen']}];
const GLIDER_EASTER_EGGS = [
  {id:'sleeping-cat', room:'den', description:'A sleeping cat purrs.'},
  {id:'fridge-word', room:'kitchen', description:'Magnet letters spell a word.'},
  {id:'dorothy-photo', room:'living_room', description:'A photo by the flue.'},
  {id:'pixel-plane', room:'living_room', description:'The book flips to a plane.'},
  {id:'pull-chain', room:'attic', description:'Light dims and brightens.'},
  {id:'dedication-plaque', room:'attic', description:'For John Calhoun, 1988.'},
  {id:'storm-cloud', room:'basement', description:'Weather shifts upstairs.'},
  {id:'monochrome-mode', room:'backyard', description:'Type DOROTHY for 1988 mode.'}
];
let score = 0;
function newGame() { score = 0; }
localStorage.setItem('glider_scores', JSON.stringify(highScores));
const name = prompt('Enter your initials');
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ #
# Workflow registry                                                   #
# ------------------------------------------------------------------ #


def test_glider_workflow_registered(bench, glider):
    assert glider["category"] == "multistep_gamedev"
    assert len(glider["steps"]) == 4
    assert glider.get("min_easter_eggs", 0) >= 8
    assert glider.get("lang") == "html"


def test_glider_steps_carry_contracts(glider):
    joined = " ".join(s["prompt"] for s in glider["steps"])
    assert "GLIDER_ROOMS" in joined
    assert "GLIDER_EASTER_EGGS" in joined
    assert "```html" in joined
    assert all(isinstance(s.get("max_tokens"), int) and s["max_tokens"] > 0 for s in glider["steps"])
    assert all(len(s.get("checks", [])) > 0 for s in glider["steps"])


def test_get_all_workflows_filter(bench):
    ids = [w["id"] for w in bench.get_all_workflows()]
    assert ids == ["glider_2026_house"]


# ------------------------------------------------------------------ #
# Extraction helpers                                                  #
# ------------------------------------------------------------------ #


def test_extract_html_document_fenced():
    text = "Sure!\n```html\n<!DOCTYPE html>\n<html><body>hi</body></html>\n```\nEnjoy."
    doc = msb.extract_html_document(text)
    assert doc is not None and doc.startswith("<!DOCTYPE html>")


def test_extract_html_document_raw():
    doc = msb.extract_html_document("prose <html><body>x</body></html> more")
    assert doc == "<html><body>x</body></html>"


def test_extract_html_document_truncated_fence():
    doc = msb.extract_html_document("```html\n<html><body><canvas>")
    assert doc is not None and "<html>" in doc


def test_extract_html_document_none_cases():
    assert msb.extract_html_document(None) is None
    assert msb.extract_html_document("") is None
    assert msb.extract_html_document("just prose, no markup") is None


def test_serialize_messages():
    out = msb.serialize_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    assert "[USER]\na" in out
    assert out.endswith("[ASSISTANT]")


def test_strip_thinking_removes_blocks():
    assert msb.strip_thinking("<think>secret</think>hello") == "hello"
    assert msb.strip_thinking(None) == ""


# ------------------------------------------------------------------ #
# Egg manifest parsing                                                #
# ------------------------------------------------------------------ #


def test_extract_egg_ids_unique_and_ordered():
    doc = "const GLIDER_EASTER_EGGS = [{id:'a'}, {id:'b'}, {id:'a'}, {id:\"c\"}];"
    assert msb.extract_egg_ids(doc) == ["a", "b", "c"]


def test_extract_egg_ids_alias_and_missing():
    assert msb.extract_egg_ids("const GLIDER_EGGS = [{id:'x'}];") == ["x"]
    assert msb.extract_egg_ids("no manifest here") == []
    assert msb.extract_egg_ids(None) == []


def test_persistent_scoreboard_signals():
    assert msb.has_persistent_scoreboard_js(GOOD_DOC)
    bad = GOOD_DOC.replace("newGame() { score = 0; }", "").replace("let score = 0;", "let score = startScore;")
    assert not msb.has_persistent_scoreboard_js(bad)
    bad2 = GOOD_DOC.replace("localStorage.setItem", "console.log")
    assert not msb.has_persistent_scoreboard_js(bad2)
    assert not msb.has_persistent_scoreboard_js(None)


# ------------------------------------------------------------------ #
# Check evaluation                                                    #
# ------------------------------------------------------------------ #


def test_evaluate_check_regex_any_of_min_eggs():
    ok, _ = msb.evaluate_check({"type": "regex", "pattern": r"\bvent\b"}, "floor VENT blows")
    assert ok
    miss, _ = msb.evaluate_check({"type": "regex", "pattern": r"\bvent\b"}, "nothing")
    assert not miss
    ok2, detail = msb.evaluate_check({"type": "any_of", "patterns": [r"\bdoor\b", r"\bexits\b"]}, "it has exits")
    assert ok2 and detail["matched"]
    ok3, detail3 = msb.evaluate_check({"type": "min_eggs", "count": 2}, "const GLIDER_EGGS=[{id:'p'},{id:'q'}];")
    assert ok3 and detail3["found"] == 2


def test_evaluate_check_unknown_type():
    ok, detail = msb.evaluate_check({"type": "wat"}, "anything")
    assert not ok and "unknown check type" in detail.get("error", "")


def test_evaluate_checks_weighted_totals():
    checks = [
        {"name": "a", "type": "regex", "pattern": "cat", "weight": 3},
        {"name": "b", "type": "regex", "pattern": "dog"},
    ]
    res = msb.evaluate_checks(checks, "cat and cat")
    assert res["weight_total"] == 4
    assert res["weight_passed"] == 3
    assert not res["all_passed"]
    assert len(res["checks"]) == 2


# ------------------------------------------------------------------ #
# Scoring                                                             #
# ------------------------------------------------------------------ #


def _fake_steps(workflow):
    """Step results that pass every check and delivered a doc each turn."""
    return [{"doc_extracted": True, "validation": {"weight_total": 10, "weight_passed": 10}} for _ in workflow["steps"]]


def test_score_workflow_perfect(bench, glider):
    score, breakdown = bench.score_workflow(
        glider,
        _fake_steps(glider),
        GOOD_DOC,
        ui_render={"ran": True, "screenshot": True},
    )
    assert score == 100.0
    assert breakdown["breakdown"]["step_content_checks"] == 40.0
    assert breakdown["breakdown"]["easter_eggs"] == 15.0
    assert breakdown["breakdown"]["ui_render"] == 15.0
    assert len(breakdown["egg_ids"]) >= glider["min_easter_eggs"]


def test_score_workflow_empty_fail(bench, glider):
    score, _ = bench.score_workflow(
        glider,
        [{"doc_extracted": False, "validation": {"weight_total": 10, "weight_passed": 0}}],
        None,
        ui_render={"ran": False},
    )
    assert score == 0.0


def test_score_workflow_sandbox_unavailable_partial_credit(bench, glider):
    score, breakdown = bench.score_workflow(
        glider,
        _fake_steps(glider),
        GOOD_DOC,
        ui_render={"ran": None, "skipped": "sandbox unavailable"},
    )
    # Everything content-side passes; only the render bonus is reduced.
    assert score == 92.0
    assert breakdown["breakdown"]["ui_render"] == 7.0


# ------------------------------------------------------------------ #
# Verification / execution                                            #
# ------------------------------------------------------------------ #


def test_verify_ui_render_no_doc(bench):
    res = bench.verify_ui_render(None)
    assert res["ran"] is False
    assert res.get("skipped") == "no final document"


def test_verify_ui_render_sandbox_unavailable(bench, monkeypatch):
    monkeypatch.setattr(msb, "grade_code", None)
    res = bench.verify_ui_render("<html><body>x</body></html>")
    assert res["ran"] is None
    assert res.get("skipped") == "sandbox unavailable"


@pytest.mark.asyncio()
async def test_run_workflow_happy_path(bench, glider, tmp_path, monkeypatch):
    turn = {
        "success": True,
        "latency": 0.5,
        "response": f"```html\n{GOOD_DOC}\n```",
        "tokens_generated": 100,
        "error": None,
    }
    calls = []

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        calls.append([m["role"] for m in messages])
        return dict(turn)

    async def huge_ctx(model):
        return 1_000_000

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(
        bench,
        "verify_ui_render",
        lambda doc: {"ran": True, "screenshot": True, "score": 100},
    )
    monkeypatch.setattr(bench, "ARTIFACTS_DIR", tmp_path)
    events: list[tuple[str, dict]] = []
    result = await bench.run_workflow(
        "test-model",
        use_proxy=True,
        workflow=glider,
        progress_callback=lambda e, d: events.append((e, d)),
    )

    assert result["success"]
    # The payload carries the COMPLETE document (not a truncated excerpt)
    # so Evaluation Payload viewers show the full generated code.
    assert result["response"] == GOOD_DOC.strip()
    assert result["response"].rstrip().endswith("</html>")
    # Exact 100 is covered by test_score_workflow_perfect; here the sample doc
    # does not satisfy every per-step engine check, so just require passing.
    assert result["score"] >= 60.0
    assert result["steps_completed"] == 4
    assert result["steps_total"] == 4
    # Conversation grows across turns: user/assistant pairs accumulate.
    assert [len(c) for c in calls] == [1, 3, 5, 7]
    assert {e for e, _ in events} == {"test_step"}
    assert len(events) == 4
    assert result["artifact"] and str(result["artifact"]).endswith(".html")
    assert result["validation"]["persistent_scoreboard"]
    # Full requested budgets survive on a large-context host.
    assert all(sr["num_predict"] == glider["steps"][i]["max_tokens"] for i, sr in enumerate(result["steps"]))
    # Every turn's raw response must be persisted (nothing the model
    # generated may be lost, even when it is not the final artifact).
    turn_files = sorted(tmp_path.glob("*__turn*.txt"))
    assert len(turn_files) == 4
    assert all(sr["response_path"] and str(tmp_path) in sr["response_path"] for sr in result["steps"])
    assert "newGame" in turn_files[0].read_text(encoding="utf-8")


@pytest.mark.asyncio()
async def test_run_workflow_keeps_most_substantial_document(bench, glider, tmp_path, monkeypatch):
    """A later truncated fragment must not replace an earlier full delivery."""
    full = {"success": True, "latency": 0.5, "response": f"```html\n{GOOD_DOC}\n```", "tokens_generated": 100, "error": None}
    fragment = {"success": True, "latency": 0.2, "response": "```html\n<html><body>partial", "tokens_generated": 10, "error": None}

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        # Full doc on turns 1 and 3; fragments on turns 2 and 4.
        return dict(full if len(messages) in (1, 5) else fragment)

    async def huge_ctx(model):
        return 1_000_000

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(
        bench,
        "verify_ui_render",
        lambda doc: {"ran": True, "screenshot": True, "score": 100},
    )
    monkeypatch.setattr(bench, "ARTIFACTS_DIR", tmp_path)
    result = await bench.run_workflow("test-model", use_proxy=True, workflow=glider)

    assert result["validation"]["persistent_scoreboard"]
    artifact = tmp_path / f"{bench._sanitize_model_filename('test-model')}__glider_2026_house.html"
    assert GOOD_DOC.strip()[:60] in artifact.read_text(encoding="utf-8")


@pytest.mark.asyncio()
async def test_run_workflow_retries_when_no_doc(bench, glider, monkeypatch):
    no_doc = {"success": True, "latency": 0.1, "response": "I cannot do that.", "tokens_generated": 5, "error": None}
    n_calls = 0

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        nonlocal n_calls
        n_calls += 1
        return dict(no_doc)

    async def huge_ctx(model):
        return 1_000_000

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(bench, "verify_ui_render", lambda doc: {"ran": False})
    result = await bench.run_workflow("m", True, glider)
    # Each step: prose answer + one redo; then the healing assembly pass
    # (attempt + redo) because no step delivered a document.
    assert n_calls == 2 * len(glider["steps"]) + 2
    assert not result["success"]
    assert result["steps_completed"] == 0
    assert any(sr["step_id"] == "assembly_pass" for sr in result["steps"])


@pytest.mark.asyncio()
async def test_empty_turn_is_retried_and_recovers(bench, glider, monkeypatch, tmp_path):
    """A zero-token turn (backend stream died) is transient -> retry succeeds."""
    import multistep_benchmark as msb

    monkeypatch.setattr(msb, "_EMPTY_TURN_RETRY_DELAY_S", 0)
    calls = {"n": 0}

    async def huge_ctx(model):
        return 1_000_000

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"success": True, "latency": 0.1, "response": "", "tokens_generated": 0, "error": None}
        return {"success": True, "latency": 0.4, "response": f"```html\n{GOOD_DOC}\n```", "tokens_generated": 90, "error": None}

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(
        bench, "verify_ui_render", lambda doc: {"ran": True, "screenshot": True, "score": 100}
    )
    monkeypatch.setattr(bench, "ARTIFACTS_DIR", tmp_path)
    result = await bench.run_workflow("m", True, glider)

    assert calls["n"] == 5  # turn 1: dead stream + recovery; turns 2-4: one call each
    assert result["success"]
    assert result["steps_completed"] == len(glider["steps"])
    assert result["steps"][0]["error"] is None


@pytest.mark.asyncio()
async def test_persists_error_when_all_empty_retries_exhausted(bench, glider, monkeypatch):
    """Still-empty after retries -> step records a diagnosable error."""
    import multistep_benchmark as msb

    monkeypatch.setattr(msb, "_EMPTY_TURN_RETRIES", 1)
    monkeypatch.setattr(msb, "_EMPTY_TURN_RETRY_DELAY_S", 0)
    calls = {"n": 0}

    async def huge_ctx(model):
        return 1_000_000

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        calls["n"] += 1
        return {"success": True, "latency": 0.1, "response": "", "tokens_generated": 0, "error": None}

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(bench, "verify_ui_render", lambda doc: {"ran": False})
    result = await bench.run_workflow("m", True, glider)

    assert not result["success"]
    # Each original turn: attempt + retry; assembly pass: attempt + retry.
    assert calls["n"] == 2 * len(glider["steps"]) + 2
    for step in result["steps"]:
        assert "empty generation after" in (step["error"] or "")


@pytest.mark.asyncio()
async def test_assembly_pass_heals_missing_docs_and_credits_delivery(
    bench, glider, monkeypatch, tmp_path
):
    """Interrupted turns trigger one assembly turn that ships ONE runnable doc."""
    import multistep_benchmark as msb

    monkeypatch.setattr(msb, "_EMPTY_TURN_RETRIES", 0)
    monkeypatch.setattr(msb, "_EMPTY_TURN_RETRY_DELAY_S", 0)
    calls = {"n": 0}

    async def huge_ctx(model):
        return 1_000_000

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"success": True, "latency": 0.3, "response": f"```html\n{GOOD_DOC}\n```", "tokens_generated": 50, "error": None}
        if calls["n"] <= 4:
            return {"success": True, "latency": 0.1, "response": "", "tokens_generated": 0, "error": None}
        return {"success": True, "latency": 0.5, "response": f"```html\n{GOOD_DOC}\n```", "tokens_generated": 80, "error": None}

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(
        bench, "verify_ui_render", lambda doc: {"ran": True, "screenshot": True, "score": 100}
    )
    monkeypatch.setattr(bench, "ARTIFACTS_DIR", tmp_path)
    result = await bench.run_workflow("m", True, glider)

    assert calls["n"] == 5  # 4 turns + 1 healing assembly pass
    assembly_steps = [sr for sr in result["steps"] if sr["step_id"] == "assembly_pass"]
    assert len(assembly_steps) == 1 and assembly_steps[0]["doc_extracted"]
    assert set(assembly_steps[0]["healed_passes"]) == {
        s["label"] for s in glider["steps"][1:]
    }
    assert result["success"]
    assert result["validation"]["breakdown"]["delivery_chain"] == 15.0
    artifact = tmp_path / f"{bench._sanitize_model_filename('m')}__glider_2026_house.html"
    assert GOOD_DOC.strip()[:60] in artifact.read_text(encoding="utf-8")


@pytest.mark.asyncio()
async def test_no_assembly_pass_when_every_turn_delivers(bench, glider, monkeypatch):
    calls = {"n": 0}

    async def huge_ctx(model):
        return 1_000_000

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        calls["n"] += 1
        return {"success": True, "latency": 0.2, "response": f"```html\n{GOOD_DOC}\n```", "tokens_generated": 40, "error": None}

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(
        bench, "verify_ui_render", lambda doc: {"ran": True, "screenshot": True, "score": 100}
    )
    await bench.run_workflow("m", True, glider)

    assert calls["n"] == len(glider["steps"])  # no extra healing turn


def test_glider_workflow_defines_assembly_contract():
    b = MultiStepBenchmark()
    glider = next(w for w in b.get_all_workflows() if w["id"] == "glider_2026_house")
    asm = glider.get("assembly")
    assert asm and "{missing}" in asm["prompt"]
    assert "runnable" in asm["prompt"].lower()
    names = [c["name"] for c in asm["checks"]]
    assert "all_rooms_present" in names and "eight_eggs_assembled" in names


# ------------------------------------------------------------------ #
# Context-window budgeting (8K-ctx hosts must still run all turns)     #
# ------------------------------------------------------------------ #


def test_extract_manifest_state_pulls_all_glider_manifests():
    doc = (
        "<html><script>\n"
        "const GLIDER_ROOMS = [{id:'den',name:'Den',exits:['kitchen']}];\n"
        "const GLIDER_EASTER_EGGS = [{id:'sleeping-cat',room:'den'}];\n"
        "</script></html>"
    )
    state = extract_manifest_state(doc)
    assert "GLIDER_ROOMS" in state and "'den'" in state
    assert "GLIDER_EASTER_EGGS" in state and "sleeping-cat" in state
    assert extract_manifest_state("no manifests here") == ""
    assert extract_manifest_state(None) == ""


def test_estimate_and_turn_budget():
    msgs = [{"role": "user", "content": "x" * 4000}]
    est = MultiStepBenchmark._estimate_tokens(msgs)
    # Conservative divisor: dense code is ~3 chars/token, not the prose ~4.
    assert 1300 <= est <= 1400
    b = MultiStepBenchmark()
    # Fits comfortably: full request honored.
    assert b._turn_budget(8192, msgs, 4000) == 4000
    # Tight window: clamped to what remains after prompt + safety margin.
    assert b._turn_budget(8192, msgs, 9000) == 8192 - est - 128
    # Over-full window: nothing left.
    assert b._turn_budget(1000, msgs, 9000) == 0


def test_compact_messages_keeps_last_doc_and_manifests():
    big_doc = (
        "<html>" + "x" * 26000 + "<script>const GLIDER_ROOMS = [{id:'den'}];"
        "const GLIDER_EASTER_EGGS = [{id:'sleeping-cat',room:'den'}];</script></html>"
    )
    long_prompt = "build the den" + " detail " * 80
    messages = [
        {"role": "user", "content": long_prompt},
        {"role": "assistant", "content": big_doc},
        {"role": "user", "content": "add the kitchen"},
        {"role": "assistant", "content": "<html>later doc</html>"},
        {"role": "user", "content": "finish it"},
    ]
    b = MultiStepBenchmark()

    compacted, changed = b._compact_messages(messages, 8192)
    assert changed
    users = [m["content"] for m in compacted if m["role"] == "user"]
    assistants = [m["content"] for m in compacted if m["role"] == "assistant"]
    # Most recent user prompt and most recent doc untouched...
    assert users[-1] == "finish it"
    assert assistants[-1] == "<html>later doc</html>"
    # ...older assistant doc reduced to manifest state; older prompts summarized.
    assert "System note" in assistants[0]
    assert "GLIDER_ROOMS" in assistants[0] and "sleeping-cat" in assistants[0]
    assert "System note" in users[0] and long_prompt[:100] in users[0]
    # The short middle prompt survives verbatim (below the compaction floor).
    assert "add the kitchen" in users
    # Small transcript on a big window: untouched.
    same, changed2 = b._compact_messages(messages[:2], 1_000_000)
    assert not changed2 and same == messages[:2]


def test_resolve_context_window_reads_proxy_runtime(monkeypatch):
    import httpx

    runtime = {
        "loaded_models": [
            {
                "name": "test-model",
                "backend_model": "test-model--latest",
                "running_settings": {"ctx-size": "8192"},
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/admin/runtime"):
            return httpx.Response(200, content=json.dumps(runtime).encode())
        return httpx.Response(404)

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    b = MultiStepBenchmark()
    with patch.object(httpx, "AsyncClient", side_effect=client_factory):
        assert asyncio.run(b._resolve_context_window("test-model")) == 8192
    # Cached afterwards (no second HTTP call needed even with dead transport).
    assert asyncio.run(b._resolve_context_window("test-model")) == 8192


def test_resolve_context_window_warms_unloaded_model(monkeypatch):
    import httpx

    runtime_empty = {"loaded_models": []}
    runtime_loaded = {
        "loaded_models": [
            {
                "name": "test-model",
                "backend_model": "test-model--latest",
                "running_settings": {"ctx-size": "8192"},
            }
        ]
    }
    scans = []
    chats = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/admin/runtime"):
            scans.append(1)
            # First scan: model not loaded yet. After the warm-up chat: loaded.
            body = runtime_empty if len(scans) == 1 else runtime_loaded
            return httpx.Response(200, content=json.dumps(body).encode())
        if request.url.path.endswith("/api/chat"):
            chats.append(1)
            assert json.loads(request.content.decode())["options"]["num_predict"] == 1
            return httpx.Response(
                200,
                content=b'{"message":{"role":"assistant","content":"OK"},"done":true}\n',
            )
        return httpx.Response(404)

    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    b = MultiStepBenchmark()
    # One proxy URL so one logical scan == exactly one /admin/runtime hit;
    # the default list would fan the scan out over several URLs.
    b.PROXY_SERVER_URLS = ["http://mock-proxy:11434"]
    with patch.object(httpx, "AsyncClient", side_effect=client_factory):
        assert asyncio.run(b._resolve_context_window("test-model")) == 8192
    assert chats, "warm-up chat must fire before re-scan"


def test_resolve_context_window_raises_loudly_when_unknown(monkeypatch):
    import context_awareness

    monkeypatch.setattr(context_awareness, "RESOLVE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(context_awareness, "POLL_INTERVAL_S", 0.01)
    b = MultiStepBenchmark()
    b.PROXY_SERVER_URLS = ["http://dead-proxy.invalid:11434"]
    b._ctx_cache.clear()
    with pytest.raises(RuntimeError, match="context window"):
        asyncio.run(b._resolve_context_window("never-loaded-model"))


@pytest.mark.asyncio()
async def test_run_workflow_fails_step_when_context_exhausted(bench, glider, monkeypatch):
    n_calls = 0

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        nonlocal n_calls
        n_calls += 1
        raise AssertionError("query must not fire when the budget is already exhausted")

    async def tiny_ctx(model):
        return 600

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", tiny_ctx)
    monkeypatch.setattr(bench, "verify_ui_render", lambda doc: {"ran": False})
    result = await bench.run_workflow("m", True, glider)
    assert n_calls == 0
    assert not result["success"]
    assert result["steps_completed"] == 0
    assert "Context window exhausted" in (result["error"] or "")
    assert len(result["steps"]) == 1
    assert "Context window exhausted" in (result["steps"][0]["error"] or "")


@pytest.mark.asyncio()
async def test_run_workflow_clamps_budget_on_small_context(bench, glider, monkeypatch):
    seen_budgets = []

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        seen_budgets.append(max_tokens)
        return {"success": False, "latency": 0.01, "response": "", "tokens_generated": 0, "error": "mock down"}

    async def small_ctx(model):
        return 8192

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", small_ctx)
    monkeypatch.setattr(bench, "verify_ui_render", lambda doc: {"ran": False})
    await bench.run_workflow("m", True, glider)
    # Glider prompts are multi-KB; on an 8K window every budget must be
    # clamped below the step's requested 9000-11000 tokens yet stay generable.
    # The healing assembly budget (if the mock produced missing docs) obeys
    # the same clamp.
    for i, budget in enumerate(seen_budgets):
        step_cap = glider["steps"][i]["max_tokens"] if i < len(glider["steps"]) else 11000
        assert budget < step_cap
        assert 512 <= budget <= 8192 - 128


def test_save_per_model_result(bench, glider, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "MODELS_DIR", tmp_path)
    model_record = {
        "model": "qwen:test/8b",
        "timestamp": "2026-08-22T00:00:00",
        "tasks": [{"test_id": glider["id"], "success": True, "score": 95.0}],
    }
    path = bench.save_per_model_result(model_record, use_proxy=True)
    assert path is not None and path.exists()
    data = json.loads(path.read_text())
    assert data["benchmark_version"].startswith("MultiStep")
    assert data["per_model"] is True
    assert data["benchmark_type"] == "proxy"
    assert data["results"][0]["tasks"][0]["test_id"] == glider["id"]


# ------------------------------------------------------------------ #
# Web routes                                                          #
# ------------------------------------------------------------------ #


@pytest.fixture()
def client():
    from web.app import active_run, active_run_lock, app

    app.config["TESTING"] = True
    with app.test_client() as c:
        with active_run_lock:
            active_run["status"] = "idle"
            active_run["current_model"] = None
            active_run["current_test"] = None
            active_run["current_category"] = None
            active_run["tests_completed"] = 0
            active_run["total_tests"] = 0
            active_run["models"] = []
            active_run["use_proxy"] = True
            active_run["results"] = []
            active_run["start_time"] = None
            active_run["saved_as"] = None
        yield c


def test_api_tests_multistep_lists_glider(client):
    res = client.get("/api/tests/multistep")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    tests = data.get("tests", [])
    ids = [t["id"] for t in tests]
    assert "glider_2026_house" in ids
    glider_entry = next(t for t in tests if t["id"] == "glider_2026_house")
    assert glider_entry["type"] == "multistep"
    assert glider_entry["steps"] >= 1


def test_api_run_multistep_requires_models(client):
    res = client.post("/api/run/multistep", json={"models": []})
    assert res.status_code == 400
    assert "error" in json.loads(res.data.decode("utf-8"))


def test_api_run_multistep_rejects_when_running(client):
    from web.app import active_run, active_run_lock

    with active_run_lock:
        active_run["status"] = "running"
    try:
        res = client.post("/api/run/multistep", json={"models": ["m"]})
        assert res.status_code == 409
    finally:
        with active_run_lock:
            active_run["status"] = "idle"


def test_progress_callback_tracks_turn_level_progress():
    """Each agentic turn advances tests_completed so the progress bar moves."""
    from web.app import active_run, active_run_lock, get_progress_callback

    cb = get_progress_callback("multistep")
    with active_run_lock:
        active_run["status"] = "idle"
    try:
        cb(
            "benchmark_start",
            {"models": ["m1"], "use_proxy": True, "total_tests": 2, "timestamp": "t"},
        )
        wf = {"model": "m1", "workflow": "glider_2026_house", "workflow_label": "Glider",
              "category": "c", "label": "T1", "step": 1, "total": 4, "num_predict": 1000}
        cb("test_step", {**wf, "step": 1})
        with active_run_lock:
            assert (active_run["tests_completed"], active_run["total_tests"]) == (0, 4)
        cb("test_step", {**wf, "step": 3})
        with active_run_lock:
            assert active_run["tests_completed"] == 2
        # Second workflow accumulates: done += 4, current total 5 turns.
        wf2 = {**wf, "workflow": "other_house", "step": 2, "total": 5}
        cb("test_step", wf2)
        with active_run_lock:
            assert (active_run["tests_completed"], active_run["total_tests"]) == (5, 9)
    finally:
        with active_run_lock:
            active_run.update({"status": "idle", "tests_completed": 0, "total_tests": 0})


def test_multistep_benchmark_start_resets_turn_counters():
    from web.app import get_progress_callback

    cb = get_progress_callback("multistep")
    wf = {"model": "m", "workflow": "w", "label": "L", "step": 2, "total": 3,
          "workflow_label": "WL", "category": "c", "num_predict": 10}
    cb("benchmark_start", {"models": ["m"], "use_proxy": True, "total_tests": 1, "timestamp": "t"})
    cb("test_step", wf)
    # A fresh run must start counting from zero again.
    cb("benchmark_start", {"models": ["m"], "use_proxy": True, "total_tests": 1, "timestamp": "t2"})
    cb("test_step", {**wf, "step": 1})
    from web.app import active_run as ar
    assert (ar["tests_completed"], ar["total_tests"]) == (0, 3)


# ------------------------------------------------------------------ #
# Truncation detection, doc selection & healing assembly              #
# ------------------------------------------------------------------ #


def test_document_is_complete_flags_truncation():
    from multistep_benchmark import document_is_complete

    complete = "<html><body><script>let score = 0;</script></body></html>"
    assert document_is_complete(complete) is True
    assert document_is_complete(complete + "\n") is True
    # Token cut inside a <script>: unclosed tag + missing </html>.
    assert document_is_complete("<html><body><script>function broken() {") is False
    # Closed script but the document never closes.
    assert document_is_complete("<html><script>x</script></body>") is False
    assert document_is_complete(None) is False
    assert document_is_complete("") is False


@pytest.mark.asyncio()
async def test_assembly_pass_fires_when_final_doc_truncated(
    bench, glider, monkeypatch, tmp_path
):
    """Every turn 'delivers' but the best doc is cut mid-tag: heal it."""
    truncated = GOOD_DOC[: GOOD_DOC.rfind("</html>")] + "\n<script>function broken() {"
    calls = {"n": 0}

    async def huge_ctx(model):
        return 1_000_000

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        calls["n"] += 1
        if calls["n"] <= 4:
            return {
                "success": True,
                "latency": 0.2,
                "response": f"```html\n{truncated}\n```",
                "tokens_generated": 60,
                "error": None,
            }
        return {
            "success": True,
            "latency": 0.4,
            "response": f"```html\n{GOOD_DOC}\n```",
            "tokens_generated": 90,
            "error": None,
        }

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(
        bench, "verify_ui_render", lambda doc: {"ran": True, "screenshot": True, "score": 100}
    )
    monkeypatch.setattr(bench, "ARTIFACTS_DIR", tmp_path)
    result = await bench.run_workflow("m", True, glider)

    assert calls["n"] == 5  # 4 turns + healing pass for the truncation
    asm = [sr for sr in result["steps"] if sr["step_id"] == "assembly_pass"]
    assert len(asm) == 1 and asm[0]["doc_extracted"]
    assert any("untruncated" in h for h in asm[0]["healed_passes"])
    artifact = tmp_path / f"{bench._sanitize_model_filename('m')}__glider_2026_house.html"
    text = artifact.read_text(encoding="utf-8")
    assert "</html>" in text and "broken() {" not in text
    assert result["success"]


@pytest.mark.asyncio()
async def test_complete_doc_outranks_longer_truncated_fragment(
    bench, glider, monkeypatch, tmp_path
):
    """A bigger token-cut fragment must NOT replace an earlier complete doc."""
    big_fragment = "```html\n<html><script>" + ("x" * 30000) + "\n"
    seen = {"n": 0}

    async def huge_ctx(model):
        return 1_000_000

    async def fake_query(model, use_proxy, messages, max_tokens, custom_keys=None):
        seen["n"] += 1
        resp = f"```html\n{GOOD_DOC}\n```" if seen["n"] == 1 else big_fragment
        return {"success": True, "latency": 0.2, "response": resp, "tokens_generated": 50, "error": None}

    monkeypatch.setattr(bench, "query_model_turn", fake_query)
    monkeypatch.setattr(bench, "_resolve_context_window", huge_ctx)
    monkeypatch.setattr(
        bench, "verify_ui_render", lambda doc: {"ran": None, "screenshot": False, "score": 0}
    )
    monkeypatch.setattr(bench, "ARTIFACTS_DIR", tmp_path)
    result = await bench.run_workflow("m", True, glider)

    artifact = tmp_path / f"{bench._sanitize_model_filename('m')}__glider_2026_house.html"
    text = artifact.read_text(encoding="utf-8")
    assert GOOD_DOC.strip()[:60] in text
    assert "x" * 1000 not in text
    assert result["validation"]["final_document_chars"] < 30000


def test_artifact_route_serves_file_and_blocks_bad_paths(client, tmp_path, monkeypatch):
    from web.app import multistep_benchmark as app_ms

    art = tmp_path / "m__glider_2026_house.html"
    art.write_text("<html><body>game</body></html>", encoding="utf-8")
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_ms, "ARTIFACTS_DIR", tmp_path)

    res = client.get("/api/multistep/artifact/m__glider_2026_house.html")
    assert res.status_code == 200
    assert b"game" in res.data
    assert res.mimetype == "text/html"

    # Unsupported extension refused even inside ARTIFACTS_DIR.
    res_json = client.get("/api/multistep/artifact/secret.json")
    assert res_json.status_code == 400

    # Path traversal refused.
    res_up = client.get("/api/multistep/artifact/%2e%2e/secret.json")
    assert res_up.status_code in (400, 404)

    # Missing artifact -> 404.
    res_missing = client.get("/api/multistep/artifact/nobody__here.html")
    assert res_missing.status_code == 404


def test_result_detail_annotates_artifact_urls(client, tmp_path, monkeypatch):
    from web.app import multistep_benchmark as app_ms

    art = tmp_path / "m__w.html"
    art.write_text("<html></html>", encoding="utf-8")
    turn = tmp_path / "m__w__turn1.txt"
    turn.write_text("raw turn output", encoding="utf-8")
    snapshot = {
        "results": [
            {
                "model": "m",
                "tasks": [
                    {
                        "test_id": "w",
                        "artifact": str(art),
                        "steps": [{"step_id": "s1", "response_path": str(turn)}],
                    }
                ],
            }
        ]
    }
    snap_name = "multistep_benchmarks_test_proxy.json"
    (tmp_path / snap_name).write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(app_ms, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(app_ms, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(app_ms, "ARTIFACTS_DIR", tmp_path)

    res = client.get(f"/api/results/{snap_name}")
    assert res.status_code == 200
    data = json.loads(res.data.decode("utf-8"))
    task = data["results"][0]["tasks"][0]
    assert task["artifact_url"] == "/api/multistep/artifact/m__w.html"
    assert task["steps"][0]["response_url"] == "/api/multistep/artifact/m__w__turn1.txt"
