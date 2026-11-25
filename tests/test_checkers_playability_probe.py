"""Hermetic tests for the checkers playability probe.

The probe runs a headless-Chromium harness inside the grading sandbox; these
tests stub the container I/O so the decision logic is covered without Docker.
"""

import base64
import json

import sandbox_exec as se


class _FakeContainer:
    def exec_run(self, *args, **kwargs):
        return 0, b""


def _dom_for(facts):
    blob = base64.b64encode(json.dumps(facts).encode("utf-8")).decode("ascii")
    return f'<html><body><div id="__probe">RESULT:{blob}</div></body></html>'.encode()


def _patch_io(monkeypatch, dom):
    monkeypatch.setattr(se, "_put_file", lambda *a, **k: None)

    def fake_read(container, path):
        if path.endswith("probe_dom.html"):
            return dom
        return b""

    monkeypatch.setattr(se, "_read_file", fake_read)


def test_probe_fails_when_board_never_responds(monkeypatch):
    _patch_io(monkeypatch, _dom_for({"playable": False, "reason": "no piece could be moved: the game offers no legal first move"}))
    result = {"ran": True, "ui_rendered": True, "error": ""}
    se._checkers_playability_probe(_FakeContainer(), "<html></html>", 30, result)

    assert result["ran"] is False
    assert "not playable" in result["error"]
    assert result["playability"]["playable"] is False
    # A functional failure, not a blank render.
    assert result["ui_rendered"] is True


def test_probe_passes_when_board_responds(monkeypatch):
    _patch_io(monkeypatch, _dom_for({"playable": True, "reason": "the AI answered a human move"}))
    result = {"ran": True, "ui_rendered": True, "error": ""}
    se._checkers_playability_probe(_FakeContainer(), "<html></html>", 30, result)

    assert result["ran"] is True
    assert result["error"] == ""
    assert result["playability"]["playable"] is True


def test_the_harness_requires_an_ai_reply():
    """The probe must engage the AI mode and fail when the AI never answers.

    The earlier probe passed any page whose board changed on a click, which is
    true of a human-vs-human game with a selection highlight. Playing one move
    and waiting for the reply is the only check that says the game is playable
    *as a game*.
    """
    page = se._CHECKERS_PROBE_HTML
    assert "findAiControl" in page and "ai_mode" in page
    assert "the AI answered a human move" in page
    assert "the AI never answered" in page
    assert "no piece could be moved" in page


def test_probe_unavailable_does_not_fail_the_game(monkeypatch):
    _patch_io(monkeypatch, b"")  # harness produced no RESULT marker
    result = {"ran": True, "ui_rendered": True, "error": ""}
    se._checkers_playability_probe(_FakeContainer(), "<html></html>", 30, result)

    assert result["ran"] is True
    assert result["playability"] == {"probe": "unavailable"}


def test_grade_code_forwards_test_id_for_ui_runs(monkeypatch):
    seen = {}

    def fake_run(code, lang, timeout, ui, test_id=None, framework=None):
        seen["test_id"] = test_id
        seen["ui"] = ui
        return {
            "ran": True,
            "output": "",
            "lint_passed": True,
            "screenshot": "x",
            "ui_rendered": True,
            "exit_code": 0,
        }

    monkeypatch.setattr(se, "run_code_once", fake_run)
    graded = se.grade_code("<html></html>", "html", ui=True, test_id="game_checkers_web")

    assert seen["test_id"] == "game_checkers_web"
    assert seen["ui"] is True
    assert graded["score"] == 100


def test_grade_code_does_not_forward_unknown_test_id_for_cli(monkeypatch):
    seen = {}

    def fake_run(code, lang, timeout, ui, test_id=None, framework=None):
        seen["test_id"] = test_id
        return {"ran": True, "output": "ok", "lint_passed": True, "exit_code": 0}

    monkeypatch.setattr(se, "run_code_once", fake_run)
    se.grade_code("print(1)", "python", test_id="logic_schedule")

    # CLI runs keep their old behaviour (no test_id unless a known fixture).
    assert seen["test_id"] is None
