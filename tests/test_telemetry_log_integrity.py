"""A torn telemetry line must not discard the rest of the file.

The bug this pins
-----------------
/api/telemetry/history read its JSONL inside a single try/except, so one
unparseable line returned 500 for the whole file. The live data made that
concrete: ``system_idle.jsonl`` has 365,326 lines and exactly one of them is
torn -- at line 172. One bad line was discarding 365,000 good points.

How the line got torn
---------------------
It is not a truncated tail. The file ends with a newline, and the damage is at
line 172 of 365,326, with the file continuing for another 365,154 lines. The
recorded line is two records sharing one line:

    ..."model_path": "none", "n_ctx{"timestamp": "202

That is the signature of a process killed *mid-append*: the kernel accepted part
of the buffer, the process died, and the next open started writing at the file
end -- inside the unfinished record. alpaca-web was OOM-killed 29 times, so
this is expected damage, not a malformed-input edge case.

Why one reader and not the other
--------------------------------
``analyzer.load_telemetry`` has always done ``except: continue`` per line,
because the analyzer reads all 33 files and one bad line must not matter. The
web reader did not. Two readers of one file format in one repo disagreed, and
the stricter one was the one a person actually looks at. These tests assert
they now agree.

The count is reported, not hidden
--------------------------------
``skipped_lines`` is returned whenever anything was dropped. Silently returning
"no data" for a file that has 365,000 points in it would be the worse failure.
"""

import json

import pytest

from web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _point(record: dict) -> str:
    return json.dumps(record)


def _good(i: int) -> dict:
    return {
        "timestamp": f"2026-06-29T19:45:{i:02d}+0000",
        "epoch_time": 1782762340.0 + i,
        "model_alias": "torn_test",
        "system": {"ram_used_gb": 10.0 + i},
        "gpus": [],
        "llama_server": {"model_path": "none", "n_ctx": 8192, "slots": {"total": 1}},
    }


# The exact shape found in the live file: a record cut mid-field, immediately
# followed by the start of the next one, all on a single line.
TORN_LINE = (
    '{"timestamp": "2026-06-29T19:45:40+0000", "epoch_time": 1782762340.8454018, '
    '"model_alias": "system_idle", "system": {"ram_total_gb": 30.63, "ram_used_gb": 26.6}, '
    '"gpus": [], "llama_server": {"model_path": "none", "n_ctx{"timestamp": "202'
)


def _write(tmp_path, lines, name="torn_test.jsonl"):
    d = tmp_path / "telemetry"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return p


def _history(client, tmp_path, limit=100):
    """Point the route at our temp file and GET it.

    The route reads TELEMETRY_DIR per request, so setting the env var here is
    enough -- no patching of module state, and no risk of leaking it.
    """
    import os

    os.environ["TELEMETRY_DIR"] = str(tmp_path / "telemetry")
    try:
        return client.get(f"/api/telemetry/history?model=torn_test&limit={limit}")
    finally:
        del os.environ["TELEMETRY_DIR"]


def test_a_torn_line_in_the_middle_does_not_discard_the_file(tmp_path, client):
    """The exact production failure: 365k good lines, one torn at 172."""
    lines = [_point(_good(i)) for i in range(5)]
    lines.insert(2, TORN_LINE)
    _write(tmp_path, lines)

    res = _history(client, tmp_path)
    assert res.status_code == 200, res.data
    data = json.loads(res.data)
    assert len(data["history"]) == 5, "the five good points must survive"
    assert data["skipped_lines"] == 1, "the dropped line must be counted, not hidden"
    assert [p["epoch_time"] for p in data["history"]] == [1782762340.0 + i for i in range(5)]


def test_the_skip_count_is_reported_for_every_torn_line(tmp_path, client):
    lines = [_point(_good(i)) for i in range(3)] + [TORN_LINE, TORN_LINE, "not json at all"]
    _write(tmp_path, lines)

    data = json.loads(_history(client, tmp_path).data)
    assert len(data["history"]) == 3
    assert data["skipped_lines"] == 3


def test_a_truncated_final_line_is_tolerated(tmp_path, client):
    """The other classic mode: a kill during the last append leaves no newline."""
    lines = [_point(_good(i)) for i in range(4)]
    lines.append('{"timestamp": "2026-06-29T19:45:40+0000", "epoch_ty')
    _write(tmp_path, lines)

    res = _history(client, tmp_path)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert len(data["history"]) == 4
    assert data["skipped_lines"] == 1


def test_a_clean_file_reports_no_skips(tmp_path, client):
    """No skipped_lines key at all when nothing was dropped.

    An always-present zero would train the reader to ignore it, which is the
    opposite of what the key is for.
    """
    _write(tmp_path, [_point(_good(i)) for i in range(3)])
    data = json.loads(_history(client, tmp_path).data)
    assert "skipped_lines" not in data
    assert len(data["history"]) == 3


def test_an_entirely_torn_file_is_empty_not_an_error(tmp_path, client):
    _write(tmp_path, [TORN_LINE, TORN_LINE, TORN_LINE])
    res = _history(client, tmp_path)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["history"] == []
    assert data["skipped_lines"] == 3


def test_the_limit_still_applies_after_skipping(tmp_path, client):
    _write(tmp_path, [_point(_good(i)) for i in range(50)] + [TORN_LINE])
    data = json.loads(_history(client, tmp_path, limit=5).data)
    assert len(data["history"]) == 5
    # The newest five, and the torn line is still reported even though it is not
    # in the window that was returned.
    assert [p["epoch_time"] for p in data["history"]] == [1782762340.0 + i for i in range(45, 50)]
    assert data["skipped_lines"] == 1


def test_blank_lines_are_not_counted_as_skips(tmp_path, client):
    _write(tmp_path, [_point(_good(0)), "", "   ", _point(_good(1))])
    data = json.loads(_history(client, tmp_path).data)
    assert len(data["history"]) == 2
    assert "skipped_lines" not in data


def test_a_genuine_io_failure_is_still_an_error(tmp_path, client):
    """Bad *data* in a log is not an error; an unreadable *file* still is.

    The old code caught Exception, so a permissions failure and a bad line were
    indistinguishable. Splitting them is what lets the line be skipped without
    also swallowing real faults.
    """
    import os

    p = _write(tmp_path, [_point(_good(0))])
    os.chmod(p, 0o000)
    if os.access(p, os.R_OK):  # running as root: the mode is not enforced
        pytest.skip("cannot make a file unreadable as this user")
    try:
        res = _history(client, tmp_path)
        assert res.status_code == 500
        assert "Failed to read telemetry" in json.loads(res.data)["error"]
    finally:
        os.chmod(p, 0o644)


def test_the_web_reader_and_the_analyzer_agree_on_a_torn_file(tmp_path, client, monkeypatch):
    """Two readers of one format, one repo.

    analyzer.load_telemetry has always skipped bad lines per line. If the web
    reader regresses to treating one as fatal they will disagree again, and the
    analyzer passing its tests would hide it.
    """
    import analyzer

    lines = [_point(_good(i)) for i in range(6)]
    lines.insert(3, TORN_LINE)
    p = _write(tmp_path, lines)
    monkeypatch.setattr(analyzer, "TELEMETRY_DIR", p.parent)

    from_analyzer = analyzer.load_telemetry("torn_test", limit=500, max_age_seconds=10**9)
    from_web = json.loads(_history(client, tmp_path).data)["history"]

    assert len(from_analyzer) == 6
    assert len(from_web) == 6
    assert [d["epoch_time"] for d in from_analyzer] == [d["epoch_time"] for d in from_web]


def test_nonascii_bytes_do_not_kill_the_read(tmp_path, client):
    """A stray non-UTF-8 byte is corruption too, and must not be fatal."""
    d = tmp_path / "telemetry"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "torn_test.jsonl"
    p.write_bytes((_point(_good(0)) + "\n").encode("utf-8") + b"\xff\xfe garbage \n" + (_point(_good(1)) + "\n").encode("utf-8"))

    res = _history(client, tmp_path)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert len(data["history"]) == 2
    assert data["skipped_lines"] == 1
