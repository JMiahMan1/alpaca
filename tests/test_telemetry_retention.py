"""The telemetry writer must not create the damage it used to, and the directory must stop growing.

Two separate defects, both found on the live stack rather than reasoned about:

**One bad line was discarding 365,000 good ones.** ``system_idle.jsonl`` had a
single record cut off mid-key (``..."model_path": "none", "n_ctx``) at line 172
of 365,326, and ``/api/telemetry/history`` 500'd for every model because the
reader treated the whole file as one unit. The reader is fixed in
``21b1ab6``; these tests cover the writer, which was still creating new damage.

**Nothing ever bounded the directory.** 33 files, 1.2 GB, 42 days, because a
per-model file is only ever appended to. Rotation bounds a hot file; the age
sweep bounds a cold one.

The torn-tail repair is the load-bearing behaviour here: the point is not to
detect a damaged record, it is to stop the *next* good record from concatenating
onto it, which is what turned one bad record into two on the live file.
"""

import ast
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import telemetry_monitor as tm

APP_SOURCE = Path(__file__).resolve().parent.parent / "telemetry_monitor.py"


@pytest.fixture
def tdir(tmp_path, monkeypatch):
    """Point the module at a scratch directory with small, predictable bounds."""
    d = tmp_path / "telemetry"
    monkeypatch.setattr(tm, "TELEMETRY_DIR", d)
    monkeypatch.setattr(tm, "TELEMETRY_MAX_BYTES", 4096)
    monkeypatch.setattr(tm, "TELEMETRY_MAX_GENERATIONS", 3)
    return d


def _record(n):
    return {"epoch_time": 1700000000 + n, "model_alias": "m", "pad": "x" * 200}


# --------------------------------------------------------------------------- #
# the torn tail                                                               #
# --------------------------------------------------------------------------- #


def test_a_truncated_record_does_not_swallow_the_next_one(tdir):
    """The exact damage found on the live stack, and the reason it mattered.

    Without a repair, O_APPEND seeks to EOF -- which is now mid-record -- and the
    next record concatenates onto the fragment. Two bad records where there was
    one, and a reader sees the good data as part of an unparseable line.
    """
    target = tdir / "m.jsonl"
    tdir.mkdir(parents=True)
    target.write_text('{"epoch_time": 1, "model_path": "none", "n_ctx')  # cut mid-key

    tm.write_telemetry_log("m", _record(2))

    lines = target.read_text().splitlines()
    assert len(lines) == 2, lines
    # The damage is still there, in its own line, isolated.
    assert "n_ctx" in lines[0] and not lines[0].rstrip().endswith("}")
    # And critically the new record is intact and readable on its own.
    assert json.loads(lines[1])["epoch_time"] == 1700000002


def test_a_complete_tail_gets_no_inserted_blank_line(tdir):
    tm.write_telemetry_log("m", _record(1))
    tm.write_telemetry_log("m", _record(2))
    tm.write_telemetry_log("m", _record(3))
    lines = (tdir / "m.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert all(line.strip() for line in lines), f"a blank line crept in: {lines!r}"
    assert [json.loads(line)["epoch_time"] for line in lines] == [
        1700000001,
        1700000002,
        1700000003,
    ]


def test_every_record_the_writer_emits_is_parseable(tdir):
    """Across every generation, not just the active file.

    Read across generations deliberately: 25 records overshoot the 4 KB cap, so
    this also proves rotation moves intact records rather than fragments.
    """
    for n in range(25):
        tm.write_telemetry_log("m", _record(n))
    parsed = []
    for path in sorted(tdir.glob("*.jsonl*")):
        for line in path.read_text().splitlines():
            parsed.append(json.loads(line)["epoch_time"])  # raises on a torn record
    assert sorted(parsed) == sorted(1700000000 + n for n in range(25)), parsed


def test_the_append_is_a_single_unbuffered_write(tdir):
    """Pin the atomicity property structurally.

    The buffered text writer accumulates a record in user space and emits it on
    close, so a kill in between leaves whatever the flush had pushed. One
    unbuffered write of one complete line to an O_APPEND descriptor is the
    smallest unit the OS will reorder or interleave.
    """
    seen = []
    real_open = open

    def spy(path, mode="r", *a, **kw):
        if str(path).endswith("m.jsonl"):
            seen.append((mode, kw.get("buffering")))
        return real_open(path, mode, *a, **kw)

    with patch("builtins.open", spy):
        tm.write_telemetry_log("m", _record(1))

    appends = [s for s in seen if "a" in s[0]]
    assert appends, f"no append seen, only {seen}"
    assert all(mode == "ab" for mode, _ in appends), appends
    assert all(buf == 0 for _, buf in appends), f"append was buffered: {appends}"


def test_a_failed_write_does_not_leave_a_half_written_file(tdir):
    """A write that raises must not leave a partial record behind."""
    tdir.mkdir(parents=True)
    with patch("builtins.open", side_effect=OSError("no space left")):
        tm.write_telemetry_log("m", _record(1))  # must not raise
    assert not (tdir / "m.jsonl").exists()


def test_several_models_stay_in_separate_files(tdir):
    for n in range(5):
        tm.write_telemetry_log("alpha", _record(n))
        tm.write_telemetry_log("beta", _record(n))
    assert len((tdir / "alpha.jsonl").read_text().splitlines()) == 5
    assert len((tdir / "beta.jsonl").read_text().splitlines()) == 5


# --------------------------------------------------------------------------- #
# rotation                                                                     #
# --------------------------------------------------------------------------- #


def test_a_file_under_the_cap_is_never_rotated(tdir):
    tm.write_telemetry_log("m", _record(1))
    size = (tdir / "m.jsonl").stat().st_size
    assert size < tm.TELEMETRY_MAX_BYTES
    assert not list(tdir.glob("*.jsonl.*")), "rotated a file that was under the cap"


def test_crossing_the_cap_rolls_the_file(tdir):
    for n in range(40):  # 40 * ~250 bytes > 4096
        tm.write_telemetry_log("m", _record(n))
    assert (tdir / "m.jsonl.1").exists(), sorted(p.name for p in tdir.iterdir())
    # The active file is back under the cap.
    assert (tdir / "m.jsonl").stat().st_size < tm.TELEMETRY_MAX_BYTES


def test_rotation_keeps_every_record_it_wrote(tdir):
    """Rotating must move data, not discard it."""
    for n in range(60):
        tm.write_telemetry_log("m", _record(n))
    epochs = []
    for path in sorted(tdir.glob("*.jsonl*")):
        for line in path.read_text().splitlines():
            epochs.append(json.loads(line)["epoch_time"])
    # Newest data first in the active file, older generations behind it.
    assert sorted(epochs) == sorted(1700000000 + n for n in range(60))
    assert len(epochs) == 60, f"lost records across rotation: {len(epochs)}"


def test_generations_are_bounded(tdir):
    """The whole point of a cap on generations: the directory stops growing."""
    for n in range(400):
        tm.write_telemetry_log("m", _record(n))
    generations = sorted(p.name for p in tdir.glob("m.jsonl.*"))
    assert len(generations) == tm.TELEMETRY_MAX_GENERATIONS, generations
    assert not list(tdir.glob("m.jsonl.4")), "generation cap not enforced"
    assert max(tdir.iterdir(), key=lambda p: p.stat().st_size).name == "m.jsonl"


def test_rotation_is_scoped_to_one_model(tdir):
    """'hot' is pushed past the cap; 'cold' is left comfortably under it."""
    for n in range(20):  # ~5 KB, over the 4 KB cap
        tm.write_telemetry_log("hot", _record(n))
        if n <= 4:  # ~1 KB, under it
            tm.write_telemetry_log("cold", _record(n))
    assert (tdir / "hot.jsonl.1").exists(), sorted(p.name for p in tdir.iterdir())
    assert not (tdir / "cold.jsonl.1").exists(), "rotated a model that was under the cap"
    assert len((tdir / "cold.jsonl").read_text().splitlines()) == 5


def test_rotation_failure_does_not_lose_the_new_record(tdir):
    """A rotation that raises must not cost us the point we just measured."""
    for n in range(40):
        tm.write_telemetry_log("m", _record(n))
    with patch.object(tm, "_rotate_if_oversized", side_effect=OSError("boom")):
        tm.write_telemetry_log("m", _record(999))  # must not raise
    assert json.loads((tdir / "m.jsonl").read_text().splitlines()[-1])["epoch_time"] == 1700000000 + 999


# --------------------------------------------------------------------------- #
# age pruning                                                                  #
# --------------------------------------------------------------------------- #


def test_prune_removes_files_older_than_the_window(tdir):
    old = tdir / "m.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text("{}\n")
    ancient = (time.time() - 40 * 86400)
    import os

    os.utime(old, (ancient, ancient))

    assert tm.prune_old_telemetry() == 1
    assert not old.exists()


def test_prune_keeps_recent_files(tdir):
    tm.write_telemetry_log("m", _record(1))
    assert tm.prune_old_telemetry() == 0
    assert (tdir / "m.jsonl").exists()


def test_prune_removes_stale_rotated_generations_too(tdir):
    import os

    tdir.mkdir(parents=True)
    gen = tdir / "m.jsonl.1"
    gen.write_text("{}\n")
    ancient = time.time() - 40 * 86400
    os.utime(gen, (ancient, ancient))
    tm.write_telemetry_log("m", _record(1))  # a fresh active file
    assert tm.prune_old_telemetry() == 1
    assert not gen.exists()
    assert (tdir / "m.jsonl").exists(), "pruning a rotated file took the active one with it"


def test_prune_honours_an_explicit_window(tdir):
    import os

    f = tdir / "m.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("{}\n")
    two_days = time.time() - 2 * 86400
    os.utime(f, (two_days, two_days))
    assert tm.prune_old_telemetry(max_age_days=1) == 1
    assert tm.prune_old_telemetry(max_age_days=30) == 0


def test_prune_on_a_missing_directory_is_quiet(tdir):
    assert tm.prune_old_telemetry() == 0


def test_prune_survives_an_unreadable_entry(tdir):
    tdir.mkdir(parents=True)
    (tdir / "m.jsonl").write_text("{}\n")
    with patch.object(Path, "unlink", side_effect=OSError("busy")):
        assert tm.prune_old_telemetry(max_age_days=0) in (0, 1)  # must not raise


# --------------------------------------------------------------------------- #
# wiring                                                                       #
# --------------------------------------------------------------------------- #


def test_the_sweep_runs_from_the_poll_loop(tmp_path, monkeypatch):
    """Not merely defined -- actually called, or retention never happens."""
    src = APP_SOURCE.read_text()
    tree = ast.parse(src)
    main_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "main")
    called = {
        node.func.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "prune_old_telemetry" in called, "the retention sweep is never called from main()"
    assert "write_telemetry_log" in called


def test_the_defaults_are_finite_and_sane():
    """Unbounded is the defect; assert the shipped defaults are not."""
    assert 0 < tm.TELEMETRY_MAX_BYTES <= 512 * 1024 * 1024
    assert 1 <= tm.TELEMETRY_MAX_GENERATIONS <= 32
    assert 0 < tm.TELEMETRY_RETENTION_DAYS <= 365
    assert tm.PRUNE_INTERVAL_S > 0


def test_the_env_vars_are_documented_where_they_are_set():
    """The two-step compose env contract: .env.example names what compose needs."""
    example = Path(__file__).resolve().parent.parent / ".env.example"
    if not example.exists():  # a stripped checkout; the assertion above still holds
        pytest.skip("no .env.example in this checkout")
    text = example.read_text()
    for name in (
        "TELEMETRY_MAX_BYTES",
        "TELEMETRY_MAX_GENERATIONS",
        "TELEMETRY_RETENTION_DAYS",
        "TELEMETRY_PRUNE_INTERVAL_S",
    ):
        assert name in text, f"{name} is read from the environment but not in .env.example"
