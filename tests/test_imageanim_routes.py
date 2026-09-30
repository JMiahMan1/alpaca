"""The /api/image/animate route.

The interesting part of this endpoint is not that it renders - imageanim's own
87 tests cover that - but that it refuses cheaply and refuses specifically. Every
knob can arrive from a browser panel, a hand-written curl, or a Raven tool, and
this box already had a service OOM-killed. So the tests below lean on three
things:

* a rejected request must cost nothing, so every limit is checked before the
  first frame is rendered (asserted by pointing ARTIFACTS_DIR at a path that
  does not exist and confirming a rejected request still 400s rather than
  blowing up trying to write there);
* the errors must name the field, because "invalid input" tells the person
  driving the panel nothing about which box to fix;
* an artifact name is a path handed to PIL.open, so traversal and symlinks have
  to be refused, not just "../" spelled literally.
"""

import base64
import io
import struct
import wave
from pathlib import Path

import pytest
from PIL import Image

from web.app import app, benchmark


def _still(width=320, height=240, seed=0):
    """A still with enough structure to pan over and crop.

    A flat colour compresses to one frame and has nothing to move across, which
    would make several assertions here vacuous.
    """
    img = Image.new("RGB", (width, height))
    px = img.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = ((x * 7 + seed) % 256, (y * 5) % 256, (x + y + seed) % 256)
    return img


def _png_bytes(img=None, **kw):
    """PNG bytes.

    NOTE: this returns raw bytes, which werkzeug puts in `form` rather than
    `files` -- see _upload(). Passing (bytes, name) as a data value does NOT
    create a file part; only a real file object does.
    """
    buf = io.BytesIO()
    (img or _still(**kw)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(raw, name="a.png"):
    """A werkzeug file part. Must be a file object, not bytes.

    Measured, not assumed: EnvironBuilder with data={"image": (b"...", "a.png")}
    yields files=[] and form=['image'] -- the bytes become a form *string*, so
    the handler legitimately sees no upload at all. (io.BytesIO(b"..."), name)
    yields files=['image'].
    """
    return io.BytesIO(raw), name


def _data_uri(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _webp_frame_count(raw):
    """Count ANMF chunks without trusting Pillow's encoder."""
    n, off = 0, 12
    while off + 8 <= len(raw):
        fourcc = raw[off : off + 4]
        size = struct.unpack("<I", raw[off + 4 : off + 8])[0]
        if fourcc == b"ANMF":
            n += 1
        off += 8 + size + (size & 1)
    return n


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client whose artifact writes land in a tmp dir, never the real one."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", artifacts)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


# ── GET: what it will accept ────────────────────────────────────────────────


def test_status_lists_the_four_kinds_with_honest_labels(client):
    body = client.get("/api/image/animate").get_json()
    assert [k["id"] for k in body["kinds"]] == ["ken_burns", "parallax", "crossfade", "sprite"]
    assert body["formats"] == ["webp", "gif"]
    assert body["default_format"] == "webp"


def test_status_states_the_image_count_each_kind_needs(client):
    """The panel must not offer crossfade with a single input slot."""
    kinds = {k["id"]: k for k in client.get("/api/image/animate").get_json()["kinds"]}
    assert kinds["ken_burns"]["needs_images"] == 1
    assert kinds["crossfade"]["needs_images"] == 2
    assert kinds["sprite"]["needs_images"] == 1
    for kind in kinds.values():
        assert kind["note"], f"{kind['id']} has no explanation"


def test_status_publishes_the_limits_the_route_enforces(client):
    body = client.get("/api/image/animate").get_json()
    assert body["limits"]["max_dimension"] == 2048
    assert body["limits"]["max_frames"] == 240
    assert body["limits"]["max_total_pixels"] == 150_000_000


def test_status_defaults_match_what_an_unadorned_request_produces(client):
    body = client.get("/api/image/animate").get_json()
    resp = client.post("/api/image/animate", data={"image": _upload(_png_bytes(), "a.png")})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    got = resp.get_json()
    assert got["size"] == body["defaults"]["size"]
    assert got["frames"] == body["defaults"]["frames"]
    assert got["duration_ms"] == body["defaults"]["duration_ms"]
    assert got["format"] == body["default_format"]


# ── POST: the happy paths ───────────────────────────────────────────────────


def test_an_uploaded_still_becomes_a_real_animated_webp(client):
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "256x192", "frames": "16"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["ok"] is True
    assert body["kind"] == "ken_burns"
    assert (body["size"], body["frames"]) == ([256, 192], 16)
    assert body["total_ms"] == body["frames"] * body["duration_ms"]

    saved = (Path(benchmark.ARTIFACTS_DIR) / body["filename"]).read_bytes()
    assert saved[:4] == b"RIFF" and saved[8:12] == b"WEBP"
    # Pillow's lossy encoder drops one of 16 near-identical ken_burns frames as a
    # duplicate, so the encoded count can be one under what was rendered. Assert
    # "a real animation of the right order", not an exact equality.
    encoded = _webp_frame_count(saved)
    assert 15 <= encoded <= 16, f"expected ~16 frames, container has {encoded}"
    assert Image.open(io.BytesIO(saved)).n_frames >= 2, "Pillow cannot play it back"


def test_the_saved_file_is_the_same_bytes_the_data_uri_carries(client):
    """Two copies that drift would mean the preview is not what gets downloaded."""
    body = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "192x144", "frames": "12", "format": "gif"},
    ).get_json()
    saved = (Path(benchmark.ARTIFACTS_DIR) / body["filename"]).read_bytes()
    assert body["data_uri"].endswith(base64.b64encode(saved).decode())


def test_download_url_resolves_through_the_existing_artifact_route(client):
    """The URL is only useful if GET /api/artifacts/<name> actually serves it."""
    body = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "160x160", "frames": "8"},
    ).get_json()
    got = client.get(body["download_url"])
    assert got.status_code == 200
    assert got.headers["Content-Type"] == "image/webp"


def test_gif_is_renderable_and_larger_than_webp_for_the_same_frames(client):
    """WebP is the default for a reason; pin the reason so it stays a choice."""

    def render(fmt):
        # A fresh BytesIO per request: one file object shared across two posts is
        # already consumed by the first, and werkzeug fails on the second with
        # "I/O operation on closed file".
        return client.post(
            "/api/image/animate",
            data={"image": _upload(_png_bytes(), "a.png"), "size": "200x150", "frames": "12", "format": fmt},
        ).get_json()

    webp, gif = render("webp"), render("gif")
    assert webp["format"] == "webp" and gif["format"] == "gif"
    assert gif["data_uri"].startswith("data:image/gif")
    assert gif["bytes"] > webp["bytes"]


def test_an_artifact_name_is_accepted_so_the_gallery_can_reuse_a_generated_image(client, tmp_path):
    (Path(benchmark.ARTIFACTS_DIR) / "sd-out-7.png").write_bytes(_png_bytes())
    body = client.post(
        "/api/image/animate",
        data={"artifact": "sd-out-7.png", "size": "128x128", "frames": "6"},
    ).get_json()
    assert body["frames"] == 6


def test_a_data_uri_is_accepted_because_the_gallery_already_holds_one(client):
    body = client.post(
        "/api/image/animate",
        data={"artifact_b64": _data_uri(_png_bytes()), "size": "128x128", "frames": "6"},
    ).get_json()
    assert body["frames"] == 6


def test_bare_base64_is_accepted_too(client):
    raw = base64.b64encode(_png_bytes()).decode()
    resp = client.post(
        "/api/image/animate", data={"artifact_b64": raw, "size": "128x128", "frames": "6"}
    )
    assert resp.status_code == 200


def test_crossfade_needs_two_stills_and_takes_them(client):
    two = client.post(
        "/api/image/animate",
        data={
            "image": _upload(_png_bytes(seed=1), "a.png"),
            "images": _upload(_png_bytes(seed=9), "b.png"),
            "kind": "crossfade",
            "size": "128x128",
            "frames": "10",
        },
    )
    assert two.status_code == 200, two.get_data(as_text=True)
    assert two.get_json()["frames"] == 10


def test_the_sprite_kind_produces_a_flipbook_from_one_sheet(client):
    # Distinct content per cell: _still alone repeats vertically every 100px, which
    # made 6 of the 12 frames identical and lossy WebP dropped them as duplicates.
    sheet = Image.new("RGB", (400, 300))
    for row in range(3):
        for col in range(4):
            cell = _still(100, 100, seed=row * 10 + col)
            cell.paste(_still(50, 50, seed=90 + col), (0, 0))  # break the repeat
            sheet.paste(cell, (col * 100, row * 100))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")

    resp = client.post(
        "/api/image/animate",
        data={
            "image": _upload(buf.getvalue(), "sheet.png"),
            "kind": "sprite",
            "sprite_cols": "4",
            "sprite_rows": "3",
            "size": "80x80",
            "frames": "12",
        },
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    saved = (Path(benchmark.ARTIFACTS_DIR) / resp.get_json()["filename"]).read_bytes()
    assert _webp_frame_count(saved) >= 6, "the sheet did not become a flipbook"
    assert Image.open(io.BytesIO(saved)).n_frames >= 2


# ── rejection: specific, named, and free ────────────────────────────────────


@pytest.mark.parametrize(
    "field,value,expected_in_message",
    [
        ("kind", "morph", "kind must be one of"),
        ("format", "mp4", "format must be one of"),
        ("size", "9999x9999", "must not exceed"),
        ("size", "8x8", "at least 16x16"),
        ("frames", "1", "at least 2"),
        ("frames", "100000", "must not exceed 240"),
        ("frames", "12.5", "must be a number"),
        ("size", "abc", "size"),
    ],
)
def test_each_bad_knob_is_a_400_that_names_the_field(client, field, value, expected_in_message):
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "128x128", "frames": "8", field: value},
    )
    assert resp.status_code == 400, resp.get_data(as_text=True)
    assert expected_in_message in resp.get_json()["error"]


def test_the_megapixel_budget_is_checked_and_explained(client):
    """frames x size is the only way to ask for gigabytes, so it gets its own check."""
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "2048x2048", "frames": "240"},
    )
    assert resp.status_code == 400
    error = resp.get_json()["error"]
    assert "megapixel" in error and "Fewer frames" in error


def test_a_rejected_request_writes_nothing(client, monkeypatch, tmp_path):
    """Cheap rejection is the whole point of validating first."""
    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", tmp_path / "does-not-exist")
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "99999x1"},
    )
    assert resp.status_code == 400
    assert not (tmp_path / "does-not-exist").exists()


def test_no_source_image_says_so_and_says_how(client):
    resp = client.post("/api/image/animate", data={"size": "128x128"})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "no source image"
    assert "artifact_b64" in body["hint"]


def test_crossfade_with_one_still_says_how_many_it_wanted(client):
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "kind": "crossfade"},
    )
    assert resp.status_code == 400
    assert "needs 2 or more stills" in resp.get_json()["error"]


def test_sprite_without_a_grid_is_refused_rather_than_guessed(client):
    resp = client.post(
        "/api/image/animate", data={"image": _upload(_png_bytes(), "a.png"), "kind": "sprite"}
    )
    assert resp.status_code == 400
    assert "sprite_cols" in resp.get_json()["error"]


def test_sprite_with_two_sheets_is_refused(client):
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "images": _upload(_png_bytes(seed=2), "b.png"), "kind": "sprite",
              "sprite_cols": "2", "sprite_rows": "2"},
    )
    assert resp.status_code == 400
    assert "exactly 1 sheet" in resp.get_json()["error"]


def test_a_file_that_is_not_an_image_is_a_400_not_a_500(client):
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(b"this is not a png"), "size": "128x128", "frames": "6"},
    )
    assert resp.status_code == 400
    assert "Pillow" in resp.get_json()["error"]


def test_bad_base64_is_refused_by_name(client):
    resp = client.post(
        "/api/image/animate",
        data={"artifact_b64": "not base64 at all !!!", "size": "128x128"},
    )
    assert resp.status_code == 400
    assert "artifact_b64" in resp.get_json()["error"]


# ── the artifact name is a path handed to PIL.open ──────────────────────────


@pytest.mark.parametrize("name", ["../../etc/passwd", "sub/dir.png", "..", "/etc/hostname"])
def test_an_artifact_name_cannot_escape_the_artifacts_directory(client, name):
    resp = client.post(
        "/api/image/animate", data={"artifact": name, "size": "128x128", "frames": "6"}
    )
    assert resp.status_code == 400, resp.get_data(as_text=True)
    assert "no artifact named" in resp.get_json()["error"] or "outside" in resp.get_json()["error"]


def test_a_symlink_pointing_out_of_the_artifacts_directory_is_refused(client, tmp_path):
    """os.path.basename alone does not stop this: the name is clean, the target is not."""
    outside = tmp_path / "secret.png"
    outside.write_bytes(_png_bytes())
    (Path(benchmark.ARTIFACTS_DIR) / "innocent.png").symlink_to(outside)

    resp = client.post(
        "/api/image/animate", data={"artifact": "innocent.png", "size": "128x128", "frames": "6"}
    )
    assert resp.status_code == 400
    assert "outside" in resp.get_json()["error"]


def test_a_missing_artifact_says_which_name(client):
    resp = client.post(
        "/api/image/animate", data={"artifact": "nope.png", "size": "128x128"}
    )
    assert resp.status_code == 400
    assert "nope.png" in resp.get_json()["error"]


# ── the knobs actually reach the renderer ────────────────────────────────────


def test_zoom_and_pan_arrive_at_the_renderer(client, monkeypatch):
    """The route passing the right numbers is the contract; the module's own
    tests cover what those numbers do."""
    seen = {}
    real_build = None

    import imageanim

    def spy(stills, spec):
        seen.update(zoom_from=spec.zoom_from, zoom_to=spec.zoom_to, pan_x=spec.pan_x, duration_ms=spec.duration_ms)
        return real_build(stills, spec)

    real_build = imageanim.build_animation
    monkeypatch.setattr(imageanim, "build_animation", spy)

    client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "128x128", "frames": "6",
              "zoom_from": "1.0", "zoom_to": "2.5", "pan_x": "-0.4", "duration_ms": "250"},
    )
    assert seen == {"zoom_from": 1.0, "zoom_to": 2.5, "pan_x": -0.4, "duration_ms": 250}


def test_a_size_can_be_given_as_separate_width_and_height(client):
    body = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "width": "150", "height": "110", "frames": "6"},
    ).get_json()
    assert body["size"] == [150, 110]


def test_a_bare_number_size_is_square(client):
    body = client.post(
        "/api/image/animate",
        data={"image": _upload(_png_bytes(), "a.png"), "size": "144", "frames": "6"},
    ).get_json()
    assert body["size"] == [144, 144]


def test_a_json_body_works_as_well_as_multipart(client):
    """A Raven tool sends JSON; a browser form sends multipart. Both must work."""
    resp = client.post(
        "/api/image/animate",
        json={"artifact_b64": _data_uri(_png_bytes()), "size": "128x128", "frames": "6"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.get_json()["frames"] == 6


def test_a_bodyless_post_is_a_named_400_not_a_500(client):
    """This route reads request.form first and falls back to JSON, so unlike the
    dominant request.get_json() or {} convention here it does not 415 on an
    empty body - it reaches the handler and reports the missing input by name."""
    resp = client.post("/api/image/animate")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "no source image"


def test_get_is_the_status_route_and_post_is_the_render_route(client):
    """Both verbs share the path, so they must not shadow each other."""
    assert client.get("/api/image/animate").status_code == 200
    assert client.post("/api/image/animate").status_code == 400


def test_a_wav_upload_is_refused_rather_than_half_rendered(client):
    """A wrong file type is the most likely real mistake; it must not be a 500."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 100)
    resp = client.post(
        "/api/image/animate",
        data={"image": _upload(buf.getvalue(), "clip.wav"), "size": "128x128", "frames": "6"},
    )
    assert resp.status_code == 400
