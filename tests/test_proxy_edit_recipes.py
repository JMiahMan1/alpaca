"""Photo-edit recipes: one base photo, an optional reference, and a named change.

The feature this replaces asked for "scene first, people second, face identity
third" and refused to run on anything but one model. Both halves were wrong.

The ordering was wrong because the thing people actually have is a photo they
like and a list of things they want different about it. So a request is now a
base photo, zero or more references, and one named change -- and adding a new
kind of change is one entry in _EDIT_RECIPES, not a new branch in a validator.

The model gate was wrong because "does this engine take several reference images"
is a property of the engine, not of a model's file name. It lives in
_REFERENCE_IMAGE_FAMILIES as data, and model_capabilities() reports it so the
UI can ask before it offers an option rather than discovering it from a 400.

These tests pin the request contract, the capability data, the honest refusals,
and the one thing that is easy to get wrong and invisible when it is: the roles
have to reach the engine as words, because the engine only ever sees an ordered
list of images.
"""

import importlib.util
import json
import os
import pathlib
import tempfile

import pytest

with tempfile.TemporaryDirectory() as tmpdir:
    os.environ["GRAMMAR_REGISTRY_DIR"] = os.path.join(tmpdir, "grammars")
    os.environ["SCHEMA_REGISTRY_DIR"] = os.path.join(tmpdir, "schemas")
    SPEC = importlib.util.spec_from_file_location(
        "alpaca_proxy_recipes", pathlib.Path(__file__).resolve().parent.parent / "alpaca-proxy.py"
    )
    proxy = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(proxy)

QWEN = "Qwen-Image-2.1-GGUF/qwen_image_2.1"
SDXL = "stable-diffusion-xl-base-1.0"


def apply(preset, images=1, roles=None, model=QWEN, prompt="a photo"):
    data = {"prompt": prompt, "preset": preset}
    if roles is not None:
        data["reference_roles"] = json.dumps(roles)
    return proxy._apply_edit_preset(data, images, model)


def ok(preset, **kw):
    """Apply and assert it succeeded, returning the (mutated data, meta)."""
    data, meta = apply(preset, **kw)
    # meta["recipe"] carries the human label; the change is the canonical role.
    assert meta.get("recipe") == proxy._EDIT_RECIPES[preset]["label"], (
        f"expected {preset!r} to apply, got recipe={meta.get('recipe')!r} change={meta.get('change')!r}"
    )
    assert meta.get("change") == proxy._EDIT_RECIPES[preset]["change"]
    return data, meta


def refuses(preset, needle, **kw):
    with pytest.raises(ValueError) as exc:
        apply(preset, **kw)
    message = str(exc.value)
    assert needle.lower() in message.lower(), f"refusal did not mention {needle!r}: {message}"
    return message


# --------------------------------------------------------------------------
# The catalogue. The UI consumes this, so its shape is a contract.
# --------------------------------------------------------------------------


def test_the_catalogue_lists_four_changes_plus_keep_the_face():
    ids = [r["id"] for r in proxy._edit_recipe_summary()]
    assert ids == ["edit.face", "edit.outfit", "edit.hair", "edit.background", "edit.identity"]


def test_every_catalogue_entry_carries_what_a_panel_needs_to_render_it():
    for row in proxy._edit_recipe_summary():
        for key in ("id", "label", "change", "expects", "needs_reference", "min_images", "max_images"):
            assert key in row, f"{row.get('id')} is missing {key}"
        assert row["label"], f"{row['id']} has no human label"
        assert row["min_images"] >= 1, f"{row['id']} should be usable from a single photo"


def test_the_catalogue_is_model_agnostic():
    """The whole point: nothing in the catalogue names a model."""
    text = json.dumps(proxy._edit_recipe_summary()).lower()
    for banned in ("qwen", "sdxl", "flux", "stable-diffusion", "2.1"):
        assert banned not in text, f"the catalogue still names a model ({banned!r})"


def test_identity_is_the_one_change_that_needs_no_reference_role():
    by_id = {r["id"]: r for r in proxy._edit_recipe_summary()}
    assert by_id["edit.identity"]["expects"] is None
    assert by_id["edit.background"]["expects"] == "scene"


# --------------------------------------------------------------------------
# Capabilities: data, so a new instruct model is one line rather than a branch.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "family", "refs", "neg"),
    [
        ("Qwen-Image-2.1-GGUF/qwen_image_2.1", "qwen-image", True, False),
        ("Qwen-Image-Edit-Rapid-AIO:q4_k", "qwen-image", True, False),
        # _model_family checks "stable-diffusion" before "sdxl", and it IS a substring
        # of this name -- so that is the honest answer, not sdxl.
        ("stable-diffusion-xl-base-1.0", "stable-diffusion", False, True),
        ("sdxl-turbo", "sdxl", False, True),
        ("flux-dev", "flux", False, True),
    ],
)
def test_capabilities_are_derived_from_the_family_not_the_exact_name(name, family, refs, neg):
    caps = proxy.model_capabilities(name)
    assert caps["family"] == family
    assert caps["reference_images"] is refs
    assert caps["negative_prompt"] is neg
    assert caps["max_reference_images"] == (4 if refs else 0)
    assert caps["model"] == name


def test_capabilities_degrade_to_unknown_rather_than_guessing():
    caps = proxy.model_capabilities("some-model-nobody-has-heard-of")
    assert caps["family"] == "unknown"
    assert caps["reference_images"] is False
    assert caps["max_reference_images"] == 0


def test_a_new_reference_family_is_one_line_of_data():
    """Adding a family must not require touching a request handler.

    This is the property that makes the feature model-agnostic rather than
    merely model-agnostic today.
    """
    assert isinstance(proxy._REFERENCE_IMAGE_FAMILIES, frozenset)
    before = proxy._REFERENCE_IMAGE_FAMILIES
    with pytest.MonkeyPatch.context() as mp:
        # A frozenset is immutable, so the only way to add a family is to replace
        # the attribute. That is exactly what adding a model means in practice.
        mp.setattr(
            proxy,
            "_REFERENCE_IMAGE_FAMILIES",
            before | {"some-new-instruct-family"},
        )
        mp.setattr(proxy, "_model_family", lambda name: "some-new-instruct-family")
        _data, meta = apply("edit.face", images=2, roles=["photo", "face"], model="whatever")
        assert meta["reference_images_supported"] is True


# --------------------------------------------------------------------------
# The base photo. One image, and it is the thing being edited.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    ["edit.face", "edit.outfit", "edit.hair", "edit.background", "edit.identity"],
)
def test_one_photo_and_no_reference_is_enough_for_every_change(preset):
    """Text-only is the common case, and it must not require a second image."""
    data, meta = ok(preset, images=1)
    assert meta["reference_roles"] == ["scene"]
    assert meta["reference_count"] == 0
    assert len(data["prompt"]) > 100, "the recipe's own instruction should reach the engine"


def test_a_photo_with_no_roles_still_applies():
    """Most callers will not send reference_roles at all. That must work."""
    _data, meta = ok("edit.outfit", images=1, roles=None)
    assert meta["reference_roles"] == ["scene"]


def test_the_first_image_is_the_photo_being_edited_whatever_it_is_called():
    """A caller who labels the base "person" still has a base photo."""
    _data, meta = ok("edit.outfit", images=2, roles=["person", "clothing"])
    assert meta["reference_roles"] == ["scene", "outfit"]
    assert meta["reference_count"] == 1


def test_a_scene_shaped_reference_is_legal_because_background_asks_for_one():
    """Both canonicalise to "scene"; the base is positional, not name-matched."""
    _data, meta = ok("edit.background", images=2, roles=["photo", "place"])
    assert meta["reference_roles"] == ["scene", "scene"]
    assert meta["reference_count"] == 1


def test_no_photo_at_all_is_refused_with_an_actionable_message():
    refuses("edit.face", "needs the photo to edit", images=0)


def test_a_role_count_that_disagrees_with_the_image_count_is_refused():
    refuses("edit.outfit", "one role for each image", images=3, roles=["photo", "outfit"])


def test_an_unknown_role_is_named_in_the_refusal():
    refuses("edit.face", "banana", images=2, roles=["photo", "banana"])


@pytest.mark.parametrize(
    ("preset", "word"),
    [
        ("edit.face", "place"),
        ("edit.hair", "photo"),
        ("edit.outfit", "scene"),
    ],
)
def test_a_second_scene_is_refused_because_a_background_is_not_a_face_reference(preset, word):
    """The guard _parse_reference_roles cannot do.

    "place" and "scene" are perfectly valid role words -- they canonicalise to
    "scene" -- so the vocabulary check passes. What makes them wrong here is that
    the *base* is already the scene: attaching a second one to a face change is
    a request whose meaning is not obvious, so it is refused by name.
    """
    refuses(preset, "unsupported reference", images=2, roles=["photo", word])


@pytest.mark.parametrize(
    ("preset", "word"),
    [
        ("edit.face", "hair"),
        ("edit.hair", "face"),
        ("edit.outfit", "style"),
        ("edit.background", "face"),
    ],
)
def test_a_reference_that_is_a_real_role_but_meaningless_here_is_still_allowed(preset, word):
    """Deliberately permissive.

    Someone changing the face may also hand over a hair photo because the face
    is partly occluded by it. Refusing would be the unintuitive behaviour this
    feature exists to remove, so the only refused case is a second base photo.
    """
    _data, meta = ok(preset, images=2, roles=["photo", word])
    assert meta["reference_count"] == 1


# --------------------------------------------------------------------------
# Role vocabulary. These are the words a person actually types.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("face", "face"),
        ("identity", "face"),
        ("outfit", "outfit"),
        ("clothing", "outfit"),
        ("wardrobe", "outfit"),
        ("costume", "outfit"),
        ("hair", "hair"),
        ("hairstyle", "hair"),
        ("haircut", "hair"),
        ("place", "scene"),
        ("backdrop", "scene"),
        ("photo", "scene"),
        ("base", "scene"),
    ],
)
def test_the_words_a_person_types_reach_the_engine_as_canonical_roles(word, expected):
    recipe = "edit.background" if expected == "scene" else f"edit.{'hair' if expected == 'hair' else expected}"
    _data, meta = ok(recipe, images=2, roles=["photo", word])
    assert meta["reference_roles"][1] == expected


def test_a_bare_reference_with_no_base_is_still_the_base():
    """One image, one role: it is the photo, not a reference."""
    _data, meta = ok("edit.hair", images=1, roles=["hair"])
    assert meta["reference_roles"] == ["scene"]
    assert meta["reference_count"] == 0


# --------------------------------------------------------------------------
# The refusal that matters: a model that cannot read references says so.
# --------------------------------------------------------------------------


def test_references_on_a_single_image_model_are_refused_by_name_and_with_an_alternative():
    message = refuses(
        "edit.face", SDXL, images=2, roles=["photo", "face"], model=SDXL, prompt="x"
    )
    assert "reference image" in message.lower()
    assert "describe the change in words" in message.lower(), "the refusal must offer the alternative"


def test_the_same_change_without_a_reference_is_fine_on_a_single_image_model():
    """The refusal is about references, not about the change being impossible."""
    _data, meta = ok("edit.outfit", images=1, model=SDXL)
    assert meta["reference_images_supported"] is False
    assert meta["reference_count"] == 0


def test_two_references_on_one_model_are_still_refused():
    refuses("edit.face", "cannot use 2 reference", images=3, roles=["photo", "face", "face"], model=SDXL)


# --------------------------------------------------------------------------
# Roles have to reach the engine as words.
# --------------------------------------------------------------------------


def test_every_reference_gets_a_caption_naming_its_position():
    """The engine sees an ordered list of images and nothing else.

    Without a caption per image the model has three pictures and no idea which
    is the person whose face it should keep, which is the usual reason a
    reference image is reported as ignored.
    """
    data, _ = ok("edit.face", images=3, roles=["photo", "face", "face"])
    prompt = data["prompt"]
    assert "<image2> is the face reference to follow." in prompt
    assert "<image3> is the face reference to follow." in prompt


def test_a_background_reference_is_captioned_as_a_place_not_a_face():
    # The caption is built from the recipe label, so the scene role reads as
    # "background / place" -- the words a person would use.
    data, _ = ok("edit.background", images=2, roles=["photo", "place"])
    assert "<image2> is the background / place reference to follow." in data["prompt"]


def test_a_reference_is_not_captioned_when_there_is_none():
    data, _ = ok("edit.identity", images=1)
    assert "reference to follow" not in data["prompt"]


def test_the_user_prompt_is_kept_and_the_instruction_comes_first():
    data, _ = ok("edit.outfit", images=1, prompt="a red wool coat")
    assert "a red wool coat" in data["prompt"]
    assert data["prompt"].rstrip().endswith("a red wool coat"), "the user's words must be last, so they win"


def test_every_recipe_instruction_says_what_must_not_change():
    """A change instruction that does not protect the rest invites collateral edits."""
    for key, recipe in proxy._EDIT_RECIPES.items():
        instruction = recipe["instruction"].lower()
        assert "keep" in instruction or "unchanged" in instruction, f"{key} does not protect anything"
        assert "<image1>" in recipe["instruction"], f"{key} never names the photo it is editing"


def test_negative_prompt_is_forced_empty_because_instruct_models_ignore_it():
    data, _ = ok("edit.face", images=1)
    assert data["negative_prompt"] == ""


# --------------------------------------------------------------------------
# The legacy recipe is frozen and still works.
# --------------------------------------------------------------------------


def test_the_legacy_qwen_identity_recipe_still_applies():
    data, meta = proxy._apply_edit_preset(
        {"prompt": "x", "preset": "qwen_image_21.identity", "reference_roles": json.dumps(["scene", "person", "face"])},
        3,
        QWEN,
    )
    assert meta.get("reference_roles") == ["scene", "person", "face"]
    assert data["negative_prompt"] == ""


def test_the_legacy_recipe_is_still_model_gated():
    """It is frozen, not relaxed. It demanded three specific roles and still does."""
    with pytest.raises(ValueError):
        proxy._apply_edit_preset(
            {"prompt": "x", "preset": "qwen_image_21.identity", "reference_roles": json.dumps(["scene", "face"])},
            2,
            QWEN,
        )


def test_the_new_recipes_needs_only_one_image_where_the_legacy_needed_three():
    """The concrete improvement, as an assertion rather than a claim."""
    ok("edit.face", images=1)
    with pytest.raises(ValueError):
        proxy._apply_edit_preset(
            {"prompt": "x", "preset": "qwen_image_21.identity", "reference_roles": json.dumps(["scene"])},
            1,
            QWEN,
        )


def test_an_unknown_preset_still_falls_through_to_the_preset_table():
    refuses("edit.does-not-exist", "unknown image preset", images=1)


def test_no_preset_at_all_is_a_pass_through():
    data, meta = proxy._apply_edit_preset({"prompt": "x"}, 1, QWEN)
    assert meta == {}
    assert data == {"prompt": "x"}



def _repo_file(*parts):
    return pathlib.Path(__file__).resolve().parent.parent.joinpath(*parts)


def test_the_recipe_default_size_fits_the_card():
    """The default must be a size the VAE encode can actually allocate.

    1024x1024 needs ~8315 MB to encode the init image, which is more than the
    8188 MiB an RTX 4060 has. sd-server reports that as "generate_image returned
    no results" and only its container log mentions memory, so the real cause is
    invisible from the dashboard -- which is exactly why the default is pinned.
    """
    assert proxy._EDIT_RECIPE_DEFAULTS["size"] == "768x768", (
        f"the recipe default size must fit 8 GB of VRAM; got {proxy._EDIT_RECIPE_DEFAULTS['size']!r}"
    )


def test_the_default_size_is_the_same_everywhere_it_is_written():
    """The proxy, the panel's size field and the JS fallback must agree.

    Three separate places spell this default, and the browser will happily POST
    a size the proxy was never validated against. Changing one and forgetting the
    other two is the bug, so all three are pinned together.
    """
    template = _repo_file("web", "templates", "index.html").read_text()
    assert 'id="sd-recipe-size" value="768x768"' in template, (
        "the panel's size default disagrees with the proxy's"
    )

    js = _repo_file("web", "static", "js", "dashboard.js").read_text()
    line = next((s for s in js.splitlines() if "getElementById('sd-recipe-size')" in s), None)
    assert line is not None, "the recipe submit handler no longer reads sd-recipe-size"
    assert "'768x768'" in line, f"the JS fallback disagrees with the proxy default: {line.strip()}"

    assert proxy._EDIT_RECIPE_DEFAULTS["size"] == "768x768"
