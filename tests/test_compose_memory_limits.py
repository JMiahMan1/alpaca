"""docker-compose memory limits: stated, in sync, and large enough to matter.

Written after I got a compose fact wrong. An earlier commit claimed
`docker compose` outside Swarm ignores `deploy.resources.limits` and that only
the top-level `mem_limit:` works. The evidence was a container inspected at a
moment when it carried no limit at all, which I read as "the deploy key was
ignored". It was not: the container predated the `mem_limit:` key and its limit
was exactly the `31G` declared under `deploy.resources.limits.memory`. So
`deploy:` is honoured on this host.

Two things are pinned here so the record cannot drift again:

* whatever cap is intended, it is *stated* somewhere in the compose file. A cap
  of 31G on a 30 GiB box is no cap at all, and that is the failure this guards.
* when both keys are written for a service, they *agree*. They are redundant by
  design, and redundant config that disagrees is worse than one of them.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = REPO_ROOT / "docker-compose.yml"

# Anything at or above this on a 30 GiB host is indistinguishable from no cap.
MEANINGFUL_CEILING_BYTES = 24 * 1024**3

# Services whose steady-state footprint is a large fraction of the box, so a
# missing cap is a real hazard rather than a theoretical one.
WATCHED = ("sd-server", "alpaca-web", "llama-server", "audio-server", "alpaca-proxy")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


@pytest.fixture(scope="module")
def services(compose) -> dict:
    assert "services" in compose, "compose file has no `services:` key"
    return compose["services"]


def _declared_caps(service: dict) -> dict[str, str]:
    """Every memory cap a service declares, keyed by which spelling used it."""
    caps: dict[str, str] = {}
    if "mem_limit" in service:
        caps["mem_limit"] = str(service["mem_limit"])
    deploy = (service.get("deploy") or {}).get("resources") or {}
    if "limits" in deploy and "memory" in deploy["limits"]:
        caps["deploy.resources.limits.memory"] = str(deploy["limits"]["memory"])
    return caps


def _as_bytes(value: str) -> int:
    """Parse a compose byte string ('16g', '512m', '1073741824')."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmg]?)b?\s*", str(value), re.IGNORECASE)
    assert m, f"cannot parse a memory value: {value!r}"
    scale = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[m.group(2).lower()]
    return int(float(m.group(1)) * scale)


@pytest.mark.parametrize("name", WATCHED)
def test_every_big_service_states_a_memory_cap(services, name):
    """A service with no cap can take the whole box, which is how it went before."""
    assert name in services, f"{name} is not in the compose file"
    caps = _declared_caps(services[name])
    assert caps, f"{name} declares no memory cap at all"


@pytest.mark.parametrize("name", WATCHED)
def test_no_declared_cap_exceeds_the_box(services, name):
    """The specific mistake: 31 GiB on a 30 GiB host reads as a cap and is not one."""
    for key, raw in _declared_caps(services[name]).items():
        assert _as_bytes(raw) < MEANINGFUL_CEILING_BYTES, (
            f"{name} declares {raw} under {key}, which is at or above this box's "
            "usable memory -- that is not a cap, it is a comment"
        )


def test_where_both_spellings_are_present_they_agree(services):
    """Redundant by design, so they must not drift apart."""
    both = {name: _declared_caps(svc) for name, svc in services.items() if len(_declared_caps(svc)) > 1}
    assert both, "expected at least one service to declare both spellings"
    for name, caps in both.items():
        sizes = {_as_bytes(v) for v in caps.values()}
        assert len(sizes) == 1, (
            f"{name} declares disagreeing memory caps: {caps}. Whichever key a given "
            "compose version reads, the answer would depend on which one it picked."
        )


def test_sd_server_cap_leaves_room_for_the_rest_of_the_stack(services):
    """sd-server is the single biggest consumer; the cap is a blast radius, not a squeeze.

    Measured steady state with Qwen-Image-2.1 loaded is ~12.6 GiB on a 30 GiB
    box shared with llama-server and the dashboard.
    """
    cap = _as_bytes(_declared_caps(services["sd-server"])["mem_limit"])
    assert cap >= 14 * 1024**3, "sd-server's cap would OOM-kill a normally-loaded Qwen-Image"
    assert cap <= 20 * 1024**3, "sd-server's cap leaves too little headroom for a larger SD model"


def test_the_compose_comment_does_not_assert_a_compose_behaviour_it_has_not_proven(compose):
    """The comment is the thing that misled me. Keep it honest.

    A comment claiming compose 'silently ignores' deploy limits, with no evidence
    recorded, is how a wrong fact survives in a repo for years.
    """
    text = COMPOSE.read_text()
    forbidden = [
        "silently ignores `deploy:`",
        "silently ignores deploy:",
        "the top-level key is the one that works",
    ]
    for phrase in forbidden:
        assert phrase not in text, (
            f"docker-compose.yml still asserts {phrase!r}. On this host `deploy:` IS honoured: a "
            "container created before `mem_limit:` existed was inspected and found carrying exactly "
            "the value declared under deploy.resources.limits.memory."
        )


def test_the_comment_records_how_to_verify_the_cap(services):
    """Both keys are inert until the container is recreated; say so where it is edited."""
    text = COMPOSE.read_text()
    assert "docker inspect" in text and "HostConfig.Memory" in text, (
        "the compose file should say how to verify the cap actually took effect"
    )
