#!/usr/bin/env python3
"""Id-contract check: HTML ids referenced by dashboard.js must exist in index.html.

Catches template/JS contract drift early (e.g. a JS edit referencing ids that only
exist in an unrestarted container's stale template). Exit 1 on missing ids.

Usage:
    python scripts/check_id_contract.py                 # check repo files
    python scripts/check_id_contract.py --served        # check live server output
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

JS_GETTERS = re.compile(
    r"""(?:getElementById|querySelector)\(\s*['"]#?([A-Za-z][\w-]{2,60})['"]\s*\)"""
)
ID_DEF = re.compile(r"""\bid\s*=\s*["']([A-Za-z][\w-]{2,60})["']""")
DYNAMIC_OK = re.compile(r"^(doc-|general-lb-|shared-lb-)")

# Ids looked up defensively with null-guards (optional element that may be absent).
OPTIONAL_GUARDED = {"sd-model-type-badge", "sd-photo-edit-prompt"}


def collect_js_ids(js_text: str) -> set[str]:
    return {m.group(1) for m in JS_GETTERS.finditer(js_text)}


def collect_html_ids(html_text: str) -> set[str]:
    return set(ID_DEF.findall(html_text))


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", action="store_true", help="check live server at --url")
    ap.add_argument("--url", default="http://localhost:5000")
    args = ap.parse_args()

    if args.served:
        js_text = fetch(f"{args.url}/static/js/dashboard.js")
        html_text = fetch(args.url)
        label = "served"
    else:
        js_text = (REPO / "web/static/js/dashboard.js").read_text()
        html_text = (REPO / "web/templates/index.html").read_text()
        label = "repo"

    js_ids = collect_js_ids(js_text)
    html_ids = collect_html_ids(html_text)
    # JS-generated markup counts as defined: id="..." inside template literals,
    # plus explicit el.id = '...' assignments.
    js_defined = set(ID_DEF.findall(js_text)) | {
        m.group(1) for m in re.finditer(r"""\.id\s*=\s*['"]([\w-]+)['"]""", js_text)
    }
    missing = sorted(
        i
        for i in js_ids - html_ids - js_defined
        if not DYNAMIC_OK.match(i) and i not in OPTIONAL_GUARDED
    )

    print(f"[{label}] JS-referenced ids: {len(js_ids)}, HTML ids: {len(html_ids)}")
    if missing:
        print(f"MISSING ({len(missing)}) - referenced in JS but absent from HTML:")
        for i in missing:
            print(f"  #{i}")
        return 1
    print("id-contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
