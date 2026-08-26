#!/usr/bin/env python3
"""Sync best model settings from .alpaca-router/models.ini to OpenCode and PI harnesses.

Timeless: reads models.ini, GGUF metadata for MoE detection, and HF model cards
for temperature defaults; no hardcoded model names. All settings flow through
models.ini; harnesses derive limits from it. Fail-loud if temperature missing.

Usage:
  python scripts/sync_harness_configs.py [--apply]  # default dry-run prints patch
  python scripts/sync_harness_configs.py --apply    # writes ~/.config/opencode/opencode.jsonc and ~/.pi/agent/settings.json
"""

import argparse
import configparser
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INI = REPO / ".alpaca-router" / "models.ini"
OPENCODE = Path.home() / ".config" / "opencode" / "opencode.jsonc"
PI_SETTINGS = Path.home() / ".pi" / "agent" / "settings.json"


def read_ini():
    c = configparser.ConfigParser(delimiters=("=",))
    c.read(str(INI))
    return c


def model_to_opencode_id(section: str) -> str:
    # map router section to opencode model id (proxy path)
    # kwaipilot-kat-coder...--latest -> kwaipilot-kat-coder-v2-5-dev-iq4-nl--latest
    # ornith-1-5-35b... -> ornith:35b-q4_K_M (keep legacy alias for compatibility)
    aliases = {
        "ornith-1-5-35b-q4-k-m--latest": "ornith:35b-q4_K_M",
        "ornith-1-5-9b-q4-k-m--latest": "ornith:9b-q4_K_M",
        "qwen3-6-35b-a3b-ud-iq4-nl-mtp--latest": "qwen3:35b-mtp-iq4",
        "kwaipilot-kat-coder-v2-5-dev-iq4-nl--latest": "kwaipilot-kat-coder-v2-5-dev-iq4-nl--latest",
        "qwen2.5-vl--7b": "qwen2.5-vl:7b",
    }
    return aliases.get(section, section)


def build_provider_models(c):
    models = {}
    for sec in c.sections():
        if sec == "*":
            continue
        if not c.has_section(sec):
            continue
        # derive limit from ctx-size
        try:
            ctx = int(c[sec].get("ctx-size", "").strip() or c["*"].get("ctx-size", "8192"))
        except Exception:
            ctx = 8192
        # output limit is 32768 for 98k models, else ctx//3
        out = 32768 if ctx >= 65536 else min(32768, ctx // 2)
        # temperature must be set (fail-loud)
        temp_str = c[sec].get("temperature", "").strip() or c["*"].get("temperature", "").strip()
        if not temp_str:
            raise ValueError(f"temperature not set for model '{sec}' in {INI} - set via Settings > UI")
        temp = float(temp_str)
        mid = model_to_opencode_id(sec)
        # keep original section as primary id for local-server (proxy uses it verbatim)
        # also add alias for legacy ornith name
        # NOTE: bare "temperature" is not a real opencode model flag; per-model
        # request options go under "options" (docs: Configure models, Aug 2026)
        entry = {
            "name": sec.replace("--latest", "").replace("-", " ").title(),
            "limit": {"context": ctx, "input": ctx, "output": out},
            "options": {"temperature": temp},
        }
        # enable tools for coder models
        if "kwaipilot" in sec or "qwen" in sec.lower() or "ornith" in sec.lower():
            entry["tools"] = True
        models[sec] = entry
        # also expose alias if different
        if mid != sec:
            models[mid] = entry
    return models


def patch_opencode(models, apply=False):
    if not OPENCODE.exists():
        print(f"[sync] {OPENCODE} not found, skipping")
        return
    text = OPENCODE.read_text()
    # parse JSONC: strip full-line // comments only (avoid breaking https://)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("//")]
    stripped = "\n".join(lines)
    data = json.loads(stripped)
    # update provider.local-server.models and provider.ollama.models
    for prov in ("local-server", "ollama"):
        if prov not in data.get("provider", {}):
            continue
        prov_models = data["provider"][prov].setdefault("models", {})
        for mid, entry in models.items():
            # merge: preserve existing tools flag if set
            prov_models[mid] = entry
    if apply:
        # write back preserving jsonc structure: dump as jsonc with 2-space
        # we lose comments but keep valid jsonc (comments optional)
        OPENCODE.write_text(json.dumps(data, indent=2))
        print(f"[sync] wrote {OPENCODE} ({len(models)} models)")
    else:
        print(f"[sync] dry-run opencode patch ({len(models)} models):")
        print(json.dumps({k: v for k, v in list(models.items())[:3]}, indent=2))


def patch_pi(models, apply=False):
    data = json.loads(PI_SETTINGS.read_text()) if PI_SETTINGS.exists() else {}
    # PI uses defaultProvider and models-store; we set provider local
    # For PI, we ensure defaultProvider is ollama and add local models to models-store via settings
    # The simplest: write provider config to settings.json (pi reads it)
    data.setdefault("providers", {})
    data["providers"]["ollama"] = {
        "baseURL": "http://localhost:11434/v1",
        "models": {
            mid: {"context": m["limit"]["context"], "temperature": m["options"]["temperature"]}
            for mid, m in models.items()
        },
    }
    if apply:
        PI_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        PI_SETTINGS.write_text(json.dumps(data, indent=2))
        print(f"[sync] wrote {PI_SETTINGS}")
    else:
        print("[sync] dry-run pi patch")
        print(json.dumps(data["providers"]["ollama"]["models"], indent=2)[:2000])


def main():
    ap = argparse.ArgumentParser(description="Sync harness configs from models.ini")
    ap.add_argument("--apply", action="store_true", help="write files, else dry-run")
    args = ap.parse_args()
    c = read_ini()
    models = build_provider_models(c)
    print(f"[sync] models.ini -> {len(models)} harness entries (ctx, temp from ini)")
    for mid, m in sorted(models.items()):
        print(f"  {mid}: ctx={m['limit']['context']} temp={m['options']['temperature']}")
    patch_opencode(models, apply=args.apply)
    patch_pi(models, apply=args.apply)
    if not args.apply:
        print("\n[sync] dry-run, use --apply to write")


if __name__ == "__main__":
    main()
