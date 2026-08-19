"""
Model Tracking & Benchmark Lifecycle Manager for Alpaca.

Tracks:
1. Newly added / discovered models (local and online)
2. Previously benchmarked items (scores, run counts, timestamps, result files)
3. Model lifecycle state (New vs Benchmarked)
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class ModelTracker:
    """Tracks model discovery, addition, and historical benchmarking lifecycle."""

    def __init__(self, data_dir: Path | str = "data"):
        self.data_dir = Path(data_dir)
        self.tracking_file = self.data_dir / "model_tracking.json"
        self._lock = threading.Lock()
        self.shared_benchmarks_dir = self.data_dir / "shared_llm_benchmarks"
        self.general_benchmarks_dir = self.data_dir / "llm_benchmarks"

    def _load_tracking_data(self) -> dict[str, Any]:
        """Load stored tracking metadata from disk."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {"models": {}}

    def _save_tracking_data(self, data: dict[str, Any]) -> None:
        """Save tracking metadata to disk."""
        try:
            self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tracking_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def record_model_seen(self, model_id: str, source: str = "local") -> None:
        """Record a model as seen/added."""
        if not model_id:
            return
        with self._lock:
            data = self._load_tracking_data()
            models = data.setdefault("models", {})
            now = datetime.now().isoformat()
            if model_id not in models:
                models[model_id] = {
                    "first_seen_at": now,
                    "added_at": now,
                    "source": source,
                    "benchmark_count": 0,
                    "last_benchmarked_at": None,
                    "latest_score": None,
                    "latest_run_type": None,
                    "latest_result_file": None,
                }
            self._save_tracking_data(data)

    def record_benchmark_result(
        self,
        model_id: str,
        score_pct: float,
        run_type: str,
        result_file: str,
        timestamp: str | None = None,
    ) -> None:
        """Record that a benchmark was executed for a model."""
        if not model_id:
            return
        with self._lock:
            data = self._load_tracking_data()
            models = data.setdefault("models", {})
            now = timestamp or datetime.now().isoformat()
            entry = models.get(model_id, {})
            entry["first_seen_at"] = entry.get("first_seen_at") or now
            entry["benchmark_count"] = entry.get("benchmark_count", 0) + 1
            entry["last_benchmarked_at"] = now
            entry["latest_score"] = round(score_pct, 1)
            entry["latest_run_type"] = run_type
            entry["latest_result_file"] = os.path.basename(result_file) if result_file else None
            models[model_id] = entry
            self._save_tracking_data(data)

    def delete_model(self, model_id: str) -> bool:
        """Remove a model's tracking entry from the persisted registry.

        Called when a model is deleted from disk (with or without its benchmark
        results) so it stops appearing in the "Previously Benchmarked" list.
        Returns True if an entry existed and was removed.
        """
        if not model_id:
            return False
        with self._lock:
            data = self._load_tracking_data()
            models = data.get("models", {})
            clean_id = model_id.strip()
            variants = {
                clean_id,
                clean_id.replace("--", ":"),
                clean_id.replace(":", "--"),
                clean_id.replace("--", "_"),
                clean_id.replace(":", "_"),
                clean_id.removesuffix(":latest"),
                clean_id + ":latest" if not clean_id.endswith(":latest") else clean_id,
                clean_id.removesuffix(".gguf"),
                clean_id.split("/")[-1],
            }
            removed = False
            keys_to_del = [
                k for k in list(models.keys()) if k in variants or any(v in k for v in variants if len(v) > 5)
            ]
            for k in keys_to_del:
                del models[k]
                removed = True
            if removed:
                self._save_tracking_data(data)
            return removed

    def clear_all(self) -> bool:
        """Clear all tracking entries and reset the registry."""
        with self._lock:
            self._save_tracking_data({"models": {}})
            return True

    def scan_historical_benchmarks(self) -> dict[str, dict[str, Any]]:
        """Scans all saved benchmark files on disk to compile complete history per model."""
        history: dict[str, dict[str, Any]] = {}

        # 1. Scan SharedLLM benchmark files (root snapshot files and per-model files)
        if self.shared_benchmarks_dir.exists():
            shared_files = list(self.shared_benchmarks_dir.glob("shared_llm_benchmarks_*.json"))
            models_sub = self.shared_benchmarks_dir / "models"
            if models_sub.exists():
                shared_files.extend(models_sub.glob("shared_*.json"))

            for fpath in sorted(shared_files, key=lambda p: str(p)):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        doc = json.load(f)
                    gen_at = doc.get("generated_at") or ""
                    for res in doc.get("results", []):
                        m_id = res.get("model")
                        if not m_id:
                            continue
                        tasks = res.get("tasks") or res.get("tests") or []
                        total = len(tasks)
                        success_cnt = sum(1 for t in tasks if isinstance(t, dict) and t.get("success"))
                        score_pct = (success_cnt / total * 100.0) if total > 0 else 0.0

                        entry = history.setdefault(
                            m_id,
                            {
                                "benchmark_count": 0,
                                "last_benchmarked_at": None,
                                "latest_score": None,
                                "latest_run_type": "shared_llm",
                                "latest_result_file": None,
                                "scores": [],
                            },
                        )
                        entry["benchmark_count"] += 1
                        entry["scores"].append(score_pct)
                        if not entry["last_benchmarked_at"] or gen_at >= entry["last_benchmarked_at"]:
                            entry["last_benchmarked_at"] = gen_at
                            entry["latest_score"] = round(score_pct, 1)
                            entry["latest_run_type"] = "shared_llm"
                            entry["latest_result_file"] = fpath.name
                except Exception:
                    pass

        # 2. Scan General benchmark files (root snapshot files and per-model files)
        if self.general_benchmarks_dir.exists():
            gen_files = list(self.general_benchmarks_dir.glob("benchmarks_*.json"))
            models_sub = self.general_benchmarks_dir / "models"
            if models_sub.exists():
                gen_files.extend(models_sub.glob("general_*.json"))

            for fpath in sorted(gen_files, key=lambda p: str(p)):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        doc = json.load(f)
                    gen_at = doc.get("generated_at") or ""
                    for res in doc.get("results", []):
                        m_id = res.get("model")
                        if not m_id:
                            continue
                        tests = []
                        if isinstance(res.get("tests"), list):
                            tests.extend(res["tests"])
                        for k, v in res.items():
                            if k.startswith("category_") and isinstance(v, dict) and isinstance(v.get("tests"), list):
                                tests.extend(v["tests"])

                        total = len(tests)
                        success_cnt = sum(1 for t in tests if isinstance(t, dict) and t.get("success"))
                        score_pct = (success_cnt / total * 100.0) if total > 0 else 0.0

                        entry = history.setdefault(
                            m_id,
                            {
                                "benchmark_count": 0,
                                "last_benchmarked_at": None,
                                "latest_score": None,
                                "latest_run_type": "general",
                                "latest_result_file": None,
                                "scores": [],
                            },
                        )
                        entry["benchmark_count"] += 1
                        entry["scores"].append(score_pct)
                        if not entry["last_benchmarked_at"] or gen_at >= entry["last_benchmarked_at"]:
                            entry["last_benchmarked_at"] = gen_at
                            entry["latest_score"] = round(score_pct, 1)
                            entry["latest_run_type"] = "general"
                            entry["latest_result_file"] = fpath.name
                except Exception:
                    pass

        return history

    def get_tracking_summary(
        self,
        current_local_models: list[str] | None = None,
        current_online_models: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Returns structured tracking summary of Newly Added models vs Previously Benchmarked items."""
        stored_data = self._load_tracking_data()
        stored_models = stored_data.get("models", {})
        history = self.scan_historical_benchmarks()

        all_model_ids: set[str] = set()
        model_source_map: dict[str, str] = {}
        model_label_map: dict[str, str] = {}

        if current_local_models:
            for m in current_local_models:
                all_model_ids.add(m)
                model_source_map[m] = "local"
                model_label_map[m] = m

        if current_online_models:
            for om in current_online_models:
                m_id = str(om.get("id") or om.get("name") or "")
                if m_id:
                    all_model_ids.add(m_id)
                    model_source_map[m_id] = str(om.get("provider", "online"))
                    model_label_map[m_id] = str(om.get("label") or om.get("name") or m_id)

        # Also include any models in stored registry or history
        for m_id in stored_models:
            all_model_ids.add(m_id)
        for m_id in history:
            all_model_ids.add(m_id)

        newly_added: list[dict[str, Any]] = []
        previously_benchmarked: list[dict[str, Any]] = []
        all_tracked: dict[str, dict[str, Any]] = {}

        for m_id in sorted(all_model_ids):
            stored = stored_models.get(m_id, {})
            hist = history.get(m_id, {})

            benchmarked = bool(hist.get("benchmark_count") or stored.get("benchmark_count"))
            bench_count = hist.get("benchmark_count") or stored.get("benchmark_count") or 0
            last_bench = hist.get("last_benchmarked_at") or stored.get("last_benchmarked_at")
            latest_score = (
                hist.get("latest_score") if hist.get("latest_score") is not None else stored.get("latest_score")
            )
            latest_file = hist.get("latest_result_file") or stored.get("latest_result_file")
            latest_type = hist.get("latest_run_type") or stored.get("latest_run_type")
            source = model_source_map.get(m_id) or stored.get("source", "unknown")
            label = model_label_map.get(m_id) or m_id

            item = {
                "id": m_id,
                "label": label,
                "source": source,
                "is_online": source not in ("local", "ollama", "router"),
                "status": "benchmarked" if benchmarked else "new",
                "is_new": not benchmarked,
                "first_seen_at": stored.get("first_seen_at"),
                "benchmark_count": bench_count,
                "last_benchmarked_at": last_bench,
                "latest_score": latest_score,
                "latest_run_type": latest_type,
                "latest_result_file": latest_file,
            }

            all_tracked[m_id] = item
            if benchmarked:
                previously_benchmarked.append(item)
            else:
                newly_added.append(item)

        # Sort benchmarked by latest benchmark timestamp descending
        previously_benchmarked.sort(
            key=lambda x: (x.get("last_benchmarked_at") or "", x.get("latest_score") or 0),
            reverse=True,
        )

        return {
            "success": True,
            "counts": {
                "total": len(all_model_ids),
                "newly_added": len(newly_added),
                "previously_benchmarked": len(previously_benchmarked),
            },
            "newly_added": newly_added,
            "previously_benchmarked": previously_benchmarked,
            "all_tracked": all_tracked,
        }


# Singleton instance
model_tracker = ModelTracker()
