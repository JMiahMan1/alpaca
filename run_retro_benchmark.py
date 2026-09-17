#!/usr/bin/env python3
"""Quick focused benchmark: retrogames with local model."""
import asyncio
import sys

sys.path.insert(0, '.')

from llm_benchmark_suite import LLMModelBenchmark


async def main():
    suite = LLMModelBenchmark()
    model = "qwen3-8-27b-ud-q3-k-xl"
    print(f"Running retrogames benchmark for: {model}")
    result = await suite.run_model_benchmarks(
        models=[model],
        use_proxy=True,
        mode="functional",
        groups=["retrogames"],
        test_ids=["retro_space_invaders"],
    )
    print("\n=== BENCHMARK RESULT ===")
    print(f"Model: {model}")
    for r in result.get("results", []):
        print(f"Result file: {r.get('model')}")
        group_scores = r.get("group_scores", [])
        for gs in group_scores:
            if gs.get("group") == "retrogames":
                print(f"Retrogames score: {gs.get('score', 'N/A')}")
                print(f"Retrogames letter: {gs.get('letter', 'N/A')}")
        for cat in r:
            if cat.startswith("category_"):
                tests = r[cat].get("tests", [])
                print(f"Category: {cat} — {len(tests)} tests run")
                for t in tests:
                    status = "PASS" if t.get("passed") else "FAIL"
                    print(f"  {t.get('test_id')}: {status} (score={t.get('score', 'N/A')})")

if __name__ == "__main__":
    asyncio.run(main())
