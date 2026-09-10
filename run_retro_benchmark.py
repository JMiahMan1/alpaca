#!/usr/bin/env python3
"""Quick focused benchmark: retrogames with openrouter:thinkingmachines/inkling:free"""
import asyncio
import sys
sys.path.insert(0, '.')

from llm_benchmark_suite import LLMModelBenchmark

async def main():
    suite = LLMModelBenchmark()
    # Use the openrouter online model identifier
    model = "openrouter:thinkingmachines/inkling:free"
    # Focus on retrogames category, functional mode only
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
        group_scores = r.get("group_scores", {})
        retrogames_score = group_scores.get("retrogames", {})
        if retrogames_score:
            print(f"Retrogames score: {retrogames_score.get('score', 'N/A')}")
            print(f"Retrogames letter: {retrogames_score.get('letter', 'N/A')}")
        # Print per-test results for retrogames
        for cat in r:
            if cat.startswith("category_"):
                tests = r[cat].get("tests", [])
                print(f"Category: {cat} — {len(tests)} tests run")
                for t in tests:
                    status = "PASS" if t.get("passed") else "FAIL"
                    print(f"  {t.get('test_id')}: {status} (score={t.get('score', 'N/A')})")

if __name__ == "__main__":
    asyncio.run(main())
