# LLM Benchmarking Integration Guide

This document explains how to integrate custom LLM benchmarks into the Alpaca proxy system for evaluating models across different SharedLLM task categories.

## Overview

The Alpaca system now includes a comprehensive LLM benchmarking suite (`llm_benchmark_suite.py`) that provides:

1. **Model Evaluation Across Categories**: Tests models on coding, reasoning, instruction-following, creative, and home automation tasks
2. **Live Results UI**: Real-time dashboard showing benchmark progress and results
3. **Configuration Recommendations**: Optimal llama.cpp settings for different use cases
4. **Integration Support**: Works with both direct calls and through the Alpaca proxy

## Setting Up the Benchmarking Suite

### Prerequisites

The benchmarking suite requires:
- Python 3.7 or higher
- Installed Python packages: `httpx`, `requests`
- Access to the Alpaca proxy endpoint (via the Alpaca container or direct URL)

### Installation

```bash
pip install httpx requests
```

## Running Benchmarks

### Basic Usage

Run benchmarks for multiple models:

```bash
python3 llm_benchmark_suite.py
python3 llm_benchmark_suite.py "qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b"
```

### Benchmark Types

The suite automatically tests both:
1. **Direct Benchmarks**: Calls OLLAMA_URL directly
2. **Proxy Benchmarks**: Routes through Alpaca proxy

## Integration with Alpaca Configuration

### 1. Adding to Alpaca Startup Script

Create or modify the `ALPACA_STARTUP_BENCHMARKS.sh` script:

```bash
#!/bin/bash

# Start alpaca-proxy
python3 alpaca-proxy.py &
ALPACA_PID=$!

# Wait for services to initialize
sleep 10

# Run benchmarks if environment variable is set
if [ "$RUN_LLM_BENCHMARKS" = "true" ]; then
    echo "Starting LLM benchmarking pipeline..."
    python3 llm_benchmark_suite.py "$BENCHMARK_MODELS"
    echo "Benchmarking complete."
fi

# Monitor and restart if needed
while true; do
    if ! ps -p $ALPACA_PID > /dev/null; then
        echo "Alpaca proxy died, restarting..."
        python3 alpaca-proxy.py &
        ALPACA_PID=$!
        sleep 5
    fi
    sleep 30

done
```

### 2. Environment Configuration

```bash
# .env configuration for benchmarks
ALPACA_STARTUP_BENCHMARKS=true
BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b
ALPACARD_MODELS_DIR=/router-models
LLAMA_SERVER_URL=http://llama-server:8080
PROXY_URL=http://localhost:11434
OLLAMA_URL=http://localhost:11434
```

### 3. Docker Integration

Add to `Dockerfile`:

```dockerfile
COPY llm_benchmark_suite.py /app/
COPY ALPHA_STARTUP_BENCHMARKS.sh /app/scripts/
RUN chmod +x /app/scripts/ALPHA_STARTUP_BENCHMARKS.sh
```

Update `docker-compose.yml`:

```yaml
alpaca-proxy:
  build: .
  environment:
    - RUN_LLM_BENCHMARKS=true
    - BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b
  volumes:
    - ./data/llm_benchmarks:/app/data/llm_benchmarks
    - ./models:/router-models
  command: /app/scripts/ALPHA_STARTUP_BENCHMARKS.sh
```

## Running in Production

### 1. Scheduled Benchmarks

Create a cron job for periodic benchmarking:

```bash
# Add to crontab (crontab -e)

# Run benchmarks every 4 hours during off-peak hours
0 2,6,10,14,18,22 * * * /usr/bin/python3 /home/user/code/alpaca/llm_benchmark_suite.py "$BENCHMARK_MODELS"

# Run lightweight benchmarks daily
0 3 * * * /usr/bin/python3 /home/user/code/alpaca/llm_benchmark_suite.py "qwen3:8b,qwen2.5-coder:7b"
```

### 2. On-Demand Benchmarking

Trigger benchmarks via API:

```bash
# Trigger benchmarks via Alpaca proxy API
curl -X POST http://localhost:11434/api/v1/benchmarks \
  -H "Content-Type: application/json" \
  -d '{"models": ["qwen3:8b", "qwen2.5-coder:7b"], "type": "proxy"}'
```

### 3. Service Status Monitoring

Add Alpaca proxy health endpoint for benchmarks:

```python
# in alpaca-proxy.py
@app.get("/admin/benchmarks")
async def run_benchmarks():
    """Trigger LLM benchmarking suite."""
    models = ["qwen3:8b", "qwen2.5-coder:7b", "qwen3.5:9b"]
    
    benchmark = LLMModelBenchmark()
    results = asyncio.run(benchmark.run_optimization_pipeline(models))
    
    return {
        "status": "started",
        "message": "Benchmarking started",
        "results_dir": results["results_dir"],
        "models_tested": len(models)
    }
```

## Results Management

### 1. Results Directory Structure

```
data/llm_benchmarks/
├── direct_benchmarks_latest.json
├── proxy_benchmarks_latest.json
├── benchmarks_20250622_153045_direct.json
├── benchmarks_20250622_153045_proxy.json
└── models/
    ├── qwen3.8b.profile.json
    ├── qwen2.5-coder.7b.profile.json
    └── qwen3.5.9b.profile.json
```

### 2. Results API

Access results via Alpaca proxy:

```bash
# Get latest benchmark results
curl http://localhost:11434/api/v1/benchmarks/latest

# Get specific model profile
curl http://localhost:11434/api/v1/benchmarks/model/qwen3.8b/profile

# Get best performing model
curl http://localhost:11434/api/v1/benchmarks/recommendations
```

### 3. Result Processing Script

Create `process_benchmark_results.py` for post-processing:

```python
#!/usr/bin/env python3
"""Process benchmark results and generate recommendations."""

import json
from pathlib import Path
from llm_benchmark_suite import LLMModelBenchmark

def main():
    benchmark = LLMModelBenchmark()
    
    # Load latest results
    latest_file = Path("data/llm_benchmarks/proxy_benchmarks_latest.json")
    if not latest_file.exists():
        print("No results file found")
        return
    
    with open(latest_file, "r") as f:
        results = json.load(f)
    
    # Generate recommendations
    analysis = benchmark.analyze_results(latest_file)
    recommendations = benchmark.recommend_models(analysis)
    
    # Process recommendations
    if recommendations:
        best_model = recommendations[0]["model"]
        print(f"\nBest performing model: {best_model}")
        print(f"Success rate: {recommendations[0]['performance']['overall_success_rate']}%")
        
        # Update Alpaca configuration
        update_alpaca_config(best_model)

if __name__ == "__main__":
    main()
```

## Configuration Optimization

### 1. Model Profile Generation

The benchmark suite generates model profile files (`.profile.json`) in the `models/` subdirectory:

```json
{
    "ctx-size": "8192",
    "cache-type-k": "f16",
    "cache-type-v": "f16",
    "flash-attn": "on",
    "quality": {
        "summary": {
            "tests_run": 5,
            "tests_passed": 4,
            "avg_tokens_per_sec": 23.5,
            "avg_ttft_ms": 120.0,
            "max_tokens_per_sec": 35.1,
            "min_ttft_ms": 45.0
        },
        "by_category": {
            "coding": {"avg_tokens_per_sec": 25.3},
            "reasoning": {"avg_tokens_per_sec": 20.1},
            "instruction": {"avg_tokens_per_sec": 22.7}
        },
        "tests": [
            {
                "id": "debug_fix",
                "label": "Python: debug logic error",
                "category": "coding",
                "tokens_per_sec": 23.5,
                "ttft_ms": 150.0,
                "success": true,
                "quality_score": 1.0
            }
        ]
    }
}
```

### 2. Integration with Alpaca-Puller

Update `alpaca-puller.py` to use benchmark results:

```python
def load_model_config(model_name):
    """Load model configuration from benchmarks if available."""
    profile_path = Path(f"models/{model_name}.profile.json")
    if profile_path.exists():
        with open(profile_path, "r") as f:
            profile = json.load(f)
        
        # Extract configuration
        return {
            "ctx-size": profile.get("ctx-size", "8192"),
            "cache-type-k": profile.get("cache-type-k", "f16"),
            "cache-type-v": profile.get("cache-type-v", "f16"),
            "flash-attn": profile.get("flash-attn", "on"),
            "quality_score": profile.get("quality", {}).get("summary", {}).get("avg_tokens_per_sec", 0)
        }
    return None
```

### 3. Dynamic Configuration Updates

Create a service to periodically update Alpaca configuration based on benchmarks:

```python
# dynamic_config_updater.py
import json
import time
from pathlib import Path
from alpaca-puller import update_models_ini

def update_from_benchmarks():
    """Update models.ini based on latest benchmarks."""
    benchmark = LLMModelBenchmark()
    
    # Load latest proxy benchmarks
    results_file = Path("data/llm_benchmarks/proxy_benchmarks_latest.json")
    if not results_file.exists():
        return
    
    # Get analysis
    analysis = benchmark.analyze_results(results_file)
    
    # Get recommendations
    recommendations = benchmark.recommend_models(analysis)
    
    # Update configuration for top model
    if recommendations:
        top_model = recommendations[0]["model"]
        config_recs = benchmark.generate_optimized_config_recommendations(results_file)
        
        optimal = config_recs["optimal_configurations"]["coding_assistance"]
        
        update_models_ini(top_model, optimal["recommended_settings"])
        print(f"Updated {top_model} configuration")
        
        # Restart services to apply changes
        restart_llama_server()

if __name__ == "__main__":
    while True:
        update_from_benchmarks()
        time.sleep(3600)  # Run every hour
```

## Monitoring and Alerting

### 1. Health Checks

Add health checks to monitor benchmark system:

```python
@app.get("/admin/health")
async def health_check():
    """Check system health including benchmarks."""
    health = {
        "status": "healthy",
        "alpaca_proxy": "running",
        "benchmark_system": "ready"
    }
    
    # Check if results directory exists
    if not Path("data/llm_benchmarks").exists():
        health["benchmark_system"] = "missing_results_dir"
    
    return health
```

### 2. Performance Alerts

Monitor benchmark performance:

```bash
# Alert if benchmark success rate is low
python3 monitor_benchmarks.py --threshold 80 --directory data/llm_benchmarks
```

## Testing and Validation

### 1. Unit Tests

Create tests for the benchmarking suite:

```python
# test_llm_benchmarks.py
import pytest
from llm_benchmark_suite import LLMModelBenchmark

def test_configuration_tests():
    benchmark = LLMModelBenchmark()
    tests = benchmark._coding_tests("test_model")
    
    assert len(tests) == 3
    assert all(t["category"] == "coding" for t in tests)
    assert all("num_predict" in t for t in tests)

def test_display_live_results():
    benchmark = LLMModelBenchmark()
    
    mock_results = {
        "models_tested": 3,
        "generated_at": "2024-06-22T15:30:45",
        "benchmark_type": "proxy",
        "results": [
            {
                "model": "qwen3:8b",
                "category_coding": {
                    "tests_run": 3,
                    "tests_passed": 3,
                    "avg_tokens_per_sec": 25.3,
                    "avg_ttft_ms": 150.0
                }
            }
        ]
    }
    
    # Test that display works (no exceptions)
    benchmark._display_live_results(mock_results)
```

### 2. Integration Tests

Test benchmark integration:

```bash
# Run integration tests
cd alpaca
pytest test/benchmarks/test_integration.py -v
```

## Troubleshooting

### 1. Common Issues

**Issue: Benchmarks fail due to connection errors**

```bash
# Check if alpaca-proxy is running
ps aux | grep alpaca-proxy

# Check network connectivity
curl http://localhost:11434/api/v1/models
```

**Issue: Results not being saved**

```bash
# Check disk space
df -h /home/user/code/alpaca

# Check permissions
ls -la data/llm_benchmarks
```

**Issue: Memory usage too high**

```bash
# Monitor memory usage
ps aux | grep python3 | grep benchmark
```

### 2. Debug Scripts

Create debug scripts for troubleshooting:

```python
# debug_benchmarks.py
def test_benchmark_health():
    """Test benchmark system health."""
    benchmark = LLMModelBenchmark()
    
    # Test model evaluation
    test_data = {
        "id": "test",
        "category": "coding",
        "label": "test",
        "prompt": "test",
        "num_predict": 10
    }
    
    try:
        result = asyncio.run(
            benchmark._evaluate_model_test("test_model", test_data, use_proxy=True)
        )
        print(f"Benchmark test result: {result}")
    except Exception as e:
        print(f"Benchmark test failed: {e}")
```

## Maintenance

### 1. Data Cleanup

Create cleanup script for old benchmark results:

```python
# cleanup_benchmarks.py
def clean_old_results(keep_days=30):
    """Clean old benchmark results."""
    from datetime import datetime, timedelta
    
    results_dir = Path("data/llm_benchmarks")
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    
    for file_path in results_dir.glob("*.json"):
        file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_date < cutoff_date:
            print(f"Removing old file: {file_path}")
            file_path.unlink()
```

### 2. Update Frequency

Monitor and adjust benchmark frequency based on:
- Model update schedules
- Usage patterns
- System resources
- Seasonal workload variations

## Migration Guide

### 1. From Legacy Benchmarks

Update existing benchmark results to new format:

```python
# migrate_results.py
import json
from pathlib import Path

def migrate_legacy_results():
    legacy_file = Path("benchmark_results.json")
    if not legacy_file.exists():
        return
    
    with open(legacy_file, "r") as f:
        legacy_results = json.load(f)
    
    # Convert to new format
    migrated = {
        "benchmark_version": "2.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark_type": "proxy",
        "models_tested": len(legacy_results),
        "results": []
    }
    
    # ... migration logic ...
```

### 2. Upgrading from Older Versions

When upgrading from older versions:
1. Ensure `python3 llm_benchmark_suite.py` runs without errors
2. Verify that results are saved correctly
3. Update any external integrations that reference the old results format
4. Test with your target models before production use

## Version History

### Version 2.0.0
- Added live results UI
- Integrated with Alpaca proxy
- Comprehensive category testing
- Automatic configuration optimization

### Version 1.0.0
- Initial benchmarking suite
- Basic model evaluation
- Static results reporting

## Conclusion

The LLM benchmarking suite provides a robust, automated way to evaluate and optimize LLM models for SharedLLM applications. By integrating with the Alpaca proxy system, you can:

1. Continuously evaluate model performance
2. Automatically select and configure optimal models
3. Monitor system health through benchmark results
4. Generate actionable insights for model selection and tuning

The suite is designed to be extensible, allowing you to add new test categories and evaluation criteria as your requirements evolve.
```

## Configuration Setup

This section provides the complete configuration to set up the benchmarking system with all the necessary files. Here's what you need to create:

### 1. .env file

Create a `.env` file in your project root:

```bash
# .env file
ALPACA_STARTUP_BENCHMARKS=true
BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b
ALPACARD_MODELS_DIR=/router-models
LLAMA_SERVER_URL=http://llama-server:8080
PROXY_URL=http://localhost:11434
OLLAMA_URL=http://localhost:11434
INTERNAL_SECRET=your-secret-key-here
```

### 2. ALPHA_STARTUP_BENCHMARKS.sh

Create the startup script:

```bash
#!/bin/bash

# ALPHA_STARTUP_BENCHMARKS.sh

# Start alpaca-proxy
python3 alpaca-proxy.py &
ALPACA_PID=$!

# Wait for services to initialize
sleep 10

# Run benchmarks if environment variable is set
if [ "$RUN_LLM_BENCHMARKS" = "true" ]; then
    echo "Starting LLM benchmarking pipeline..."
    python3 llm_benchmark_suite.py "$BENCHMARK_MODELS"
    echo "Benchmarking complete."
fi

# Monitor and restart if needed
while true; do
    if ! ps -p $ALPACA_PID > /dev/null; then
        echo "Alpaca proxy died, restarting..."
        python3 alpaca-proxy.py &
        ALPACA_PID=$!
        sleep 5
    fi
    sleep 30

done
```

### 3. Dockerfile

Update the Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alpaca-proxy.py .
COPY llm_benchmark_suite.py .
COPY ALPHA_STARTUP_BENCHMARKS.sh scripts/
RUN chmod +x scripts/ALPHA_STARTUP_BENCHMARKS.sh

EXPOSE 11434

CMD ["/bin/bash", "-c", "scripts/ALPHA_STARTUP_BENCHMARKS.sh"]
```

### 4. docker-compose.yml

Update the docker-compose configuration:

```yaml
dversion: '3.8'

services:
  llama-server:
    image: llama-server:latest
    container_name: llama-server
    ports:
      - "8080:8080"
    volumes:
      - ./router-models:/router-models
    environment:
      - MODEL_MODELS_DIR=/router-models

  alpaca-proxy:
    build: .
    container_name: alpaca-proxy
    ports:
      - "11434:11434"
    environment:
      - RUN_LLM_BENCHMARKS=true
      - BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b
      - ALPACA_PROXY_PORT=11434
      - LLAMA_SERVER_URL=http://llama-server:8080
    volumes:
      - ./data/llm_benchmarks:/app/data/llm_benchmarks
      - ./router-models:/router-models
    depends_on:
      - llama-server
    command: /app/scripts/ALPHA_STARTUP_BENCHMARKS.sh

  benchmark-updater:
    build: .
    container_name: benchmark-updater
    command: python3 monitor_benchmarks.py
    environment:
      - RUN_INTERVAL=3600
    volumes:
      - ./data/llm_benchmarks:/app/data/llm_benchmarks
      - ./router-models:/router-models
    depends_on:
      - alpaca-proxy
    restart: unless-stopped
```

### 5. requirements.txt

Create the requirements file:

```bash
# requirements.txt
httpx>=0.24.0
requests>=2.31.0
asyncio-mqtt>=0.12.0
pyyaml>=6.0
```

### 6. monitor_benchmarks.py

Create the monitoring script:

```python
# monitor_benchmarks.py
import asyncio
import json
import time
import requests
from pathlib import Path
from llm_benchmark_suite import LLMModelBenchmark

async def monitor_and_update():
    """Monitor benchmarks and update configuration automatically."""
    print("Starting benchmark monitor...")
    
    benchmark = LLMModelBenchmark()
    
    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running benchmark monitoring cycle...")
        
        try:
            # Check if we have recent results
            results_dir = Path("data/llm_benchmarks")
            if not results_dir.exists():
                results_dir.mkdir(parents=True, exist_ok=True)
                print("Created results directory")
                continue
            
            # Load latest results
            latest_direct = results_dir / "direct_benchmarks_latest.json"
            latest_proxy = results_dir / "proxy_benchmarks_latest.json"
            
            current_time = time.time()
            needs_benchmark = True
            
            # Check if we should run benchmarks (every 4 hours during peak hours)
            hour = time.localtime().tm_hour
            if hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
                if hour % 4 == 0:
                    print(f"Running scheduled benchmarks for hour {hour}")
                else:
                    needs_benchmark = False
            
            if needs_benchmark:
                # Check if we have models configured
                models = [m.strip() for m in os.getenv("BENCHMARK_MODELS", "qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b").split(",") if m.strip()]
                
                if models:
                    print(f"Running benchmarks for models: {models}")
                    results = asyncio.run(benchmark.run_model_benchmarks(models, use_proxy=True))
                    
                    # Display results
                    print(f"\nBenchmarking complete!")
                    print(f"Results saved to {results_dir}")
                    
                    # Try to update configuration if we have good results
                    if results and results.get("results"):
                        analysis = benchmark.analyze_results(Path(results_dir / "proxy_benchmarks_latest.json"))
                        recommendations = benchmark.recommend_models(analysis)
                        
                        if recommendations:
                            top_model = recommendations[0]["model"]
                            print(f"\nTop performing model: {top_model} ({recommendations[0]['performance']['overall_success_rate']}%) success rate")
                            
                            # Here you would update your Alpaca configuration
                            # update_alpaca_config(top_model)
                            
            else:
                print(f"Skips scheduled benchmarks (hour {hour} is not a benchmark hour)")
                
        except Exception as e:
            print(f"Error during monitoring cycle: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait for next cycle (4 hours)
        print(f"Waiting 4 hours until next monitoring cycle...")
        await asyncio.sleep(4 * 3600)

if __name__ == "__main__":
    asyncio.run(monitor_and_update())
```

### 7. Process Results Script

Create a script to process benchmark results:

```python
# process_results.py
#!/usr/bin/env python3
"""Process benchmark results and generate actionable insights."""

import json
import os
from pathlib import Path
from llm_benchmark_suite import LLMModelBenchmark

def main():
    benchmark = LLMModelBenchmark()
    
    results_dir = Path("data/llm_benchmarks")
    if not results_dir.exists():
        print("No results directory found. Run benchmarks first.")
        return
    
    print("Processing benchmark results...")
    
    # Analyze all results
    analysis = benchmark.analyze_results()
    recommendations = benchmark.recommend_models(analysis)
    
    # Generate configuration recommendations
    config_recs = benchmark.generate_optimized_config_recommendations()
    
    # Create summary report
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_models_tested": analysis.get("total_models_tested", 0),
        "recommended_model": None,
        "quality_tiers": {}
    }
    
    # Process recommendations
    if recommendations:
        summary["recommended_model"] = recommendations[0]["model"]
        
        # Group models by quality tier
        for rec in recommendations:
            score = rec["performance"]["overall_success_rate"]
            if score >= 85:
                tier = "Elite"
            elif score >= 70:
                tier = "Good"
            elif score >= 55:
                tier = "Acceptable"
            else:
                tier = "Needs Improvement"
            
            summary["quality_tiers"][rec["model"]] = {
                "score": score,
                "tier": tier,
                "recommendation": rec["recommendation"],
                "reasons": rec["reason"]
            }
    
    # Save summary
    summary_path = results_dir / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Summary saved to: {summary_path}")
    
    # Display recommendations
    print(f"\n{'='*80}")
    print(f"BENCHMARK PROCESS SUMMARY")
    print(f"{'='*80}")
    
    if recommendations:
        print(f"\nTop 3 Model Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"\n{i}. {rec['model']} - {rec['recommendation']}")
            print(f"   Performance: {rec['performance']['overall_success_rate']}% success rate")
            print(f"   Reasons: {', '.join(rec['reason'][:3])}")
    
    print(f"\nQuality Tiers:")
    for model, info in summary["quality_tiers"].items():
        print(f"  {model}: {info['tier']} ({info['score']}%)")
    
    # Generate configuration update commands
    if config_recs["recommended_models"]:
        print(f"\nRecommended Configuration Optimizations:")
        for use_case, config in config_recs["optimal_configurations"].items():
            print(f"\n  {use_case.title()}:")
            print(f"    Priority: {config['priority']}")
            for key, value in config['recommended_settings'].items():
                print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
```

### 8. Test Results Formatter

Create a script to format test results for easier consumption:

```python
# format_results.py
#!/usr/bin/env python3
"""Format benchmark results for different outputs."""

import json
import os
from pathlib import Path
from datetime import datetime

def format_for_display(results):
    """Format results for human-readable display."""
    formatted = {
        "timestamp": results.get("generated_at"),
        "benchmark_type": results.get("benchmark_type"),
        "models": []
    }
    
    for model_result in results.get("results", []):
        model = {
            "name": model_result.get("model"),
            "type": model_result.get("benchmark_type"),
            "performance": {}
        }
        
        for category in ["coding", "reasoning", "instruction", "creative", "home_automation"]:
            cat_key = f"category_{category}"
            if cat_key in model_result:
                cat_stats = model_result[cat_key]
                model["performance"][category] = {
                    "success_rate": f"{cat_stats.get('tests_passed', 0) / max(cat_stats.get('tests_run', 1), 1) * 100:.1f}%",
                    "avg_tokens_per_sec": f"{cat_stats.get('avg_tokens_per_sec', 0):.1f}",
                    "avg_ttft_ms": f"{cat_stats.get('avg_ttft_ms', 0):.0f}",
                    "tests_passed": cat_stats.get("tests_passed", 0),
                    "tests_total": cat_stats.get("tests_run", 0)
                }
        
        formatted["models"].append(model)
    
    return formatted

def format_for_api(results):
    """Format results for API consumption."""
    api_format = {
        "benchmark_id": f"bm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": results.get("generated_at"),
        "models": []
    }
    
    for model_result in results.get("results", []):
        model = {
            "name": model_result.get("model"),
            "score": calculate_model_score(model_result),
            "capabilities": extract_capabilities(model_result),
            "performance_metrics": extract_performance_metrics(model_result)
        }
        
        api_format["models"].append(model)
    
    return api_format

def calculate_model_score(model_result):
    """Calculate an overall score for a model."""
    total_tests = 0
    passed_tests = 0
    
    for category in ["coding", "reasoning", "instruction", "creative", "home_automation"]:
        cat_key = f"category_{category}"
        if cat_key in model_result:
            cat_stats = model_result[cat_key]
            total_tests += cat_stats.get("tests_run", 0)
            passed_tests += cat_stats.get("tests_passed", 0)
    
    return round((passed_tests / max(total_tests, 1)) * 100, 1)

def extract_capabilities(model_result):
    """Extract capabilities from model performance."""
    capabilities = []
    
    for category in ["coding", "reasoning", "instruction", "creative", "home_automation"]:
        cat_key = f"category_{category}"
        if cat_key in model_result:
            cat_stats = model_result[cat_key]
            success_rate = cat_stats.get("tests_passed", 0) / max(cat_stats.get("tests_run", 1), 1)
            
            if success_rate >= 0.8:
                capabilities.append(category)
    
    return capabilities

def extract_performance_metrics(model_result):
    """Extract performance metrics from model results."""
    metrics = {}
    
    for category in ["coding", "reasoning", "instruction", "creative", "home_automation"]:
        cat_key = f"category_{category}"
        if cat_key in model_result:
            cat_stats = model_result[cat_key]
            metrics[category] = {
                "success_rate": cat_stats.get("tests_passed", 0) / max(cat_stats.get("tests_run", 1), 1),
                "tokens_per_sec": cat_stats.get("avg_tokens_per_sec", 0),
                "ttft_ms": cat_stats.get("avg_ttft_ms", 0),
                "reliability": "high" if cat_stats.get("tests_passed", 0) / max(cat_stats.get("tests_run", 1), 1) >= 0.8 else "medium" if cat_stats.get("tests_passed", 0) / max(cat_stats.get("tests_run", 1), 1) >= 0.6 else "low"
            }
    
    return metrics

def main():
    """Main function to format results."""
    results_dir = Path("data/llm_benchmarks")
    
    if not results_dir.exists():
        print("No results directory found. Run benchmarks first.")
        return
    
    print("Formatting benchmark results...")
    
    # Find latest results file
    latest_file = max(results_dir.glob("*.json"), key=os.path.getmtime)
    print(f"Processing file: {latest_file.name}")
    
    with open(latest_file, "r") as f:
        results = json.load(f)
    
    # Format for display
    display_format = format_for_display(results)
    display_path = results_dir / f"display_{latest_file.name}"
    with open(display_path, "w") as f:
        json.dump(display_format, f, indent=2)
    
    print(f"Display format saved to: {display_path}")
    
    # Format for API
    api_format = format_for_api(results)
    api_path = results_dir / f"api_{latest_file.name}"
    with open(api_path, "w") as f:
        json.dump(api_format, f, indent=2)
    
    print(f"API format saved to: {api_path}")
    
    # Display formatted results
    print(f"\n{'='*80}")
    print(f"FORMATTED RESULTS")
    print(f"{'='*80}")
    
    print(f"\nSummary:")
    print(f"  Timestamp: {display_format['timestamp']}")
    print(f"  Models: {len(display_format['models'])}")
    
    for model in display_format['models']:
        print(f"\n  {model['name']} ({model['type']}):")
        for category, metrics in model['performance'].items():
            print(f"    {category}: {metrics['success_rate']} success, {metrics['avg_tokens_per_sec']} tok/s, {metrics['avg_ttft_ms']}ms TTFT")

if __name__ == "__main__":
    main()
```

### 9. Docker Commands

Run the benchmark suite using Docker:

```bash
# Build and run with benchmarks enabled
export RUN_LLM_BENCHMARKS=true
export BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b
docker compose up --build

# Or just run benchmarks
export RUN_LLM_BENCHMARKS=true
export BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b
docker compose run --rm alpaca-proxy

# Run benchmarks on existing deployment
export RUN_LLM_BENCHMARKS=true
export BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b,qwen3.5:9b
docker compose exec alpaca-proxy python3 llm_benchmark_suite.py "$BENCHMARK_MODELS"

# Check results
mkdir -p data/llm_benchmarks
docker cp alpaca-proxy:/app/data/llm_benchmarks ./

# View latest results
cat data/llm_benchmarks/proxy_benchmarks_latest.json | jq .results[0].model
```

### 10. Monitoring and Alerts

Set up monitoring for the benchmark system:

```bash
# Create monitoring script
# monitor.sh

#!/bin/bash

# Check if Alpaca is running
if ! pgrep -f "alpaca-proxy.py" > /dev/null; then
    echo "ERROR: Alpaca proxy is not running"
    exit 1
fi

# Check if benchmark results are being created
RESULTS_DIR="data/llm_benchmarks"
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p $RESULTS_DIR
    echo "Created results directory: $RESULTS_DIR"
fi

# Check if results are recent (within last 4 hours)
LATEST_RESULTS="$RESULTS_DIR/proxy_benchmarks_latest.json"
if [ -f "$LATEST_RESULTS" ]; then
    LAST_MODIFIED=$(stat -c %Y "$LATEST_RESULTS")
    CURRENT_TIME=$(date +%s)
    HOURS_SINCE=$(( (CURRENT_TIME - LAST_MODIFIED) / 3600 ))
    
    if [ $HOURS_SINCE -gt 6 ]; then
        echo "WARNING: No benchmark results for $HOURS_SINCE hours"
    fi
else
    echo "WARNING: No benchmark results file found"
fi

# Check disk space
DISK_USAGE=$(df -h /home/user/code/alpaca | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "WARNING: Disk usage is high ($DISK_USAGE%)"
fi

echo "Monitoring completed successfully"
```

Make executable:

```bash
chmod +x monitor.sh
```

Run with cron:

```bash
# Add to crontab
crontab -e

# Run monitoring every hour
0 * * * * /home/user/code/alpaca/monitor.sh
```

## Quick Start Summary

1. **Set up environment variables** in `.env`
2. **Create all required files** (scripts, Dockerfile, docker-compose.yml)
3. **Run the benchmarks**:
   ```bash
   export RUN_LLM_BENCHMARKS=true
   export BENCHMARK_MODELS=qwen3:8b,qwen2.5-coder:7b
   docker compose up --build
   ```
4. **Monitor results**:
   ```bash
   mkdir -p data/llm_benchmarks
   docker cp alpaca-proxy:/app/data/llm_benchmarks ./
   cat data/llm_benchmarks/proxy_benchmarks_latest.json
   ```

This setup provides a complete, automated LLM benchmarking system integrated with the Alpaca proxy, ready to evaluate and optimize your LLM models for SharedLLM use cases.