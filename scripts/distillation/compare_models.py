#!/usr/bin/env python3
"""
Compare teacher, baseline student, and distilled student models.

This script evaluates all three models on the test set and provides
detailed comparison metrics.

Usage:
    uv run python scripts/distillation/compare_models.py \
      --teacher gemma3:27b \
      --student-baseline models/gemma3-270m-student-v1 \
      --student-distilled models/gemma3-270m-distilled-v1 \
      --test-data data/synthetic/test.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
from tqdm import tqdm
from collections import defaultdict


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_with_ollama(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7
) -> Tuple[str, float]:
    """Generate using Ollama API. Returns (output, latency_ms)."""
    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 256,
            }
        }
    )
    latency = (time.time() - start) * 1000
    response.raise_for_status()
    return response.json()["message"]["content"], latency


def parse_json_output(text: str) -> Dict[str, str]:
    """Parse JSON from model output."""
    try:
        # Try to find JSON in the output
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
        return {}
    except:
        return {}


def compute_metrics(
    predictions: List[Dict[str, str]],
    references: List[Dict[str, str]]
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {
        "exact_match": 0,
        "summary_match": 0,
        "branch_match": 0,
        "valid_json": 0,
    }

    for pred, ref in zip(predictions, references):
        # Valid JSON
        if pred:
            metrics["valid_json"] += 1

            # Exact match
            if pred == ref:
                metrics["exact_match"] += 1

            # Field matches
            if pred.get("summary") == ref.get("summary"):
                metrics["summary_match"] += 1
            if pred.get("branch") == ref.get("branch"):
                metrics["branch_match"] += 1

    # Convert to percentages
    n = len(predictions)
    return {k: (v / n) * 100 for k, v in metrics.items()}


def evaluate_model(
    model_name: str,
    test_data: List[Dict[str, Any]],
    backend: str = "ollama",
    verbose: bool = True
) -> Tuple[Dict[str, float], List[Dict[str, str]], float]:
    """
    Evaluate a model on test data.

    Returns:
        - metrics: Dict of evaluation metrics
        - predictions: List of parsed outputs
        - avg_latency: Average latency in ms
    """
    predictions = []
    references = []
    latencies = []

    iterator = tqdm(test_data, desc=f"Evaluating {model_name}") if verbose else test_data

    for example in iterator:
        messages = example["messages"]

        # Extract reference
        assistant_msg = [m for m in messages if m["role"] == "assistant"]
        if assistant_msg:
            ref_text = assistant_msg[0]["content"]
            ref_json = parse_json_output(ref_text)
            references.append(ref_json)
        else:
            references.append({})

        # Generate prediction
        input_messages = [m for m in messages if m["role"] in ["system", "user"]]

        try:
            if backend == "ollama":
                output, latency = generate_with_ollama(model_name, input_messages)
            else:
                # TODO: Add vLLM support
                output, latency = "", 0

            pred_json = parse_json_output(output)
            predictions.append(pred_json)
            latencies.append(latency)

        except Exception as e:
            if verbose:
                print(f"\nError: {e}")
            predictions.append({})
            latencies.append(0)

    # Compute metrics
    metrics = compute_metrics(predictions, references)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return metrics, predictions, avg_latency


def print_comparison_table(results: Dict[str, Dict]):
    """Print a nice comparison table."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    # Header
    print(f"{'Metric':<25} {'Teacher':<15} {'Student-Base':<15} {'Student-Dist':<15}")
    print("-"*80)

    # Metrics
    metrics = ["exact_match", "summary_match", "branch_match", "valid_json"]
    for metric in metrics:
        row = f"{metric:<25}"
        for model in ["teacher", "student_baseline", "student_distilled"]:
            if model in results:
                value = results[model]["metrics"].get(metric, 0)
                row += f"{value:>14.1f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)

    print("-"*80)

    # Latency
    row = f"{'avg_latency_ms':<25}"
    for model in ["teacher", "student_baseline", "student_distilled"]:
        if model in results:
            value = results[model]["latency"]
            row += f"{value:>14.1f}"
        else:
            row += f"{'N/A':>15}"
    print(row)

    # Speedup
    if "teacher" in results and "student_distilled" in results:
        teacher_latency = results["teacher"]["latency"]
        student_latency = results["student_distilled"]["latency"]
        speedup = teacher_latency / student_latency if student_latency > 0 else 0
        print(f"\nSpeedup: {speedup:.1f}x faster than teacher")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare teacher and student models")
    parser.add_argument(
        "--teacher",
        default="gemma3:27b",
        help="Teacher model name"
    )
    parser.add_argument(
        "--student-baseline",
        type=Path,
        help="Path to baseline student model"
    )
    parser.add_argument(
        "--student-distilled",
        type=Path,
        help="Path to distilled student model"
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/synthetic/test.jsonl"),
        help="Test dataset"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm"],
        default="ollama",
        help="Backend for inference"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparison.json"),
        help="Output file for detailed results"
    )

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_data = load_jsonl(args.test_data)
    print(f"Loaded {len(test_data)} test examples")

    results = {}

    # Evaluate teacher
    if args.teacher:
        print(f"\n{'='*80}")
        print(f"Evaluating Teacher: {args.teacher}")
        print(f"{'='*80}")
        metrics, predictions, latency = evaluate_model(
            args.teacher,
            test_data,
            args.backend
        )
        results["teacher"] = {
            "metrics": metrics,
            "predictions": predictions,
            "latency": latency,
        }
        print(f"\nTeacher Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.1f}%")
        print(f"  avg_latency: {latency:.1f}ms")

    # Evaluate baseline student
    if args.student_baseline:
        print(f"\n{'='*80}")
        print(f"Evaluating Baseline Student: {args.student_baseline}")
        print(f"{'='*80}")
        # TODO: Load and evaluate baseline student model
        # For now, skip
        print("Baseline student evaluation not yet implemented")

    # Evaluate distilled student
    if args.student_distilled:
        print(f"\n{'='*80}")
        print(f"Evaluating Distilled Student: {args.student_distilled}")
        print(f"{'='*80}")
        # TODO: Load and evaluate distilled student model
        # For now, skip
        print("Distilled student evaluation not yet implemented")

    # Print comparison
    if results:
        print_comparison_table(results)

    # Save detailed results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        # Remove predictions to keep file small
        output_results = {
            k: {**v, "predictions": []}
            for k, v in results.items()
        }
        json.dump(output_results, f, indent=2)

    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
