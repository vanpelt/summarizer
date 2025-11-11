"""
Evaluate summarizer models using Ollama API.

This script evaluates both the fine-tuned gemma3-summary-v1 model
and the stock gemma3:270m model for generating JSON summaries and branch names.
"""

import weave
import time
import re
import asyncio
import argparse
import requests
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_prompt_template() -> str:
    """Load the prompt template from prompt.txt."""
    prompt_path = Path(__file__).parent.parent / "prompt.txt"
    with open(prompt_path, "r") as f:
        # Read everything except the example request at the end
        content = f.read()
        # Extract just the template (everything before "Request:")
        lines = content.split("\n")
        template_lines = []
        for line in lines:
            if line.strip() == "Request:":
                break
            template_lines.append(line)
        return "\n".join(template_lines).strip() + "\n\nRequest:\n"


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSONL file.

    Each line has format:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "USER REQUEST"},
        {"role": "assistant", "content": '{"summary": "...", "branch": "..."}'}
    ]}
    """
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            messages = data["messages"]

            # Extract user request
            user_message = next((m for m in messages if m["role"] == "user"), None)
            assistant_message = next((m for m in messages if m["role"] == "assistant"), None)

            if user_message and assistant_message:
                request = user_message["content"]
                expected_output = json.loads(assistant_message["content"])

                dataset.append({
                    "request": request,
                    "expected_summary": expected_output.get("summary", ""),
                    "expected_branch": expected_output.get("branch", ""),
                })

    return dataset


class SummarizerModelOllama(weave.Model):
    """
    Weave Model wrapper for summarizer models using Ollama API.
    """

    ollama_url: str = "http://localhost:11434"
    model_name: str = "gemma3-summary-v1"
    temperature: float = 0.1
    prompt_template: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Initialize model after Pydantic construction."""
        super().model_post_init(__context)
        # Load prompt template if not provided
        if not self.prompt_template:
            self.prompt_template = load_prompt_template()
        self._test_connection()

    def _test_connection(self):
        """Test Ollama connection."""
        print("="*80)
        print("Testing Ollama connection")
        print(f"URL: {self.ollama_url}")
        print(f"Model: {self.model_name}")
        print("="*80)

        try:
            # Test API endpoint
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if self.model_name in model_names:
                print(f"✅ Model '{self.model_name}' is available")
            else:
                print(f"⚠️  Model '{self.model_name}' not found")
                print(f"Available models: {', '.join(model_names)}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print(f"\nMake sure Ollama is running and accessible at {self.ollama_url}")
            raise

        print("="*80 + "\n")

    @weave.op()
    def predict(self, request: str, expected_summary: str = "", expected_branch: str = "") -> Dict:
        """
        Generate summary and branch name from a request.

        Args:
            request: User request describing the desired code change
            expected_summary: Expected summary (for reference, not used in prediction)
            expected_branch: Expected branch (for reference, not used in prediction)

        Returns:
            Dictionary with generated JSON and metadata
        """
        start_time = time.time()

        # Construct full prompt from template
        full_prompt = self.prompt_template + request

        # Retry logic: up to 3 attempts
        max_retries = 3
        generated_text = None
        timed_out = False
        last_error = None

        for attempt in range(max_retries):
            gen_start = time.time()
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                        "format": {
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "branch": {"type": "string"},
                            },
                            "required": ["summary", "branch"]
                        },
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": 256,
                        }
                    },
                    timeout=60,  # 1 minute timeout
                )
                response.raise_for_status()
                result = response.json()

                generated_text = result["response"]

                # Success! Break out of retry loop
                break

            except requests.exceptions.Timeout as e:
                timed_out = True
                last_error = f"Timeout after 60s"
                print(f"⏰ Timeout on attempt {attempt + 1}/{max_retries} for request: {request[:50]}...")
                if attempt < max_retries - 1:
                    print(f"   Retrying...")
                else:
                    print(f"   Max retries reached")
                    generated_text = f"Error: Timeout after {max_retries} attempts"

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"❌ Ollama API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying...")
                else:
                    print(f"   Max retries reached")
                    generated_text = f"Error: {e}"

        generation_time = time.time() - gen_start
        total_time = time.time() - start_time

        return {
            "generated_output": generated_text or f"Error: {last_error}",
            "total_time_seconds": round(total_time, 2),
            "generation_time_seconds": round(generation_time, 2),
            "timed_out": timed_out,
            "expected_summary": expected_summary,
            "expected_branch": expected_branch,
        }


# Weave Scorer Functions
@weave.op()
def json_validity_score(model_output: Dict, request: str, expected_summary: str, expected_branch: str) -> Dict:
    """Check if the output is valid JSON with required fields."""
    generated = model_output.get("generated_output", "")

    # Try to parse as JSON
    is_valid_json = False
    has_summary = False
    has_branch = False
    parsed_json = None
    summary = None
    branch = None

    try:
        parsed_json = json.loads(generated)
        is_valid_json = True
        has_summary = "summary" in parsed_json
        has_branch = "branch" in parsed_json
        summary = parsed_json.get("summary", "")
        branch = parsed_json.get("branch", "")
    except json.JSONDecodeError:
        # Try to extract JSON from text if it's wrapped in other content
        json_match = re.search(r'\{[^{}]*"summary"[^{}]*"branch"[^{}]*\}', generated)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
                is_valid_json = True
                has_summary = "summary" in parsed_json
                has_branch = "branch" in parsed_json
                summary = parsed_json.get("summary", "")
                branch = parsed_json.get("branch", "")
            except json.JSONDecodeError:
                pass

    return {
        "is_valid_json": is_valid_json,
        "has_summary_field": has_summary,
        "has_branch_field": has_branch,
        "has_both_fields": has_summary and has_branch,
        "extracted_summary": summary,
        "extracted_branch": branch,
    }


@weave.op()
def summary_quality_score(model_output: Dict, request: str, expected_summary: str, expected_branch: str) -> Dict:
    """Evaluate the quality of the summary field."""
    generated = model_output.get("generated_output", "")

    # First parse JSON to get summary
    summary = None
    try:
        parsed = json.loads(generated)
        summary = parsed.get("summary", "")
    except json.JSONDecodeError:
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"summary"[^{}]*"branch"[^{}]*\}', generated)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                summary = parsed.get("summary", "")
            except json.JSONDecodeError:
                pass

    if summary is None:
        return {
            "word_count": 0,
            "is_valid_word_count": False,
            "is_title_case": False,
            "has_no_punctuation": False,
            "summary_quality_score": 0,
        }

    # Count words
    words = summary.split()
    word_count = len(words)
    is_valid_word_count = 2 <= word_count <= 4

    # Check title case (each word starts with uppercase)
    is_title_case = all(word[0].isupper() if word else False for word in words)

    # Check for punctuation (should not have any at the end)
    has_no_punctuation = not summary.strip().endswith(('.', '!', '?', ',', ';'))

    # Calculate quality score
    quality_score = 0
    if is_valid_word_count:
        quality_score += 40
    if is_title_case:
        quality_score += 30
    if has_no_punctuation:
        quality_score += 30

    return {
        "word_count": word_count,
        "is_valid_word_count": is_valid_word_count,
        "is_title_case": is_title_case,
        "has_no_punctuation": has_no_punctuation,
        "summary_quality_score": quality_score,
    }


@weave.op()
def branch_quality_score(model_output: Dict, request: str, expected_summary: str, expected_branch: str) -> Dict:
    """Evaluate the quality of the branch field."""
    generated = model_output.get("generated_output", "")

    # First parse JSON to get branch
    branch = None
    try:
        parsed = json.loads(generated)
        branch = parsed.get("branch", "")
    except json.JSONDecodeError:
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"summary"[^{}]*"branch"[^{}]*\}', generated)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                branch = parsed.get("branch", "")
            except json.JSONDecodeError:
                pass

    if branch is None:
        return {
            "slash_count": 0,
            "has_single_slash": False,
            "word_count_after_slash": 0,
            "is_valid_word_count": False,
            "is_kebab_case": False,
            "is_lowercase": False,
            "has_valid_chars": False,
            "has_valid_prefix": False,
            "branch_quality_score": 0,
        }

    # Count slashes
    slash_count = branch.count("/")
    has_single_slash = slash_count == 1

    # Extract part after slash
    parts = branch.split("/")
    prefix = parts[0] if len(parts) > 0 else ""
    suffix = parts[1] if len(parts) > 1 else ""

    # Count words after slash (separated by -)
    words_after_slash = suffix.split("-") if suffix else []
    word_count_after_slash = len(words_after_slash)
    is_valid_word_count = 1 <= word_count_after_slash <= 4

    # Check kebab-case (words separated by -)
    is_kebab_case = "-" in suffix or word_count_after_slash == 1

    # Check lowercase
    is_lowercase = branch.islower()

    # Check valid characters (only a-z, 0-9, -, /)
    has_valid_chars = bool(re.match(r'^[a-z0-9/-]+$', branch))

    # Check for valid prefix (bug/, feat/, chore/, test/, etc.)
    valid_prefixes = ['bug', 'feat', 'feature', 'fix', 'chore', 'test', 'docs', 'refactor', 'style', 'perf']
    has_valid_prefix = prefix in valid_prefixes

    # Calculate quality score
    quality_score = 0
    if has_single_slash:
        quality_score += 20
    if is_valid_word_count:
        quality_score += 20
    if is_kebab_case:
        quality_score += 15
    if is_lowercase:
        quality_score += 15
    if has_valid_chars:
        quality_score += 15
    if has_valid_prefix:
        quality_score += 15

    return {
        "slash_count": slash_count,
        "has_single_slash": has_single_slash,
        "word_count_after_slash": word_count_after_slash,
        "is_valid_word_count": is_valid_word_count,
        "is_kebab_case": is_kebab_case,
        "is_lowercase": is_lowercase,
        "has_valid_chars": has_valid_chars,
        "has_valid_prefix": has_valid_prefix,
        "branch_quality_score": quality_score,
    }


@weave.op()
def overall_quality_score(model_output: Dict, request: str, expected_summary: str, expected_branch: str) -> Dict:
    """Combined quality score based on JSON validity, summary, and branch quality."""
    # Get individual scores
    json_score = json_validity_score(model_output, request, expected_summary, expected_branch)
    summary_score = summary_quality_score(model_output, request, expected_summary, expected_branch)
    branch_score = branch_quality_score(model_output, request, expected_summary, expected_branch)

    # Calculate overall score (0-100)
    score = 0

    # JSON validity: 20 points
    if json_score["has_both_fields"]:
        score += 20

    # Summary quality: 40 points
    score += summary_score["summary_quality_score"] * 0.4

    # Branch quality: 40 points
    score += branch_score["branch_quality_score"] * 0.4

    return {
        "overall_quality_score": round(score, 2),
        "is_high_quality": score >= 80,
        "is_acceptable_quality": score >= 60,
        "is_perfect": score >= 95,
    }


def main():
    """Main evaluation script using Ollama API for summarizer models."""
    parser = argparse.ArgumentParser(description="Evaluate summarizer models via Ollama")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--dataset",
        choices=["test", "val", "both"],
        default="test",
        help="Which dataset to use (default: test)"
    )
    parser.add_argument(
        "--models",
        default="gemma3-summary-v1:latest,gemma3:270m",
        help="Comma-separated list of model names to evaluate (default: gemma3-summary-v1,gemma3:270m)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (default: all)"
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Summarizer Model Evaluation")
    print("="*80 + "\n")

    # Initialize Weave
    weave_project = "summarizer"
    print("Initializing Weave...")
    weave.init(weave_project)
    print(f"✅ Weave initialized: {weave_project}\n")

    # Load datasets
    data_dir = Path(__file__).parent.parent / "data" / "gpt5nano"
    datasets = []

    if args.dataset in ["test", "both"]:
        test_data = load_dataset(data_dir / "test.jsonl")
        if args.limit:
            test_data = test_data[:args.limit]
        datasets.append(("test", test_data))

    if args.dataset in ["val", "both"]:
        val_data = load_dataset(data_dir / "val.jsonl")
        if args.limit:
            val_data = val_data[:args.limit]
        datasets.append(("val", val_data))

    # Parse model names
    model_names = [m.strip() for m in args.models.split(",")]

    # Run evaluations for each model and dataset
    all_results = []

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*80}\n")

        # Create model
        model = SummarizerModelOllama(
            ollama_url=args.ollama_url,
            model_name=model_name,
        )

        for dataset_name, dataset in datasets:
            print(f"\n{'-'*80}")
            print(f"Dataset: {dataset_name} ({len(dataset)} examples)")
            print(f"{'-'*80}\n")

            # Create Weave Evaluation
            evaluation = weave.Evaluation(
                name=f"{model_name.replace(':', '-').replace('/', '-')}-{dataset_name}",
                dataset=dataset,
                scorers=[
                    json_validity_score,
                    summary_quality_score,
                    branch_quality_score,
                    overall_quality_score,
                ],
            )

            # Run evaluation
            print(f"Running evaluation...")
            eval_start = time.time()
            results = asyncio.run(evaluation.evaluate(model))
            eval_time = time.time() - eval_start

            # Print summary
            print(f"\n{'='*80}")
            print(f"Evaluation Complete: {model_name} on {dataset_name}")
            print(f"{'='*80}")
            print(f"Total evaluation time: {eval_time/60:.2f} minutes ({eval_time:.2f} seconds)")
            print(f"Average time per example: {eval_time/len(dataset):.2f} seconds")
            print(f"{'='*80}\n")

            all_results.append({
                "model": model_name,
                "dataset": dataset_name,
                "results": results,
                "time": eval_time,
            })

    # Final summary
    print(f"\n{'='*80}")
    print("All Evaluations Complete!")
    print(f"{'='*80}")
    print(f"Evaluated {len(model_names)} model(s) on {len(datasets)} dataset(s)")
    print(f"\nView detailed results in Weave:")
    print(f"https://wandb.ai/{weave_project}")
    print(f"{'='*80}\n")

    return all_results


if __name__ == "__main__":
    main()
