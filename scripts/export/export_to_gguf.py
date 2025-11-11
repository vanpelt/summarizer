#!/usr/bin/env python3
"""
Export trained Unsloth model to GGUF format for Ollama
"""

import os
import sys
import torch
from unsloth import FastLanguageModel

# Disable compilation for export
torch._dynamo.config.disable = True

def export_gguf(
    model_path: str,
    output_name: str = "gemma3-summary-v1",
    quantization: str = "Q4_K_M",
):
    """
    Export model to GGUF format

    Args:
        model_path: Path to the trained adapter model
        output_name: Name for the output GGUF file
        quantization: Quantization method (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)
    """
    print("=" * 60)
    print(f"Exporting {model_path} to GGUF")
    print("=" * 60)

    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Export to GGUF
    print(f"\nExporting with quantization: {quantization}")
    print(f"Output name: {output_name}")

    try:
        model.save_pretrained_gguf(
            output_name,
            tokenizer,
            quantization_method=quantization,
        )
        print(f"\n✅ Successfully exported to: {output_name}-{quantization}-unsloth.gguf")

    except Exception as e:
        print(f"\n❌ Export failed with {quantization}: {e}")
        print("\nTrying fallback quantization: Q8_0")

        # Fallback to Q8_0 if unsupported
        model.save_pretrained_gguf(
            output_name,
            tokenizer,
            quantization_method="Q8_0",
        )
        print(f"\n✅ Successfully exported to: {output_name}-Q8_0-unsloth.gguf")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Unsloth model to GGUF")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/gemma3-270m-student-unsloth-v1",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="gemma3-summary-v1",
        help="Output GGUF file name prefix",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "BF16"],
        help="Quantization method",
    )

    args = parser.parse_args()

    export_gguf(
        model_path=args.model_path,
        output_name=args.output_name,
        quantization=args.quantization,
    )
