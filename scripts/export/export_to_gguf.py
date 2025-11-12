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

    # Convert model_path to absolute path to avoid issues when changing directories
    model_path = os.path.abspath(model_path)

    # Create parent output directory (models/gguf/)
    # Unsloth will create the model-specific subdirectory itself
    output_parent = "models/gguf"
    os.makedirs(output_parent, exist_ok=True)

    # Full output directory (for display and Modelfile creation)
    output_dir = f"{output_parent}/{output_name}"
    print(f"\nOutput directory: {output_dir}")

    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Prepare model for inference (important for merged models)
    model = FastLanguageModel.for_inference(model)

    # Export to GGUF
    print(f"\nExporting with quantization: {quantization}")
    print(f"Output directory: {output_dir}")

    try:
        # save_pretrained_gguf takes a directory path where to save the GGUF files
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method=quantization,
        )

        # Unsloth creates the GGUF with the model's base name, not our output_name
        # Find and rename the generated GGUF file
        quant_upper = quantization.upper().replace("-", "_")
        model_base_name = os.path.basename(model_path.rstrip("/"))
        generated_gguf = f"{model_base_name}.{quant_upper}.gguf"
        expected_gguf = f"{output_name}-{quantization}-unsloth.gguf"

        # Move from current directory to output directory with correct name
        if os.path.exists(generated_gguf):
            os.rename(generated_gguf, os.path.join(output_dir, expected_gguf))
            print(f"\n✅ Successfully exported to: {output_dir}/{expected_gguf}")
        else:
            print(f"\n⚠️  Warning: Expected GGUF file not found: {generated_gguf}")
            print(f"   Looking for any .gguf files...")
            import glob
            gguf_files = glob.glob("*.gguf")
            if gguf_files:
                print(f"   Found: {gguf_files}")
                # Move the first one found
                os.rename(gguf_files[0], os.path.join(output_dir, expected_gguf))
                print(f"\n✅ Moved {gguf_files[0]} to: {output_dir}/{expected_gguf}")

        # Create Modelfile
        _create_modelfile(expected_gguf, output_dir)

    except Exception as e:
        print(f"\n❌ Export failed with {quantization}: {e}")
        print("\nTrying fallback quantization: Q8_0")

        # Fallback to Q8_0 if unsupported
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method="Q8_0",
        )

        # Handle Q8_0 filename
        model_base_name = os.path.basename(model_path.rstrip("/"))
        generated_gguf = f"{model_base_name}.Q8_0.gguf"
        expected_gguf = f"{output_name}-Q8_0-unsloth.gguf"

        if os.path.exists(generated_gguf):
            os.rename(generated_gguf, os.path.join(output_dir, expected_gguf))
        else:
            import glob
            gguf_files = glob.glob("*.gguf")
            if gguf_files:
                os.rename(gguf_files[0], os.path.join(output_dir, expected_gguf))

        print(f"\n✅ Successfully exported to: {output_dir}/{expected_gguf}")

        # Create Modelfile
        _create_modelfile(expected_gguf, output_dir)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Location: {output_dir}")
    print("=" * 60)


def _create_modelfile(gguf_filename: str, output_dir: str):
    """Create Modelfile for Ollama import."""
    modelfile_content = f"""FROM {gguf_filename}
TEMPLATE \"\"\"{{{{- $systemPromptAdded := false }}}}
{{{{- range $i, $_ := .Messages }}}}
{{{{- $last := eq (len (slice $.Messages $i)) 1 }}}}
{{{{- if eq .Role "user" }}}}<start_of_turn>user
{{{{- if (and (not $systemPromptAdded) $.System) }}}}
{{{{- $systemPromptAdded = true }}}}
{{{{ $.System }}}}
{{{{ end }}}}
{{{{ .Content }}}}<end_of_turn>
{{{{ if $last }}}}<start_of_turn>model
{{{{ end }}}}
{{{{- else if eq .Role "assistant" }}}}<start_of_turn>model
{{{{ .Content }}}}{{{{ if not $last }}}}<end_of_turn>
{{{{ end }}}}
{{{{- end }}}}
{{{{- end }}}}
\"\"\"
PARAMETER stop "<end_of_turn>"
PARAMETER top_k 64
PARAMETER top_p 0.95
"""
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    print(f"✅ Created Modelfile at: {modelfile_path}")


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
