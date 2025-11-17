#!/usr/bin/env python3
"""
Export trained Unsloth model to GGUF format for Ollama
"""

import os
import sys
import torch
from pathlib import Path
from unsloth import FastLanguageModel

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import SYSTEM_PROMPT, SYSTEM_PROMPT_TWO_LINE

# Disable compilation for export
torch._dynamo.config.disable = True


def detect_model_format(model_path: str) -> str:
    """
    Detect whether a model was trained on JSON or two-line format.

    Heuristic: Check if the model path contains 'two-line' or 'twoline'.

    Args:
        model_path: Path to the model directory

    Returns:
        'two-line' or 'json'
    """
    model_path_lower = model_path.lower()
    if 'two-line' in model_path_lower or 'twoline' in model_path_lower:
        return 'two-line'
    return 'json'


def export_gguf(
    model_path: str,
    output_name: str = "gemma3-summary-v1",
    quantization: str = "Q4_K_M",
    format: str = "auto",
):
    """
    Export model to GGUF format

    Args:
        model_path: Path to the trained adapter model
        output_name: Name for the output GGUF file
        quantization: Quantization method (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)
        format: Output format ('json', 'two-line', or 'auto' to detect from path)
    """
    print("=" * 60)
    print(f"Exporting {model_path} to GGUF")
    print("=" * 60)

    # Detect or use specified format
    if format == "auto":
        detected_format = detect_model_format(model_path)
        print(f"Auto-detected format: {detected_format}")
    else:
        detected_format = format
        print(f"Using specified format: {detected_format}")

    # Select appropriate system prompt
    if detected_format == "two-line":
        system_prompt = SYSTEM_PROMPT_TWO_LINE
        print("Using TWO-LINE system prompt")
    else:
        system_prompt = SYSTEM_PROMPT
        print("Using JSON system prompt")

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

        # Standard naming: gemma3-270m-summarizer-Q4_KM.gguf
        expected_gguf = f"gemma3-270m-summarizer-{quant_upper}.gguf"

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
        _create_modelfile(expected_gguf, output_dir, system_prompt)

        # Create HuggingFace config files (system prompt + params)
        _create_hf_config_files(output_dir, system_prompt)

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
        expected_gguf = f"gemma3-270m-summarizer-Q8_0.gguf"

        if os.path.exists(generated_gguf):
            os.rename(generated_gguf, os.path.join(output_dir, expected_gguf))
        else:
            import glob
            gguf_files = glob.glob("*.gguf")
            if gguf_files:
                os.rename(gguf_files[0], os.path.join(output_dir, expected_gguf))

        print(f"\n✅ Successfully exported to: {output_dir}/{expected_gguf}")

        # Create Modelfile
        _create_modelfile(expected_gguf, output_dir, system_prompt)

        # Create HuggingFace config files (system prompt + params)
        _create_hf_config_files(output_dir, system_prompt)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Location: {output_dir}")
    print("=" * 60)


def _create_modelfile(gguf_filename: str, output_dir: str, system_prompt: str):
    """Create Modelfile for Ollama import with embedded default system prompt."""
    # Escape the default system prompt for Modelfile SYSTEM directive
    # This makes it available as the default system message
    default_system = system_prompt.strip()

    modelfile_content = f"""FROM {gguf_filename}

# Default system prompt (used when user doesn't provide one)
SYSTEM \"\"\"
{default_system}
\"\"\"

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


def _create_hf_config_files(output_dir: str, system_prompt: str):
    """
    Create HuggingFace configuration files for Ollama integration.

    Creates three files:
    - 'system': Contains the default system prompt
    - 'params': JSON file with sampling parameters
    - 'template': Go template for chat formatting (matches gemma3:270m exactly)

    These files allow HuggingFace to configure the model when used with Ollama
    via 'ollama run hf.co/username/repo'
    """
    import json

    # Create 'system' file with default system prompt
    system_path = os.path.join(output_dir, "system")
    with open(system_path, 'w') as f:
        f.write(system_prompt.strip())
    print(f"✅ Created HuggingFace system file at: {system_path}")

    # Create 'params' file with sampling parameters
    # Matching official Ollama gemma3:270m parameters exactly
    params = {
        "top_k": 64,
        "top_p": 0.95,
        "stop": ["<end_of_turn>"]
    }

    params_path = os.path.join(output_dir, "params")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"✅ Created HuggingFace params file at: {params_path}")

    # Create 'template' file with proper Gemma3 chat template
    # This matches the official gemma3:270m template exactly
    # Key feature: Handles system prompt insertion with $systemPromptAdded flag
    template_content = """{{- $systemPromptAdded := false }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<start_of_turn>user
{{- if (and (not $systemPromptAdded) $.System) }}
{{- $systemPromptAdded = true }}
{{ $.System }}
{{ end }}
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""

    template_path = os.path.join(output_dir, "template")
    with open(template_path, 'w') as f:
        f.write(template_content)
    print(f"✅ Created HuggingFace template file at: {template_path}")


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
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "json", "two-line"],
        help="Output format: 'auto' detects from model path, 'json' or 'two-line' to specify (default: auto)",
    )

    args = parser.parse_args()

    export_gguf(
        model_path=args.model_path,
        output_name=args.output_name,
        quantization=args.quantization,
        format=args.format,
    )
