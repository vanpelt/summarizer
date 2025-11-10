import os
from pathlib import Path
from huggingface_hub import InferenceClient

client = InferenceClient()

# Load prompt from prompt.txt
prompt_path = Path(__file__).parent / "prompt.txt"
prompt_content = prompt_path.read_text().strip()

completion = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {
            "role": "user",
            "content": prompt_content
        }
    ],
)

print(completion.choices[0].message)
