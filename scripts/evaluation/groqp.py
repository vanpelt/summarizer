import os
from pathlib import Path
from groq import Groq

client = Groq()

# Load prompt from prompt.txt
prompt_path = Path(__file__).parent / "prompt.txt"
prompt_content = prompt_path.read_text().strip()

completion = client.chat.completions.create(
    model=os.getenv("MODEL", "llama-3.1-8b-instant"),
    messages=[
        {
            "role": "user",
            "content": prompt_content
        }
    ],
    response_format={"type": "json_object"}
)

print(completion.choices[0].message)
