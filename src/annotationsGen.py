import os
import json
from openai import OpenAI




client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY"))

SOURCE_FOLDER = "chunks"
OUTPUT_FOLDER = "../Data/annotationsGPT4.1_caribbean"
MAX_FILES = 12

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Initial Prompt
# -----------------------------

SYSTEM_PROMPT = """
You are an advanced information extraction model trained to annotate humanitarian-related quantities in text. You need to extract
and annotate explicit numerical values (written in digits, letters, or a combination) from the given text. When annotating, use the
exact form in which the number appears in the text (e.g., "four" instead of "4").
Numbers that represent dates (years like "2022", full dates like "12/10/2023", or even "March 15") must never be annotated.
These are not quantities and must be completely ignored.
Modifiers, which are words that modify a number, should be limited to terms like "over", "about", "on average", or "at least". These
are the only cases where the modifier field should be included. Words like "total", "to date", and similar terms are not considered
modifiers, and should not be annotated as such. If there is no modifier, omit the modifier attribute altogether.
For each number, follow these steps:
• Quantity: The number itself, in its exact form (e.g., "four", "10", "20,000").
• Unit: The entity or measure the number refers to (e.g., "people", "tents", "US$", "per cent", "tonnes of debris").
• Event Type: The nature of the event:
– EventP: When the number refers to people-related events.
– EventA: When the number refers to aid-related events (supplies, assistance).
– EventO: For other events that don’t fit in the previous categories.
• Event Description: A brief description of the event (e.g., "displaced", "supplied", "surveyed").
• Modifier: If applicable, include the modifier (e.g., "over", "about").

Quantity (STRICT FIELD RULES):
- The quantity MUST be copied EXACTLY as it appears in the text (preserve spaces, commas, and formatting; never normalize).
- ALWAYS output quantity as a STRING inside quotes, including standalone integers.
- The quantity field contains ONLY the number itself:
  * The modifier goes in the modifier field, NEVER inside quantity.
    Correct: {"quantity": "5 million", "modifier": "more than", ...}
    Wrong:   {"quantity": "more than 5 million", ...}
  * Currency symbols and units go in the unit field, NEVER inside quantity.
    Correct: {"quantity": "74 million", "unit": "US$", ...}
    Wrong:   {"quantity": "US$74 million", ...}
    Correct: {"quantity": "5.5", "unit": "tons of supplies", ...}
    Wrong:   {"quantity": "5.5 tons", ...}
- A range like "between US$6 and 7 billion" or "28–32 per cent" is TWO annotations,
  one per bound, each with its own unit (e.g. "US$", "per cent").

Examples:
Correct: "300 000"
Wrong: 300000
Wrong: "300000"
Wrong: 300000.0
The output must be a JSON array of objects in the following format:
{
"quantity": "<String>",
"unit": "<Unit>",
"eventType": "<EventP | EventA | EventO>",
"eventDescription": "<Short description of the event>",
"modifier": "<Modifier (if any)>"
}
"""

# -----------------------------
# Processing
# -----------------------------

files = sorted([f for f in os.listdir(SOURCE_FOLDER) if f.endswith(".txt")])[:MAX_FILES]
for filename in files:

    try:

        with open(os.path.join(SOURCE_FOLDER, filename), "r", encoding="utf-8") as f:
            text = f.read()

        prompt = SYSTEM_PROMPT + "\n\nText:\n" + text + "\n\nReturn ONLY a valid JSON array."

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You must respond ONLY with valid JSON. Do not include explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # tolerate markdown code fences around the JSON
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename.replace(".txt", ".json")
        )

        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2)

        print(f"✔ Processed {filename}")

    except Exception as e:
        print(f"Error with {filename}: {e}")

print("Done!")