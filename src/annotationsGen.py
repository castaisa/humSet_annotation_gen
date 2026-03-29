import os
import json
from openai import OpenAI




client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

SOURCE_FOLDER = "../Data/text_sources"
OUTPUT_FOLDER = "../Data/annotationsGPT4.1"
MAX_FILES = 1000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Initial Prompt
# -----------------------------

SYSTEM_PROMPT = """
You are an advanced information extraction model trained to annotate humanitarian-related quantities in text. You need to extract
and annotate explicit numerical values (written in digits, letters, or a combination) from the given text. When annotating, use the
exact form in which the number appears in the text (e.g., "four" instead of "4").
If the number appears as a standalone integer (e.g., 30, 10, 198), annotate it as an integer (without quotes) in the output JSON.
All other numbers (fractions, percentages, numbers with units like "4 kg") should still be represented as strings. Besides, numbers
that represent dates (years like "2022", full dates like "12/10/2023", or even "March 15") must never be annotated. These are not
quantities and must be completely ignored.
Modifiers, which are words that modify a number, should be limited to terms like "over", "about", "on average", or "at least". These
are the only cases where the modifier field should be included. Words like "total", "to date", and similar terms are not considered
modifiers, and should not be annotated as such. If there is no modifier, omit the modifier attribute altogether.
For each number, follow these steps:
• Quantity: The number itself, in its exact form (e.g., "four", "10", "20,000").
• Unit: The entity the number refers to (e.g., "people", "tents").
• Event Type: The nature of the event:
– EventP: When the number refers to people-related events.
– EventA: When the number refers to aid-related events (supplies, assistance).
– EventO: For other events that don’t fit in the previous categories.
• Event Description: A brief description of the event (e.g., "displaced", "supplied", "surveyed").
• Modifier: If applicable, include the modifier (e.g., "over", "about").

Quantity (STRICT TEXT RULE):
- The quantity MUST be copied EXACTLY as it appears in the text.
- Preserve spaces, commas, and formatting.
- NEVER normalize numbers.
- NEVER remove spaces.
- NEVER convert to integers or numeric values.
- ALWAYS output quantity as a STRING inside quotes.

Examples:
Correct: "300 000"
Wrong: 300000
Wrong: "300000"
Wrong: 300000.0
The output should be in the following JSON format:
1 {
2 "quantity": <String>,
3 "unit": "<Unit>",
4 "eventType": "<EventP | EventA | EventO>",
5 "eventDescription": "<Short description of the event>",
6 "modifier": "<Modifier (if any)>"
7 }
Now, I will present you with some texts. Acknowledge this.
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