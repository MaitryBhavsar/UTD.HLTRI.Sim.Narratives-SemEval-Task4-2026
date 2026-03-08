#!/usr/bin/env python3
"""
Contrastive narrative similarity method using Gemini instead of OpenAI.

Copy of `contrastive_nocurriculum.py` but calling Google Gemini.
Same prompts, demos, and JSONL I/O; only the API client differs.
"""

import os
import json
import time
from typing import Dict, Any, List

import dataclasses

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Optional; environment variables can be set without python-dotenv.
    pass

import yaml
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
import google.generativeai as genai


GEMINI_MODEL = "gemini-2.5-flash"

# API key must come from environment; no hardcoded secrets.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running this script.")


class MinimumDelay:
    """Context manager to enforce minimum delay between API calls."""

    def __init__(self, delay: float | int):
        self.delay = delay
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        seconds = end - self.start
        if self.delay > seconds:
            time.sleep(self.delay - seconds)


@retry(wait=wait_random_exponential(min=1, max=90), stop=stop_after_attempt(3))
def generate_with_gemini(model, delay: float | int, contents: str):
    """Generate response using Gemini API with retry logic and minimum delay."""
    try:
        with MinimumDelay(delay):
            response = model.generate_content(contents=contents)
            return response
    except Exception as e:  # noqa: BLE001
        print(f"Exception: {e}")
        raise e


def read_jsonl(path: str):
    """Read JSONL file line by line, skipping empty lines and handling JSON errors."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping line {lineno} in {os.path.abspath(path)} due to JSON error: {e}")
                continue


def write_jsonl(path: str, data: List[Dict[str, Any]]):
    """Write data to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


@dataclasses.dataclass
class ChatCompletionConfig:
    """Configuration for chat completion-style requests."""

    seed: int
    delay: int
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str
    user_prompt: str
    response_format: dict | None = None


def add_demos(demo_file_path: str, message_parts: List[str], user_prompt_template: str) -> List[str]:
    """Add demo examples to the prompt parts for few-shot learning."""
    for demo in read_jsonl(demo_file_path):
        user_content = user_prompt_template.format(
            anchor_story=demo["anchor_text"],
            story_a=demo["text_a"],
            story_b=demo["text_b"],
        )
        message_parts.append(f"User:\n{user_content}\n\n")
        message_parts.append(f"Assistant:\n{json.dumps(demo['output'], ensure_ascii=False)}\n\n")
    return message_parts


def build_gemini_prompt(
    system_prompt: str,
    user_prompt_template: str,
    demo_file_path: str,
    anchor_story: str,
    story_a: str,
    story_b: str,
) -> str:
    """Build full prompt for Gemini (system + demos + current example)."""
    parts: List[str] = [system_prompt, "\n\n---\n\n"]
    parts.append("Here are some examples:\n\n")
    parts = add_demos(demo_file_path, parts, user_prompt_template)
    parts.append("---\n\nNow respond to this example:\n\n")
    parts.append(
        user_prompt_template.format(
            anchor_story=anchor_story,
            story_a=story_a,
            story_b=story_b,
        )
    )
    parts.append("\n\nRespond with valid JSON only (no markdown, no extra text).")
    return "".join(parts)


def parse_gemini_response(response_text: str) -> Dict[str, Any] | None:
    """Parse Gemini response text as JSON. Handles markdown code blocks."""
    if not response_text or not response_text.strip():
        return None
    text = response_text.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip() if end != -1 else text[start:].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip() if end != -1 else text[start:].strip()
    if not text.startswith("{"):
        idx = text.find("{")
        if idx != -1:
            end = text.rfind("}")
            if end != -1:
                text = text[idx : end + 1]
            else:
                text = text[idx:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main() -> None:
    """Run the Gemini contrastive judge on a Track A JSONL file."""
    # Load configuration from YAML (same fields as main OpenAI method).
    CONFIG_FILE_PATH = "contra.yaml"
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        CONFIG = ChatCompletionConfig(**config)

    SYSTEM_PROMPT = CONFIG.system_prompt
    USER_PROMPT_TEMPLATE = CONFIG.user_prompt

    # Input and output files
    INPUT_FILE = "data/dev_track_a.jsonl"
    OUTPUT_FILE = "output/contrastive_results_dev_track_a_gemini.jsonl"
    DEMO_FILE = "contra_demos_1stage.jsonl"

    # Initialize Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": CONFIG.temperature,
        "max_output_tokens": CONFIG.max_tokens,
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=generation_config,
    )

    all_results: List[Dict[str, Any]] = []

    print(f"Processing {INPUT_FILE} using {DEMO_FILE} (Gemini: {GEMINI_MODEL})...")
    print("Total examples to process: ", end="", flush=True)

    total_examples = sum(1 for _ in read_jsonl(INPUT_FILE))
    print(f"{total_examples}")

    for line in tqdm(read_jsonl(INPUT_FILE), total=total_examples, desc="Examples", unit="ex"):
        prompt = build_gemini_prompt(
            SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE,
            DEMO_FILE,
            line["anchor_text"],
            line["text_a"],
            line["text_b"],
        )

        try:
            response = generate_with_gemini(model, delay=CONFIG.delay, contents=prompt)
            raw_text = response.text if response and hasattr(response, "text") else None
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ API error: {e}")
            raw_text = None

        if raw_text:
            output_content = parse_gemini_response(raw_text)
            all_results.append({"input": line, "output": output_content})
        else:
            all_results.append({"input": line, "output": None})

    print(f"\nWriting results to {OUTPUT_FILE}...")
    write_jsonl(OUTPUT_FILE, all_results)
    print(f"✓ Done! Processed {len(all_results)} examples.")

    # Calculate and print metrics when gold labels are available.
    print("\n" + "=" * 50)
    print("METRICS")
    print("=" * 50)

    y_true: List[str] = []
    y_pred: List[str] = []

    for result in all_results:
        if result["output"] is None:
            continue
        true_closer = result["input"].get("text_a_is_closer")
        if true_closer is None:
            continue
        actual = "A" if true_closer is True else "B"
        predicted = result["output"].get("result", {}).get("selected_story")
        if predicted in ("A", "B"):
            y_true.append(actual)
            y_pred.append(predicted)

    if y_true and y_pred:
        from sklearn.metrics import accuracy_score, classification_report

        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=["A", "B"]))
    else:
        print("No valid predictions to calculate metrics.")


if __name__ == "__main__":
    main()

