#!/usr/bin/env python3
"""
Contrastive narrative similarity method using non-curriculum demos.
Replicates the functionality of contrastive.ipynb but uses
`contra_demos_copy.jsonl` as demonstrations and evaluates on the SemEval
dev Track A set (`dev_track_a.jsonl`).
"""

import os
import json
import time
from typing import Dict, Any, List, Optional

import dataclasses
import yaml
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI, BadRequestError
from openai.types.chat import ChatCompletion
from tqdm import tqdm


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
def chat(client: OpenAI, delay: float | int, **kwargs) -> ChatCompletion | None:
    """Make a chat completion request with retry logic and minimum delay."""
    try:
        with MinimumDelay(delay):
            return client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        print(f"Bad Request: {e}")
        if "safety" in e.message:
            return None
        raise e
    except Exception as e:
        print(f"Exception: {e}")
        raise e


def read_jsonl(path):
    """Read JSONL file line by line, skipping empty lines and handling JSON errors."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # skip empty lines
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping line {lineno} in {os.path.abspath(path)} due to JSON error: {e}")
                continue


def write_jsonl(path, data):
    """Write data to JSONL file."""
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")


@dataclasses.dataclass
class ChatCompletionConfig:
    """Configuration for chat completion requests."""
    seed: int
    delay: int
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str
    user_prompt: str
    response_format: dict | None = None


def add_demos(demo_file_path, message, user_prompt_template):
    """Add demo examples to the message list for few-shot learning."""
    for demo in read_jsonl(demo_file_path):
        message.append({
            "role": "user",
            "content": user_prompt_template.format(
                anchor_story=demo["anchor_text"],
                story_a=demo["text_a"],
                story_b=demo["text_b"]
            )
        })
        message.append({
            "role": "assistant",
            "content": json.dumps(demo["output"])
        })
    return message


def main():
    """Main execution function."""
    # Load configuration from YAML
    CONFIG_FILE_PATH = 'contra.yaml'
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = yaml.safe_load(f)
        CONFIG = ChatCompletionConfig(**config)
    
    SYSTEM_PROMPT = CONFIG.system_prompt
    USER_PROMPT_TEMPLATE = CONFIG.user_prompt

    # Input and output files
    INPUT_FILE = 'data/dev_track_a.jsonl'
    OUTPUT_FILE = 'output/contrastive_results_dev_track_a.jsonl'
    DEMO_FILE = 'contra_demos_1stage.jsonl'

    # Initialize OpenAI client with API key from environment (anonymized)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY environment variable is not set')
    client = OpenAI(api_key=api_key)
    
    all_results = []
    
    print(f"Processing {INPUT_FILE} using {DEMO_FILE}...")
    print(f"Total examples to process: ", end="", flush=True)
    
    # Count total examples first
    total_examples = sum(1 for _ in read_jsonl(INPUT_FILE))
    print(f"{total_examples}")
    
    # Process each example with progress bar
    for idx, line in enumerate(
        tqdm(read_jsonl(INPUT_FILE), total=total_examples, desc="Examples", unit="ex"),
        start=1,
    ):
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages = add_demos(DEMO_FILE, messages, USER_PROMPT_TEMPLATE)
        messages.append({
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                anchor_story=line["anchor_text"],
                story_a=line["text_a"],
                story_b=line["text_b"]
            )
        })

        response = chat(
            client,
            delay=CONFIG.delay,
            model=CONFIG.model,
            messages=messages,
            temperature=CONFIG.temperature,
            response_format=CONFIG.response_format,
            seed=CONFIG.seed,
        )

        if response is not None:
            try:
                output_content = json.loads(response.choices[0].message.content)
                all_results.append(
                    {
                        "input": line,
                        "output": output_content,
                    }
                )
            except json.JSONDecodeError as e:
                print(f"  ✗ JSON decode error: {e}")
                all_results.append(
                    {
                        "input": line,
                        "output": None,
                    }
                )
        else:
            all_results.append(
                {
                    "input": line,
                    "output": None,
                }
            )
    
    # Write results
    print(f"\nWriting results to {OUTPUT_FILE}...")
    write_jsonl(OUTPUT_FILE, all_results)
    print(f"✓ Done! Processed {len(all_results)} examples.")
    
    # Calculate and print metrics
    print("\n" + "="*50)
    print("METRICS")
    print("="*50)
    
    y_true = []
    y_pred = []
    
    for result in all_results:
        if result["output"] is None:
            continue
        
        # Map True Label: True -> 'A', False -> 'B'
        true_closer = result['input'].get('text_a_is_closer')
        if true_closer is None:
            continue
            
        actual = 'A' if true_closer is True else 'B'
        predicted = result['output'].get('result', {}).get('selected_story')
        
        if predicted:
            y_true.append(actual)
            y_pred.append(predicted)
    
    if y_true and y_pred:
        from sklearn.metrics import classification_report, accuracy_score
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=['A', 'B']))
    else:
        print("No valid predictions to calculate metrics.")


if __name__ == "__main__":
    main()
