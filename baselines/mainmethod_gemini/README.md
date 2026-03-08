# SemEval-2026 Task 4 Track A — Contrastive LLM judge (Gemini)

This baseline mirrors the **main contrastive LLM judge** in the repository, but
uses **Google Gemini** instead of the OpenAI Chat Completions API.

It runs the same three-aspect contrastive judge:

- **Inputs**: `anchor_text`, `text_a`, `text_b`, and (for dev) `text_a_is_closer`
- **System + user prompts**: defined in `contra.yaml`
- **Few-shot demos**: from `contra_demos_1stage.jsonl`
- **Outputs**: JSONL with the model's structured decision and accuracy (if gold labels are present)

## Setup

From this directory:

```bash
pip install -r requirements.txt
```

You must also set a Gemini API key (no key is stored in this repository), for example:

```bash
export GOOGLE_API_KEY="<your-gemini-api-key>"
# or
export GEMINI_API_KEY="<your-gemini-api-key>"
```

## Data

We follow the official SemEval-2026 Task 4 Track A format.

Place the **dev** file `dev_track_a.jsonl` (with `text_a_is_closer` labels) in this
baseline's `data/` directory:

- `baselines/mainmethod_gemini/data/dev_track_a.jsonl`

You obtain this file from the task organizers; it is **not** redistributed here.

Each line must be a JSON object with at least:

- `anchor_text`
- `text_a`
- `text_b`
- `text_a_is_closer` (boolean; dev labels)

## Running the Gemini main method

From this `mainmethod_gemini` folder:

```bash
python contrastive_nocurriculum_gemini.py
```

By default, the script expects:

- **config**: `contra.yaml`
- **demos**: `contra_demos_1stage.jsonl`
- **input**: `data/dev_track_a.jsonl`
- **output**: `output/contrastive_results_dev_track_a_gemini.jsonl`

The script prints the number of processed examples and, if the input contains
`text_a_is_closer`, an overall accuracy and a classification report.

## Notes

- This folder is self-contained for the Gemini variant (config, demos, script, and requirements).
- No API keys, user names, or absolute file-system paths are stored in this directory.
- The method is conceptually identical to the main OpenAI-based judge; only the backend LLM client differs.

