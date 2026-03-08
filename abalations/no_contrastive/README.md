# Ablation 2 — Simplified decision (no aspect structure)

This ablation removes the **three-aspect structure** (abstract theme / course
of action / outcomes) from the output schema.

Instead of a rich `result` object with multiple fields, the model outputs only:

- `selected_story` ("A" or "B")
- `final_explanation` (a concise justification)

The prompts and general contrastive comparison remain similar; only the schema
and supervision are simplified via `contra_noaspects.yaml`.

## Setup

From this directory:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="<your-key-here>"
```

## Data

As in the main method, place the dev file at:

- `data/dev_track_a.jsonl`

with fields:

- `anchor_text`
- `text_a`
- `text_b`
- `text_a_is_closer` (boolean; dev labels)

## Running the ablation

From `abalations/no_contrastive`:

```bash
python contrastive_nocurriculum_simple_noaspects.py
```

Defaults:

- **config**: `contra_noaspects.yaml`
- **demos**: `contra_demos_copy.jsonl`
- **input**: `data/dev_track_a.jsonl`
- **output**: `output/contrastive_results_dev_track_a_simple_noaspects.jsonl`

The script reports the number of processed examples and accuracy when gold
labels are available.

