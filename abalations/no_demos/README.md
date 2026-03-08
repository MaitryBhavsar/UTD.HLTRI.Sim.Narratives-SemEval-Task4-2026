# Ablation 1 — No demonstrations

This ablation removes **all few-shot demonstrations** from the contrastive LLM
judge. For each example, the model sees only:

- the **system prompt** (from `contra.yaml`), and
- a **single user prompt** instantiated with `anchor_text`, `text_a`, and `text_b`.

Everything else (config, input, output, metrics) matches the main method.

## Setup

From the repository root (or this folder):

```bash
pip install -r requirements.txt
```

> You can also reuse the main repository `requirements.txt`; this local file lists
> only the packages actually needed for this ablation.

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="<your-key-here>"
```

## Data

Place the SemEval-2026 Task 4 Track A dev file here:

- `data/dev_track_a.jsonl`

The format is the same as the main method:

- `anchor_text`
- `text_a`
- `text_b`
- `text_a_is_closer` (boolean; dev labels)

## Running the ablation

From `abalations/no_demos`:

```bash
python contrastive_no_demos.py
```

By default, the script expects:

- **config**: `contra.yaml` (copied from the main method)
- **input**: `data/dev_track_a.jsonl`
- **output**: `output/contrastive_results_dev_track_a_no_demos.jsonl`

The script prints the number of processed examples and reports accuracy and a
classification report when `text_a_is_closer` is present.

