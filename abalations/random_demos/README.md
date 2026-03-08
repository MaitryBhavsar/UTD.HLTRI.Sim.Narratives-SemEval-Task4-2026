# Ablation 3 — Alternative / random demos

This ablation uses an **alternative demonstration file** for the contrastive LLM
judge:

- Instead of `contra_demos_1stage.jsonl`, it uses `contra_demos_nocurriculum.jsonl`,
  which changes the set (and possibly ordering) of demonstrations presented to the model.

The decision schema and prompts remain contrastive; only the demos change.

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

Fields:

- `anchor_text`
- `text_a`
- `text_b`
- `text_a_is_closer` (boolean; dev labels)

## Running the ablation

From `abalations/random_demos`:

```bash
python contrastive_nocurriculum_nodemoset.py
```

Defaults:

- **config**: `contra.yaml`
- **demos**: `contra_demos_nocurriculum.jsonl`
- **input**: `data/dev_track_a.jsonl`
- **output**: `output/contrastive_results_dev_track_a_nodemoset.jsonl`

The script prints the number of processed examples and accuracy when gold
labels are provided.

