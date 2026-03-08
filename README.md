# Narrative Similarity – Main Method and Baselines

This repository contains our main contrastive LLM method and baselines for
SemEval-2026 Task 4 Track A (narrative similarity).

## Main method: contrastive LLM judge

The script `contrastive_nocurriculum.py` implements a three-aspect
contrastive judge that:

- takes JSONL input with `anchor_text`, `text_a`, `text_b`, and
  (for dev) `text_a_is_closer`,
- uses an LLM (via the OpenAI Chat Completions API) with a contrastive
  system prompt (`contra.yaml`) and few-shot demonstrations
  (`contra_demos_1stage.jsonl`),
- returns a JSONL file with model outputs and reports accuracy when gold
  labels are available.

### Dependencies

```bash
pip install -r requirements.txt
```

You must also set an OpenAI API key, e.g.:

```bash
export OPENAI_API_KEY="<your-key-here>"
```

### Data

We follow the official SemEval-2026 Task 4 Track A format.

- Place the **dev** file `dev_track_a.jsonl` (with `text_a_is_closer` labels)
  inside the `data/` directory of this repository.
  You obtain this file from the task organizers (we do not redistribute it).

### Running the main method

From the repository root:

```bash
python contrastive_nocurriculum.py
```

By default, the script expects:

- config: `contra.yaml`
- demos: `contra_demos_1stage.jsonl`
- input: `data/dev_track_a.jsonl`
- output: `output/contrastive_results_dev_track_a.jsonl`

The script prints the number of processed examples and, if the input contains
`text_a_is_closer`, an overall accuracy and classification report.

Further baselines (e.g. MiniLM embedding baseline) can be added in separate
subdirectories and documented here.

## Baseline: MiniLM embedding

See `baselines/minilm_embedding/README.md`.

## Baseline: LLM prompt chaining (v3)

Three-stage prompt-chaining baseline (normalize → segment → compare).

See `baselines/llm_prompt_chaining_v3/README.md`.

## Baseline: Contrastive LLM judge (Gemini)

Gemini-based version of the main contrastive LLM judge (same prompts/demos, different backend).

See `baselines/mainmethod_gemini/README.md`.

## Ablations (main method variants)

The `abalations/` directory contains controlled variants of the main contrastive judge:

- `abalations/no_demos/` — no few-shot demonstrations (single prompt per example).
- `abalations/no_contrastive/` — simplified decision schema (no aspect structure).
- `abalations/random_demos/` — alternative demonstration file (`contra_demos_nocurriculum.jsonl`).

Each ablation folder includes its own `README.md`, script, config, and any extra
demo files required to reproduce results.
