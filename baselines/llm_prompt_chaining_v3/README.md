# SemEval-2026 Task 4 Track A — LLM prompt-chaining baseline (v3)

This baseline implements a **multi-stage prompting pipeline** for narrative similarity:

- **Stage 1 (Normalize)**: rewrite Anchor/A/B into a neutral shared style (reduce distractors).
- **Stage 2 (Segment)**: convert each normalized story into 4–8 chronological plot beats.
- **Stage 3 (Compare)**: decide whether **Story A** or **Story B** is closer to the Anchor, using:
  abstract theme, course-of-action, and outcomes.

## Setup

From this directory:

```bash
pip install -r requirements.txt
```

Set an API key (not included in this repository):

```bash
export OPENAI_API_KEY="<your-key-here>"
```

## Data

Place a Track A JSONL file in `data/` (we do not redistribute the official SemEval data).

Each line must contain:

- `anchor_text`
- `text_a`
- `text_b`

If the file also contains gold `text_a_is_closer` (dev labels), the script will report accuracy.

## Run (Track A)

```bash
python -m src.predict_track_a --input data/dev_track_a.jsonl --output output/track_a.jsonl
```

Options:

- `--config`: YAML config (default: `config.yaml`)
- `--save-intermediates`: store stage outputs per example (large output)
- `--limit N`: run only first \(N\) examples

## Package submission

```bash
python -m src.package_submission --track-a output/track_a.jsonl --zip submission_track_a.zip
```

## Output format

The output is a JSONL file with the same examples in the same order. The field:

- `text_a_is_closer`: the **prediction** (True/False), or `null` if a prediction failed.

If the input contained `text_a_is_closer`, it is preserved as:

- `gold_text_a_is_closer`

## Reproducibility notes

LLM-based methods can still vary across runs due to backend/model changes. This baseline uses:

- low `temperature` (default 0.2)
- an optional `seed` (see `config.yaml`)

but exact determinism is not guaranteed.

