# SemEval-2026 Task 4 Track A — MiniLM embedding baseline

Sentence-embedding baseline for **Track A** only: `sentence-transformers/all-MiniLM-L6-v2` + cosine similarity.

## Task format

- **Input JSONL**: each line has `anchor_text`, `text_a`, `text_b` (and in dev, optionally `text_a_is_closer`).
- **Output JSONL**: same lines in same order, with `text_a_is_closer` set to our prediction (True/False).

## Setup

```bash
pip install -r requirements.txt
```

Place dev/test Track A JSONL files in `data/`.

## Usage

### Predict (Track A)

```bash
python -m src.predict_track_a --input data/dev_track_a.jsonl --output output/track_a.jsonl
```

Options:

- `--input` — path to input JSONL
- `--output` — path to write output JSONL (default: `output/track_a.jsonl`)
- `--batch-size` — batch size for encoding (default: 64)
- `--chunking` — enable chunking (split stories into ~200–300 word chunks, average embeddings)

If the input contains gold `text_a_is_closer`, accuracy is computed and printed.

### Package submission

```bash
python -m src.package_submission --track-a output/track_a.jsonl --zip submission_track_a.zip
```

Creates a zip with exactly `track_a.jsonl` at the root (no folders).

## Project layout

```
embedding_baseline/
  data/           # put dev/test JSONL here
  output/
  src/
    io_utils.py
    embedder.py
    predict_track_a.py
    package_submission.py
  requirements.txt
  README.md
```
