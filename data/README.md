# Data for SemEval-2026 Task 4 Track A

This repository does **not** redistribute the official SemEval-2026 Task 4 data.

To run the main method:

1. Register for SemEval-2026 Task 4 and download the development data.
2. Extract the file `dev_track_a.jsonl` from the dev zip.
3. Place `dev_track_a.jsonl` into this `data/` directory.

The file must be in JSONL format, with one object per line, containing at
least the fields:

- `anchor_text`
- `text_a`
- `text_b`
- `text_a_is_closer` (boolean; dev labels)

The script `contrastive_nocurriculum.py` reads `data/dev_track_a.jsonl` by
default.
