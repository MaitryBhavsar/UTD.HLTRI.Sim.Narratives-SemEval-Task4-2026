"""JSONL read/write helpers. UTF-8, one JSON object per line."""

import json
from pathlib import Path
from typing import List


def read_jsonl(path: str | Path) -> List[dict]:
    """Read a JSONL file; return list of dicts (one per non-empty line)."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: List[dict]) -> None:
    """Write rows to a JSONL file; one JSON object per line, UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
