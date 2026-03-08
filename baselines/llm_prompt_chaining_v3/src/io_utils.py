from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {p}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {lineno} of {p}")
            yield obj


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

