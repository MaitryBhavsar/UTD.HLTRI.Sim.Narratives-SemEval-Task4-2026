from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Track A output as a submission zip.")
    parser.add_argument(
        "--track-a",
        required=True,
        help="Path to predicted Track A JSONL (will be stored as track_a.jsonl in zip).",
    )
    parser.add_argument("--zip", required=True, help="Path to write the submission zip.")
    args = parser.parse_args()

    track_a_path = Path(args.track_a)
    zip_path = Path(args.zip)

    if not track_a_path.exists():
        raise FileNotFoundError(f"Missing file: {track_a_path.resolve()}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(track_a_path, arcname="track_a.jsonl")

    print(f"Wrote: {zip_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

