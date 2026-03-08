"""Package Track A submission: zip with track_a.jsonl at root (no folders)."""

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Track A submission zip")
    parser.add_argument("--track-a", required=True, help="Path to track_a.jsonl")
    parser.add_argument("--zip", required=True, help="Output zip path")
    args = parser.parse_args()

    track_a_path = Path(args.track_a)
    zip_path = Path(args.zip)
    if not track_a_path.exists():
        print(f"Error: file not found: {track_a_path}")
        return 1

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(track_a_path, "track_a.jsonl")
    print(f"Created {zip_path} with track_a.jsonl at root")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
