"""SemEval-2026 Task 4 Track A predictor: MiniLM embeddings + cosine similarity."""

import argparse
import sys
from pathlib import Path

import numpy as np

from .io_utils import read_jsonl, write_jsonl
from .embedder import MiniLMEmbedder, preprocess_text


def chunk_by_words(text: str, words_per_chunk: int = 250) -> list[str]:
    """Split text into chunks of ~words_per_chunk words (whitespace-split)."""
    text = preprocess_text(text)
    words = text.split()
    if not words:
        return [""]
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        chunks.append(chunk)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Track A: MiniLM + cosine similarity")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL (Track A)")
    parser.add_argument("--output", "-o", default="output/track_a.jsonl", help="Output JSONL path")
    parser.add_argument("--batch-size", type=int, default=64, help="Encode batch size")
    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Split stories into ~200-300 word chunks and average embeddings",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    rows = read_jsonl(input_path)
    if not rows:
        print("Error: no rows in input", file=sys.stderr)
        return 1

    embedder = MiniLMEmbedder()

    if not args.chunking:
        # Embed all three texts per row
        anchors = [r["anchor_text"] for r in rows]
        text_a_list = [r["text_a"] for r in rows]
        text_b_list = [r["text_b"] for r in rows]
        all_texts = anchors + text_a_list + text_b_list
        embs = embedder.encode(all_texts, batch_size=args.batch_size, normalize=True)
        n = len(rows)
        anchor_embs = embs[:n]
        a_embs = embs[n : 2 * n]
        b_embs = embs[2 * n :]
        sim_a = np.sum(anchor_embs * a_embs, axis=1)
        sim_b = np.sum(anchor_embs * b_embs, axis=1)
    else:
        # Chunk each story, embed chunks, average then normalize
        sim_a_list = []
        sim_b_list = []
        for r in rows:
            anchor_chunks = chunk_by_words(r["anchor_text"], 250)
            a_chunks = chunk_by_words(r["text_a"], 250)
            b_chunks = chunk_by_words(r["text_b"], 250)
            all_chunks = anchor_chunks + a_chunks + b_chunks
            chunk_embs = embedder.encode(all_chunks, batch_size=args.batch_size, normalize=True)
            na, nb, nc = len(anchor_chunks), len(a_chunks), len(b_chunks)
            anchor_emb = chunk_embs[:na].mean(axis=0)
            a_emb = chunk_embs[na : na + nb].mean(axis=0)
            b_emb = chunk_embs[na + nb :].mean(axis=0)
            anchor_emb = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-9)
            a_emb = a_emb / (np.linalg.norm(a_emb) + 1e-9)
            b_emb = b_emb / (np.linalg.norm(b_emb) + 1e-9)
            sim_a_list.append(np.dot(anchor_emb, a_emb))
            sim_b_list.append(np.dot(anchor_emb, b_emb))
        sim_a = np.array(sim_a_list)
        sim_b = np.array(sim_b_list)

    # Predict: text_a closer iff sim_a > sim_b; tie-break True
    pred = np.greater(sim_a, sim_b) | np.isclose(sim_a, sim_b)

    # Build output rows (preserve order; set text_a_is_closer to prediction)
    output_rows = []
    for i, row in enumerate(rows):
        out_row = dict(row)
        out_row["text_a_is_closer"] = bool(pred[i])
        output_rows.append(out_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")

    # Accuracy if gold present in input
    gold = [r.get("text_a_is_closer") for r in rows]
    if all(g is not None for g in gold):
        correct = sum(1 for g, p in zip(gold, pred) if g == p)
        acc = correct / len(gold)
        print(f"Accuracy: {acc:.4f} ({correct}/{len(gold)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
