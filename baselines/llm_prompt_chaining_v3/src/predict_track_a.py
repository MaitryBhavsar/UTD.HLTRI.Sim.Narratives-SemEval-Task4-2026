from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

from .agent import AgentConfig, ThreeStageNarrativeSimilarityAgent
from .io_utils import read_jsonl, write_jsonl


def _load_config(path: Path) -> AgentConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got: {type(data)}")
    return AgentConfig(**data)


def _get_prediction(stage3_output: Dict[str, Any]) -> Optional[Tuple[str, bool]]:
    try:
        sel = stage3_output["result"]["selected_story"]
    except Exception:
        return None
    if sel not in {"A", "B"}:
        return None
    return sel, (sel == "A")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track A baseline: LLM prompt-chaining (normalize->segment->compare)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Track A JSONL (must contain anchor_text, text_a, text_b).",
    )
    parser.add_argument(
        "--output",
        default="output/track_a.jsonl",
        help="Where to write predictions JSONL (default: output/track_a.jsonl).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="YAML config path (default: config.yaml in this baseline folder).",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Store stage1/stage2/stage3 outputs per example (large).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N examples (0 = all).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")

    cfg = _load_config(config_path)
    agent = ThreeStageNarrativeSimilarityAgent(cfg=cfg)

    rows = list(read_jsonl(input_path))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    results: List[Dict[str, Any]] = []
    y_true: List[bool] = []
    y_pred: List[bool] = []

    for ex in tqdm(rows, desc="Examples", unit="ex"):
        anchor = ex.get("anchor_text", "")
        text_a = ex.get("text_a", "")
        text_b = ex.get("text_b", "")

        gold = ex.get("text_a_is_closer", None)
        if "text_a_is_closer" in ex:
            ex = dict(ex)
            ex["gold_text_a_is_closer"] = ex.get("text_a_is_closer")

        pipeline_out: Optional[Dict[str, Any]] = None
        stage3_out: Optional[Dict[str, Any]] = None
        pred_bool: Optional[bool] = None

        try:
            pipeline_out = agent.process(anchor, text_a, text_b)
            stage3_out = pipeline_out.get("stage3_output")
            if isinstance(stage3_out, dict):
                pred = _get_prediction(stage3_out)
                if pred is not None:
                    _, pred_bool = pred
        except Exception as e:
            # Keep going; record the error string for traceability.
            pipeline_out = {"error": str(e)}

        out_row = dict(ex)
        if pred_bool is not None:
            out_row["text_a_is_closer"] = pred_bool
        else:
            out_row["text_a_is_closer"] = None

        if args.save_intermediates:
            out_row["llm_pipeline"] = pipeline_out

        results.append(out_row)

        if isinstance(gold, bool) and isinstance(pred_bool, bool):
            y_true.append(gold)
            y_pred.append(pred_bool)

    write_jsonl(output_path, results)
    print(f"Wrote: {output_path.resolve()}")

    if y_true and y_pred:
        try:
            from sklearn.metrics import accuracy_score, classification_report

            acc = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {acc:.4f} ({len(y_true)} examples with gold labels)")
            print(classification_report(y_true, y_pred, labels=[True, False]))
        except Exception as e:
            print(f"Could not compute metrics: {e}")

    # Print effective config for reproducibility
    print("\nEffective config:")
    print(yaml.safe_dump(asdict(cfg), sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

