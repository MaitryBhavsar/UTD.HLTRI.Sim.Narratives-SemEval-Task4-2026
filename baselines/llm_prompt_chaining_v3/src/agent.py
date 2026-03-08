from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_client import ChatRequest, OpenAIChatClient


STAGE1_SYSTEM_MESSAGE = """You are a narrative normalizer for a narrative-similarity task.

Task:
Rewrite each input story into a simple, neutral, third-person narrative using a shared style.
Preserve the story's plot-critical meaning, but reduce superficial distractors by:
- Replacing proper names with stable generic roles (e.g., Person1, Person2, Friend, Thief).
- Generalizing specific time/place details to generic terms unless causally necessary.
- Removing sentiment-heavy phrasing; keep plot-relevant facts only.
- Removing background lore/subplots that do not affect the main plot arc.

Hard rules:
- Do NOT invent events. Do NOT remove plot-critical events.
- Keep the rewrite short (about 3–8 sentences).
- Keep chronological order of events.
- Output valid JSON ONLY with keys: anchor_text, text_a, text_b (each a string)."""

STAGE1_USER_TEMPLATE = """Input stories (JSON):
{input_json}
"""


STAGE2_SYSTEM_MESSAGE = """You are a story segmenter for a narrative-similarity task.

Input:
A JSON object with keys: "anchor_text", "text_a", "text_b". Each value is a normalized story string.

Task:
For EACH story, split the story into 4–8 chronological plot beats (segments) for the MAIN plot arc.
Each beat must be in story-world chronological order.

Beat labels (choose exactly one per beat; use role in the overall arc):
- PREMISE: starting situation / context needed for the arc
- GOAL: what the main character wants/needs (explicit or strongly implied)
- ACTION: a deliberate step taken toward/within the arc
- CONFLICT: an obstacle, pressure, or problem that forces a response
- TURNING_POINT: a key pivot that changes direction or raises stakes
- OUTCOME: the final result/resolution at the end of the story (EXACTLY ONE per story)

For each beat, output an object with fields:
- label: one of {PREMISE, GOAL, ACTION, CONFLICT, TURNING_POINT, OUTCOME}
- text: 1–2 simple sentences describing this beat (faithful to the input story)
- function: a 3–7 word plot-function phrase describing this beat’s role
- gist: a compact 5–12 word event summary derived from the beat text
- story_effect: a short phrase describing why this beat matters for progression

Hard rules:
- Do NOT invent facts beyond the normalized story text.
- Preserve chronological order.
- Ensure EXACTLY ONE OUTCOME beat per story.
- Output valid JSON ONLY with keys: anchor_text, text_a, text_b.
  Each value is a list of beat objects as specified above. No extra text."""

STAGE2_USER_TEMPLATE = """Normalized stories (Stage 1 output JSON):
{stage1_json}
"""


STAGE3_SYSTEM_MESSAGE = """You are an expert annotator specializing in narrative similarity.

Goal:
Determine which of two candidate stories (Story A or Story B) is most similar to an Anchor story
based on three core aspects:

1) Abstract Theme: The defining constellation of problems, central ideas, and core motifs
   (excluding concrete setting).
2) Course of Action: The sequence of events, actions, conflicts, turning points, and the order
   in which they occur.
3) Outcomes: The results of the plot at the end of the text (e.g., conflict resolution, characters'
   fates, moral lessons). Outcomes do NOT include intermediate statuses that later change.

Ignore:
- Writing style
- Concrete setting/time period (unless it is required by the events)
- Character and location names
- Text length
- Level of detail in which events are described

Input:
Stage 2 JSON. Each story is a list of chronological beats with labels, gist, function, story_effect.

Instructions:
- Use ONLY the Stage 2 content; do not add new story facts.
- Compare Anchor vs Story A and Anchor vs Story B for EACH aspect independently:
  - For Abstract Theme: infer a concise theme statement for each story from the beats (exclude setting),
    then decide which candidate matches the anchor's theme more closely.
  - For Course of Action: compare the event sequence and key pivots/obstacles, including their order.
  - For Outcomes: compare the OUTCOME beats (final result).
- Then select the overall closer story.

Output JSON ONLY (no extra text) with this schema:
{
  "result": {
    "contra_reason": {
      "theme": "<contrastive theme reasoning>",
      "course": "<contrastive course-of-action reasoning>",
      "outcomes": "<contrastive outcome reasoning>"
    },
    "selected_story": "A" or "B",
    "contributing_aspects": ["abstract_theme","course_of_action","outcomes"],
    "final_explanation": "1–3 concise sentences summarizing why the selected story is closer."
  }
}

Contributing aspects:
Include an aspect in contributing_aspects only if it meaningfully supports the selection.
If an aspect is roughly a tie, omit it."""

STAGE3_USER_TEMPLATE = """Stage 2 JSON (beats for Anchor, Story A, Story B):
{stage2_json}
"""


def _extract_json_from_response(response: str, stage_name: str) -> str:
    resp = (response or "").strip()
    if not resp:
        raise ValueError(f"{stage_name} response is empty")

    if "```" in resp:
        lines = resp.splitlines()
        kept: List[str] = []
        in_fence = False
        for line in lines:
            s = line.strip()
            if s.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence and s.lower() == "json":
                continue
            kept.append(line)
        resp = "\n".join(kept).strip()

    if not resp.startswith("{"):
        start = resp.find("{")
        end = resp.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"{stage_name} response does not contain a JSON object")
        resp = resp[start : end + 1].strip()

    return resp


def _validate_stage1(obj: Dict[str, Any]) -> None:
    for key in ("anchor_text", "text_a", "text_b"):
        if key not in obj:
            raise ValueError(f"Stage 1 missing key: {key}")
        if not isinstance(obj[key], str):
            raise ValueError(f"Stage 1 key '{key}' must be a string")


def _validate_stage2(obj: Dict[str, Any]) -> None:
    allowed_labels = {"PREMISE", "GOAL", "ACTION", "CONFLICT", "TURNING_POINT", "OUTCOME"}
    for key in ("anchor_text", "text_a", "text_b"):
        if key not in obj:
            raise ValueError(f"Stage 2 missing key: {key}")
        beats = obj[key]
        if not isinstance(beats, list) or len(beats) == 0:
            raise ValueError(f"Stage 2 key '{key}' must be a non-empty list")

        outcome_count = 0
        for i, beat in enumerate(beats):
            if not isinstance(beat, dict):
                raise ValueError(f"Stage 2 beat {key}[{i}] must be an object")
            for field in ("label", "text", "function", "gist", "story_effect"):
                if field not in beat or not isinstance(beat[field], str):
                    raise ValueError(f"Stage 2 beat {key}[{i}] missing/invalid '{field}'")
            label = beat["label"].strip()
            if label not in allowed_labels:
                raise ValueError(f"Stage 2 beat {key}[{i}] has invalid label '{label}'")
            if label == "OUTCOME":
                outcome_count += 1

        if outcome_count != 1:
            raise ValueError(
                f"Stage 2 story '{key}' must have exactly one OUTCOME beat; got {outcome_count}"
            )


def _validate_stage3(obj: Dict[str, Any]) -> None:
    if "result" not in obj or not isinstance(obj["result"], dict):
        raise ValueError("Stage 3 output must have a 'result' object")

    res = obj["result"]
    if res.get("selected_story") not in ("A", "B"):
        raise ValueError("Stage 3 'selected_story' must be 'A' or 'B'")

    if "contra_reason" not in res or not isinstance(res["contra_reason"], dict):
        raise ValueError("Stage 3 must include object 'contra_reason'")
    for k in ("theme", "course", "outcomes"):
        if k not in res["contra_reason"] or not isinstance(res["contra_reason"][k], str):
            raise ValueError(f"Stage 3 contra_reason.{k} must be a string")

    if "contributing_aspects" not in res or not isinstance(res["contributing_aspects"], list):
        raise ValueError("Stage 3 'contributing_aspects' must be a list")
    allowed = {"abstract_theme", "course_of_action", "outcomes"}
    for x in res["contributing_aspects"]:
        if x not in allowed:
            raise ValueError(f"Stage 3 contributing_aspects item '{x}' is invalid")

    if "final_explanation" not in res or not isinstance(res["final_explanation"], str):
        raise ValueError("Stage 3 must include string 'final_explanation'")


@dataclass(frozen=True)
class AgentConfig:
    model: str = "gpt-4o"
    temperature: float = 0.2
    seed: Optional[int] = 123
    min_delay_seconds: float = 0.0
    stage1_max_tokens: int = 2048
    stage2_max_tokens: int = 4096
    stage3_max_tokens: int = 4096


class ThreeStageNarrativeSimilarityAgent:
    def __init__(self, *, cfg: AgentConfig, client: Optional[OpenAIChatClient] = None) -> None:
        self.cfg = cfg
        self.client = client or OpenAIChatClient(min_delay_seconds=cfg.min_delay_seconds)

    def process_stage1(self, anchor_text: str, text_a: str, text_b: str) -> Dict[str, Any]:
        input_obj = {"anchor_text": anchor_text, "text_a": text_a, "text_b": text_b}
        user_message = STAGE1_USER_TEMPLATE.format(
            input_json=json.dumps(input_obj, ensure_ascii=False, indent=2)
        )
        response = self.client.generate(
            system_message=STAGE1_SYSTEM_MESSAGE,
            user_message=user_message,
            request=ChatRequest(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.stage1_max_tokens,
                seed=self.cfg.seed,
            ),
        )
        cleaned = _extract_json_from_response(response, "Stage 1")
        parsed = json.loads(cleaned)
        _validate_stage1(parsed)
        return parsed

    def process_stage2(self, stage1_output: Dict[str, Any]) -> Dict[str, Any]:
        user_message = STAGE2_USER_TEMPLATE.format(
            stage1_json=json.dumps(stage1_output, ensure_ascii=False, indent=2)
        )
        response = self.client.generate(
            system_message=STAGE2_SYSTEM_MESSAGE,
            user_message=user_message,
            request=ChatRequest(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.stage2_max_tokens,
                seed=self.cfg.seed,
            ),
        )
        cleaned = _extract_json_from_response(response, "Stage 2")
        parsed = json.loads(cleaned)
        _validate_stage2(parsed)
        return parsed

    def process_stage3(self, stage2_output: Dict[str, Any]) -> Dict[str, Any]:
        user_message = STAGE3_USER_TEMPLATE.format(
            stage2_json=json.dumps(stage2_output, ensure_ascii=False, indent=2)
        )
        response = self.client.generate(
            system_message=STAGE3_SYSTEM_MESSAGE,
            user_message=user_message,
            request=ChatRequest(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.stage3_max_tokens,
                seed=self.cfg.seed,
            ),
        )
        cleaned = _extract_json_from_response(response, "Stage 3")
        parsed = json.loads(cleaned)
        _validate_stage3(parsed)
        return parsed

    def process(self, anchor_text: str, text_a: str, text_b: str) -> Dict[str, Any]:
        stage1 = self.process_stage1(anchor_text, text_a, text_b)
        stage2 = self.process_stage2(stage1)
        stage3 = self.process_stage3(stage2)
        return {"stage1_output": stage1, "stage2_output": stage2, "stage3_output": stage3}


TwoStageAgent = ThreeStageNarrativeSimilarityAgent

