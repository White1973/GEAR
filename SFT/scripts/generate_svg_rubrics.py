"""Batch generate SVG evaluation rubrics for SFT training data.

Reads a JSONL file (each line has "messages" and "images" fields),
sends each sample to a vision LLM with a rubric-designer prompt,
and writes the results (original data + rubric) to an output JSONL.

The rubrics cover four physical-world dimensions:
  1. SPATIAL_GROUNDING  — visual element positions match the image
  2. ACTION_LEGALITY    — SVG code is syntactically valid and renderable
  3. LOGIC_FRESHNESS    — reasoning is non-repetitive and incrementally refined
  4. STRATEGIC_PROGRESS — each revision improves similarity toward the target
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import openai as _openai

    OpenAIError = _openai.OpenAIError
except Exception:
    OpenAIError = Exception

# ============================================================
# RUBRIC PROMPT — SVG Domain
# ============================================================

SVG_RUBRIC_PROMPT = """\
# Role
You are a Rubric Designer for SVG code generation. Analyze the target image \
and the solver's response, then generate evaluation rubrics across four pillars.

# Context
## [System Rules] {system_rules}
## [Trajectory History] {history_log}
## [Current Visual State] [Provided via attached Image]

================================================================================
# STEP 1: IMAGE & CODE ANALYSIS (fill before generating rubrics)
================================================================================

1.1 VISUAL ELEMENTS
  Identify every distinct visual element in the image:
    shapes (rect, circle, ellipse, line, path, polygon, text …),
    their bounding-box positions (x, y, width, height),
    fill/stroke colors, stroke-width, opacity, transforms.
  Also note the canvas viewBox if apparent.

1.2 SPATIAL RELATIONSHIPS
  For each pair of major elements describe their spatial relation:
    above/below, left-of/right-of, overlapping, containing/contained,
    alignment (centered, left-aligned …).

1.3 CODE SANITY
  Check the solver's <answer> SVG code for:
    - Well-formed XML / valid SVG tags
    - No orphaned or mismatched tags
    - viewBox present and consistent with element coordinates
    - No elements far outside the viewBox
    - No zero-width/height rects, zero-radius circles, or invisible fills \
      (fill="none" + stroke="none")

1.4 TEMPORAL DIFF (from history_log, if available)
  STALE_PHRASES   : spatial descriptions repeated verbatim from previous turn
  CHANGED_CODE     : specific SVG attributes or elements that changed
  FRESHNESS_SCORE  : fraction of unchanged text; <0.60 = fresh, 0.60–0.85 = partial, >0.85 = stale

1.5 STRATEGIC DELTA
  Compare current SVG code to the image (or to the description from history):
    - Distance reduction: did the new code move elements closer to their \
      target positions / colors / sizes?
    - Element coverage: did the new code add any missing elements or remove \
      incorrectly added ones?
    BEST_REVISION_ACTION : the single most impactful change the solver should \
      make next

================================================================================
# STEP 2: RUBRIC GENERATION
================================================================================

Generate rubrics in this fixed order. Include optional rules only when triggered.

## Rule 1 — SPATIAL_GROUNDING  (weight 14, always)
(a)[5pts] <observation> correctly describes position (x, y or relative) and \
  size of EVERY major visual element (shape type + location); Score 0 if any \
  element is wrong or missing.
(b)[5pts] <observation> states correct spatial relationships between elements \
  (above/below, left/right, overlapping, containment); Score 0 if any major \
  relation is wrong.
(c)[4pts] Colors and stroke styles mentioned in <observation> match the image; \
  Score 0 if a color is clearly wrong; auto-PASS if image is grayscale/monochrome.

## Rule 2 — ACTION_LEGALITY  (weight 12, always, VETO)
(a)[6pts] The SVG code in <answer> is syntactically well-formed: all tags are \
  closed, attribute values are quoted, no stray characters.
(b)[6pts] The SVG code in <answer> is renderable: viewBox exists, no elements \
  reference undefined ids, no negative/zero dimensions that make elements \
  invisible.
VETO: If this rule scores 0, ALL other rubric scores collapse to 0.

## Rule 3 — LOGIC_FRESHNESS  (weight 5, always)
(a)[2pts] TC_State_Evolution: If history is available, <observation> and \
  <reasoning> contain NEW information compared to the previous turn; \
  FRESHNESS <0.60 = PASS; 0.60–0.85 = PARTIAL; >0.85 = FAIL; \
  auto-PASS if no history or first turn.
(b)[2pts] TC_Monotony: <reasoning> is not a near-copy of the previous turn's \
  reasoning (overlap >85%); auto-PASS if <3 turns.
(c)[1pt] TC_Refinement: <reasoning> identifies a specific issue from the \
  previous turn's rendering and proposes a concrete fix; auto-PASS if first turn.

## Rule 4 — STRATEGIC_PROGRESS  (weight 5 normal / 8 if DETOUR, always)
(a)[40%] The SVG code in <answer> changes at least one element that moves it \
  closer to the target image (position, color, size, or adding a missing element).
(b)[35%] The most impactful change proposed in <reasoning> is actually \
  implemented in <answer>.
(c)[25%] <prediction> gives a concrete expectation about visual improvement \
  (e.g., "similarity should increase to ~0.85"), not a vague restatement.

================================================================================
# OUTPUT FORMAT
================================================================================

Return ONLY the JSON in a ```json block.

{{
  "turn_id": "{turn_id}",
  "turn_analysis": {{
    "visual_elements": [
      {{"type": "rect|circle|path|text|...", "position": "description", "color": "color"}}
    ],
    "spatial_relationships": ["elementA is above/below/left-of/overlapping elementB"],
    "code_sanity": {{"well_formed": true, "renderable": true, "issues": []}},
    "freshness_score": 0.0,
    "best_revision_action": "description of most impactful change",
    "summary": "One sentence: key elements, their layout, and the main improvement needed."
  }},
  "rubrics": [
    {{
      "rule_id": "SPATIAL_GROUNDING",
      "description": "PASS(14): (a)[5pts]<observation> correct position+size for ALL elements; Score 0 if wrong/missing. (b)[5pts]spatial relations correct; Score 0 if major relation wrong. (c)[4pts]colors match; Score 0 if clearly wrong; auto-PASS if monochrome.",
      "weight": 14
    }},
    {{
      "rule_id": "ACTION_LEGALITY",
      "description": "PASS(12): (a)[6pts]SVG well-formed: tags closed, attrs quoted, no stray chars. (b)[6pts]SVG renderable: viewBox exists, no invisible/negative-dim elements. VETO: if 0 here, all other scores → 0.",
      "weight": 12
    }},
    {{
      "rule_id": "LOGIC_FRESHNESS",
      "description": "PASS(5): TC_State[2pts]: freshness<0.60 PASS / >0.85 FAIL; auto-PASS first turn. TC_Monotony[2pts]: overlap>85% FAIL; auto-PASS<3 turns. TC_Refinement[1pt]: specific fix proposed; auto-PASS first turn.",
      "weight": 5
    }},
    {{
      "rule_id": "STRATEGIC_PROGRESS",
      "description": "PASS(5): (a)[40%]code changes move closer to target. (b)[35%]reasoning's top fix is implemented. (c)[25%]prediction gives concrete expectation, not vague.",
      "weight": 5
    }}
  ]
}}
"""


# ============================================================
# Config & Model
# ============================================================

@dataclass
class AsyncOpenAIModelConfig:
    model_name: str = "auto"
    max_completion_tokens: int = field(default=8192)
    temperature: float = field(default=1.0)
    top_p: float = field(default=1.0)
    frequency_penalty: float = field(default=0.0)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class AsyncOpenAIModel:
    def __init__(self, **kwargs):
        self.config = AsyncOpenAIModelConfig(**kwargs)
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )
        self.cost = 0.0
        self.n_calls = 0

    @retry(
        wait=wait_exponential(multiplier=5, min=0, max=300),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((
            OpenAIError,
        )),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit, retrying in {retry_state.next_action.sleep} seconds..."
        ),
    )
    async def get_response(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        call_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "max_completion_tokens": kwargs.pop(
                "max_completion_tokens", self.config.max_completion_tokens
            ),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "top_p": kwargs.pop("top_p", self.config.top_p),
            "frequency_penalty": kwargs.pop(
                "frequency_penalty", self.config.frequency_penalty
            ),
            **self.config.model_kwargs,
            **kwargs,
        }

        for i in range(5):
            response = await self.client.chat.completions.create(**call_kwargs)
            logger.debug(f"AsyncOpenAIModel get response (attempt {i})")
            if response.choices:
                break
            else:
                logger.warning(f"openai api response choices is empty, retry {i}")

        _usage_info = response.usage.model_dump() if response.usage else {}
        usage_info = {
            "total_tokens": _usage_info.get("total_tokens", 0) or 0,
            "prompt_tokens": _usage_info.get("prompt_tokens", 0) or 0,
            "completion_tokens": _usage_info.get("completion_tokens", 0) or 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        if (
            "completion_tokens_details" in _usage_info
            and _usage_info["completion_tokens_details"] is not None
            and _usage_info["completion_tokens_details"].get("reasoning_tokens") is not None
        ):
            usage_info["completion_tokens_details"]["reasoning_tokens"] = (
                _usage_info["completion_tokens_details"].get("reasoning_tokens", 0)
            )

        self.n_calls += 1

        try:
            reasoning_content = response.choices[0].message.reasoning_content
        except Exception:
            reasoning_content = ""

        tool_calls = []
        if response.choices[0].message.tool_calls is not None:
            for tool_call in response.choices[0].message.tool_calls:
                tool_calls.append(tool_call.model_dump())

        return {
            "content": response.choices[0].message.content or "",
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "extra": {
                "response": response.model_dump(),
                "usage": usage_info,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            },
        }


# ============================================================
# Helper utilities
# ============================================================

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try multiple strategies to extract a JSON object from a string."""
    if not text:
        return None

    # Strategy 1 – fenced block
    fence_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2 – first balanced brace span
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    # Strategy 3 – raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _image_to_base64(image_path: str) -> Optional[str]:
    """Read image from file path and return base64-encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        return None


# ============================================================
# Data extraction for SVG SFT format
# ============================================================

def _extract_system_text(messages: List[Dict[str, Any]]) -> str:
    """Extract system prompt text from messages list."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        return item.get("text", "")
    return ""


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract user prompt text from messages list (text only, no images)."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "\n".join(parts)
    return ""


def _extract_assistant_text(messages: List[Dict[str, Any]]) -> str:
    """Extract last assistant response text from messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        return item.get("text", "")
    return ""


def _extract_image_path(data_item: Dict[str, Any]) -> Optional[str]:
    """Extract the first image path from the 'images' field."""
    images = data_item.get("images", [])
    if images and len(images) > 0:
        return images[0]
    return None


def _build_history_log(messages: List[Dict[str, Any]]) -> str:
    """Build trajectory history string from multi-turn messages.

    For single-turn SFT data this will be 'None (Initial Turn)'.
    """
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if len(assistant_msgs) <= 1:
        return "None (Initial Turn)"

    history_parts = []
    turn_idx = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        content = str(content).strip()

        if role == "assistant":
            turn_idx += 1
            # Truncate for history brevity
            snippet = content[:300] + "..." if len(content) > 300 else content
            history_parts.append(f"[Turn {turn_idx}] assistant: {snippet}")
        elif role == "user" and turn_idx > 0:
            snippet = content[:200] + "..." if len(content) > 200 else content
            history_parts.append(f"[Turn {turn_idx}] user feedback: {snippet}")

    # Exclude the last assistant turn (that's the current response)
    if history_parts:
        # Remove last assistant entry
        for i in range(len(history_parts) - 1, -1, -1):
            if history_parts[i].startswith("[Turn") and "assistant:" in history_parts[i]:
                history_parts = history_parts[:i]
                break

    return "\n".join(history_parts) if history_parts else "None (Initial Turn)"


# ============================================================
# Rubric Generator
# ============================================================

class RubricGenerator:
    """Generate SVG evaluation rubrics for SFT training data.

    Usage:
        generator = RubricGenerator(api_key="...", base_url="...", model_name="...")
        asyncio.run(generator.process_jsonl(input_path, output_path))
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "gemini-2.5-pro",
        max_completion_tokens: int = 8192,
        temperature: float = 1.0,
        max_concurrent: int = 5,
    ):
        self.model = AsyncOpenAIModel(
            model_name=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
        self.model.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model.config.model_name = model_name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ----------------------------------------------------------
    # Single-item generation
    # ----------------------------------------------------------

    async def _generate_rubric_for_item(
        self,
        data_item: Dict[str, Any],
        line_num: int,
    ) -> Dict[str, Any]:
        """Generate rubric for a single JSONL data item."""
        async with self._semaphore:
            messages = data_item.get("messages", [])
            image_path = _extract_image_path(data_item)
            turn_id = f"turn_{line_num:06d}"

            # Extract context
            system_rules = _extract_system_text(messages)
            history_log = _build_history_log(messages)
            solver_response = _extract_assistant_text(messages)

            # Validate image
            if not image_path:
                logger.warning(f"[{turn_id}] No image path found, skipping.")
                return {**data_item, "rubric": None, "rubric_error": "No image path", "turn_id": turn_id}

            if not os.path.exists(image_path):
                logger.warning(f"[{turn_id}] Image file not found: {image_path}")
                return {**data_item, "rubric": None, "rubric_error": f"Image not found: {image_path}", "turn_id": turn_id}

            # Generate rubric
            rubric_text, prompt_msg = await self._generate_rubric_with_retry(
                image_path=image_path,
                system_rules=system_rules,
                history_log=history_log,
                solver_response=solver_response,
                turn_id=turn_id,
            )

            # Parse rubric JSON
            rubric_dict = _try_parse_json(rubric_text) if rubric_text else None

            result = {
                **data_item,
                "rubric_prompt": prompt_msg,
                "turn_id": turn_id,
                "rubric": rubric_dict,
                "rubric_raw": rubric_text,
            }
            if rubric_dict is None:
                result["rubric_error"] = "Failed to parse rubric JSON"

            return result

    async def _generate_rubric_with_retry(
        self,
        image_path: str,
        system_rules: str,
        history_log: str,
        solver_response: str,
        turn_id: str,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], Optional[List]]:
        """Generate rubric by calling the LLM with retry logic."""
        prompt = SVG_RUBRIC_PROMPT.format(
            system_rules=system_rules[:2000] if system_rules else "SVG code generation task",
            history_log=history_log,
            turn_id=turn_id,
        )

        base64_img = _image_to_base64(image_path)
        if not base64_img:
            logger.error(f"[{turn_id}] Failed to encode image: {image_path}")
            return None, None

        # Build message: image + rubric prompt + solver response
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                    {"type": "text", "text": f"\n\n[Solver Response]\n{solver_response}\n\n[Constraint]\nReturn ONLY the JSON in a ```json block."},
                ],
            }
        ]

        for attempt in range(max_retries):
            logger.info(f"[{turn_id}] Attempt {attempt + 1}/{max_retries}")
            try:
                response = await self.model.get_response(messages)
                response_text = response.get("content", "")

                if response_text:
                    parsed = _try_parse_json(response_text)
                    if parsed is not None:
                        logger.info(f"[{turn_id}] Rubric generated successfully.")
                        return response_text, messages
                    else:
                        logger.warning(f"[{turn_id}] JSON parse failed (attempt {attempt + 1}).")
                        continue
                else:
                    logger.warning(f"[{turn_id}] Empty response (attempt {attempt + 1}).")
                    continue

            except OpenAIError as e:
                wait_time = 3 * (2 ** attempt)
                logger.warning(f"[{turn_id}] API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"[{turn_id}] Unexpected error: {e}", exc_info=True)
                break

        logger.error(f"[{turn_id}] All {max_retries} attempts failed.")
        return None, None

    # ----------------------------------------------------------
    # Batch processing
    # ----------------------------------------------------------

    async def process_jsonl_async(
        self,
        input_path: str,
        output_path: str,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Process an entire JSONL file and write rubrics to output."""
        data_items = []
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data_items.append((line_num, item))
                except json.JSONDecodeError as e:
                    logger.error(f"Line {line_num}: JSON decode error: {e}")
                    continue
                if max_items and len(data_items) >= max_items:
                    break

        logger.info(f"Loaded {len(data_items)} items from {input_path}")

        tasks = [
            self._generate_rubric_for_item(item, line_num)
            for line_num, item in data_items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        fail_count = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Item {i + 1} raised exception: {result}")
                    fail_count += 1
                else:
                    if result.get("rubric") is not None:
                        success_count += 1
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        fail_count += 1

        logger.info(
            f"Done. Success: {success_count}, Failed: {fail_count}, "
            f"Total: {len(data_items)}. Output: {output_path}"
        )
        return [r for r in results if not isinstance(r, Exception)]

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper."""
        return asyncio.run(self.process_jsonl_async(input_path, output_path, max_items))


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SVG evaluation rubrics for JSONL training data."
    )
    parser.add_argument(
        "--input", "-i", type=str,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", type=str, default=os.getenv("RUBRIC_TEACHER_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logger.info(f"Starting SVG rubric generation pipeline")
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model:  {args.model}")

    generator = RubricGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_concurrent=args.max_concurrent,
    )

    start_time = time.time()
    results = generator.process_jsonl(
        input_path=args.input,
        output_path=args.output,
        max_items=args.max_items,
    )
    elapsed = time.time() - start_time

    total = len(results)
    success = sum(1 for r in results if r.get("rubric") is not None)
    logger.info(f"Pipeline complete in {elapsed:.1f}s — Success: {success}/{total}")

    if args.verbose:
        for r in results[:3]:
            rubric = r.get("rubric")
            tid = r.get("turn_id", "?")
            if rubric:
                rubrics = rubric.get("rubrics", [])
                print(f"\n[{tid}] {len(rubrics)} rubrics:")
                for rb in rubrics:
                    print(f"  [{rb.get('weight',0):2d}] {rb.get('rule_id','?')}: {rb.get('description','')[:80]}...")
            else:
                print(f"\n[{tid}] FAILED: {r.get('rubric_error', 'unknown')}")


if __name__ == "__main__":
    main()
