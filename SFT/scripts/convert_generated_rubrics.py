#!/usr/bin/env python3
"""Convert generated rubric records into LLaMA-Factory multimodal SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def prompt_to_text(prompt: Any) -> str:
    """Flatten an OpenAI-style rubric prompt and replace its image with ``<image>``."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for message in prompt or []:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") in {"image", "image_url"}:
                parts.append("<image>")
    return "\n".join(part for part in parts if part).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--system",
        default="You are GEAR's Rubricator. Produce task- and trajectory-specific grounded evaluation rubrics.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            images = item.get("images", [])
            prompt = prompt_to_text(item.get("rubric_prompt"))
            placeholder_count = prompt.count("<image>")
            if images and placeholder_count == 0:
                prompt = "<image>\n" + prompt
            rubric = item.get("rubric_raw") or item.get("rubric")
            if not prompt or not rubric:
                skipped += 1
                continue
            if not isinstance(rubric, str):
                rubric = json.dumps(rubric, ensure_ascii=False)
            record = {
                "messages": [
                    {"role": "system", "content": args.system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rubric},
                ],
                "images": images,
            }
            final_placeholders = sum(m["content"].count("<image>") for m in record["messages"])
            if final_placeholders != len(images):
                raise ValueError(
                    f"line {line_number}: placeholders={final_placeholders}, images={len(images)}"
                )
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1
    print(f"converted={converted}, skipped={skipped}, output={args.output}")


if __name__ == "__main__":
    main()
