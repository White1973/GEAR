#!/usr/bin/env python3
"""Normalize FrozenLake solver samples and make a reproducible train/validation split.

Accepted input uses OpenAI-style ``messages``. User content may be a string or a list containing
``text`` and ``image`` items. Output uses the LLaMA-Factory ``messages`` + ``images`` schema.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def normalize_item(sample: dict[str, Any]) -> dict[str, Any]:
    messages_out: list[dict[str, str]] = []
    images: list[str] = list(sample.get("images", []))

    for message in sample["messages"]:
        role = message["role"]
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            text_parts: list[str] = []
            placeholders = 0
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image":
                    image = str(part.get("image", "")).removeprefix("file://")
                    if image:
                        images.append(image)
                        placeholders += 1
            text = ("<image>\n" * placeholders + "".join(text_parts)).strip()
        messages_out.append({"role": role, "content": text})

    placeholder_count = sum(m["content"].count("<image>") for m in messages_out)
    if placeholder_count != len(images):
        raise ValueError(f"image placeholders={placeholder_count}, images={len(images)}")
    return {"messages": messages_out, "images": images}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                items.append(normalize_item(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1")
    items = load_jsonl(args.input)
    random.Random(args.seed).shuffle(items)
    val_size = max(1, round(len(items) * args.val_ratio)) if items else 0
    write_jsonl(args.val_output, items[:val_size])
    write_jsonl(args.train_output, items[val_size:])
    print(f"train={len(items) - val_size}, val={val_size}")


if __name__ == "__main__":
    main()
