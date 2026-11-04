#!/usr/bin/env python3
"""Validate LLaMA-Factory multimodal ``messages`` JSONL data."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    count = 0
    with args.jsonl.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            count += 1
            try:
                item = json.loads(line)
                messages = item["messages"]
                images = item.get("images", [])
                roles = [message.get("role") for message in messages]
                if roles[:2] != ["system", "user"] or roles[-1:] != ["assistant"]:
                    errors.append(f"line {line_number}: unexpected roles {roles}")
                placeholders = sum(str(message.get("content", "")).count("<image>") for message in messages)
                if placeholders != len(images):
                    errors.append(
                        f"line {line_number}: placeholders={placeholders}, images={len(images)}"
                    )
                if not args.allow_missing_images:
                    for image in images:
                        if not Path(image).is_file():
                            errors.append(f"line {line_number}: missing image {image}")
            except Exception as exc:
                errors.append(f"line {line_number}: {exc}")

    if errors:
        for error in errors[:50]:
            print(error)
        raise SystemExit(f"validation failed: {len(errors)} error(s) across {count} item(s)")
    print(f"validation passed: {count} item(s)")


if __name__ == "__main__":
    main()
