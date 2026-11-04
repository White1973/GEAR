#!/usr/bin/env python3
"""Reproducibly sample records from a JSONL file."""

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = min(args.n, len(lines))
    selected = random.Random(args.seed).sample(lines, count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    print(f"sampled={count}, total={len(lines)}, output={args.output}")


if __name__ == "__main__":
    main()
