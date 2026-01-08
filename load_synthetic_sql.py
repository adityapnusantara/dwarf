#!/usr/bin/env python3
"""
Utility to load gretelai/synthetic_text_to_sql and format rows into chat-style prompts.

For each example we produce:
- system: table schema from sql_context
- user: natural language request from sql_prompt
- assistant: SQL query from sql
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def format_example(example: dict) -> dict:
    system = (
        "You are an expert SQL assistant. "
        "Use ONLY the provided table schema to write a correct SQL query. "
        "Return valid SQL without explanations.\n"
        f"Table schema:\n{example['sql_context']}"
    ).strip()
    user = example["sql_prompt"]
    assistant = example["sql"]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Format gretelai/synthetic_text_to_sql as chat prompts.")
    parser.add_argument("--split", default="train", help="Dataset split to load (default: train).")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of rows (default: 10; 0 disables).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("formatted_synthetic_sql.jsonl"),
        help="Output JSONL path (default: formatted_synthetic_sql.jsonl).",
    )
    args = parser.parse_args()

    ds = load_dataset("gretelai/synthetic_text_to_sql", split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            formatted = format_example(row)
            f.write(json.dumps(formatted, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(ds)} rows to {args.out}")


if __name__ == "__main__":
    main()
