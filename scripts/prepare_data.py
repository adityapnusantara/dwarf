#!/usr/bin/env python3
"""
Prepare gretelai/synthetic_text_to_sql as chat_template data.

- system: schema from sql_context with explicit SQL assistant instruction
- user: natural language request from sql_prompt
- assistant: SQL query from sql
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

PROMPT_TEXT_TO_SQL = '''You are an expert SQL assistant. Use ONLY the provided table schema to write a correct SQL query. Return valid SQL without explanations.
Table schema:
{table_info}
Question: {input}
Output format enclose the generated SQL query in a code block::
```
-- Your SQL query
```
'''

# PROMPT_TEXT_TO_SQL = '''Task Overview:
# You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

# Database Engine:
# SQLite

# Database Schema:
# {table_info}
# This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

# Question:
# {input}

# Instructions:
# - Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
# - The generated query should return all of the information asked in the question without any missing or extra information.
# - Before generating the final SQL query, please think through the steps of how to write the query.

# Output Format:
# In your answer, please enclose the generated SQL query in a code block:
# ```
# -- Your SQL query
# ```
# '''

def format_example(example: dict) -> dict:
    user = PROMPT_TEXT_TO_SQL.format(
        table_info=example["sql_context"],
        input=example["sql_prompt"],
    ).strip()
    assistant = f"```{example['sql']}```"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Format gretelai/synthetic_text_to_sql as chat prompts.")
    parser.add_argument("--split", default="train", help="Dataset split to load (default: train).")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Limit number of rows (default: 200; 0 disables and loads full split).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/train.jsonl"),
        help="Output JSONL path (default: data/train_sample.jsonl).",
    )
    args = parser.parse_args()

    ds = load_dataset("gretelai/synthetic_text_to_sql", split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            formatted = format_example(row)
            f.write(json.dumps(formatted, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(ds)} rows to {args.out}")


if __name__ == "__main__":
    main()