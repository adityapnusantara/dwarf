"""Run text-to-SQL evaluation on the finetuned model using the test set."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sqlalchemy import create_engine
from tqdm import tqdm

from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.schema_converter import convert_schema_text
DEFAULT_MODEL_PATH = ROOT_DIR / "outputs" / "gemma-3-1b-sql-qlora" / "merged"
DEFAULT_TEST_PATH = ROOT_DIR / "data" / "test.jsonl"
DATABASE_DIR = ROOT_DIR / "database"

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


@dataclass
class Example:
    db_id: str
    question: str
    sql: str
    sql_complexity: str
    question_style: str


def load_examples(path: Path, limit: Optional[int] = None) -> List[Example]:
    rows: List[Example] = []
    with path.open() as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            payload: Dict[str, str] = json.loads(line)
            rows.append(
                Example(
                    db_id=payload["db_id"],
                    question=payload["question"],
                    sql=payload["sql"],
                    sql_complexity=payload.get("sql_complexity"),
                    question_style=payload.get("question_style"),
                )
            )
    return rows


def resolve_db_path(db_id: str) -> Path:
    matches = list(DATABASE_DIR.rglob(f"{db_id}.sqlite"))
    if not matches:
        raise FileNotFoundError(f"Could not find database for db_id={db_id}")
    return matches[0]


def build_chat_model(model_path: Path) -> ChatHuggingFace:
    llm = HuggingFacePipeline.from_model_id(
        model_id=str(model_path),
        task="text-generation",
        model_kwargs={"trust_remote_code": True, "device_map": "auto"},
    )
    return ChatHuggingFace(llm=llm, temperature=0)


def post_process_sql(sql: str) -> Optional[str]:
    # Extract text after "<start_of_turn>model" and keep the first SELECT...; block.
    pattern = r"<start_of_turn>model(.*)"
    match_answer = re.search(pattern, sql, re.DOTALL)
    if match_answer:
        answer = match_answer.group(1).strip()
        match_sql = re.search(r"(SELECT.*?);", answer, re.DOTALL | re.IGNORECASE)
        if match_sql:
            return match_sql.group(1).strip()
    return None


def build_sql_chain(chat: ChatHuggingFace, database: SQLDatabase):
    prompt_template = PromptTemplate(
        template=PROMPT_TEXT_TO_SQL,
        input_variables=["table_info", "input"],
    )

    def _schema(_: Dict) -> str:
        return convert_schema_text(database.get_table_info())

    return (
        RunnablePassthrough.assign(table_info=_schema)
        | prompt_template
        | chat
        | StrOutputParser()
    )


def evaluate(
    examples: Iterable[Example],
    chat: ChatHuggingFace,
    output_path: Optional[Path] = None,
) -> None:
    total = 0
    success = 0
    fail = 0
    skip = 0

    result_file = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_file = output_path.open("w", encoding="utf-8")

    examples_list = list(examples)
    progress = tqdm(examples_list, desc="Evaluating", unit="q")

    for example in progress:
        total += 1
        db_path = resolve_db_path(example.db_id)
        engine = create_engine(f"sqlite:///{db_path}")
        database = SQLDatabase(engine, sample_rows_in_table_info=3)

        if example.sql:
            try:
                database.run(example.sql)
            except Exception:  # noqa: BLE001
                skip += 1
                progress.set_postfix(success=success, fail=fail, skip=skip)
                if result_file:
                    result_file.write(
                        json.dumps(
                            {
                                "db_id": example.db_id,
                                "question": example.question,
                                "ground_truth_sql": example.sql,
                                "sql_complexity": example.sql_complexity,
                                "question_style": example.question_style,
                                "model_raw_output": None,
                                "parsed_sql": None,
                                "status": "skip",
                            }
                        )
                        + "\n"
                    )
                continue

        sql_chain = build_sql_chain(chat, database)
        raw_sql = sql_chain.invoke({"input": example.question})
        # print(f"Raw SQL:\n{raw_sql}\n")
        parsed_sql = post_process_sql(raw_sql)
        # print(f"Parsed SQL:\n{parsed_sql}\n")
        # print(f"Ground Truth SQL:\n{example.sql}\n")

        if not parsed_sql:
            fail += 1
            progress.set_postfix(success=success, fail=fail, skip=skip)
            if result_file:
                result_file.write(
                        json.dumps(
                            {
                                "db_id": example.db_id,
                                "question": example.question,
                                "ground_truth_sql": example.sql,
                                "sql_complexity": example.sql_complexity,
                                "question_style": example.question_style,
                                "model_raw_output": raw_sql,
                                "parsed_sql": None,
                                "status": "fail",
                            }
                        )
                    + "\n"
                )
            continue

        try:
            database.run(parsed_sql)
            success += 1
            status = "success"
        except Exception:  # noqa: BLE001
            fail += 1
            status = "fail"

        progress.set_postfix(success=success, fail=fail, skip=skip)
        if result_file:
            result_file.write(
                json.dumps(
                    {
                        "db_id": example.db_id,
                        "question": example.question,
                        "ground_truth_sql": example.sql,
                        "sql_complexity": example.sql_complexity,
                        "question_style": example.question_style,
                        "model_raw_output": raw_sql,
                        "parsed_sql": parsed_sql,
                        "status": status,
                    }
                )
                + "\n"
            )

    attempted = total - skip
    success_rate = (success / attempted * 100) if attempted else 0
    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"Skipped (ground truth failed): {skip}")
    print(f"Attempted: {attempted}")
    print(f"Execution Success: {success} ({success_rate:.2f}%)")
    if result_file:
        result_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned text-to-SQL model on the provided test set."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the merged model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help=f"Path to JSONL test file (default: {DEFAULT_TEST_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to run (default: all)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Write per-example results (JSONL) to this path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.test_path, limit=args.limit)
    chat = build_chat_model(args.model_path)
    evaluate(examples, chat, output_path=args.output_file)


if __name__ == "__main__":
    main()
