"""Utilities for reshaping LangChain SQL table info into compact DDL + inserts."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


def convert_schema_text(schema_text: str) -> str:
    """
    Convert the verbose `SQLDatabase.get_table_info()` output into a prompt-friendly
    format: `CREATE TABLE ...;` followed by `INSERT INTO ... VALUES (...)` rows.
    """
    blocks = _split_table_blocks(schema_text)
    converted: List[str] = []

    for block in blocks:
        create_stmt, comment = _separate_create_and_comment(block)
        if not create_stmt:
            continue

        cleaned_create = _ensure_trailing_semicolon(create_stmt)
        converted.append(cleaned_create)

        table_name = _extract_table_name(create_stmt)
        columns, rows = _parse_sample_rows(comment)
        if table_name and columns and rows:
            converted.append(_build_insert_statement(table_name, columns, rows))

    return "\n".join(converted)


def _split_table_blocks(schema_text: str) -> List[str]:
    # Split on each CREATE TABLE occurrence while keeping the delimiter.
    raw_blocks = re.split(r"(?=CREATE TABLE)", schema_text, flags=re.IGNORECASE)
    return [block.strip() for block in raw_blocks if block.strip()]


def _separate_create_and_comment(block: str) -> Tuple[str, str]:
    if "/*" not in block:
        return block.strip(), ""
    create_part, comment_part = block.split("/*", 1)
    comment_part = comment_part.split("*/", 1)[0]
    return create_part.strip(), comment_part.strip()


def _ensure_trailing_semicolon(statement: str) -> str:
    statement = statement.rstrip()
    return statement if statement.endswith(";") else f"{statement};"


def _extract_table_name(create_stmt: str) -> str:
    match = re.search(r"CREATE TABLE\s+[`\"\[]?([\w]+)[`\"\]]?", create_stmt, re.IGNORECASE)
    return match.group(1) if match else ""


def _parse_sample_rows(comment: str) -> Tuple[List[str], List[List[str]]]:
    lines = [line.strip() for line in comment.splitlines() if line.strip()]
    if not lines:
        return [], []

    # Drop the "3 rows from <table> table:" header if present.
    if "rows from" in lines[0]:
        lines = lines[1:]
    if not lines:
        return [], []

    columns = _split_row(lines[0])
    data_lines = lines[1:]

    rows = []
    for line in data_lines:
        values = _split_row(line)
        if columns and len(values) != len(columns):
            continue
        rows.append(values)
    return columns, rows


def _split_row(line: str) -> List[str]:
    if "\t" in line:
        return line.split("\t")
    # Fallback to splitting on multiple spaces.
    return re.split(r"\s{2,}", line.strip())


def _build_insert_statement(
    table_name: str, columns: Sequence[str], rows: Iterable[Sequence[str]]
) -> str:
    column_list = ", ".join(columns)
    safe_table = table_name.strip('\"`[]')
    value_groups = []
    for row in rows:
        values = ", ".join(_format_value(value) for value in row)
        value_groups.append(f"({values})")
    values_str = ", ".join(value_groups)
    return f"INSERT INTO {safe_table} ({column_list}) VALUES {values_str};"


def _format_value(value: str) -> str:
    value = value.strip()
    if value.upper() == "NULL" or value == "":
        return "NULL"
    if re.fullmatch(r"-?\d+", value) or re.fullmatch(r"-?\d+\.\d+", value):
        return value
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


if __name__ == "__main__":
    import sys

    input_text = sys.stdin.read()
    print(convert_schema_text(input_text))
