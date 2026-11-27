"""DuckDB SQL Router & Read-Only Query Engine for CSV and Excel (.xlsx/.xlsm) files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import openpyxl
from pydantic import BaseModel, ConfigDict, Field

from src.config import DATA_DIR, DEFAULT_GROQ_MODEL
from src.schemas import RetrievedChunk

TABULAR_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx", ".xlsm")

_AGGREGATION_RE = re.compile(
    r"\b("
    r"how\s+many|count|total|sum|average|avg|mean|median|min|minimum|max|maximum|"
    r"highest|lowest|top\s+\d+|list\s+all|all\s+employees|all\s+rows|"
    r"group\s+by|per\s+hub|per\s+department|per\s+site|per\s+sheet|"
    r"breakdown|calculate|compute|aggregate"
    r")\b",
    re.IGNORECASE,
)

_FORBIDDEN_SQL_RE = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"copy|export|import|pragma|call|load|install|vacuum|checkpoint|merge"
    r")\b",
    re.IGNORECASE,
)


class TabularSQLPlan(BaseModel):
    """Strict structured SQL plan returned by Groq for DuckDB execution."""

    model_config = ConfigDict(extra="forbid")

    sql_query: str = Field(
        ...,
        description="Single read-only DuckDB SELECT (or WITH ... SELECT) query using exact table and column names.",
    )
    source_files: list[str] = Field(
        ...,
        description="Exact source filename(s) (e.g. employee_directory.csv) queried by the SQL statement.",
    )
    reasoning: str = Field(
        ...,
        description="Brief 1-sentence explanation of what the SQL query computes.",
    )


@dataclass(frozen=True)
class TabularTableInfo:
    """Metadata and schema preview for a DuckDB table backed by a CSV or Excel sheet."""

    table_name: str
    source_file: str
    sheet_name: str | None
    row_count: int
    columns: list[tuple[str, str]]
    sample_rows: list[dict[str, Any]]


def _sanitize_sql_identifier(raw: str, fallback: str = "col") -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _dedupe_identifiers(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, raw in enumerate(names):
        base = _sanitize_sql_identifier(raw, fallback=f"col_{idx + 1}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def _coerce_csv_or_cell_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    return str(val).strip()


def load_tabular_connection(
    data_dir: Path = DATA_DIR,
) -> tuple[duckdb.DuckDBPyConnection, list[TabularTableInfo]]:
    """Load all .csv and .xlsx/.xlsm files from data_dir into an in-memory DuckDB connection."""
    conn = duckdb.connect(":memory:")
    tables: list[TabularTableInfo] = []
    used_table_names: set[str] = set()

    if not data_dir.exists():
        return conn, tables

    def _unique_table_name(base: str) -> str:
        candidate = _sanitize_sql_identifier(base, fallback="table")
        if candidate not in used_table_names:
            used_table_names.add(candidate)
            return candidate
        i = 2
        while f"{candidate}_{i}" in used_table_names:
            i += 1
        final = f"{candidate}_{i}"
        used_table_names.add(final)
        return final

    tabular_files = sorted(
        p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in TABULAR_EXTENSIONS
    )

    for fp in tabular_files:
        ext = fp.suffix.lower()
        if ext == ".csv":
            tname = _unique_table_name(fp.stem)
            try:
                conn.execute(
                    f'CREATE TABLE "{tname}" AS SELECT * FROM read_csv_auto(?, header=true, normalize_names=true)',
                    [str(fp)],
                )
                cols_info = [(r[0], r[1]) for r in conn.execute(f'DESCRIBE "{tname}"').fetchall()]
                row_cnt = int(conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0])
                sample_cur = conn.execute(f'SELECT * FROM "{tname}" LIMIT 3')
                sample_cols = [d[0] for d in sample_cur.description]
                samples = [dict(zip(sample_cols, row)) for row in sample_cur.fetchall()]
                tables.append(
                    TabularTableInfo(
                        table_name=tname,
                        source_file=fp.name,
                        sheet_name=None,
                        row_count=row_cnt,
                        columns=cols_info,
                        sample_rows=samples,
                    )
                )
            except Exception:
                conn.execute(f'DROP TABLE IF EXISTS "{tname}"')
                continue

        elif ext in (".xlsx", ".xlsm"):
            try:
                wb = openpyxl.load_workbook(fp, read_only=True, data_only=True)
            except Exception:
                continue
            try:
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    row_iter = ws.iter_rows(values_only=True)
                    headers: list[str] = []
                    for raw_row in row_iter:
                        if raw_row and any(c is not None and str(c).strip() != "" for c in raw_row):
                            headers = _dedupe_identifiers(
                                [str(c).strip() if c is not None else f"col_{i + 1}" for i, c in enumerate(raw_row)]
                            )
                            break
                    if not headers:
                        continue

                    data_rows: list[tuple[Any, ...]] = []
                    for raw_row in row_iter:
                        if not raw_row or not any(c is not None and str(c).strip() != "" for c in raw_row):
                            continue
                        padded = [
                            _coerce_csv_or_cell_value(raw_row[i] if i < len(raw_row) else None)
                            for i in range(len(headers))
                        ]
                        data_rows.append(tuple(padded))

                    if not data_rows:
                        continue

                    # Infer DuckDB column types from Python values
                    col_defs: list[str] = []
                    for c_idx, col_name in enumerate(headers):
                        non_null = [r[c_idx] for r in data_rows if r[c_idx] is not None and r[c_idx] != ""]
                        if non_null and all(isinstance(v, bool) for v in non_null):
                            sql_type = "BOOLEAN"
                        elif non_null and all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
                            sql_type = "BIGINT"
                        elif non_null and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
                            sql_type = "DOUBLE"
                        else:
                            sql_type = "VARCHAR"
                            data_rows = [
                                tuple(
                                    str(r[j]) if (j == c_idx and r[j] is not None) else r[j]
                                    for j in range(len(headers))
                                )
                                for r in data_rows
                            ]
                        col_defs.append(f'"{col_name}" {sql_type}')

                    base_name = fp.stem if len(wb.sheetnames) == 1 else f"{fp.stem}__{sheet_name}"
                    tname = _unique_table_name(base_name)
                    conn.execute(f'CREATE TABLE "{tname}" ({", ".join(col_defs)})')
                    placeholders = ", ".join(["?"] * len(headers))
                    conn.executemany(f'INSERT INTO "{tname}" VALUES ({placeholders})', data_rows)

                    cols_info = [(r[0], r[1]) for r in conn.execute(f'DESCRIBE "{tname}"').fetchall()]
                    samples = [dict(zip(headers, r)) for r in data_rows[:3]]
                    tables.append(
                        TabularTableInfo(
                            table_name=tname,
                            source_file=fp.name,
                            sheet_name=sheet_name,
                            row_count=len(data_rows),
                            columns=cols_info,
                            sample_rows=samples,
                        )
                    )
            finally:
                wb.close()

    return conn, tables


def validate_readonly_sql(sql: str) -> str:
    """Ensure generated SQL is a single read-only SELECT or WITH query; raise ValueError otherwise."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("Generated SQL query is empty.")
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not permitted.")
    lowered = cleaned.lower()
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise ValueError("Only SELECT or WITH queries are permitted.")
    if _FORBIDDEN_SQL_RE.search(cleaned):
        raise ValueError("Disallowed non-read-only SQL keyword detected.")
    return cleaned


def should_route_to_tabular_sql(query: str, tables: list[TabularTableInfo]) -> bool:
    """Decide whether a query expresses an aggregation/counting intent over loaded CSV/Excel tables."""
    if not tables or not query.strip():
        return False
    if not _AGGREGATION_RE.search(query):
        return False

    q_lower = query.lower()
    q_tokens = set(re.findall(r"[a-z0-9]+", q_lower))

    # Build vocabulary of table names, filenames, sheet names, and column names
    schema_tokens: set[str] = {"csv", "excel", "xlsx", "spreadsheet", "table", "sheet", "rows", "directory"}
    for t in tables:
        schema_tokens.update(re.findall(r"[a-z0-9]+", t.source_file.lower()))
        schema_tokens.update(re.findall(r"[a-z0-9]+", t.table_name.lower()))
        if t.sheet_name:
            schema_tokens.update(re.findall(r"[a-z0-9]+", t.sheet_name.lower()))
        for col_name, _ in t.columns:
            schema_tokens.update(re.findall(r"[a-z0-9]+", col_name.lower()))
        for row in t.sample_rows:
            for v in row.values():
                if v is not None:
                    schema_tokens.update(re.findall(r"[a-z0-9]+", str(v).lower()))

    return bool(q_tokens & schema_tokens)


def format_schemas_for_sql_prompt(tables: list[TabularTableInfo]) -> str:
    """Format DuckDB table schemas and sample rows for Text-to-SQL generation."""
    blocks: list[str] = []
    for t in tables:
        col_str = ", ".join(f"{c} ({tp})" for c, tp in t.columns)
        sheet_label = f" | sheet: {t.sheet_name}" if t.sheet_name else ""
        sample_str = json.dumps(t.sample_rows, default=str)
        blocks.append(
            f"Table `{t.table_name}` (source_file: {t.source_file}{sheet_label}, rows: {t.row_count})\n"
            f"  Columns: {col_str}\n"
            f"  Sample rows: {sample_str}"
        )
    return "\n\n".join(blocks)


def execute_tabular_sql_query(
    query: str,
    conn: duckdb.DuckDBPyConnection,
    tables: list[TabularTableInfo],
    groq_client: Any,
    model: str = DEFAULT_GROQ_MODEL,
    max_rows: int = 50,
) -> dict[str, Any]:
    """Generate a read-only DuckDB SQL query via Groq, execute it, and package results as RetrievedChunks."""
    schema_text = format_schemas_for_sql_prompt(tables)
    schema_dict = TabularSQLPlan.model_json_schema()

    sys_msg = (
        "You are a precise DuckDB SQL generator. Convert the user's question into a single read-only "
        "DuckDB SELECT (or WITH ... SELECT) query using ONLY the exact table and column names below.\n"
        "Use ILIKE for case-insensitive string matching when filtering text values.\n\n"
        f"{schema_text}"
    )
    user_msg = f"Question: {query}\n\nReturn JSON matching TabularSQLPlan."

    completion = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_completion_tokens=1024,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "tabular_sql_plan", "strict": True, "schema": schema_dict},
        },
    )
    raw_plan = completion.choices[0].message.content or ""
    plan = TabularSQLPlan.model_validate_json(raw_plan)
    safe_sql = validate_readonly_sql(plan.sql_query)

    cursor = conn.execute(safe_sql)
    result_cols = [desc[0] for desc in cursor.description] if cursor.description else []
    raw_rows = cursor.fetchmany(max_rows)
    formatted_rows = [dict(zip(result_cols, row)) for row in raw_rows]

    if not formatted_rows:
        raise ValueError("DuckDB SQL query returned 0 rows.")

    valid_sources = {t.source_file for t in tables}
    matched_sources = [sf for sf in plan.source_files if sf in valid_sources] or [tables[0].source_file]

    rows_lines = [
        " | ".join(f"{k}: {v}" for k, v in row_dict.items())
        for row_dict in formatted_rows
    ]
    chunk_text = (
        f"[DuckDB SQL Execution]\n"
        f"Query: {safe_sql}\n"
        f"Reasoning: {plan.reasoning}\n"
        f"Result ({len(formatted_rows)} row(s)):\n"
        + "\n".join(rows_lines)
    )

    chunks = [
        RetrievedChunk(
            chunk_id=f"{sf}::duckdb_sql_result",
            text=chunk_text,
            source_file=sf,
            page_or_section="DuckDB SQL Query Result",
            rerank_score=1.0,
        )
        for sf in matched_sources[:2]
    ]

    usage = None
    if getattr(completion, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
        }

    return {
        "sql_query": safe_sql,
        "reasoning": plan.reasoning,
        "source_files": matched_sources,
        "columns": result_cols,
        "rows": formatted_rows,
        "row_count": len(formatted_rows),
        "chunks": chunks,
        "token_usage": usage,
    }
