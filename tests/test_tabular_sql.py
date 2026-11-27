"""Unit and integration tests for DuckDB Tabular SQL Router & Read-Only Query Engine."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import openpyxl
import pytest

from src.retrieval.tabular_sql import (
    TabularSQLPlan,
    execute_tabular_sql_query,
    load_tabular_connection,
    should_route_to_tabular_sql,
    validate_readonly_sql,
)


@pytest.fixture()
def tabular_sandbox():
    root = Path(tempfile.mkdtemp(prefix="ragquest_duckdb_"))
    try:
        (root / "employee_directory.csv").write_text(
            "employee_id,full_name,department,hub_location\n"
            "NV-EMP-1001,Dr. Elena Rostova,Robotics Engineering,Zurich Lab\n"
            "NV-EMP-1018,Marcus Vance,Information Security,Austin HQ\n"
            "NV-EMP-1042,Priya Nair,People Operations,Bengaluru R&D Center\n"
            "NV-EMP-1089,Devon Brooks,Robotics Engineering,Austin HQ\n",
            encoding="utf-8",
        )
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Budgets"
        ws.append(["cost_center", "hub", "annual_budget_usd"])
        ws.append(["CC-ZUR-90", "Zurich Lab", 450000])
        ws.append(["CC-AUS-12", "Austin HQ", 620000])
        wb.save(root / "ops_budgets.xlsx")
        wb.close()
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_tabular_connection_loads_csv_and_excel(tabular_sandbox: Path) -> None:
    conn, tables = load_tabular_connection(tabular_sandbox)
    try:
        names = {t.source_file: t for t in tables}
        assert "employee_directory.csv" in names
        assert "ops_budgets.xlsx" in names
        assert names["employee_directory.csv"].row_count == 4
        assert names["ops_budgets.xlsx"].row_count == 2
        total_budget = conn.execute(
            f'SELECT SUM(annual_budget_usd) FROM "{names["ops_budgets.xlsx"].table_name}"'
        ).fetchone()[0]
        assert total_budget == 1070000
    finally:
        conn.close()


def test_should_route_to_tabular_sql_intent_classifier(tabular_sandbox: Path) -> None:
    conn, tables = load_tabular_connection(tabular_sandbox)
    try:
        assert should_route_to_tabular_sql("How many employees work in Austin HQ?", tables) is True
        assert should_route_to_tabular_sql("What is the total annual_budget_usd across all hubs?", tables) is True
        # Prose policy questions must NOT route to SQL
        assert should_route_to_tabular_sql("What is the lodging cap in Zurich under POL-EXP-402?", tables) is False
        assert should_route_to_tabular_sql("What are the SEV-1 incident notification SLAs?", tables) is False
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_sql",
    [
        "",
        "DROP TABLE employee_directory",
        "DELETE FROM employee_directory WHERE 1=1",
        "SELECT * FROM employee_directory; DROP TABLE employee_directory",
        "PRAGMA table_info('employee_directory')",
        "COPY employee_directory TO 'evil.csv'",
    ],
)
def test_validate_readonly_sql_blocks_mutations_and_injection(bad_sql: str) -> None:
    with pytest.raises(ValueError):
        validate_readonly_sql(bad_sql)


def test_execute_tabular_sql_query_returns_exact_result_chunk(tabular_sandbox: Path) -> None:
    conn, tables = load_tabular_connection(tabular_sandbox)
    try:
        plan_json = TabularSQLPlan(
            sql_query="SELECT hub_location, COUNT(*) AS employee_count FROM employee_directory GROUP BY hub_location ORDER BY employee_count DESC",
            source_files=["employee_directory.csv"],
            reasoning="Count employees grouped by hub_location.",
        ).model_dump_json()

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=plan_json))],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=45, total_tokens=345),
        )

        res = execute_tabular_sql_query(
            query="How many employees are in each hub?",
            conn=conn,
            tables=tables,
            groq_client=mock_groq,
        )
        assert res["row_count"] == 3
        assert res["rows"][0]["hub_location"] == "Austin HQ"
        assert res["rows"][0]["employee_count"] == 2
        assert res["chunks"][0].source_file == "employee_directory.csv"
        assert "Austin HQ" in res["chunks"][0].text
    finally:
        conn.close()
