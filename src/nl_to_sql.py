"""
Natural Language -> SQL.

A doctor types a question in plain English. The local LLM writes a SQL query,
we check it is safe, run it on the database and show the answer.

How we stop the LLM from inventing tables or columns ("grounding"):
  1. The prompt contains the REAL schema (every table and column) plus a short
     description of what each column means and a few sample rows.
  2. A few example question -> SQL pairs show the expected style.
  3. If the query fails, the error message is sent back to the LLM once so it
     can fix its own mistake (self-correction).

Safety:
  - only SELECT queries are allowed (no DELETE, DROP, UPDATE ...)
  - the database is opened READ-ONLY, so even a bad query can't change data
"""
import re
import sqlite3
from pathlib import Path

from config import DB_PATH
from src.ollama_client import chat_json

COLUMN_NOTES = """Column meanings:
- patients.patient_id: text id like 'P0001'
- patients.sex: 'male' or 'female'
- patients.art_regimen: the HIV medicine, e.g. 'Biktarvy', 'Triumeq', 'Dovato', 'Genvoya', 'Symtuza', 'Cabenuva (injectable)'
- patients.missed_doses_last_month: number of missed HIV medicine doses last month
- patients.viral_load: HIV copies/mL. viral_load < 50 means undetectable (suppressed); viral_load >= 50 means detectable
- patients.cd4_count: cells/mm3; below 200 is low
- patients.last_visit_date and notes.visit_date: text in 'YYYY-MM-DD' format
- alcohol_flags.flagged: 1 = the NLP found CURRENT unhealthy alcohol use in the note, 0 = not found
- alcohol_flags.evidence: the sentence from the note that the NLP used
- alcohol_flags has exactly one row per patient; join it on patient_id
- alcohol_flags contains EVERY patient (flagged or not), so to count or list flagged patients you MUST add WHERE flagged = 1
- patient_overview is a VIEW with one row per patient that ALREADY combines patients and alcohol_flags
  (all patient columns + flagged + evidence). PREFER patient_overview for any question that mixes
  patient details (sex, age, regimen, viral load, CD4, missed doses) with alcohol flags - then no JOIN is needed.
  The column 'flagged' does NOT exist in the patients table.
- To filter dates by month use LIKE, e.g. last_visit_date LIKE '2026-03-%' for March 2026.
- When asked "which" or "what is the most common", return only the column that answers the question."""

FEW_SHOT = """Examples:
Question: How many patients were not flagged?
{"sql": "SELECT COUNT(*) AS n FROM alcohol_flags WHERE flagged = 0;"}

Question: How many male patients are there?
{"sql": "SELECT COUNT(*) AS n FROM patients WHERE sex = 'male';"}

Question: List the patients on Triumeq who were flagged for alcohol use.
{"sql": "SELECT patient_id FROM patient_overview WHERE flagged = 1 AND art_regimen = 'Triumeq';"}

Question: What is the average age for flagged and for non-flagged patients?
{"sql": "SELECT flagged, AVG(age) AS avg_age FROM patient_overview GROUP BY flagged;"}

Question: What is the average number of missed doses for each sex?
{"sql": "SELECT sex, AVG(missed_doses_last_month) AS avg_missed FROM patients GROUP BY sex;"}"""

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b", re.I)


def get_schema_text(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    parts = []
    for (sql,) in conn.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'view')"):
        parts.append(sql.strip() + ";")
    parts.append("\nSample rows:")
    for table in ["patient_overview"]:
        cur = conn.execute(f"SELECT * FROM {table} LIMIT 2")
        cols = [c[0] for c in cur.description]
        for row in cur.fetchall():
            parts.append(f"{table}: " + ", ".join(f"{c}={str(v)[:60]!r}" for c, v in zip(cols, row)))
    conn.close()
    return "\n".join(parts) + "\n\n" + COLUMN_NOTES


def build_system_prompt(schema_text):
    return (
        "You write SQLite queries for a hospital HIV clinic database.\n"
        "Use ONLY the tables and columns listed below. Never invent columns.\n"
        "Write ONE read-only SELECT query. For percentages, return a number from 0 to 100.\n"
        'Reply with JSON only: {"sql": "<the query>"}\n\n'
        "Database schema:\n" + schema_text + "\n\n" + FEW_SHOT
    )


def is_safe(sql):
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return False, "Only one statement is allowed."
    if not re.match(r"^(select|with)\b", s, re.I):
        return False, "Only SELECT queries are allowed."
    if FORBIDDEN.search(s):
        return False, "Query contains a forbidden keyword."
    return True, ""


def run_sql(sql, db_path=DB_PATH):
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)  # read-only
    try:
        cur = conn.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        return cols, cur.fetchall()
    finally:
        conn.close()


def ask(question, system_prompt=None, max_attempts=2):
    """Returns dict: question, sql, columns, rows, error, attempts."""
    system_prompt = system_prompt or build_system_prompt(get_schema_text())
    user = "Question: " + question
    result = {"question": question, "sql": "", "columns": [], "rows": [], "error": "", "attempts": 0}
    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt
        reply = chat_json(system_prompt, user)
        sql = str(reply.get("sql", "")).strip()
        result["sql"] = sql
        safe, why = is_safe(sql) if sql else (False, "The model did not return a query.")
        if not safe:
            error = why
        else:
            try:
                result["columns"], result["rows"] = run_sql(sql)
                result["error"] = ""
                return result
            except sqlite3.Error as e:
                error = str(e)
        result["error"] = error
        # Self-correction: show the model its query and the error, and ask again
        user = (f"Question: {question}\n\nYour previous query was:\n{sql}\n"
                f"It failed with this error: {error}\nWrite a corrected query.")
    return result
