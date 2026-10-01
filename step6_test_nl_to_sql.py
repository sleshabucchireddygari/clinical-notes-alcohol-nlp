"""
STEP 6 - Test how accurate the NL-to-SQL part is.

We have 15 test questions, each with a correct ("gold") SQL query written by
hand. For each question we run the LLM's query AND the gold query and compare
the RESULTS (not the SQL text - two different queries can give the same
correct answer). This is called "execution accuracy".

Output: results/NL_TO_SQL_RESULTS.md
Run:    python step6_test_nl_to_sql.py
"""
import os

import pandas as pd

from config import OLLAMA_MODEL, RESULTS_DIR
from src.nl_to_sql import ask, build_system_prompt, get_schema_text, run_sql
from src.ollama_client import OllamaError, check_ollama

TEST_QUESTIONS = [
    ("How many patients are in the clinic?",
     "SELECT COUNT(*) FROM patients"),
    ("How many patients were flagged for unhealthy alcohol use?",
     "SELECT COUNT(*) FROM alcohol_flags WHERE flagged = 1"),
    ("How many patients are on Biktarvy?",
     "SELECT COUNT(*) FROM patients WHERE art_regimen = 'Biktarvy'"),
    ("Which HIV medicine regimen is the most common?",
     "SELECT art_regimen FROM patients GROUP BY art_regimen ORDER BY COUNT(*) DESC LIMIT 1"),
    ("How many female patients were flagged for alcohol use?",
     "SELECT COUNT(*) FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id "
     "WHERE a.flagged = 1 AND p.sex = 'female'"),
    ("What is the average age of flagged patients?",
     "SELECT AVG(p.age) FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id "
     "WHERE a.flagged = 1"),
    ("How many patients have a detectable viral load?",
     "SELECT COUNT(*) FROM patients WHERE viral_load >= 50"),
    ("List the IDs of flagged patients who have a detectable viral load.",
     "SELECT p.patient_id FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id "
     "WHERE a.flagged = 1 AND p.viral_load >= 50"),
    ("How many flagged patients missed 3 or more doses last month?",
     "SELECT COUNT(*) FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id "
     "WHERE a.flagged = 1 AND p.missed_doses_last_month >= 3"),
    ("What is the average CD4 count for flagged and for non-flagged patients?",
     "SELECT a.flagged, AVG(p.cd4_count) FROM patients p JOIN alcohol_flags a "
     "ON p.patient_id = a.patient_id GROUP BY a.flagged"),
    ("What percentage of flagged patients have a detectable viral load, compared with non-flagged patients?",
     "SELECT a.flagged, 100.0 * SUM(CASE WHEN p.viral_load >= 50 THEN 1 ELSE 0 END) / COUNT(*) "
     "FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id GROUP BY a.flagged"),
    ("How many patients have a CD4 count below 200?",
     "SELECT COUNT(*) FROM patients WHERE cd4_count < 200"),
    ("How many patients had their last visit in August 2026?",
     "SELECT COUNT(*) FROM patients WHERE last_visit_date LIKE '2026-08-%'"),
    ("List the flagged patients older than 50.",
     "SELECT p.patient_id FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id "
     "WHERE a.flagged = 1 AND p.age > 50"),
    ("How many patients on Dovato were flagged for alcohol use?",
     "SELECT COUNT(*) FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id "
     "WHERE a.flagged = 1 AND p.art_regimen = 'Dovato'"),
]


def normalise(rows):
    """Make results comparable: round numbers, ignore row and column order."""
    out = []
    for row in rows:
        vals = [round(v, 1) if isinstance(v, float) else v for v in row]
        out.append(tuple(sorted(str(v) for v in vals)))
    return sorted(out)


def main():
    try:
        check_ollama()
    except OllamaError as e:
        print(e)
        return
    system_prompt = build_system_prompt(get_schema_text())

    records = []
    for i, (question, gold_sql) in enumerate(TEST_QUESTIONS, start=1):
        print(f"Question {i}/{len(TEST_QUESTIONS)}: {question}")
        result = ask(question, system_prompt)
        _, gold_rows = run_sql(gold_sql)
        correct = not result["error"] and normalise(result["rows"]) == normalise(gold_rows)
        print("   ", "CORRECT" if correct else "WRONG", "|", result["sql"] or result["error"])
        records.append({
            "Question": question, "Correct": "yes" if correct else "no",
            "Attempts": result["attempts"], "LLM SQL": result["sql"].replace("|", "/"),
            "Error": result["error"],
        })

    df = pd.DataFrame(records)
    n_ok = (df.Correct == "yes").sum()
    report = (f"# NL-to-SQL results\n\nModel: {OLLAMA_MODEL} (local, via Ollama)\n\n"
              f"**Execution accuracy: {n_ok}/{len(df)} ({100 * n_ok / len(df):.0f}%)**\n\n"
              f"Self-corrected after an error: {(df.Attempts > 1).sum()} question(s)\n\n"
              + df.to_markdown(index=False) + "\n\n"
              "Note: matching is strict (e.g. 0.35 vs 35% counts as wrong), so read the "
              "WRONG rows - some may be reasonable answers in a different format.\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "NL_TO_SQL_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nExecution accuracy: {n_ok}/{len(df)}")
    print("Saved results/NL_TO_SQL_RESULTS.md")


if __name__ == "__main__":
    main()
