"""
STEP 4 - Put everything into a SQL database (SQLite) so it can be queried.

Tables created in data/clinic.db:
  patients       one row per patient (age, sex, HIV medicine, viral load, CD4 ...)
  notes          the doctor notes
  alcohol_flags  the NLP result for each patient: flagged 1/0 + the evidence sentence
  patient_overview  a VIEW that joins patients + alcohol_flags (one row per patient)

This turns messy free text into structured data - the whole point of clinical
NLP. After this, a doctor can ask questions like "Do flagged patients have
more detectable viral loads?" (see step 5).

The flags come from the LLM if you ran it in step 2; otherwise from the rules.
Note: the correct answers (gold labels) are NOT put in the database - the
database only contains what the NLP found, just like in real life.

Run:  python step4_build_database.py
"""
import os
import sqlite3

import pandas as pd

from config import DATA_DIR, DB_PATH, RESULTS_DIR


def main():
    patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
    notes = pd.read_csv(os.path.join(DATA_DIR, "clinic_notes.csv"))
    preds = pd.read_csv(os.path.join(RESULTS_DIR, "predictions.csv"))

    if "pred_llm" in preds and preds.pred_llm.notna().all():
        method = "llm"
    else:
        method = "rules"
        print("Note: LLM results not complete for all notes, so using the rules method's flags.")

    flags = pd.DataFrame({
        "patient_id": preds.patient_id,
        "note_id": preds.note_id,
        "flagged": preds[f"pred_{method}"].astype(int),
        "evidence": preds[f"evidence_{method}"].fillna(""),
        "method": method,
    })

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY,
            age INTEGER,
            sex TEXT,
            art_regimen TEXT,
            missed_doses_last_month INTEGER,
            viral_load INTEGER,
            cd4_count INTEGER,
            last_visit_date TEXT
        );
        CREATE TABLE notes (
            note_id TEXT PRIMARY KEY,
            patient_id TEXT REFERENCES patients(patient_id),
            visit_date TEXT,
            note_text TEXT
        );
        CREATE TABLE alcohol_flags (
            patient_id TEXT REFERENCES patients(patient_id),
            note_id TEXT REFERENCES notes(note_id),
            flagged INTEGER,
            evidence TEXT,
            method TEXT
        );
        -- A ready-joined view, so the LLM rarely needs to write a JOIN itself
        CREATE VIEW patient_overview AS
            SELECT p.patient_id, p.age, p.sex, p.art_regimen, p.missed_doses_last_month,
                   p.viral_load, p.cd4_count, p.last_visit_date,
                   a.flagged, a.evidence
            FROM patients p JOIN alcohol_flags a ON p.patient_id = a.patient_id;
    """)
    patients.to_sql("patients", conn, if_exists="append", index=False)
    notes[["note_id", "patient_id", "visit_date", "note_text"]].to_sql(
        "notes", conn, if_exists="append", index=False)
    flags.to_sql("alcohol_flags", conn, if_exists="append", index=False)
    conn.commit()

    n_flag = conn.execute("SELECT SUM(flagged) FROM alcohol_flags").fetchone()[0]
    conn.close()
    print(f"Created {DB_PATH}")
    print(f"  patients: {len(patients)}   notes: {len(notes)}   flagged by '{method}': {n_flag}")
    print("Next: python step5_ask_questions.py")


if __name__ == "__main__":
    main()
