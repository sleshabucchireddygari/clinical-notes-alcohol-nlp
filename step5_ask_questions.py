"""
STEP 5 - Ask the clinic database questions in plain English.

Examples you can try:
  How many patients were flagged for unhealthy alcohol use?
  Compare the percentage of patients with a detectable viral load between flagged and non-flagged patients.
  Show the evidence sentences for flagged female patients.
  What is the average CD4 count for flagged vs not flagged patients?

Run (interactive):    python step5_ask_questions.py
Run (one question):   python step5_ask_questions.py "How many patients are on Biktarvy?"
Type 'quit' to exit.
"""
import sys

import pandas as pd

from src.nl_to_sql import ask, build_system_prompt, get_schema_text
from src.ollama_client import OllamaError, check_ollama

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 160)


def show(result):
    print("\nSQL written by the LLM:\n  " + (result["sql"] or "(none)"))
    if result["attempts"] > 1:
        print(f"  (needed {result['attempts']} attempts - the first query had an error and was self-corrected)")
    if result["error"]:
        print("Could not answer: " + result["error"])
        return
    df = pd.DataFrame(result["rows"], columns=result["columns"])
    print(f"\nAnswer ({len(df)} row{'s' if len(df) != 1 else ''}):")
    print(df.head(20).to_string(index=False))
    if len(df) > 20:
        print(f"... and {len(df) - 20} more rows")


def main():
    try:
        check_ollama()
    except OllamaError as e:
        print(e)
        return
    system_prompt = build_system_prompt(get_schema_text())

    if len(sys.argv) > 1:
        show(ask(" ".join(sys.argv[1:]), system_prompt))
        return

    print("Ask a question about the clinic (type 'quit' to exit).")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            show(ask(q, system_prompt))


if __name__ == "__main__":
    main()
