"""
STEP 2 - Run the four NLP methods on the 150 clinic notes.

Output: results/predictions.csv  (one row per note, one column per method)

The LLM part is the slow one (it runs on your laptop). Answers are saved in
results/llm_cache.json, so if you stop the script and run it again it
continues where it left off instead of starting over.

Run everything:          python step2_classify_notes.py
Skip the LLM (fast):     python step2_classify_notes.py --no-llm
Try the LLM on 10 notes: python step2_classify_notes.py --limit 10
"""
import argparse
import json
import os
import time

import pandas as pd

from config import DATA_DIR, OLLAMA_MODEL, RESULTS_DIR
from src.classifiers import MLClassifier, keyword_classify, llm_classify, rules_classify
from src.ollama_client import OllamaError, check_ollama

CACHE_PATH = os.path.join(RESULTS_DIR, "llm_cache.json")


def run_llm(notes, limit=None):
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)

    rows = notes if limit is None else notes.head(limit)
    results = {}
    start = time.time()
    for i, (note_id, text) in enumerate(zip(rows.note_id, rows.note_text), start=1):
        key = f"{OLLAMA_MODEL}|{note_id}"
        if key not in cache:
            cache[key] = llm_classify(text)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1)
        results[note_id] = cache[key]
        elapsed = time.time() - start
        print(f"  LLM: note {i}/{len(rows)} done  ({elapsed:.0f}s so far)", end="\r")
    print()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM method")
    parser.add_argument("--limit", type=int, default=None, help="only send N notes to the LLM")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    train = pd.read_csv(os.path.join(DATA_DIR, "train_notes.csv"))
    clinic = pd.read_csv(os.path.join(DATA_DIR, "clinic_notes.csv"))
    out = clinic[["note_id", "patient_id", "label", "category", "note_text"]].copy()

    print("1/4 Keyword search ...")
    kw = [keyword_classify(t) for t in clinic.note_text]
    out["pred_keyword"] = [r["label"] for r in kw]
    out["evidence_keyword"] = [r["evidence"] for r in kw]

    print("2/4 Rules with negation handling ...")
    ru = [rules_classify(t) for t in clinic.note_text]
    out["pred_rules"] = [r["label"] for r in ru]
    out["evidence_rules"] = [r["evidence"] for r in ru]

    print("3/4 Machine learning (training on 400 notes) ...")
    ml = MLClassifier().train(train.note_text, train.label)
    mlr = [ml.classify(t) for t in clinic.note_text]
    out["pred_ml"] = [r["label"] for r in mlr]
    out["evidence_ml"] = [r["evidence"] for r in mlr]

    if args.no_llm:
        print("4/4 LLM skipped (--no-llm).")
    else:
        print(f"4/4 LLM ({OLLAMA_MODEL} via Ollama) - this can take 5-20 minutes ...")
        try:
            check_ollama()
            llm = run_llm(clinic, args.limit)
            out["pred_llm"] = out.note_id.map(lambda n: llm[n]["label"] if n in llm else None)
            out["evidence_llm"] = out.note_id.map(lambda n: llm[n]["evidence"] if n in llm else None)
            out["reason_llm"] = out.note_id.map(lambda n: llm[n]["reason"] if n in llm else None)
            out["evidence_in_note_llm"] = out.note_id.map(
                lambda n: llm[n]["evidence_in_note"] if n in llm else None)
        except OllamaError as e:
            print("\n  LLM skipped:\n  " + str(e))

    out.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)
    print("\nSaved results/predictions.csv")
    print("Next: python step3_evaluate.py")


if __name__ == "__main__":
    main()
