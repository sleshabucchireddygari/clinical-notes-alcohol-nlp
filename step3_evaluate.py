"""
STEP 3 - Measure how accurate each method is.

Compares every method's answer with the gold label (the correct answer).

  Precision = of the patients we FLAGGED, how many really drink too much?
              (low precision = doctors waste time on false alarms)
  Recall    = of the patients who REALLY drink too much, how many did we find?
              (low recall = patients who need help are missed)
  F1        = one number that balances precision and recall

Also shows accuracy for each KIND of sentence (negated, past use, family
history, ...) so we can see exactly where each method fails.

Output: results/RESULTS.md and results/errors_<method>.csv
Run:    python step3_evaluate.py
"""
import os

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from config import OLLAMA_MODEL, RESULTS_DIR

METHODS = {
    "keyword": "Keyword search",
    "rules": "Rules + negation",
    "ml": "TF-IDF + logistic regression",
    "llm": f"LLM ({OLLAMA_MODEL}, local)",
}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "predictions.csv"))
    methods = [m for m in METHODS if f"pred_{m}" in df and df[f"pred_{m}"].notna().any()]

    summary, by_category = [], {}
    for m in methods:
        d = df[df[f"pred_{m}"].notna()]
        y, p = d.label.astype(int), d[f"pred_{m}"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        summary.append({
            "Method": METHODS[m], "Notes": len(d),
            "Precision": round(precision_score(y, p, zero_division=0), 2),
            "Recall": round(recall_score(y, p, zero_division=0), 2),
            "F1": round(f1_score(y, p, zero_division=0), 2),
            "False alarms (FP)": int(fp), "Missed (FN)": int(fn),
        })
        by_category[METHODS[m]] = (d.label == d[f"pred_{m}"]).groupby(d.category).mean()
        cols = ["note_id", "category", "label", f"pred_{m}", f"evidence_{m}", "note_text"]
        d.loc[d.label != d[f"pred_{m}"], cols].to_csv(
            os.path.join(RESULTS_DIR, f"errors_{m}.csv"), index=False)

    summary = pd.DataFrame(summary)
    cat = (pd.DataFrame(by_category) * 100).round(0).astype(int).astype(str) + "%"
    counts = df.category.value_counts().rename("n notes")
    cat = cat.join(counts).sort_values("n notes", ascending=False)

    lines = ["# Results", "",
             "Test set: 150 synthetic HIV-clinic notes (gold labels known).", "",
             "## Overall", "", summary.to_markdown(index=False), "",
             "## Accuracy by type of sentence", "", cat.to_markdown(), ""]
    if "llm" in methods:
        d = df[df.pred_llm.notna()]
        ok = d.evidence_in_note_llm.fillna(False).astype(bool)
        has_ev = d.evidence_llm.fillna("").astype(str).str.len() > 0
        lines += ["## LLM evidence check (hallucination test)", "",
                  f"The LLM gave an evidence quote for {has_ev.sum()} notes. "
                  f"{(ok & has_ev).sum()} of those quotes were found word-for-word in the note; "
                  f"{(~ok & has_ev).sum()} were not (paraphrased or made up).", ""]
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("Saved results/RESULTS.md and results/errors_<method>.csv (open these to see mistakes)")
    print("Next: python step4_build_database.py")


if __name__ == "__main__":
    main()
