# Finding Unhealthy Alcohol Use in Clinical Notes with NLP + Natural-Language-to-SQL

A small end-to-end clinical NLP project on **synthetic** HIV-clinic data:

1. **Read** free-text doctor notes and flag patients with **current unhealthy alcohol use**, quoting the sentence used as evidence.
2. **Compare** four NLP approaches, from keyword search to a local LLM, and measure where each one fails.
3. **Store** the results in a SQL database and let a clinician **ask questions in plain English** (Natural Language → SQL).

Everything runs **locally** with a free open-source LLM through [Ollama](https://ollama.com). No real patient data is used.

---

## Why this problem

Unhealthy alcohol use is common among people living with HIV and is linked to missed ART doses and worse outcomes. Screening fields in the health record (like AUDIT-C) are often skipped or out of date. The information is frequently written **only in the free-text note**, for example *"drinks 6–8 beers on weekends"* or *"smelled of alcohol at visit."* Nobody can read thousands of notes by hand, so NLP is used to find these patients.

The hard part is **context**. The word "alcohol" appearing in a note does not mean the patient drinks too much:

| Note says | Correct answer | Why it's tricky |
|---|---|---|
| "Denies alcohol use." | No | negation |
| "Quit drinking in 2015, sober since." | No | past use |
| "FHx: mother with alcoholism." | No | family member, not the patient |
| "Social drinker, 1 glass of wine on weekends." | No | light drinking |
| "Pt appeared intoxicated and smelled of EtOH." | **Yes** | no amount, only indirect signs |
| "Denies IVDU but reports drinking 8 tall cans of beer every day." | **Yes** | a negation word applies to a different thing |

---

## Pipeline

```
 synthetic notes ──► 4 NLP methods ──► evaluation (precision / recall / F1, per sentence type)
                          │
                          ▼
             SQLite database: patients + notes + alcohol_flags (flag + evidence)
                          │
                          ▼
   "Do flagged patients have more detectable viral loads?"  ──LLM──►  SQL  ──►  answer
```

## The four NLP methods

| # | Method | Idea |
|---|---|---|
| 1 | **Keyword search** | Flag the note if any alcohol word appears |
| 2 | **Rules + negation** | Sentence-level rules inspired by NegEx: handle "denies", scope enders like "but", past use, family history and amount |
| 3 | **TF-IDF + logistic regression** | Classic supervised ML trained on 400 labelled notes |
| 4 | **Local LLM** (Llama 3.2 via Ollama) | Prompted with clinical definitions and a few examples. Returns JSON `{yes/no, evidence, reason}` at temperature 0 |

**Evidence check (hallucination test):** for the LLM, the code checks whether the quoted evidence sentence really appears word-for-word in the note.

## Test design

- **Training notes (400)** and **test notes (150)** use *different wording* for the same ideas. This checks that methods generalise to new phrasing instead of memorising sentences, which is the real challenge with clinical notes.
- The rules were written by looking **only at training notes**.
- Every test note has a gold label and a **sentence category** (negated, past use, family history…), so the errors can be analysed by type.


## Results

Test set: 150 synthetic HIV-clinic notes (52 with current unhealthy alcohol use). Full tables in [`results/RESULTS.md`](results/RESULTS.md).

| Method | Precision | Recall | F1 | False alarms | Missed |
|:--|--:|--:|--:|--:|--:|
| Keyword search | 0.41 | 1.00 | 0.58 | 75 | 0 |
| Rules + negation | 1.00 | 0.69 | 0.82 | 0 | 16 |
| TF-IDF + logistic regression | 0.94 | 0.65 | 0.77 | 2 | 18 |
| **LLM (Llama 3.2 3B, local)** | 0.77 | **0.98** | **0.86** | 15 | 1 |

**Where each method fails (accuracy by type of sentence):**

| Type of sentence | Keyword | Rules | ML | LLM |
|:--|--:|--:|--:|--:|
| Negated ("denies alcohol") | 0% | 100% | 94% | 100% |
| Past use ("quit in 2015") | 0% | 100% | 100% | **53%** |
| Family history ("father drank") | 0% | 100% | 100% | **20%** |
| Heavy / binge drinking | 100% | 57–58% | 58–86% | 100% |
| Indirect signs ("smelled of EtOH") | 100% | 67% | 33% | 100% |

**LLM evidence check:** the LLM quoted evidence for 131 notes; 130 of the quotes were found word-for-word in the note.

**Key lessons**

- **Keyword search** finds every true case, but more than half of its flags are false alarms. It cannot tell "denies alcohol" from "drinks daily."
- **Rules** never raised a false alarm, but missed 16 of 52 cases because the test notes used wording they were not written for ("nightly", "heavy episodic drinking", "morning shakes", "relapsed").
- **TF-IDF + logistic regression** struggled with new wording too, especially indirect signs (33%). It also risks learning shortcuts: in this data, heavy drinkers miss more ART doses, so the model can partly rely on adherence words instead of alcohol words.
- **The LLM** had the best recall (0.98) and F1 (0.86), and handled new wording and indirect signs well with no training. But the small 3B model often ignored the instructions about **past use** and **family history**, which caused 15 of its false alarms.
- **The rules and the LLM fail in opposite places**, so a **hybrid** should do better than either: rules to screen out past use and family history, and the LLM to find current use. A larger model (e.g. Llama 3.1 8B) is another option to test.
- **The evaluation design matters as much as the model.** The test notes use different wording from the training notes. Testing on the same wording would have made every method look near-perfect.


## Natural Language → SQL

Once the flags are stored, a clinician can ask:

```
Question> Compare the percentage of patients with a detectable viral load between flagged and non-flagged patients.

SQL written by the LLM:
  SELECT a.flagged, 100.0 * SUM(CASE WHEN p.viral_load >= 50 THEN 1 ELSE 0 END) / COUNT(*) ...
```

These features keep the generated SQL reliable and safe:

- **Grounding:** the prompt contains the real schema, column meanings (e.g. *viral_load ≥ 50 = detectable*), sample rows and few-shot examples, so the model does not invent tables or columns.
- **Self-correction:** if a query fails, the error is sent back to the model once to fix.
- **Safety:** only single `SELECT` statements are allowed, and the database is opened **read-only**.
- **Evaluation:** 15 test questions with hand-written gold SQL, scored by **execution accuracy**. The results are compared, not the SQL text.

**NL-to-SQL results** (15 test questions, Llama 3.2 3B): the first version scored **8/15**. Most errors came from the model forgetting to JOIN `patients` with `alcohol_flags`, or misreading the date format. After adding a pre-joined view (`patient_overview`) and clearer column notes, accuracy rose to **12/15**. The remaining errors come from the small model sometimes ignoring the instruction to use the view. Details: [`results/NL_TO_SQL_RESULTS.md`](results/NL_TO_SQL_RESULTS.md).

## Privacy

All data is synthetic. The LLM runs **locally**, so no text is sent to an external API. With real clinical notes, the same design would run inside an institution-approved, HIPAA-compliant environment with IRB approval.

## How to run

Requirements: Python 3.9+ and [Ollama](https://ollama.com) with `ollama pull llama3.2:3b`.

```bash
pip install -r requirements.txt
python step1_generate_data.py      # synthetic patients and notes
python step2_classify_notes.py     # run the 4 NLP methods (LLM step takes ~5-20 min)
python step3_evaluate.py           # precision / recall / F1 + error analysis
python step4_build_database.py     # build SQLite database
python step5_ask_questions.py      # ask questions in plain English
python step6_test_nl_to_sql.py     # NL-to-SQL accuracy test
```

To use a different model, change `OLLAMA_MODEL` in `config.py`.

## Project structure

```
config.py                 settings (model name, paths)
step1_generate_data.py    synthetic data with gold labels
step2_classify_notes.py   runs all four NLP methods
step3_evaluate.py         metrics + error analysis
step4_build_database.py   SQLite database with NLP results
step5_ask_questions.py    interactive NL-to-SQL
step6_test_nl_to_sql.py   NL-to-SQL evaluation
src/classifiers.py        keyword, rules, ML and LLM classifiers
src/nl_to_sql.py          schema grounding, safety checks, self-correction
src/ollama_client.py      small client for the local LLM
```

## Limitations and next steps

- Synthetic notes are cleaner and shorter than real notes. Real notes have copy-forward text, templates and typos.
- Next steps:
  - validate on real, de-identified notes with a gold standard from clinician chart review (with inter-rater agreement)
  - try clinical language models such as ClinicalBERT
  - separate current, past and uncertain use into three labels
  - apply the same approach to other conditions, such as injection drug use

---
Author: Slesha Bucchireddy Gari
