"""
Four ways to decide if a note shows CURRENT unhealthy alcohol use.
They go from simplest to smartest:

  1. keyword  - flag the note if any alcohol word appears
  2. rules    - look at each sentence and handle negation ("denies"),
                past use ("quit"), family history ("father") and amount
  3. ml       - TF-IDF + logistic regression trained on labelled notes
  4. llm      - a local LLM (via Ollama) reads the note, answers yes/no,
                and quotes the sentence it used as evidence

Every method returns the same thing:
  {"label": 1 or 0, "evidence": "text that led to the decision"}
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ollama_client import chat_json

ALCOHOL_TERMS = re.compile(
    r"\b(alcohol\w*|etoh|drink\w*|drunk|beers?|wine|vodka|whiskey|liquor|"
    r"shots?|intoxicated|audit-c|bal|sober)\b", re.IGNORECASE)


def split_sentences(text):
    """Split a note into sentences (on new lines, '. ' and ';')."""
    parts = re.split(r"\n|(?<=[.;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# 1. KEYWORD SEARCH
# ---------------------------------------------------------------------------
def keyword_classify(note):
    for sentence in split_sentences(note):
        if ALCOHOL_TERMS.search(sentence):
            return {"label": 1, "evidence": sentence}
    return {"label": 0, "evidence": ""}


# ---------------------------------------------------------------------------
# 2. RULES (a simple version of the classic "NegEx" idea)
#    These rules were written by looking ONLY at the training notes, the same
#    way you would write them before seeing new data. New wording in the test
#    notes ("ETOH: denies", "Hx", "relapsed", "shakes") can break them.
# ---------------------------------------------------------------------------
NEGATION_BEFORE = re.compile(r"\b(denies|denied|no|none|never|does not|doesn't|not)\b", re.I)
SCOPE_ENDERS = re.compile(r"\b(but|however|although|though)\b", re.I)
PAST_CUES = re.compile(r"\b(quit|stopped|former|formerly|remission|sober|history of|"
                       r"no longer)\b", re.I)
FAMILY_CUES = re.compile(r"\b(father|mother|brother|sister|uncle|aunt|family|"
                         r"grandfather|grandmother|parents?)\b", re.I)
HEAVY_CUES = re.compile(
    r"\b(daily|every day|a day|per day|every night|most days|pint|"
    r"binge\w*|shots|drunk|intoxicated|odor|slurred|bal|audit-c|"
    r"every weekend|most weekends|heavily)\b", re.I)
DRINKS_PER_WEEK = re.compile(r"(\d+)\s*\+?\s*(?:standard\s+)?drinks?\s*(?:per week|weekly|a week|/week)", re.I)


def _is_negated(sentence, term_match):
    before = sentence[:term_match.start()]
    cues = list(NEGATION_BEFORE.finditer(before))
    if not cues:
        return False
    # "Denies drug use BUT drinks 6 beers" -> the negation stops at "but"
    text_between = before[cues[-1].end():]
    return not SCOPE_ENDERS.search(text_between)


def rules_classify(note):
    for sentence in split_sentences(note):
        term = ALCOHOL_TERMS.search(sentence)
        if not term:
            continue
        if _is_negated(sentence, term):
            continue                                   # "Denies alcohol use."
        if PAST_CUES.search(sentence):
            continue                                   # "Quit drinking in 2015."
        if FAMILY_CUES.search(sentence):
            continue                                   # "Father had alcohol use disorder."
        per_week = DRINKS_PER_WEEK.search(sentence)
        if HEAVY_CUES.search(sentence) or (per_week and int(per_week.group(1)) >= 8):
            return {"label": 1, "evidence": sentence}
    return {"label": 0, "evidence": ""}


# ---------------------------------------------------------------------------
# 3. MACHINE LEARNING (TF-IDF + logistic regression)
# ---------------------------------------------------------------------------
class MLClassifier:
    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    def train(self, notes, labels):
        self.model.fit(notes, labels)
        return self

    def classify(self, note):
        label = int(self.model.predict([note])[0])
        # Evidence = the words in this note that pushed the model most toward "yes"
        tfidf = self.model.named_steps["tfidf"]
        clf = self.model.named_steps["clf"]
        vec = tfidf.transform([note])
        words = tfidf.get_feature_names_out()
        scores = [(vec[0, i] * clf.coef_[0][i], words[i]) for i in vec.nonzero()[1]]
        top = [w for s, w in sorted(scores, reverse=True)[:4] if s > 0]
        return {"label": label, "evidence": "top words: " + ", ".join(top)}


# ---------------------------------------------------------------------------
# 4. LLM (local model through Ollama)
# ---------------------------------------------------------------------------
LLM_SYSTEM_PROMPT = """You are a clinical NLP assistant. You read one clinical note and decide
whether it documents CURRENT unhealthy alcohol use BY THE PATIENT.

Answer "yes" if the note shows ANY of these:
- heavy drinking: several drinks every day, or more than 7 drinks/week (women) or 14 drinks/week (men)
- binge drinking: 4+ drinks (women) or 5+ drinks (men) on one occasion, or getting drunk
- signs of intoxication or alcohol harm: smells of alcohol, intoxicated, withdrawal shakes, alcohol-related ED visit
- a positive alcohol screen (for example a positive AUDIT-C)
- a relapse back to drinking

Answer "no" if:
- the patient denies alcohol use
- drinking is only in the PAST (quit, sober, in remission)
- only a FAMILY MEMBER drinks
- the patient drinks only lightly or occasionally
- alcohol is not mentioned

Reply with JSON only, in exactly this format:
{"unhealthy_alcohol_use": "yes" or "no",
 "evidence": "the exact sentence copied word-for-word from the note that supports your answer, or an empty string if alcohol is not mentioned",
 "reason": "one short sentence explaining your answer"}

Examples:
Note text: "Denies alcohol use. Smokes 1 ppd."
{"unhealthy_alcohol_use": "no", "evidence": "Denies alcohol use.", "reason": "Patient denies drinking."}

Note text: "Father had alcohol use disorder. Pt drinks 8 beers daily."
{"unhealthy_alcohol_use": "yes", "evidence": "Pt drinks 8 beers daily.", "reason": "Patient drinks heavily every day; the father's history is not the patient."}

Note text: "Quit drinking in 2015, sober since."
{"unhealthy_alcohol_use": "no", "evidence": "Quit drinking in 2015, sober since.", "reason": "Drinking was in the past."}
"""


def _normalise(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def llm_classify(note):
    reply = chat_json(LLM_SYSTEM_PROMPT, "Note text:\n" + note)
    answer = str(reply.get("unhealthy_alcohol_use", "")).strip().lower()
    evidence = str(reply.get("evidence", "") or "")
    return {
        "label": 1 if answer.startswith("y") else 0,
        "evidence": evidence,
        "reason": str(reply.get("reason", "")),
        # Hallucination check: is the quoted evidence really in the note?
        "evidence_in_note": bool(evidence) and _normalise(evidence) in _normalise(note),
        "parsed_ok": answer in ("yes", "no"),
    }
