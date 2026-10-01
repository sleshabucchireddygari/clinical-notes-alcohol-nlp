"""
STEP 1 - Create fake (synthetic) patients and doctor notes.

No real patient data is used anywhere in this project.

What this makes:
  data/train_notes.csv   400 notes used only to TRAIN the machine-learning model
  data/clinic_notes.csv  150 notes for our fake HIV clinic (used to TEST every method)
  data/patients.csv      one row per clinic patient (age, sex, viral load, CD4, ...)

Every note has a hidden "gold label" (the correct answer), so we can measure
how accurate each NLP method is:
  label = 1  -> the note shows CURRENT unhealthy alcohol use
  label = 0  -> it does not

Each note also has a "category" (e.g. negated, past_use, family_history) so we
can see WHICH kinds of sentences each method gets wrong.

IMPORTANT: training notes and clinic notes use DIFFERENT wording for the same
idea. That way the ML model can't just memorise sentences - it has to
generalise, like it would have to with real notes.

Run:  python step1_generate_data.py
"""
import os
import random
from datetime import date, timedelta

import pandas as pd

from config import DATA_DIR, SEED

# ---------------------------------------------------------------------------
# Alcohol sentences. Each category has a TRAIN list and a TEST list of phrasings.
# {n}, {w}, {b}, {yr}, {y}, {bal}, {ac} are filled with random numbers.
# ---------------------------------------------------------------------------
ALCOHOL_PHRASES = {
    # ---------------- label 1: current unhealthy alcohol use ----------------
    "heavy_drinking": (1, {
        "train": [
            "Drinks {n} beers daily.",
            "Reports drinking about {n} glasses of wine every night.",
            "Pt states {pr} drinks a pint of vodka most days.",
            "Consumes approximately {w} drinks per week.",
            "Drinks {n} cans of beer every day after work.",
        ],
        "test": [
            "Admits to {n} beers nightly.",
            "ETOH: about a fifth of whiskey per day per pt.",
            "Pt endorses drinking {w} standard drinks weekly, mostly liquor.",
            "Says {pr} has been drinking a bottle of wine a night since a recent divorce.",
            "Reports {n} tall cans of malt liquor a day.",
        ],
    }),
    "binge_drinking": (1, {
        "train": [
            "Reports binge drinking most weekends, up to {b} drinks per occasion.",
            "Had {b} shots at a party Saturday; says this happens every weekend.",
            "Drinks heavily on weekends, around {b} drinks each Friday and Saturday.",
        ],
        "test": [
            "Endorses heavy episodic drinking, {b}+ drinks on Fri and Sat nights.",
            "Goes out drinking every weekend, at least a 12-pack per night.",
            "Pt reports getting drunk most weekends, typically {b} or more drinks.",
        ],
    }),
    "indirect_signs": (1, {
        "train": [
            "Strong odor of alcohol noted during exam.",
            "Presented to ED intoxicated last week, BAL {bal}.",
            "Slurred speech at visit; pt admits {pr} had been drinking this morning.",
        ],
        "test": [
            "Pt appeared intoxicated at today's appointment and smelled of EtOH.",
            "Recent ED visit for a fall while drunk, BAL {bal} mg/dL.",
            "Tremulous; reports morning shakes relieved by a drink.",
        ],
    }),
    "positive_screen": (1, {
        "train": [
            "AUDIT-C score {ac} today (positive screen).",
            "Alcohol screening positive, AUDIT-C {ac}.",
        ],
        "test": [
            "AUDIT-C = {ac}, screen positive for unhealthy alcohol use.",
            "Positive AUDIT-C ({ac}/12) at intake.",
        ],
    }),
    "tricky_positive": (1, {
        "train": [
            "Father with alcohol use disorder. Pt reports {pr} drinks {n} beers daily.",
            "Denies illicit drug use but drinks {w} drinks weekly.",
        ],
        "test": [
            "Denies IVDU but reports drinking {n} tall cans of beer every day.",
            "Mother was an alcoholic; pt reports {pr} now drinks about {w} drinks a week.",
            "Previously sober for 2 years but relapsed recently, now drinking {n} beers a day.",
        ],
    }),
    # ---------------- label 0: NOT current unhealthy use ----------------
    "negated": (0, {
        "train": [
            "Denies alcohol use.",
            "Does not drink alcohol.",
            "No alcohol use.",
        ],
        "test": [
            "ETOH: denies.",
            "Pt denies any EtOH, tobacco, or drug use.",
            "Never drinks alcohol, per patient.",
            "Alcohol: none.",
        ],
    }),
    "past_use": (0, {
        "train": [
            "Quit drinking in {yr}, sober since.",
            "History of alcohol use disorder, in remission for {y} years.",
        ],
        "test": [
            "Former heavy drinker, has not had a drink since {yr}.",
            "Hx EtOH use disorder, sustained remission, attends AA weekly.",
            "Stopped drinking {y} years ago after pancreatitis.",
        ],
    }),
    "family_history": (0, {
        "train": [
            "Father had alcohol use disorder.",
            "Brother died of alcoholic liver disease.",
        ],
        "test": [
            "FHx: mother with alcoholism.",
            "Family history significant for alcohol abuse in father and uncle.",
        ],
    }),
    "light_drinking": (0, {
        "train": [
            "Drinks 1-2 beers socially once a month.",
            "Occasional glass of wine with dinner.",
        ],
        "test": [
            "Social drinker, 1 glass of wine on weekends.",
            "Rare alcohol use, a beer at holidays.",
            "Drinks 1 beer a few times per month.",
        ],
    }),
    "no_mention": (0, {"train": [""], "test": [""]}),
}

# How often each category appears (roughly 35% of notes are positive)
CATEGORY_WEIGHTS = {
    "heavy_drinking": 10, "binge_drinking": 7, "indirect_signs": 6,
    "positive_screen": 5, "tricky_positive": 7,
    "negated": 20, "past_use": 10, "family_history": 8,
    "light_drinking": 12, "no_mention": 15,
}

REGIMENS = ["Biktarvy", "Biktarvy", "Biktarvy", "Triumeq", "Dovato",
            "Genvoya", "Symtuza", "Cabenuva (injectable)"]
SYMPTOMS = ["No new complaints.", "Reports mild fatigue.", "Some trouble sleeping.",
            "Occasional headaches.", "Feels well overall.", "Reports low mood lately."]
TOBACCO = ["Smokes 1/2 ppd.", "Never smoker.", "Former smoker, quit 2018.",
           "Smokes 1 ppd.", "Vapes daily.", "Denies tobacco."]
DRUGS = ["Denies illicit drug use.", "Occasional marijuana use.", "", "Denies IVDU.", ""]
PLANS = ["Continue current ART. Labs in 3 months.",
         "Adherence counseling provided. RTC 3 months.",
         "Continue ART. Refer to behavioral health for support.",
         "Refill ART. Discussed PrEP for partner. RTC 6 months.",
         "Repeat viral load in 4 weeks."]


def fill(phrase, rng, sex):
    return phrase.format(
        pr="he" if sex == "male" else "she",
        n=rng.randint(4, 12), w=rng.randint(15, 40), b=rng.randint(5, 12),
        yr=rng.randint(2012, 2022), y=rng.randint(2, 10),
        bal=rng.randint(120, 310), ac=rng.randint(5, 12),
    )


def make_patient_and_note(pid, split, rng):
    categories = list(CATEGORY_WEIGHTS)
    category = rng.choices(categories, weights=[CATEGORY_WEIGHTS[c] for c in categories])[0]
    label, phrases = ALCOHOL_PHRASES[category]
    age = rng.randint(22, 71)
    sex = rng.choice(["male", "male", "female"])
    alcohol_sentence = fill(rng.choice(phrases[split]), rng, sex)
    regimen = rng.choice(REGIMENS)
    # Fake but realistic pattern: patients who drink heavily miss more doses
    # and more often have a detectable viral load.
    missed = rng.choice([0, 0, 1, 2, 3, 4, 6]) if label else rng.choice([0, 0, 0, 0, 1, 1, 2, 3])
    detectable = rng.random() < (0.45 if label else 0.12)
    viral_load = rng.randint(200, 45000) if detectable else rng.choice([0, 0, 20, 30, 40])
    cd4 = rng.randint(90, 650) if detectable else rng.randint(300, 1200)
    visit = date(2026, 3, 1) + timedelta(days=rng.randint(0, 200))

    adherence = ("Reports taking ART every day without missed doses" if missed == 0
                 else f"Reports missing {missed} doses of ART last month")
    vl_text = "undetectable (<50 copies/mL)" if viral_load < 50 else f"{viral_load:,} copies/mL"
    social = " ".join(s for s in [rng.choice(TOBACCO), alcohol_sentence, rng.choice(DRUGS)] if s)

    note = (
        f"HIV CLINIC FOLLOW-UP NOTE\n"
        f"Date: {visit.isoformat()}\n"
        f"SUBJECTIVE: {age}-year-old {sex} with HIV on {regimen} here for routine follow-up. "
        f"{adherence}. {rng.choice(SYMPTOMS)}\n"
        f"SOCIAL HISTORY: {social}\n"
        f"OBJECTIVE: BP {rng.randint(108, 152)}/{rng.randint(66, 96)}. "
        f"Viral load {vl_text}. CD4 {cd4} cells/mm3.\n"
        f"ASSESSMENT/PLAN: {rng.choice(PLANS)}"
    )
    patient = {
        "patient_id": pid, "age": age, "sex": sex, "art_regimen": regimen,
        "missed_doses_last_month": missed, "viral_load": viral_load, "cd4_count": cd4,
        "last_visit_date": visit.isoformat(),
    }
    note_row = {
        "note_id": f"N{pid[1:]}", "patient_id": pid, "visit_date": visit.isoformat(),
        "note_text": note, "label": label, "category": category,
    }
    return patient, note_row


def main():
    rng = random.Random(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    train_rows = [make_patient_and_note(f"T{i:04d}", "train", rng)[1] for i in range(1, 401)]
    pd.DataFrame(train_rows).to_csv(os.path.join(DATA_DIR, "train_notes.csv"), index=False)

    patients, notes = [], []
    for i in range(1, 151):
        p, n = make_patient_and_note(f"P{i:04d}", "test", rng)
        patients.append(p)
        notes.append(n)
    pd.DataFrame(patients).to_csv(os.path.join(DATA_DIR, "patients.csv"), index=False)
    clinic = pd.DataFrame(notes)
    clinic.to_csv(os.path.join(DATA_DIR, "clinic_notes.csv"), index=False)

    print("Created data/train_notes.csv  (400 training notes)")
    print("Created data/clinic_notes.csv (150 clinic notes for testing)")
    print("Created data/patients.csv     (150 fake patients)")
    print(f"Clinic notes with unhealthy alcohol use: {clinic.label.sum()} of {len(clinic)}")
    print("\nExample note:\n" + "-" * 60)
    print(clinic.loc[clinic.label == 1, "note_text"].iloc[0])


if __name__ == "__main__":
    main()
