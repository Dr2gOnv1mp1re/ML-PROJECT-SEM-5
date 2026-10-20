"""
train_classifier.py
───────────────────
Layer 3: Train the NLP question classifier on VQA v2 data.

Uses:
  - TF-IDF vectorizer to convert questions into feature vectors
  - Logistic Regression to classify:
      (a) question_type  — e.g. what_color, how_many, yes_no, what_is, where, who, ...
      (b) answer_type    — yes/no, number, other

Saves trained models to:
  models/nlp/tfidf_vectorizer.pkl
  models/nlp/qt_classifier.pkl
  models/nlp/at_classifier.pkl

Run (after VQA v2 is downloaded):
    .\\venv\\Scripts\\python.exe ml\\layer3_nlp\\train_classifier.py
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
VQA_DIR      = Path("data/raw/vqa_v2")
MODEL_DIR    = Path("models/nlp")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_Q_FILE = VQA_DIR / "v2_OpenEnded_mscoco_train2014_questions.json"
VAL_Q_FILE   = VQA_DIR / "v2_OpenEnded_mscoco_val2014_questions.json"
TRAIN_A_FILE = VQA_DIR / "v2_mscoco_train2014_annotations.json"
VAL_A_FILE   = VQA_DIR / "v2_mscoco_val2014_annotations.json"


# ── Question-type label function (mirrors rules in classifier.py) ─────────────
RULES = [
    (r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|should)\b", "yes_no"),
    (r"^how many\b",                                                                "how_many"),
    (r"^how much\b",                                                                "how_many"),
    (r"^what (color|colour)\b",                                                     "what_color"),
    (r"(color|colour) (is|are|of)\b",                                               "what_color"),
    (r"^where\b",                                                                   "where"),
    (r"^who\b|^whose\b",                                                            "who"),
    (r"^what (type|kind|sort|species|breed|brand|make|model|sport|activity|action|event|game|food|meal|dish)\b", "what_type"),
    (r"^what (is|are).*(doing|happening|going on|taking place)\b",                 "what_action"),
    (r"^(what time|when)\b",                                                        "when"),
    (r"^how (old|tall|long|far|wide|deep|big|large|small|fast|slow)\b",           "how_much"),
    (r"^how\b",                                                                     "how"),
    (r"^why\b",                                                                     "why"),
    (r"^what\b",                                                                    "what_is"),
    (r"^which\b",                                                                   "which"),
]


def rule_label(question: str) -> str:
    q = question.strip().lower()
    q = re.sub(r"[?!.,;]", "", q)
    for pattern, qt in RULES:
        if re.search(pattern, q):
            return qt
    return "other"


def preprocess(question: str) -> str:
    q = question.strip().lower()
    return re.sub(r"[^a-z0-9 ]", "", q)


# ── Load Data ─────────────────────────────────────────────────────────────────
def load_vqa_data() -> pd.DataFrame:
    files_exist = all(f.exists() for f in [TRAIN_Q_FILE, VAL_Q_FILE, TRAIN_A_FILE, VAL_A_FILE])
    if not files_exist:
        print("[ERROR] VQA v2 files not found. Please run data/scripts/download_vqa.py first.")
        sys.exit(1)

    print("Loading questions and annotations ...")
    records = []

    for q_file, a_file, split in [
        (TRAIN_Q_FILE, TRAIN_A_FILE, "train"),
        (VAL_Q_FILE,   VAL_A_FILE,   "val")
    ]:
        with open(q_file, encoding="utf-8") as f:
            questions = {q["question_id"]: q["question"] for q in json.load(f)["questions"]}

        with open(a_file, encoding="utf-8") as f:
            annotations = json.load(f)["annotations"]

        for ann in annotations:
            qid        = ann["question_id"]
            question   = questions.get(qid, "")
            answer_type = ann.get("answer_type", "other")   # "yes/no", "number", "other"
            qt_label   = rule_label(question)

            records.append({
                "question"     : preprocess(question),
                "question_type": qt_label,
                "answer_type"  : answer_type,
                "split"        : split
            })

    df = pd.DataFrame(records)
    print(f"  Total samples: {len(df):,}")
    print(f"  question_type distribution:\n{df['question_type'].value_counts()}\n")
    print(f"  answer_type distribution:\n{df['answer_type'].value_counts()}\n")
    return df


# ── Train ─────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    X = df["question"].values
    y_qt = df["question_type"].values
    y_at = df["answer_type"].values

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30_000,
        sublinear_tf=True,
        min_df=2
    )
    X_vec = vectorizer.fit_transform(X)

    # Split
    X_train, X_val, y_qt_train, y_qt_val, y_at_train, y_at_val = train_test_split(
        X_vec, y_qt, y_at, test_size=0.15, random_state=42, stratify=y_qt
    )

    # -- Question Type classifier --
    print("\nTraining question_type classifier ...")
    qt_clf = LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="saga",
        n_jobs=-1
    )
    qt_clf.fit(X_train, y_qt_train)
    qt_preds = qt_clf.predict(X_val)
    print(f"  question_type accuracy: {accuracy_score(y_qt_val, qt_preds):.4f}")
    print(classification_report(y_qt_val, qt_preds, zero_division=0))

    # -- Answer Type classifier --
    print("\nTraining answer_type classifier ...")
    at_clf = LogisticRegression(
        max_iter=500,
        C=5.0,
        solver="saga",
        n_jobs=-1
    )
    at_clf.fit(X_train, y_at_train)
    at_preds = at_clf.predict(X_val)
    print(f"  answer_type accuracy: {accuracy_score(y_at_val, at_preds):.4f}")
    print(classification_report(y_at_val, at_preds, zero_division=0))

    # -- Save models --
    print("\nSaving models ...")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(qt_clf,     MODEL_DIR / "qt_classifier.pkl")
    joblib.dump(at_clf,     MODEL_DIR / "at_classifier.pkl")

    print(f"  Saved to {MODEL_DIR.resolve()}")
    print("\n[SUCCESS] Training complete! Classifier now runs in ML mode.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_vqa_data()
    train(df)
