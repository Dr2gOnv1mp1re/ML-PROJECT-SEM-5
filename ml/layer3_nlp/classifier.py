"""
classifier.py
─────────────
Layer 3: Question Understanding / Intent Classification

Classifies a user's natural-language question into:
  - question_type  : e.g. "list_all", "how_many", "what_color", "where", "yes_no", "inspect_element", "scene_context", "what_is", "other"
  - answer_type    : e.g. "list", "number", "color", "location", "yes/no", "element_detail", "scene", "object", "other"
  - focus_words    : key nouns / objects being asked about (e.g. ["bus", "person", "clock"])
  - target_id      : extracted object index if user mentions "#1", "object 2", etc.
"""

import re
from typing import Dict, List, Any, Optional

RULES = [
    # Full scene / Element recognition requests
    (r"^(what('s| is) (in|inside|on) this image|what do you see|list all|describe (the )?image|tell me what|what is being identified|what is identified|recognize (everything|each and every|all)|identify (all|everything)|show all|what is here)\b", "list_all", "list"),
    (r"(each and every|every single|all elements|all objects|all items)\b", "list_all", "list"),
    
    # Specific object inspection (e.g. "tell me about #2", "describe object 1")
    (r"(object|element|item)\s*(#|\bnumber\b)?\s*(\d+)", "inspect_element", "element_detail"),
    (r"#\s*(\d+)", "inspect_element", "element_detail"),

    # Apparel & Clothing inquiries (e.g., "what is person 1 wearing?")
    (r"(wearing|dressed in|clothes|clothing|attire|coat|jacket|shoes|boots|pants|trousers|shirt)\b", "what_wearing", "apparel"),

    # Balcony, building, floor, tree, tire inquiries
    (r"\b(balcony|balconies|railing|railings)\b", "balcony_inquiry", "element_detail"),
    (r"\b(building|facade|architecture)\b", "building_inquiry", "element_detail"),
    (r"\b(floor|ground|pavement|sidewalk|street|paver|pavers|road)\b", "floor_inquiry", "element_detail"),
    (r"\b(tree|trees|leaves|foliage|plant|plants)\b", "tree_inquiry", "element_detail"),
    (r"\b(tyre|tyres|tire|tires|wheel|wheels|hub)\b", "tire_inquiry", "element_detail"),

    # Scene, background, setting, environment
    (r"(scene|background|backdrop|setting|environment|location type|place|outdoors?|indoors?|lighting|where was this taken)\b", "scene_context", "scene"),

    # Count questions
    (r"^(how many|how much|count of|number of|total number)\b", "how_many", "number"),

    # Color questions
    (r"^what (color|colour)\b", "what_color", "color"),
    (r"(color|colour) (is|are|of)\b", "what_color", "color"),

    # Location / Where
    (r"^where\b", "where", "location"),
    (r"^(in what|on what|at what) (location|place|side|position|direction)\b", "where", "location"),
    (r"(where is|where are|position of)\b", "where", "location"),

    # Yes / No verification questions
    (r"^(is there|are there|do you see|can you find|does it have|has it got|was there)\b", "yes_no", "yes/no"),
    (r"^(is|are|was|were|do|does|did|can|could|has|have|had|will)\b", "yes_no", "yes/no"),

    # Who / People
    (r"^(who|whose)\b", "who", "person"),

    # What type / kind
    (r"^what (type|kind|sort|species|breed|brand|make|model)\b", "what_type", "object"),

    # What action / doing
    (r"^what (is|are).*(doing|happening|going on)\b", "what_action", "action"),

    # General What is
    (r"^(what is|what are|what's)\b", "what_is", "object"),
    (r"^what\b", "what_is", "object"),

    # Which
    (r"^which\b", "which", "object"),
]

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "this", "that", "these",
    "those", "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "be", "do", "does", "did", "has", "have", "had", "will", "would",
    "can", "could", "should", "what", "where", "who", "how", "why", "which",
    "color", "colour", "many", "much", "number", "type", "kind", "there", "image",
    "picture", "photo", "see", "show", "tell", "me", "about"
}


import os
import joblib
from pathlib import Path

class QuestionClassifier:
    """
    Classifies natural-language questions about an image using a hybrid approach:
    1. Scikit-learn TF-IDF + LogisticRegression trained on VQA v2 / Visual Genome
    2. Rule-based entity & slot extraction for object IDs, locations, and actions
    """
    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.vectorizer = None
        self.qt_clf = None
        self.at_clf = None
        self.is_ml_active = False

        # Attempt to load trained models from default or specified directory
        search_dirs = []
        if model_dir:
            search_dirs.append(Path(model_dir))
        search_dirs.extend([
            Path("models/nlp"),
            Path(__file__).parent.parent.parent / "models" / "nlp"
        ])

        for m_dir in search_dirs:
            vec_p = m_dir / "tfidf_vectorizer.pkl"
            qt_p = m_dir / "qt_classifier.pkl"
            at_p = m_dir / "at_classifier.pkl"
            if vec_p.exists() and qt_p.exists() and at_p.exists():
                try:
                    self.vectorizer = joblib.load(vec_p)
                    self.qt_clf = joblib.load(qt_p)
                    self.at_clf = joblib.load(at_p)
                    self.is_ml_active = True
                    print(f"[QuestionClassifier] Loaded trained NLP models from {m_dir.resolve()}")
                    break
                except Exception as e:
                    print(f"[QuestionClassifier] Warning: Failed to load pickled models from {m_dir}: {e}")

    def classify(self, question: str) -> Dict[str, Any]:
        q_raw = question.strip()
        q_lower = q_raw.lower()
        q_clean = re.sub(r"[?!.,;:]", "", q_lower)

        # 1. Baseline ML prediction if model is loaded
        ml_qt = None
        ml_at = None
        if self.is_ml_active and self.vectorizer is not None:
            try:
                features = self.vectorizer.transform([q_lower])
                ml_qt = str(self.qt_clf.predict(features)[0])
                ml_at = str(self.at_clf.predict(features)[0])
            except Exception:
                ml_qt, ml_at = None, None

        # 2. Rule-based intent detection
        rule_qt = "other"
        rule_at = "other"

        # Check target index like "#2" or "object 1"
        target_id = None
        match_id = re.search(r"(?:object|element|item)?\s*#?\s*(\d+)", q_lower)
        if match_id and ("object" in q_lower or "element" in q_lower or "#" in q_lower):
            try:
                target_id = int(match_id.group(1))
            except ValueError:
                target_id = None

        for pattern, qt, at in RULES:
            if re.search(pattern, q_clean):
                rule_qt = qt
                rule_at = at
                break

        # Extract person index (e.g. "person 1", "person #2", "first person")
        person_idx = None
        m_person = re.search(r"person\s*#?\s*(\d+)", q_lower)
        if m_person:
            try:
                person_idx = int(m_person.group(1))
            except ValueError:
                person_idx = 1
        elif "first person" in q_lower or "1st person" in q_lower:
            person_idx = 1
        elif "second person" in q_lower or "2nd person" in q_lower:
            person_idx = 2
        elif "third person" in q_lower or "3rd person" in q_lower:
            person_idx = 3
        elif "person" in q_lower:
            person_idx = 1

        # Priority resolution:
        # Give immediate priority to specific domain queries (apparel, structural elements, inspection, listing)
        domain_types = (
            "what_wearing", "balcony_inquiry", "building_inquiry",
            "floor_inquiry", "tree_inquiry", "tire_inquiry",
            "list_all", "inspect_element", "scene_context"
        )

        if target_id is not None and rule_qt in ("other", "what_is"):
            final_qt = "inspect_element"
            final_at = "element_detail"
        elif rule_qt in domain_types:
            final_qt = rule_qt
            final_at = rule_at
        elif ml_qt is not None and ml_qt != "other":
            final_qt = ml_qt
            final_at = ml_at if ml_at else rule_at
        else:
            final_qt = rule_qt
            final_at = rule_at

        focus = self._extract_focus(q_clean)

        return {
            "raw_question": q_raw,
            "question_type": final_qt,
            "answer_type": final_at,
            "target_id": target_id,
            "person_index": person_idx,
            "focus_words": focus,
            "ml_predicted": self.is_ml_active,
            "ml_question_type": ml_qt,
            "ml_answer_type": ml_at
        }

    @staticmethod
    def _extract_focus(q_lower: str) -> List[str]:
        tokens = re.findall(r"\b[a-z]+\b", q_lower)
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

