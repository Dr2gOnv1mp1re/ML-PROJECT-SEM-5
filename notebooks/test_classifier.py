import sys
sys.path.insert(0, '.')
from ml.layer3_nlp.classifier import QuestionClassifier
from pathlib import Path

clf = QuestionClassifier(model_dir=Path('models/nlp'))

test_questions = [
    "What color is the car?",
    "How many people are in the image?",
    "Is there a dog in the picture?",
    "Where is the cat sitting?",
    "Who is wearing a hat?",
    "What is the man doing?",
    "What type of vehicle is parked outside?",
    "Are the bananas ripe?",
    "How old is the building?",
    "What sport is being played?",
    "Is the sky blue?",
    "How many cars are parked?",
    "What is the woman holding?",
]

print("=" * 70)
print("  LAYER 3 - QUESTION CLASSIFIER RESULTS (ML mode)")
print("=" * 70)
print("  {:<42} {:<14} {}".format("Question", "Type", "Answer Type"))
print("  " + "-" * 68)
for q in test_questions:
    r = clf.classify(q)
    print("  {:<42} {:<14} {}".format(q, r["question_type"], r["answer_type"]))
print("=" * 70)
