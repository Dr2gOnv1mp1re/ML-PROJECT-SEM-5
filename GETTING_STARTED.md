# 🚀 Getting Started — Conversational Image Recognition Chatbot

## What's Inside This ZIP

```
chatbot/
├── README.md                        ← Project overview
├── requirements.txt                 ← All Python dependencies
├── PROJECT_STRUCTURE.md             ← Full folder map
├── GETTING_STARTED.md               ← This file
├── .env.example                     ← Environment config template
│
├── ml/
│   ├── layer1_detection/
│   │   └── detector.py              ← Layer 1: YOLO Object Detector class
│   ├── layer3_nlp/
│   │   ├── classifier.py            ← Layer 3: NLP Question Classifier class
│   │   └── train_classifier.py      ← Training script (needs VQA v2 data)
│
├── models/
│   └── nlp/
│       ├── tfidf_vectorizer.pkl     ← Pre-trained TF-IDF vectorizer
│       ├── qt_classifier.pkl        ← Pre-trained question-type classifier (99.78% acc)
│       └── at_classifier.pkl        ← Pre-trained answer-type classifier (99.38% acc)
│
├── data/
│   └── scripts/
│       ├── download_coco.py         ← Downloads MS COCO val2017 (~1 GB)
│       ├── download_vqa.py          ← Downloads VQA v2 questions & annotations
│       └── download_vg.py           ← Downloads Visual Genome annotations
│
├── notebooks/
│   ├── test_yolo.py                 ← Quick YOLO detection test (Layer 1)
│   ├── test_classifier.py           ← Quick NLP classifier test (Layer 3)
│   └── sample_bus.jpg               ← Sample test image
│   └── sample_bus_detected.jpg      ← YOLO-annotated output image
│
└── backend/ frontend/ tests/        ← Scaffolded (to be implemented in Layer 4+)
```

---

## ⚙️ Setup (One-Time)

### Requirements
- Python 3.10+ (tested on 3.14)
- ~3 GB free disk space (for packages)
- Internet connection (for YOLO model weights on first run)

### Step 1 — Create virtual environment
```bash
py -m venv venv
```

### Step 2 — Activate it
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Unblock DLLs (Windows only — fixes security policy issue)
```powershell
Get-ChildItem -Path ".\venv" -Recurse -Include "*.pyd","*.dll" | Unblock-File
```

---

## 🧪 Testing the Project

### Test 1 — Layer 1: Object Detection (YOLO)
Detects objects in the sample image. Downloads YOLOv8 weights (~6 MB) automatically on first run.

```bash
py notebooks/test_yolo.py
```

**Expected output:**
```
Found: 4 object(s)
  bus       87.3%  [22.9, 231.3, 805.0, 756.8]
  person    86.6%  [48.6, 398.6, 245.3, 902.7]
  person    85.3%  [669.5, 392.2, 809.7, 877.0]
  person    82.5%  [221.5, 405.8, 345.0, 857.5]
```
Also saves `notebooks/sample_bus_detected.jpg` with bounding boxes drawn.

---

### Test 2 — Layer 3: NLP Question Classifier (ML Mode)
Classifies question type and answer type using pre-trained models.

```bash
py notebooks/test_classifier.py
```

**Expected output:**
```
[Classifier] Loaded ML models from models/nlp
Question                                   Type           Answer Type
What color is the car?                     what_color     other
How many people are in the image?          how_many       number
Is there a dog in the picture?             yes_no         yes/no
Where is the cat sitting?                  where          other
...
```

---

### Test 3 — Layer 1 Detector Module
Test the reusable detector class directly on any image:

```python
from ml.layer1_detection.detector import ObjectDetector
from pathlib import Path

detector = ObjectDetector(model_path="yolov8n.pt", conf_thresh=0.3)
result = detector.detect(Path("notebooks/sample_bus.jpg"))

print(f"Found {result['num_objects']} objects: {result['labels']}")
for d in result['detections']:
    print(f"  {d['label']} ({d['confidence']*100:.1f}%) @ {d['bbox']}")
```

---

### Test 4 — NLP Classifier on your own question
```python
from ml.layer3_nlp.classifier import QuestionClassifier
from pathlib import Path

clf = QuestionClassifier(model_dir=Path("models/nlp"))
result = clf.classify("What color is the bus?")
print(result)
# {'question_type': 'what_color', 'answer_type': 'other', 'focus_words': ['bus'], 'mode': 'ml'}
```

---

## 📥 Downloading Datasets (Optional — for training)

These are large and NOT included in the ZIP:

```bash
# MS COCO val2017 images + annotations (~1.1 GB)
py data/scripts/download_coco.py

# VQA v2 questions + annotations (~35 MB)
py data/scripts/download_vqa.py

# Visual Genome annotations (~500 MB)
py data/scripts/download_vg.py
```

---

## 📊 Model Accuracy (trained on VQA v2 — 658,111 samples)

| Model | Accuracy |
|---|---|
| Question Type Classifier | **99.78%** |
| Answer Type Classifier | **99.38%** |

---

## 🏗️ Layers Status

| Layer | Purpose | Status |
|---|---|---|
| Layer 1 | Object Detection (YOLOv8) | ✅ Complete |
| Layer 2 | Visual Knowledge (Visual Genome) | 🔧 Scaffolded |
| Layer 3 | NLP Question Classifier | ✅ Complete |
| Layer 4 | Reasoning & Response Generation | 🔧 Scaffolded |
| Backend | FastAPI server | 🔧 Scaffolded |
| Frontend | Chat UI | 🔧 Scaffolded |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | PyTorch + YOLOv8 (Ultralytics) |
| NLP | scikit-learn (TF-IDF + Logistic Regression) |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |
| Data | Pandas, NumPy, OpenCV |
