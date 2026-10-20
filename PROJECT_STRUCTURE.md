conversational-image-recognition-chatbot/
│
├── README.md                        # Project overview & setup guide
├── requirements.txt                 # All Python dependencies
├── .env                             # Environment variables (API keys, paths)
├── .gitignore
│
├── data/                            # ── DATASETS ──────────────────────────
│   ├── raw/                         # Downloaded datasets (untouched)
│   │   ├── coco/                    # MS COCO images + annotations
│   │   ├── visual_genome/           # VG objects, attributes, relationships
│   │   └── vqa_v2/                  # VQA v2 questions + answers
│   ├── processed/                   # Cleaned, ready-to-use data
│   └── scripts/                     # Download & preprocessing scripts
│       ├── download_coco.py
│       ├── download_vg.py
│       ├── download_vqa.py
│       └── preprocess.py
│
├── ml/                              # ── MACHINE LEARNING ──────────────────
│   ├── layer1_detection/            # Layer 1: Object Detection (YOLO)
│   │   ├── detector.py              # Main detection class
│   │   └── train_yolo.py            # Fine-tune YOLO on COCO
│   ├── layer2_knowledge/            # Layer 2: Visual Knowledge (VG)
│   │   ├── knowledge_graph.py       # Build attribute/relationship graph
│   │   └── vg_parser.py             # Parse Visual Genome annotations
│   ├── layer3_nlp/                  # Layer 3: NLP / Question Understanding
│   │   ├── classifier.py            # TF-IDF + Logistic Regression
│   │   └── train_classifier.py      # Train on VQA v2
│   └── layer4_reasoning/            # Layer 4: Reasoning + Response Gen
│       ├── reasoner.py              # Combine visual + NLP info
│       └── response_generator.py    # Generate natural-language answer
│
├── models/                          # ── SAVED MODELS ──────────────────────
│   ├── yolo/                        # Trained YOLO weights (.pt files)
│   ├── nlp/                         # Trained classifier (joblib .pkl)
│   └── reasoning/                   # Reasoning/response model artifacts
│
├── backend/                         # ── FASTAPI BACKEND ───────────────────
│   ├── main.py                      # FastAPI app entry point
│   ├── api/
│   │   ├── routes.py                # API endpoints (/upload, /ask)
│   │   └── schemas.py               # Request/Response Pydantic models
│   └── core/
│       ├── pipeline.py              # Orchestrates all 4 layers
│       └── config.py                # App configuration / paths
│
├── frontend/                        # ── WEB UI ────────────────────────────
│   ├── templates/
│   │   └── index.html               # Main chatbot page
│   └── static/
│       ├── css/style.css
│       └── js/app.js                # Chat logic, image upload, API calls
│
├── notebooks/                       # ── EXPERIMENTS ───────────────────────
│   ├── 01_explore_coco.ipynb
│   ├── 02_explore_vg.ipynb
│   ├── 03_explore_vqa.ipynb
│   ├── 04_yolo_detection.ipynb
│   ├── 05_nlp_classifier.ipynb
│   └── 06_full_pipeline.ipynb
│
└── tests/                           # ── UNIT TESTS ────────────────────────
    ├── test_detector.py
    ├── test_classifier.py
    └── test_pipeline.py
