# OmniVision AI — Conversational Multi-Element Image Recognition Chatbot

A conversational vision AI chatbot powered by pretrained Deep Learning models (**YOLOv8** + **MobileNetV3**) that recognizes **each and every element** in any uploaded image and directly reports each object with its **confidence score**.

---

## 🌟 Key Features

1. **Exhaustive Multi-Element Object Detection**:
   - Detects all elements across 80 common object classes (people, vehicles, animals, accessories, food, furniture, electronics, etc.).
   - Captures not just the main subject, but all secondary objects, accessories, and background elements.
   - Configurable sensitivity slider to reveal faint or distant items.
2. **Direct Confidence Scores**:
   - Clean, transparent output reporting the exact recognition confidence score for every detected item.
3. **Attribute & Spatial Extraction**:
   - **Dominant Color Identification**: Analyzes object bounding box crops to extract dominant colors.
   - **Prominence Classification**: Categorizes elements into *Foreground / Prominent*, *Midground*, or *Background & Fine Detail*.
   - **Spatial Grid Mapping**: Locates objects across horizontal and vertical coordinates (`dead-center`, `upper-left`, `bottom-right`, etc.).
4. **Conversational Visual QA**:
   - Interactive chatbot answering questions on counts, specific object details, highest/lowest confidence items, colors, and locations.
   - Clickable suggestion chips for rapid follow-ups.
5. **Interactive Visual Studio**:
   - Toggle between **Annotated View** (colored bounding boxes) and **Clean Photo**.
   - Interactive Element Catalog: Click any element card to highlight its bounding box on the image and query the chatbot.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Backend Server
```bash
py -3.14 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Open in Browser
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 💬 Example Chatbot Interaction

### Initial Detection Output upon Image Upload:
```markdown
### 🎯 Object Detection Results
Detected 6 object(s) in the image:

- #1 Bus: 87.3% confidence
- #2 Person: 86.6% confidence
- #3 Person: 85.3% confidence
- #4 Person: 82.5% confidence
- #5 Person: 26.1% confidence
- #6 Stop Sign: 25.5% confidence
```

### Example Follow-Up Questions:
- **"How many people are there?"**  
  > There are 4 persons detected:  
  > - #2 (86.6%), #3 (85.3%), #4 (82.5%), #5 (26.1%)

- **"Which object has the highest confidence?"**  
  > The object with the highest confidence is **#1 Bus** with **87.3%** confidence.

- **"Tell me about object #1"**  
  > **Object #1: Bus**  
  > - Confidence Score: 87.3%  
  > - Bounding Box Coordinates: `[x1=22.9, y1=231.3, x2=805.0, y2=756.8]`  
  > - Position: dead-center  

---

## 📁 Project Architecture

```
chatbot/
├── backend/
│   ├── main.py                     # FastAPI application entry point
│   ├── api/
│   │   ├── routes.py               # /api/upload, /api/ask, /api/samples
│   │   └── schemas.py              # Pydantic request/response schemas
│   └── core/
│       └── pipeline.py             # 4-Layer orchestration pipeline
├── frontend/
│   ├── templates/
│   │   └── index.html              # Dual-panel inspection canvas & chat UI
│   └── static/
│       ├── css/style.css           # Modern dark-mode glassmorphism design
│       └── js/app.js               # Interactive canvas & chat controller
└── ml/
    ├── layer1_detection/
    │   └── detector.py             # YOLOv8 multi-element detector + color extractor
    ├── layer2_knowledge/
    │   └── scene_analyzer.py       # MobileNetV3 scene & lighting analyzer
    ├── layer3_nlp/
    │   └── classifier.py           # Natural-language intent parser
    └── layer4_reasoning/
        └── response_generator.py   # Direct object detection & confidence score generator
```

---

## 🔌 API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the web interface |
| `/api/upload` | `POST` | Upload an image for full element detection |
| `/api/sample/{name}` | `POST` | Load a built-in test sample image |
| `/api/ask` | `POST` | Ask a natural language question about the active image |
| `/api/samples` | `GET` | List available sample images |
| `/api/health` | `GET` | Service status and capabilities |
