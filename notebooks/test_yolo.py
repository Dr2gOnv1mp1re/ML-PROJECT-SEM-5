"""
test_yolo.py
────────────
Quick Layer 1 test — downloads YOLOv8n weights, runs detection on
a sample image, prints detected objects + confidence scores.

Run:
    .\\venv\\Scripts\\python.exe notebooks\\test_yolo.py
"""

import urllib.request
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ── 1. Download a sample test image ──────────────────────────────────────────
IMG_URL  = "https://ultralytics.com/images/bus.jpg"
IMG_PATH = Path("notebooks/sample_bus.jpg")

if not IMG_PATH.exists():
    print("📥 Downloading sample image...")
    urllib.request.urlretrieve(IMG_URL, IMG_PATH)
    print(f"   Saved → {IMG_PATH}")
else:
    print(f"   Using existing image: {IMG_PATH}")

# ── 2. Load YOLOv8 nano model (auto-downloads ~6 MB weights) ─────────────────
print("\n🔧 Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")   # nano = fastest, good for testing

# ── 3. Run detection ──────────────────────────────────────────────────────────
print("\n🔍 Running object detection...")
results = model(str(IMG_PATH), conf=0.3)   # conf=0.3 → 30% confidence threshold

# ── 4. Print results ──────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  LAYER 1 — DETECTION RESULTS")
print("="*50)

result = results[0]
boxes  = result.boxes

if boxes is None or len(boxes) == 0:
    print("  No objects detected.")
else:
    print(f"  Image : {IMG_PATH}")
    print(f"  Found : {len(boxes)} object(s)\n")
    print(f"  {'#':<4} {'Class':<20} {'Confidence':>12} {'Bounding Box (x1,y1,x2,y2)'}")
    print(f"  {'-'*70}")
    for i, box in enumerate(boxes):
        cls_id  = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf    = float(box.conf[0])
        coords  = [round(float(v), 1) for v in box.xyxy[0]]
        print(f"  {i+1:<4} {cls_name:<20} {conf:>11.1%}  {coords}")

# ── 5. Save annotated image ───────────────────────────────────────────────────
OUT_PATH = Path("notebooks/sample_bus_detected.jpg")
annotated = result.plot()          # numpy array with boxes drawn
Image.fromarray(annotated).save(OUT_PATH)

print(f"\n  Annotated image saved → {OUT_PATH}")
print("="*50)
print("\n✅ Layer 1 (YOLO) is working correctly!")
