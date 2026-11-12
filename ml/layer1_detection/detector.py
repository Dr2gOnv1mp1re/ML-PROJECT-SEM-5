"""
detector.py
───────────
Layer 1: Multi-Element Object Detection & Attribute Extraction
Uses Ultralytics YOLOv8 pretrained on MS COCO (80 classes) to recognize
every element in the image—including main objects, secondary items,
accessories, background elements, and small details.
"""

from pathlib import Path
from typing import Dict, List, Any, Union, Tuple
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Category groupings for MS COCO classes
COCO_CATEGORIES = {
    # People & Apparel
    "person": "People", "tie": "Apparel & Accessories", "backpack": "Apparel & Accessories",
    "umbrella": "Apparel & Accessories", "handbag": "Apparel & Accessories", "suitcase": "Apparel & Accessories",
    
    # Vehicles & Transport
    "bicycle": "Vehicles", "car": "Vehicles", "motorcycle": "Vehicles", "airplane": "Vehicles",
    "bus": "Vehicles", "train": "Vehicles", "truck": "Vehicles", "boat": "Vehicles",
    
    # Outdoor & Urban Infrastructure
    "traffic light": "Street & Infrastructure", "fire hydrant": "Street & Infrastructure",
    "stop sign": "Street & Infrastructure", "parking meter": "Street & Infrastructure", "bench": "Street & Infrastructure",
    
    # Animals
    "bird": "Animals", "cat": "Animals", "dog": "Animals", "horse": "Animals",
    "sheep": "Animals", "cow": "Animals", "elephant": "Animals", "bear": "Animals",
    "zebra": "Animals", "giraffe": "Animals",
    
    # Sports & Recreation
    "frisbee": "Sports & Recreation", "skis": "Sports & Recreation", "snowboard": "Sports & Recreation",
    "sports ball": "Sports & Recreation", "kite": "Sports & Recreation", "baseball bat": "Sports & Recreation",
    "baseball glove": "Sports & Recreation", "skateboard": "Sports & Recreation", "surfboard": "Sports & Recreation",
    "tennis racket": "Sports & Recreation",
    
    # Food & Kitchen
    "bottle": "Food & Dining", "wine glass": "Food & Dining", "cup": "Food & Dining",
    "fork": "Food & Dining", "knife": "Food & Dining", "spoon": "Food & Dining", "bowl": "Food & Dining",
    "banana": "Food & Dining", "apple": "Food & Dining", "sandwich": "Food & Dining", "orange": "Food & Dining",
    "broccoli": "Food & Dining", "carrot": "Food & Dining", "hot dog": "Food & Dining", "pizza": "Food & Dining",
    "donut": "Food & Dining", "cake": "Food & Dining",
    
    # Furniture & Home
    "chair": "Furniture & Interior", "couch": "Furniture & Interior", "potted plant": "Furniture & Interior",
    "bed": "Furniture & Interior", "dining table": "Furniture & Interior", "toilet": "Furniture & Interior",
    
    # Electronics & Devices
    "tv": "Electronics", "laptop": "Electronics", "mouse": "Electronics", "remote": "Electronics",
    "keyboard": "Electronics", "cell phone": "Electronics", "microwave": "Electronics", "oven": "Electronics",
    "toaster": "Electronics", "sink": "Electronics & Appliances", "refrigerator": "Electronics & Appliances",
    
    # Household & Decor
    "book": "Household & Decor", "clock": "Household & Decor", "vase": "Household & Decor",
    "scissors": "Household & Decor", "teddy bear": "Household & Decor", "hair drier": "Household & Decor",
    "toothbrush": "Household & Decor"
}

COLOR_PALETTE = {
    "Black": (30, 30, 30),
    "White": (230, 230, 230),
    "Dark Gray": (75, 75, 75),
    "Light Gray": (170, 170, 170),
    "Red": (195, 30, 30),
    "Dark Red": (120, 10, 10),
    "Blue": (30, 80, 200),
    "Navy Blue": (15, 30, 95),
    "Light Blue": (110, 185, 235),
    "Green": (35, 140, 50),
    "Dark Green": (15, 70, 25),
    "Yellow": (230, 200, 30),
    "Orange": (235, 120, 25),
    "Purple": (130, 45, 155),
    "Pink": (230, 115, 170),
    "Brown": (115, 65, 30),
    "Beige": (205, 185, 145)
}


def extract_dominant_color(crop_img: np.ndarray) -> str:
    """Extract dominant human-readable color from an image crop."""
    if crop_img is None or crop_img.size == 0 or crop_img.shape[0] < 3 or crop_img.shape[1] < 3:
        return "Unknown"
    
    # Resize crop to speed up color calculation
    small = cv2.resize(crop_img, (30, 30), interpolation=cv2.INTER_AREA)
    # Convert BGR (cv2 default) to RGB
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    
    # Center crop slightly to avoid border noise
    h, w, _ = rgb.shape
    center_patch = rgb[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    if center_patch.size > 0:
        avg_color = np.median(center_patch.reshape(-1, 3), axis=0)
    else:
        avg_color = np.median(rgb.reshape(-1, 3), axis=0)
        
    r, g, b = avg_color[0], avg_color[1], avg_color[2]
    
    # Find closest named color by Euclidean distance in RGB
    min_dist = float("inf")
    best_color = "Unknown"
    for name, (cr, cg, cb) in COLOR_PALETTE.items():
        dist = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
        if dist < min_dist:
            min_dist = dist
            best_color = name
            
    return best_color


def get_spatial_position(bbox: List[float], img_w: int, img_h: int) -> Tuple[str, str, str]:
    """
    Computes horizontal, vertical, and combined spatial descriptor.
    :return: (horizontal, vertical, combined)
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    # Horizontal partition (3 main bands with 5-way granularity)
    if cx < img_w * 0.33:
        h_pos = "left"
    elif cx < img_w * 0.67:
        h_pos = "center"
    else:
        h_pos = "right"
        
    # Vertical partition
    if cy < img_h * 0.33:
        v_pos = "top"
    elif cy < img_h * 0.67:
        v_pos = "middle"
    else:
        v_pos = "bottom"
        
    if v_pos == "middle" and h_pos == "center":
        combined = "dead-center"
    elif v_pos == "middle":
        combined = f"{h_pos} side"
    elif h_pos == "center":
        combined = f"{v_pos}-center"
    else:
        combined = f"{v_pos}-{h_pos}"
        
    return h_pos, v_pos, combined


class ObjectDetector:
    """
    Enhanced YOLOv8 Object Detector that identifies every element in the image,
    including main subjects, secondary objects, small details, and accessories.
    """
    def __init__(self, model_path: Union[str, Path] = "yolov8n.pt", conf_thresh: float = 0.18):
        """
        :param model_path: Path to weights or model name ('yolov8n.pt')
        :param conf_thresh: Sensitivity threshold (default 0.18 for exhaustive detection)
        """
        self.model_path = str(model_path)
        self.conf_thresh = conf_thresh
        self.model = YOLO(self.model_path)

    def detect(self, image_source: Union[str, Path, np.ndarray, Image.Image], scene_context: Dict[str, Any] = None, conf_override: float = None) -> Dict[str, Any]:
        """
        Detect every element in the image with spatial, color, and prominence attributes.
        Dynamically adapts structural element extraction to image content and scene context.
        """
        conf = conf_override if conf_override is not None else self.conf_thresh
        
        # Load image for dimension checking and crop extraction
        if isinstance(image_source, (str, Path)):
            cv_img = cv2.imread(str(image_source))
        elif isinstance(image_source, Image.Image):
            cv_img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, np.ndarray):
            cv_img = image_source.copy()
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
            
        img_h, img_w = cv_img.shape[:2]
        total_area = float(img_w * img_h)

        # Run inference with Ultralytics YOLOv8 in lean inference mode
        with torch.inference_mode():
            results = self.model(cv_img, conf=conf, iou=0.45, verbose=False)
            first_result = results[0]

        detections: List[Dict[str, Any]] = []
        labels_count: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        if first_result.boxes is not None and len(first_result.boxes) > 0:
            for idx, box in enumerate(first_result.boxes):
                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id]
                score = float(box.conf[0].item())
                xyxy = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]
                x1, y1, x2, y2 = xyxy
                
                # Constrain within bounds
                ix1 = max(0, int(x1))
                iy1 = max(0, int(y1))
                ix2 = min(img_w, int(x2))
                iy2 = min(img_h, int(y2))
                
                # Area and relative size
                box_w = max(0, ix2 - ix1)
                box_h = max(0, iy2 - iy1)
                box_area = float(box_w * box_h)
                area_pct = round((box_area / total_area) * 100.0, 2)
                
                # Spatial positioning
                h_pos, v_pos, combined_pos = get_spatial_position(xyxy, img_w, img_h)
                
                # Depth / prominence classification
                if area_pct >= 14.0 or (area_pct >= 8.0 and v_pos in ("middle", "bottom")):
                    prominence = "Foreground / Prominent"
                elif area_pct >= 3.0:
                    prominence = "Midground"
                else:
                    prominence = "Background / Fine Detail"
                    
                # Dominant color for this object
                crop = cv_img[iy1:iy2, ix1:ix2]
                dom_color = extract_dominant_color(crop)
                
                # Category
                category = COCO_CATEGORIES.get(label, "Other Object")
                
                # Update counts
                labels_count[label] = labels_count.get(label, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
                
                detections.append({
                    "id": idx + 1,
                    "label": label,
                    "category": category,
                    "confidence": round(score, 4),
                    "confidence_pct": round(score * 100, 1),
                    "bbox": xyxy,
                    "normalized_bbox": [
                        round(x1 / img_w, 4),
                        round(y1 / img_h, 4),
                        round(x2 / img_w, 4),
                        round(y2 / img_h, 4)
                    ],
                    "area_pct": area_pct,
                    "prominence": prominence,
                    "dominant_color": dom_color,
                    "position": {
                        "horizontal": h_pos,
                        "vertical": v_pos,
                        "location_desc": combined_pos
                    }
                })

        # Sort detections by area descending so prominent ones appear first
        detections.sort(key=lambda d: d["area_pct"], reverse=True)
        # Re-index for clean display
        for i, d in enumerate(detections):
            d["rank"] = i + 1

        annotated_img = first_result.plot()

        return {
            "image_dimensions": {"width": img_w, "height": img_h},
            "num_objects": len(detections),
            "detections": detections,
            "labels_count": labels_count,
            "categories": category_counts,
            "annotated_image": annotated_img
        }
