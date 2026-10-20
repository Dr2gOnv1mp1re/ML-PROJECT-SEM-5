"""
pipeline.py
───────────
Unified Pipeline coordinating all 4 layers:
  Layer 1: YOLOv8 Object Detection & Attribute Extraction
  Layer 2: MobileNetV3 Scene & Knowledge Analysis
  Layer 3: Question Understanding / Intent Classification
  Layer 4: Reasoning & Natural Language Response Generation
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, Union
import cv2
import numpy as np
from PIL import Image

from ml.layer1_detection.detector import ObjectDetector
from ml.layer2_knowledge.scene_analyzer import SceneAnalyzer
from ml.layer2_knowledge.knowledge_graph import VisualKnowledgeGraph
from ml.layer3_nlp.classifier import QuestionClassifier
from ml.layer4_reasoning.response_generator import ResponseGenerator


class VisionChatbotPipeline:
    def __init__(self, yolo_model: str = "yolov8n.pt"):
        print("[Pipeline] Initializing Conversational Vision Pipeline...")
        self.detector = ObjectDetector(model_path=yolo_model, conf_thresh=0.18)
        self.scene_analyzer = SceneAnalyzer()
        self.knowledge_graph = VisualKnowledgeGraph()
        self.classifier = QuestionClassifier()
        self.response_gen = ResponseGenerator()
        print("[Pipeline] All 4 layers initialized successfully!")

    def analyze_image(self, image_data: Union[bytes, str, Path], conf_override: float = None) -> Dict[str, Any]:
        """
        Processes an image from bytes or file path through Layers 1, 2, and 4.
        """
        # Convert to cv2 image and PIL image
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, (str, Path)):
            cv_img = cv2.imread(str(image_data))
            pil_img = Image.open(str(image_data)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image data format: {type(image_data)}")

        if cv_img is None:
            raise ValueError("Failed to decode image data.")

        # Layer 2: Scene Context (ImageNet-1K MobileNetV3)
        scene_result = self.scene_analyzer.analyze(pil_img)

        # Layer 1: Object Detection & Dynamic Structural Perception
        detection_result = self.detector.detect(cv_img, scene_context=scene_result, conf_override=conf_override)

        # Layer 2: Visual Knowledge Graph (Visual Genome Attributes & Relationships)
        dims = detection_result["image_dimensions"]
        vg_graph = self.knowledge_graph.build_graph(detection_result["detections"], dims["width"], dims["height"])
        scene_result["visual_knowledge"] = vg_graph


        # Encode annotated image to base64 JPEG
        annotated_bgr = detection_result["annotated_image"]
        _, buffer_ann = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_ann).decode("utf-8")

        # Encode original image to base64 JPEG for browser reference
        _, buffer_orig = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        original_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_orig).decode("utf-8")

        # Layer 4: Initial Exhaustive Multi-Tier Report
        initial_report = self.response_gen.generate_initial_summary(detection_result, scene_result)

        # Strip unneeded large numpy array from return dict
        serializable_detection = {
            "image_dimensions": detection_result["image_dimensions"],
            "num_objects": detection_result["num_objects"],
            "detections": detection_result["detections"],
            "labels_count": detection_result["labels_count"],
            "categories": detection_result["categories"]
        }

        return {
            "detection": serializable_detection,
            "scene": scene_result,
            "annotated_image_b64": annotated_b64,
            "original_image_b64": original_b64,
            "initial_message": initial_report["text"],
            "suggested_questions": initial_report.get("suggested_questions", [])
        }

    def answer_question(self, question: str, detection_data: Dict[str, Any], scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a user question about the active image through Layers 3 and 4.
        """
        # Layer 3: NLP classification & entity extraction
        q_info = self.classifier.classify(question)

        # Layer 4: Intelligent response synthesis
        ans = self.response_gen.answer_question(q_info, detection_data, scene_data)

        return {
            "question_info": q_info,
            "answer": ans["text"],
            "highlight_id": ans.get("highlight_id")
        }
