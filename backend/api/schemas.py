"""
schemas.py
──────────
Pydantic data schemas for API requests and responses.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class QuestionResponse(BaseModel):
    question: str
    answer: str
    highlight_id: Optional[int] = None
    question_type: str


class DetectionSummary(BaseModel):
    num_objects: int
    labels_count: Dict[str, int]
    categories: Dict[str, int]
    detections: List[Dict[str, Any]]
    image_dimensions: Dict[str, int]


class UploadResponse(BaseModel):
    session_id: str
    detection: DetectionSummary
    scene: Dict[str, Any]
    annotated_image: str
    original_image: str
    initial_message: str
    suggested_questions: List[str]
