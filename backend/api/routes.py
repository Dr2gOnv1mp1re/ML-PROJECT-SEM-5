"""
routes.py
─────────
FastAPI API route handlers for image upload, conversational Q&A, and sample loading.
"""

import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from backend.api.schemas import QuestionRequest, QuestionResponse, UploadResponse
from backend.core.pipeline import VisionChatbotPipeline

router = APIRouter(prefix="/api")

# Singleton pipeline instance initialized at startup
pipeline: Optional[VisionChatbotPipeline] = None

# In-memory session store: session_id -> { "detection": ..., "scene": ..., ... }
SESSIONS: Dict[str, Dict[str, Any]] = {}


def get_pipeline() -> VisionChatbotPipeline:
    global pipeline
    if pipeline is None:
        pipeline = VisionChatbotPipeline()
    return pipeline


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.18),
    session_id: Optional[str] = Form(None)
):
    """
    Accepts an uploaded image file, processes all elements and scene context,
    and initializes an interactive conversational session.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    pipe = get_pipeline()
    contents = await file.read()

    try:
        result = pipe.analyze_image(contents, conf_override=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

    sid = session_id if session_id else str(uuid.uuid4())
    SESSIONS[sid] = {
        "detection": result["detection"],
        "scene": result["scene"],
        "annotated_image": result["annotated_image_b64"],
        "original_image": result["original_image_b64"]
    }

    return UploadResponse(
        session_id=sid,
        detection=result["detection"],
        scene=result["scene"],
        annotated_image=result["annotated_image_b64"],
        original_image=result["original_image_b64"],
        initial_message=result["initial_message"],
        suggested_questions=result["suggested_questions"]
    )


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    """
    Answers a natural-language question about the currently analyzed image.
    """
    sid = req.session_id or "default"
    session_data = SESSIONS.get(sid)
    if not session_data:
        # Check if any session exists
        if SESSIONS:
            # Fall back to latest session
            latest_sid = list(SESSIONS.keys())[-1]
            session_data = SESSIONS[latest_sid]
        else:
            raise HTTPException(status_code=400, detail="No image has been uploaded yet. Please upload an image first!")

    pipe = get_pipeline()
    ans_result = pipe.answer_question(
        question=req.question,
        detection_data=session_data["detection"],
        scene_data=session_data["scene"]
    )

    return QuestionResponse(
        question=req.question,
        answer=ans_result["answer"],
        highlight_id=ans_result.get("highlight_id"),
        question_type=ans_result["question_info"].get("question_type", "other")
    )


@router.get("/samples")
async def list_samples():
    """Lists available sample test images."""
    samples = []
    sample_dir = Path("notebooks")
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in sample_dir.glob(ext):
            if "detected" not in p.name:
                samples.append({
                    "name": p.name,
                    "path": str(p),
                    "size_kb": round(p.stat().st_size / 1024, 1)
                })
    return {"samples": samples}


@router.post("/sample/{sample_name}", response_model=UploadResponse)
async def load_sample(sample_name: str, confidence: float = 0.18):
    """Loads and processes a sample image directly."""
    sample_path = Path("notebooks") / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")

    pipe = get_pipeline()
    try:
        result = pipe.analyze_image(sample_path, conf_override=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample processing failed: {str(e)}")

    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "detection": result["detection"],
        "scene": result["scene"],
        "annotated_image": result["annotated_image_b64"],
        "original_image": result["original_image_b64"]
    }

    return UploadResponse(
        session_id=sid,
        detection=result["detection"],
        scene=result["scene"],
        annotated_image=result["annotated_image_b64"],
        original_image=result["original_image_b64"],
        initial_message=result["initial_message"],
        suggested_questions=result["suggested_questions"]
    )


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "YOLOv8 + MobileNetV3 + Layer 3 NLP Classifier",
        "vocabulary_classes": 80,
        "capabilities": [
            "Multi-Element Recognition",
            "Prominence & Depth Stratification",
            "Spatial Zone Mapping",
            "Dominant Color Extraction",
            "Scene Context & Atmosphere",
            "Conversational Natural Language Q&A"
        ]
    }
