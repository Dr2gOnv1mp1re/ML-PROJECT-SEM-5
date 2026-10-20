"""
main.py
───────
FastAPI application entry point for the Conversational Image Recognition Chatbot.
"""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes import router as api_router, get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm vision models so user requests respond quickly
    print("[Main] Warming up pretrained vision models...")
    get_pipeline()
    print("[Main] Pretrained models loaded and ready!")
    yield
    print("[Main] Server shutting down.")


app = FastAPI(
    title="Conversational Image Recognition Chatbot",
    description="Multi-tier object recognition and conversational visual QA",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Mount static files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
static_dir = frontend_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_index():
    index_file = frontend_dir / "templates" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Conversational Image Recognition Chatbot API is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
