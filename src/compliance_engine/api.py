from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .engine import ComplianceEngine

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

engine = ComplianceEngine(
    rules_path=DATA_DIR / "ais175_rules.json",
    clauses_path=DATA_DIR / "processed" / "clauses.json",
)

app = FastAPI(
    title="AIS-175 Compliance API",
    version="1.0.0",
    description="Upload an AIS-175 PDF, index its text, and run compliance queries.",
)


class HealthResponse(BaseModel):
    status: str
    chunks_loaded: int
    rules_loaded: int


class IngestResponse(BaseModel):
    message: str
    file: str
    chunks_created: int


class SourceItem(BaseModel):
    chunk_id: str
    page: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        examples=["What does AIS-175 mention about braking performance?"],
    )
    top_k: int = Field(default=3, ge=1, le=10, examples=[3])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {
        "status": "ok",
        "chunks_loaded": len(engine.chunks),
        "rules_loaded": len(engine.rules),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(..., description="PDF document to ingest")) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    target = UPLOADS_DIR / file.filename
    content = await file.read()
    target.write_bytes(content)

    chunk_count = engine.ingest(target)
    return {
        "message": "PDF ingested successfully",
        "file": str(target.relative_to(BASE_DIR)),
        "chunks_created": chunk_count,
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    result = engine.ask(payload.question, top_k=payload.top_k)
    return {
        "answer": result.answer,
        "sources": result.sources,
    }
