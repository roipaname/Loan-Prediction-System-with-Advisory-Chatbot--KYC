"""
backend/main.py
================
LAPAS API — FastAPI backend serving the Streamlit frontend.

Run:
    .venv/bin/python -m uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.deps import get_classifier
from backend.routers import advisory, applicants, comparisons

app = FastAPI(title="LAPAS API", version="1.0.0")


@app.on_event("startup")
def _warm_up_classifier() -> None:
    # Load the xgboost-backed classifier before any request can trigger the
    # vector-store import (which pulls in torch) — see backend/deps.py.
    get_classifier()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(applicants.router)
app.include_router(advisory.router)
app.include_router(comparisons.router)


@app.get("/health")
def health():
    return {"status": "ok"}
