"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine
from .models import Base
from .routers import sessions, consent, submit, results, clinician, admin, mock_data, ml_router, predict

# Create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ASD Governance Screening API",
    description="Clinician-Augmented Agentic AI for Longitudinal Autism Screening Support",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(consent.router)
app.include_router(submit.router)
app.include_router(results.router)
app.include_router(clinician.router)
app.include_router(admin.router)
app.include_router(mock_data.router)
app.include_router(ml_router.router)
app.include_router(predict.router)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
