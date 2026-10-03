"""
DiaTwin FastAPI Application

Main entry point for the backend API server.
Initializes the inference service and mounts all routes.
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.core.config import config as cfg
from backend.app.api.routes import router
from backend.app.services.inference import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, cleanup on shutdown."""
    print("=" * 60)
    print("  DiaTwin — Metabolic Digital Twin Engine")
    print("  Starting up...")
    print("=" * 60)

    inference_service.load()

    n_patients = len(inference_service.patient_profiles)
    print(f"  Loaded {n_patients} patient profiles")
    print(f"  Model device: {inference_service.device}")
    print(f"  API: http://localhost:{cfg.api.port}{cfg.api.api_prefix}")
    print("=" * 60)

    yield

    print("DiaTwin shutting down...")


app = FastAPI(
    title="DiaTwin API",
    description=(
        "Multimodal Metabolic Digital Twin for Real-Time Glycemic "
        "Trajectory Forecasting and Counterfactual Intervention Simulation"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router, prefix=cfg.api.api_prefix)


@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "name": "DiaTwin",
        "version": "0.1.0",
        "status": "operational",
        "patients_loaded": len(inference_service.patient_profiles),
        "model_loaded": inference_service.model is not None,
        "device": inference_service.device,
        "endpoints": {
            "docs": "/docs",
            "patients": f"{cfg.api.api_prefix}/patients",
            "predict": f"{cfg.api.api_prefix}/predict",
            "simulate": f"{cfg.api.api_prefix}/simulate",
            "alerts": f"{cfg.api.api_prefix}/alerts",
        },
        "disclaimer": (
            "DiaTwin is an investigational decision-support prototype. "
            "Treatment adjustments must be verified with venous/capillary "
            "blood testing and licensed medical judgment."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=True,
    )
