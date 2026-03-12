"""
main.py — FastAPI Application Entry Point
==========================================
Bootstraps the Maritime Intelligence Agent API service:

  * Configures structured JSON logging
  * Loads NLP models onto the GPU during startup via lifespan context
  * Mounts the v1 API router
  * Exposes /health and /ready liveness / readiness probes

Startup sequence
----------------
1. FastAPI lifespan() begins.
2. ModelRegistry.get().load_all() downloads / loads FinBERT + spaCy.
3. Service is marked ready; health checks start returning 200.
4. Uvicorn begins accepting inbound requests.
5. On shutdown, GPU cache is cleared and any held resources released.

Logging
-------
All logs are emitted as structured JSON (via python-json-logger) so they
can be ingested by Loki, Datadog, or any log aggregation platform without
additional parsing rules.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger

from app.api.routes import router
from app.models.nlp import ModelRegistry

# ---------------------------------------------------------------------------
# Logging setup  (must happen before any other import triggers a logger)
# ---------------------------------------------------------------------------

_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def _configure_logging() -> None:
    """
    Replace the root handler with a structured JSON formatter.

    Each log line is a single JSON object with fields:
    timestamp, level, name, message, and any extras passed as keyword args.
    """
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(_LOG_LEVEL)


_configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_TITLE = "Maritime Intelligence Agent — AI Service"
APP_DESCRIPTION = (
    "GPU-accelerated NLP service that analyses maritime news headlines "
    "for supply-chain risk using FinBERT sentiment analysis and spaCy NER."
)
APP_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Lifespan context manager (replaces deprecated on_event handlers)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application startup and shutdown side-effects.

    Startup
    -------
    * Log GPU availability and driver information.
    * Load all NLP models via ModelRegistry (blocks until complete).
    * Attach the registry to app.state for optional access in tests.

    Shutdown
    --------
    * Flush the CUDA memory cache so the GPU is clean for any sibling
      containers sharing the same device.
    """
    # ---- Startup -------------------------------------------------------
    startup_start = time.perf_counter()
    logger.info(
        "Starting %s v%s",
        APP_TITLE,
        APP_VERSION,
        extra={"pytorch_version": torch.__version__},
    )

    # Log GPU details before model loading so any driver issues are visible
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda
        logger.info(
            "GPU available",
            extra={
                "device": device_name,
                "vram_gb": round(vram_total, 2),
                "cuda_version": cuda_version,
                "cudnn_version": torch.backends.cudnn.version(),
            },
        )
    else:
        logger.warning(
            "CUDA not available -- running on CPU. "
            "Check nvidia-container-toolkit and docker-compose GPU passthrough.",
        )

    # Load NLP models (may take 30-90 s on first cold start)
    registry = ModelRegistry.get()
    registry.load_all()
    app.state.model_registry = registry

    startup_elapsed = (time.perf_counter() - startup_start) * 1_000
    logger.info(
        "Service ready",
        extra={
            "startup_time_ms": round(startup_elapsed, 1),
            "device": registry.device_label,
        },
    )

    yield  # Application runs here

    # ---- Shutdown -------------------------------------------------------
    logger.info("Shutting down %s", APP_TITLE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS -- tighten allowed origins for production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount versioned API routes
app.include_router(router)


# ---------------------------------------------------------------------------
# Liveness probe  (/health)
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    tags=["Observability"],
    summary="Liveness probe",
    description=(
        "Returns 200 once the HTTP server is up, regardless of whether "
        "models have finished loading.  Use /ready for a readiness gate."
    ),
)
async def health() -> JSONResponse:
    """
    GET /health

    Lightweight liveness check consumed by Docker HEALTHCHECK and
    load-balancer readiness polls.

    Returns
    -------
    JSON
        ``status``, ``gpu_active`` flag, device label, and current
        GPU memory statistics (when CUDA is available).
    """
    gpu_active = torch.cuda.is_available()

    payload: dict = {
        "status": "ok",
        "version": APP_VERSION,
        "gpu_active": gpu_active,
        "device": (
            torch.cuda.get_device_name(0) if gpu_active else "cpu"
        ),
    }

    if gpu_active:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        payload["gpu_memory"] = {
            "total_gb": round(total_mem / (1024 ** 3), 2),
            "free_gb": round(free_mem / (1024 ** 3), 2),
            "used_gb": round((total_mem - free_mem) / (1024 ** 3), 2),
            "utilisation_pct": round(((total_mem - free_mem) / total_mem) * 100, 1),
        }

    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Readiness probe  (/ready)
# ---------------------------------------------------------------------------


@app.get(
    "/ready",
    tags=["Observability"],
    summary="Readiness probe",
    description=(
        "Returns 200 only after all NLP models have been loaded into GPU "
        "memory and the service can accept /analyze requests."
    ),
)
async def ready() -> JSONResponse:
    """
    GET /ready

    Kubernetes-style readiness probe.  n8n workflows should poll this
    endpoint after container startup before sending the first /analyze
    request.

    Returns
    -------
    200 OK        — models loaded, service is ready.
    503 Unavailable — models are still loading.
    """
    registry: ModelRegistry = ModelRegistry.get()

    if not registry.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "detail": "NLP models are still initialising. Retry shortly.",
            },
        )

    return JSONResponse(
        content={
            "status": "ready",
            "models_loaded": True,
            "device": registry.device_label,
            "gpu_active": registry.gpu_active,
        }
    )


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("DEV_RELOAD", False)),
        log_level=_LOG_LEVEL.lower(),
        access_log=True,
    )
