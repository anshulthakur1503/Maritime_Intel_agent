"""
routes.py — API Route Definitions
===================================
Defines the v1 API router mounted by main.py.

Endpoints
---------
POST /api/v1/analyze
    Accepts a news headline and returns a full intelligence assessment:
    sentiment, entities, detected locations, and a 1-10 Risk Score.

The router depends on a single shared InferenceService instance injected
via FastAPI's dependency-injection system, keeping request handlers thin
and business logic fully testable in isolation.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.services.inference import AnalysisResult, InferenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Intelligence Engine"])


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_inference_service() -> InferenceService:
    """
    FastAPI dependency that provides the shared InferenceService instance.

    lru_cache(maxsize=1) ensures only one InferenceService is constructed
    per process regardless of how many requests arrive concurrently.
    """
    return InferenceService()


InferenceServiceDep = Annotated[InferenceService, Depends(_get_inference_service)]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""

    headline: str = Field(
        ...,
        min_length=3,
        max_length=1_000,
        description="A news headline or short paragraph to analyse.",
        examples=["Port of Shanghai closed due to severe weather, disrupting Asia-Pacific supply chains"],
    )

    @field_validator("headline")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Normalise leading/trailing whitespace before validation."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("headline must not be blank.")
        return stripped


class EntitySchema(BaseModel):
    """Serialisable representation of a single named entity."""

    text: str = Field(..., description="Surface form of the entity as it appears in the text.")
    label: str = Field(..., description="spaCy entity label (e.g. GPE, ORG, LOC).")
    start: int = Field(..., description="Character offset start in the original headline.")
    end: int = Field(..., description="Character offset end in the original headline.")


class SentimentSchema(BaseModel):
    """Serialisable FinBERT sentiment output."""

    label: str = Field(..., description="Winning sentiment label: positive | neutral | negative.")
    score: float = Field(..., description="Softmax confidence for the winning label (0-1).")
    all_scores: Dict[str, float] = Field(
        ...,
        description="Confidence scores for every sentiment class.",
    )


class AnalyzeResponse(BaseModel):
    """
    Complete intelligence assessment returned by POST /analyze.

    The response is designed to be consumed directly by n8n workflow nodes
    or stored in PostgreSQL without further transformation.
    """

    headline: str = Field(..., description="The original headline (trimmed).")
    sentiment: SentimentSchema
    entities: List[EntitySchema] = Field(
        ...,
        description="All named entities detected in the headline.",
    )
    locations: List[EntitySchema] = Field(
        ...,
        description="Location-type entities only (GPE, LOC, FAC).",
    )
    risk_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Composite supply-chain risk score: 1 (minimal) to 10 (critical).",
    )
    score_breakdown: Dict[str, float] = Field(
        ...,
        description="Intermediate scoring components for explainability.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Wall-clock time for the complete NLP pipeline (milliseconds).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_response(result: AnalysisResult) -> AnalyzeResponse:
    """Map an AnalysisResult domain object to the API response schema."""
    return AnalyzeResponse(
        headline=result.headline,
        sentiment=SentimentSchema(
            label=result.sentiment.label,
            score=result.sentiment.score,
            all_scores=result.sentiment.all_scores,
        ),
        entities=[
            EntitySchema(
                text=e.text,
                label=e.label,
                start=e.start,
                end=e.end,
            )
            for e in result.entities
        ],
        locations=[
            EntitySchema(
                text=loc.text,
                label=loc.label,
                start=loc.start,
                end=loc.end,
            )
            for loc in result.locations
        ],
        risk_score=result.risk_score,
        score_breakdown=result.score_breakdown,
        processing_time_ms=result.processing_time_ms,
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a maritime news headline",
    description=(
        "Runs FinBERT sentiment analysis and spaCy NER on the supplied headline, "
        "then calculates a 1-10 supply-chain Risk Score.\n\n"
        "**Risk Score interpretation**\n"
        "| Range | Level | Suggested action |\n"
        "|-------|-------|-----------------|\n"
        "| 1-3 | Low | Monitor only |\n"
        "| 4-6 | Medium | Flag for review |\n"
        "| 7-8 | High | Trigger n8n alert workflow |\n"
        "| 9-10 | Critical | Immediate escalation |"
    ),
)
async def analyze_headline(
    request: AnalyzeRequest,
    service: InferenceServiceDep,
) -> AnalyzeResponse:
    """
    POST /api/v1/analyze

    Accept a news headline, run the full intelligence pipeline, and
    return the structured assessment.

    Parameters
    ----------
    request:
        JSON body containing the headline string.
    service:
        Injected InferenceService (singleton per process).

    Returns
    -------
    AnalyzeResponse
        Sentiment, entities, locations, risk score and score breakdown.

    Raises
    ------
    422 Unprocessable Entity
        If the request body fails Pydantic validation.
    503 Service Unavailable
        If the NLP models have not finished loading.
    500 Internal Server Error
        For any unexpected inference failure.
    """
    try:
        result: AnalysisResult = service.analyze(request.headline)
    except RuntimeError as exc:
        # Models not loaded yet -- service is still warming up
        logger.error("Models not ready: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLP models are still loading. Retry in a few seconds.",
        ) from exc
    except ValueError as exc:
        logger.warning("Invalid request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected inference failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analysis.",
        ) from exc

    return _build_response(result)
