"""
inference.py — Risk Scoring Service
====================================
Orchestrates the full NLP pipeline for a single news headline and produces
a structured AnalysisResult that includes:

  * Raw FinBERT sentiment (label + per-class scores)
  * All spaCy entities and the location subset
  * A deterministic Risk Score on a 1-10 integer scale

Risk Score algorithm
--------------------
The score is built from weighted signals so analysts can trace exactly why
a particular headline received a given score.  Starting from a neutral
baseline of 5 the engine applies the following adjustments:

  Sentiment adjustment
  --------------------
  negative  -> +3   (most significant single signal)
  neutral   ->  0   (no change)
  positive  -> -2   (risk reduction, capped at floor of 1)

  Location multiplier
  -------------------
  Each distinct location entity adds +0.5, capped at +2.0 total.
  Rationale: a headline mentioning "Suez Canal, Egypt, Red Sea" is more
  geographically specific -- and therefore more actionable -- than one
  without any location context.

  Confidence weight
  -----------------
  The FinBERT confidence score (0-1) scales the sentiment adjustment.
  A barely-negative headline (confidence 0.55) scores less harshly than a
  definitively-negative one (confidence 0.97).

  Formula
  -------
  score = BASE_SCORE
        + (sentiment_delta * confidence)
        + location_bonus

  where sentiment_delta is +3, 0, or -2 and location_bonus in [0.0, 2.0].
  The result is clamped to the integer range [1, 10].

All floats are rounded only at the final clamping step so intermediate
precision does not compound rounding errors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

from app.models.nlp import EntityResult, ModelRegistry, SentimentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk-score constants  (all tunable without logic changes)
# ---------------------------------------------------------------------------

BASE_SCORE: float = 5.0

# Sentiment deltas applied *before* confidence weighting
_SENTIMENT_DELTA: Dict[str, float] = {
    "negative": +3.0,
    "neutral":   0.0,
    "positive": -2.0,
}

# Each unique location entity adds this much to the raw score
_LOCATION_BONUS_PER_ENTITY: float = 0.5
_LOCATION_BONUS_CAP: float = 2.0

SCORE_MIN: int = 1
SCORE_MAX: int = 10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """
    Complete intelligence assessment for a single news headline.

    Attributes
    ----------
    headline:
        The original input text, stored verbatim for audit traceability.
    sentiment:
        Label and confidence scores from FinBERT.
    entities:
        Every named entity detected by spaCy.
    locations:
        Subset of entities whose label is GPE, LOC, or FAC.
    risk_score:
        Integer 1-10 composite risk score (10 = maximum risk).
    score_breakdown:
        Intermediate scoring components for explainability / debugging.
    processing_time_ms:
        Wall-clock time for the full pipeline in milliseconds.
    """

    headline: str
    sentiment: SentimentResult
    entities: List[EntityResult]
    locations: List[EntityResult]
    risk_score: int
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class InferenceService:
    """
    Stateless service that runs the full NLP pipeline and produces an
    AnalysisResult for each submitted headline.

    The class holds a reference to the shared ModelRegistry but owns no
    mutable state of its own, making it safe to use as a FastAPI
    dependency (single instance injected at request time).
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        """
        Parameters
        ----------
        registry:
            Optional ModelRegistry override (useful for unit tests).
            Defaults to the process-wide singleton.
        """
        self._registry: ModelRegistry = registry or ModelRegistry.get()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def analyze(self, headline: str) -> AnalysisResult:
        """
        Run the full NLP pipeline on *headline* and return an AnalysisResult.

        Parameters
        ----------
        headline:
            Raw news headline string (ideally <= 512 tokens for FinBERT).

        Returns
        -------
        AnalysisResult
            Complete analysis including risk score and score breakdown.

        Raises
        ------
        RuntimeError
            Propagated from ModelRegistry if models were never loaded.
        ValueError
            If *headline* is empty or contains only whitespace.
        """
        headline = headline.strip()
        if not headline:
            raise ValueError("headline must be a non-empty string.")

        t_start = time.perf_counter()

        sentiment: SentimentResult = self._registry.predict_sentiment(headline)
        entities: list[EntityResult] = self._registry.extract_entities(headline)
        locations: list[EntityResult] = self._registry.extract_locations(headline)

        risk_score, breakdown = self._calculate_risk_score(sentiment, locations)

        elapsed_ms = (time.perf_counter() - t_start) * 1_000

        logger.info(
            "Analyzed headline | sentiment=%s (%.3f) | locations=%d | risk=%d | %.1fms",
            sentiment.label,
            sentiment.score,
            len(locations),
            risk_score,
            elapsed_ms,
        )

        return AnalysisResult(
            headline=headline,
            sentiment=sentiment,
            entities=entities,
            locations=locations,
            risk_score=risk_score,
            score_breakdown=breakdown,
            processing_time_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Risk scoring logic
    # ------------------------------------------------------------------

    def _calculate_risk_score(
        self,
        sentiment: SentimentResult,
        locations: list[EntityResult],
    ) -> tuple[int, Dict[str, float]]:
        """
        Compute the composite risk score from NLP signals.

        Parameters
        ----------
        sentiment:
            FinBERT output for the headline.
        locations:
            Location-type entities extracted from the headline.

        Returns
        -------
        (score, breakdown)
            score    : integer in [SCORE_MIN, SCORE_MAX]
            breakdown: dict of intermediate values for explainability
        """
        # --- Step 1: Sentiment contribution (confidence-weighted) ------
        sentiment_delta = _SENTIMENT_DELTA.get(sentiment.label.lower(), 0.0)
        weighted_sentiment = sentiment_delta * sentiment.score

        # --- Step 2: Location bonus (capped) ---------------------------
        unique_locations = len({loc.text.lower() for loc in locations})
        location_bonus = min(
            unique_locations * _LOCATION_BONUS_PER_ENTITY,
            _LOCATION_BONUS_CAP,
        )

        # --- Step 3: Assemble and clamp --------------------------------
        raw_score = BASE_SCORE + weighted_sentiment + location_bonus
        final_score = int(round(max(SCORE_MIN, min(SCORE_MAX, raw_score))))

        breakdown: Dict[str, float] = {
            "base_score": BASE_SCORE,
            "sentiment_delta_raw": sentiment_delta,
            "sentiment_confidence": round(sentiment.score, 6),
            "weighted_sentiment_contribution": round(weighted_sentiment, 4),
            "unique_location_count": float(unique_locations),
            "location_bonus": round(location_bonus, 4),
            "raw_score_before_clamp": round(raw_score, 4),
            "final_risk_score": float(final_score),
        }

        return final_score, breakdown
