"""
nlp.py — NLP Model Loader
=========================
Manages lifecycle of all NLP models used by the Intelligence Engine:
  - ProsusAI/finbert  -> Financial sentiment analysis (GPU)
  - en_core_web_sm    -> Named-entity recognition / location extraction

Design decisions
----------------
* Singleton pattern via module-level ModelRegistry so models are loaded
  once at startup and reused across every request.
* Both models are pinned to cuda:0 when CUDA is available; the service
  degrades gracefully to CPU if the GPU is absent (logs a warning).
* @dataclass result types give callers clear, typed return values instead
  of raw dicts, making downstream code easier to test and refactor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import spacy
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINBERT_MODEL_ID: str = "ProsusAI/finbert"
SPACY_MODEL_ID: str = "en_core_web_sm"

# spaCy entity labels that represent geopolitical / location entities
LOCATION_ENTITY_LABELS = frozenset({"GPE", "LOC", "FAC"})


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SentimentResult:
    """Structured output from the FinBERT sentiment pipeline."""

    label: str                              # "positive" | "neutral" | "negative"
    score: float                            # Softmax confidence for `label`
    all_scores: Dict[str, float] = field(default_factory=dict)  # All label scores


@dataclass
class EntityResult:
    """A single named entity extracted by spaCy."""

    text: str
    label: str    # e.g. "GPE", "ORG", "LOC"
    start: int    # Character offset start in original text
    end: int      # Character offset end in original text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_device() -> tuple:
    """
    Determine the compute device available at runtime.

    Returns
    -------
    (device_int, device_label)
        device_int  : 0 for CUDA, -1 for CPU (HuggingFace pipeline convention)
        device_label: human-readable string used in logs and health responses
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(
            "GPU detected: %s (%.1f GB VRAM) -- models will run on cuda:0.",
            device_name,
            vram_gb,
        )
        return 0, f"cuda:0 ({device_name})"

    logger.warning(
        "CUDA not available. Models will fall back to CPU. "
        "Verify nvidia-container-toolkit and Docker GPU passthrough."
    )
    return -1, "cpu"


# ---------------------------------------------------------------------------
# Model Registry (singleton)
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Lazy-loading singleton that owns all NLP model instances.

    Usage
    -----
    Retrieve the shared instance via ModelRegistry.get().
    Models are loaded once on the first call to load_all() and remain
    resident for the lifetime of the process.
    """

    _instance: Optional["ModelRegistry"] = None

    def __init__(self) -> None:
        self._finbert_pipeline = None
        self._spacy_nlp = None
        self._device_int, self._device_label = _resolve_device()
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "ModelRegistry":
        """Return the process-wide ModelRegistry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """
        Load every model into memory (and onto the GPU when available).
        Safe to call multiple times -- subsequent calls are no-ops.
        """
        if self._loaded:
            logger.debug("Models already loaded; skipping.")
            return

        self._load_finbert()
        self._load_spacy()
        self._loaded = True
        logger.info("All NLP models loaded on device: %s.", self._device_label)

    @property
    def is_loaded(self) -> bool:
        """Return True once load_all() has completed successfully."""
        return self._loaded

    @property
    def device_label(self) -> str:
        """Human-readable device description, e.g. 'cuda:0 (RTX 3050)'."""
        return self._device_label

    @property
    def gpu_active(self) -> bool:
        """Return True when inference is running on a CUDA device."""
        return self._device_int == 0

    # ------------------------------------------------------------------
    # Inference -- Sentiment
    # ------------------------------------------------------------------

    def predict_sentiment(self, text: str) -> SentimentResult:
        """
        Run FinBERT sentiment classification on *text*.

        Parameters
        ----------
        text:
            Raw news headline or short paragraph (truncated to 512 tokens).

        Returns
        -------
        SentimentResult
            Winning label, its confidence, and scores for all three labels.

        Raises
        ------
        RuntimeError
            If models have not been loaded via load_all().
        """
        self._ensure_loaded()

        # top_k=None returns all label scores, not just the top-1
        raw: List[Dict] = self._finbert_pipeline(
            text,
            top_k=None,
            truncation=True,
            max_length=512,
        )

        # Build a score map: {"positive": 0.xx, "neutral": 0.xx, "negative": 0.xx}
        score_map: Dict[str, float] = {
            item["label"]: round(item["score"], 6) for item in raw
        }

        best_label: str = max(score_map, key=lambda k: score_map[k])

        return SentimentResult(
            label=best_label,
            score=score_map[best_label],
            all_scores=score_map,
        )

    # ------------------------------------------------------------------
    # Inference -- Named Entity Recognition
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> List[EntityResult]:
        """
        Run spaCy NER on *text* and return every detected entity.

        Parameters
        ----------
        text:
            Raw input string of arbitrary length.

        Returns
        -------
        list[EntityResult]
            Every entity token found, in document order.
        """
        self._ensure_loaded()
        doc = self._spacy_nlp(text)
        return [
            EntityResult(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        ]

    def extract_locations(self, text: str) -> List[EntityResult]:
        """
        Return only location-type entities (GPE, LOC, FAC).

        This is the primary method used by the risk-scoring engine since
        location specificity is the strongest signal for supply-chain impact.
        """
        return [
            entity
            for entity in self.extract_entities(text)
            if entity.label in LOCATION_ENTITY_LABELS
        ]

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_finbert(self) -> None:
        """Download (or load from cache) and warm-up the FinBERT pipeline."""
        logger.info("Loading FinBERT from '%s' ...", FINBERT_MODEL_ID)

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)

        # Move model weights to GPU *before* wrapping in a pipeline so that
        # the pipeline does not silently keep a CPU copy.
        if self._device_int == 0:
            model = model.to("cuda:0")

        self._finbert_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=self._device_int,   # 0 = cuda:0, -1 = cpu
        )
        logger.info("FinBERT loaded successfully.")

    def _load_spacy(self) -> None:
        """
        Load the spaCy model and, when available, transfer the tagger /
        parser tensors to the GPU via spaCy's prefer_gpu() API.
        """
        logger.info("Loading spaCy model '%s' ...", SPACY_MODEL_ID)

        if self._device_int == 0:
            # prefer_gpu() moves spaCy's internal Thinc ops to the GPU.
            activated = spacy.prefer_gpu()
            logger.info(
                "spaCy GPU acceleration: %s.",
                "enabled" if activated else "unavailable (falling back to CPU)",
            )

        self._spacy_nlp = spacy.load(SPACY_MODEL_ID)
        logger.info("spaCy model loaded successfully.")

    def _ensure_loaded(self) -> None:
        """Raise an informative error if inference is attempted before loading."""
        if not self._loaded:
            raise RuntimeError(
                "ModelRegistry has not been initialised. "
                "Call ModelRegistry.get().load_all() during application startup."
            )
