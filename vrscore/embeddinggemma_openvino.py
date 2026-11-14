"""OpenVINO-only EmbeddingGemma 2 adapter.

Keep this module optional: importing it does not import OpenVINO or load model
weights. Run inference in :mod:`vrscore.embeddinggemma_openvino_worker` when the
caller needs dependency/process isolation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
SUPPORTED_DIMENSIONS = frozenset({128, 256, 512, 768})


class OpenVINOEmbeddingGemmaEncoder:
    """Encode retrieval queries and documents with Sentence Transformers OpenVINO."""

    def __init__(
        self,
        *,
        model: Any = None,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        dimension: int = 128,
        device: str = "cpu",
        batch_size: int = 8,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dimension not in SUPPORTED_DIMENSIONS:
            raise ValueError("dimension must be 128, 256, 512, or 768")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dimension = dimension
        self.batch_size = batch_size
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "OpenVINO inference requires the isolated environment in "
                    "requirements-openvino.txt (sentence-transformers[openvino])"
                ) from exc
            model = SentenceTransformer(
                model_id,
                revision=revision,
                device=device,
                backend="openvino",
                config_kwargs={"vision_config": None, "audio_config": None},
                model_kwargs=dict(model_kwargs or {}),
            )
        self.model = model

    def _encode(self, texts: list[str], **prompt: str) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        result = self.model.encode(
            texts,
            **prompt,
            truncate_dim=self.dimension,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in row] for row in result]
        if len(vectors) != len(texts) or any(
            len(row) != self.dimension
            or not all(math.isfinite(value) for value in row)
            or not any(row)
            for row in vectors
        ):
            raise ValueError("Model returned invalid embeddings")
        return vectors

    def encode_query(self, text: str) -> list[float]:
        """Encode one query using EmbeddingGemma's SearchQuery prompt."""
        return self._encode([text], prompt_name="SearchQuery")[0]

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode queries in one batch."""
        return self._encode(list(texts), prompt_name="SearchQuery")

    def encode_documents(
        self, texts: Sequence[str], *, titles: Sequence[str | None] | None = None
    ) -> list[list[float]]:
        """Encode documents as ``title: ... | text: ...`` without a task prompt."""
        if titles is None:
            titles = [None] * len(texts)
        if len(titles) != len(texts):
            raise ValueError("titles and texts must have the same length")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        formatted = [
            f"title: {title.strip() if title and title.strip() else 'none'}"
            f" | text: {text}"
            for title, text in zip(titles, texts, strict=True)
        ]
        return self._encode(formatted, prompt="")
