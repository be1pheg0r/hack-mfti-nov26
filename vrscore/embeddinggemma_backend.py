"""Optional ONNX/OpenVINO EmbeddingGemma adapter.

Requires the corresponding Sentence Transformers optional backend extra. The
adapter intentionally leaves the core ``vrscore.embeddinggemma`` module and
its configuration untouched.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
SUPPORTED_BACKENDS = frozenset({"torch", "onnx", "openvino"})
SUPPORTED_DIMENSIONS = frozenset({128, 256, 512, 768})


class EmbeddingGemmaBackendEncoder:
    """Encode EmbeddingGemma query/documents using a selected ST backend.

    Args:
        backend: ``"torch"``, ``"onnx"``, or ``"openvino"``.
        model: Optional pre-loaded SentenceTransformer-compatible object.
            Supplying it avoids importing Sentence Transformers at construction.
        model_id: Hugging Face model ID or local model directory.
        revision: Immutable model revision used when loading from the Hub.
        dimension: Matryoshka output dimension (128, 256, 512, or 768).
        device: Sentence Transformers device, normally ``"cpu"`` here.
        batch_size: Default inference batch size.
        model_kwargs: Optional backend-specific kwargs forwarded to ST.
    """

    def __init__(
        self,
        *,
        backend: str = "torch",
        model: Any = None,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        dimension: int = 128,
        device: str = "cpu",
        batch_size: int = 8,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(SUPPORTED_BACKENDS)}")
        if dimension not in SUPPORTED_DIMENSIONS:
            raise ValueError("dimension must be 128, 256, 512, or 768")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.backend = backend
        self.model_id = model_id
        self.revision = revision
        self.dimension = dimension
        self.batch_size = batch_size

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                model_id,
                revision=revision,
                device=device,
                backend=backend,
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
        normalized = []
        for row in vectors:
            norm = math.sqrt(sum(component * component for component in row))
            normalized.append([value / norm for value in row])
        return normalized

    def encode_query(self, text: str) -> list[float]:
        """Encode one retrieval query with the ``SearchQuery`` prompt."""
        return self._encode([text], prompt_name="SearchQuery")[0]

    def encode_documents(
        self, texts: Sequence[str], *, titles: Sequence[str | None] | None = None
    ) -> list[list[float]]:
        """Encode documents in EmbeddingGemma's title/text format."""
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

    def export_quantized_onnx(
        self,
        output_dir: str | Path,
        *,
        quantization_config: str = "avx2",
        file_suffix: str | None = None,
    ) -> Path:
        """Export dynamic-int8 ONNX from this loaded ONNX backend model.

        ``output_dir`` receives a saved model and its quantized ONNX file. This
        operation is intentionally explicit: no export or quantization occurs
        during normal construction.
        """
        if self.backend != "onnx":
            raise ValueError("dynamic ONNX quantization requires backend='onnx'")
        if not hasattr(self.model, "save_pretrained"):
            raise TypeError("model must be a SentenceTransformer ONNX model")
        try:
            from sentence_transformers.backend import (
                export_dynamic_quantized_onnx_model,
            )
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers[onnx] to export quantized ONNX"
            ) from exc

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination)
        export_dynamic_quantized_onnx_model(
            model=self.model,
            quantization_config=quantization_config,
            model_name_or_path=str(destination),
            push_to_hub=False,
            create_pr=False,
            file_suffix=file_suffix,
        )
        return destination


def load_embeddinggemma_backend(**kwargs: Any) -> EmbeddingGemmaBackendEncoder:
    """Factory alias for integrations preferring a descriptive function."""
    return EmbeddingGemmaBackendEncoder(**kwargs)
