"""Local text-only EmbeddingGemma 2 retrieval embeddings."""

from __future__ import annotations

import math
import os
import re
from importlib.metadata import version
from typing import Any, Sequence

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"


class EmbeddingGemmaEncoder:
    """Encode queries and documents with the model's asymmetric search prompts."""

    def __init__(
        self,
        *,
        model: Any = None,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        dimension: int = 128,
        device: str = "cpu",
        batch_size: int = 8,
        cpu_threads: int | None = None,
        sort_by_length: bool = False,
        backend: str = "torch",
        max_seq_length: int | None = None,
        quantization: str = "none",
    ) -> None:
        if dimension not in (128, 256, 512, 768):
            raise ValueError("dimension must be 128, 256, 512, or 768")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if cpu_threads is not None and cpu_threads < 1:
            raise ValueError("cpu_threads must be positive when specified")
        if backend not in {"torch", "onnx", "openvino"}:
            raise ValueError("backend must be torch, onnx, or openvino")
        if quantization not in {"none", "dynamic_int8"}:
            raise ValueError("quantization must be none or dynamic_int8")
        if quantization != "none" and (
            backend != "torch" or not device.startswith("cpu")
        ):
            raise ValueError("dynamic_int8 requires the torch CPU backend")
        if max_seq_length is not None and (
            isinstance(max_seq_length, bool)
            or not isinstance(max_seq_length, int)
            or max_seq_length < 1
        ):
            raise ValueError("max_seq_length must be a positive integer")
        self.dimension = dimension
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.cache_fingerprint = None
        if cpu_threads is not None and device.startswith("cpu"):
            import torch

            torch.set_num_threads(cpu_threads)
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            dtype = (
                torch.bfloat16
                if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else torch.float32
            )
            model = SentenceTransformer(
                model_id,
                revision=revision,
                device=device,
                backend=backend,
                config_kwargs={"vision_config": None, "audio_config": None},
                model_kwargs={"dtype": dtype} if backend == "torch" else {},
            )
            if max_seq_length is not None:
                model.max_seq_length = max_seq_length
            # Only immutable Hub revisions are eligible for cross-process reuse.
            if (
                backend == "torch"
                and re.fullmatch(r"[0-9a-f]{40}", revision)
                and not os.path.exists(model_id)
            ):
                self.cache_fingerprint = {
                    "encoder": model_id,
                    "revision": revision,
                    "dimension": dimension,
                    "prompt": "document:title: {title or none} | text: {text};"
                    "normalized:v1",
                    "backend": backend,
                    "device": device,
                    "dtype": str(dtype) if backend == "torch" else backend,
                    "max_seq_length": model.max_seq_length,
                    "batch_size": batch_size,
                    "sort_by_length": sort_by_length,
                    "cpu_threads": cpu_threads,
                    "quantization": quantization,
                    "torch_version": version("torch"),
                    "transformers_version": version("transformers"),
                    "sentence_transformers_version": version("sentence-transformers"),
                }
        elif max_seq_length is not None:
            model.max_seq_length = max_seq_length
        if quantization == "dynamic_int8":
            import torch

            model[0].auto_model = torch.ao.quantization.quantize_dynamic(
                model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
            )
        self.model = model

    def _encode(self, texts: list[str], **prompt: str) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if self.sort_by_length and len(texts) > 1:
            order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
            texts_to_encode = [texts[index] for index in order]
        else:
            order = list(range(len(texts)))
            texts_to_encode = texts
        result = self.model.encode(
            texts_to_encode,
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
        restored: list[list[float]] = [[] for _ in vectors]
        for encoded_index, original_index in enumerate(order):
            restored[original_index] = vectors[encoded_index]
        return restored

    def encode_query(self, text: str) -> list[float]:
        """Encode a resume as an asymmetric search query."""
        return self.encode_queries([text])[0]

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode multiple resumes together as asymmetric search queries."""
        return self._encode(list(texts), prompt_name="SearchQuery")

    def encode_documents(
        self, texts: Sequence[str], *, titles: Sequence[str | None] | None = None
    ) -> list[list[float]]:
        """Encode vacancy content with the prescribed title and text format."""
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
