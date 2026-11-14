"""Persistent JSON-lines worker for isolated OpenVINO embedding inference.

Input lines are JSON objects: ``{"op":"query","texts":[...]}`` or
``{"op":"documents","texts":[...],"titles":[...]}``. Output is one JSON
object per line. Text contents are never logged; errors contain exception type
and message only. Use a dedicated environment installed from
``requirements-openvino.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from vrscore.embeddinggemma_openvino import OpenVINOEmbeddingGemmaEncoder


def _handle(
    encoder: OpenVINOEmbeddingGemmaEncoder, request: dict[str, Any]
) -> dict[str, Any]:
    op = request.get("op")
    texts = request.get("texts")
    if not isinstance(texts, list):
        raise ValueError("request field 'texts' must be a list")
    if op == "query":
        vectors = encoder.encode_queries(texts)
    elif op == "documents":
        titles = request.get("titles")
        if titles is not None and not isinstance(titles, list):
            raise ValueError("request field 'titles' must be a list")
        vectors = encoder.encode_documents(texts, titles=titles)
    else:
        raise ValueError("op must be 'query' or 'documents'")
    return {"vectors": vectors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/embeddinggemma-2")
    parser.add_argument(
        "--revision",
        default="f9972ac22e696a105c482202646979e33e4b5a1e",
    )
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args(argv)
    base_site = os.environ.get("MFTI_BASE_SITE_PACKAGES")
    if base_site:
        sys.path.append(base_site)
    try:
        encoder = OpenVINOEmbeddingGemmaEncoder(
            model_id=args.model_id,
            revision=args.revision,
            dimension=args.dimension,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as exc:
        print(
            f"OpenVINO worker initialization failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("each request must be a JSON object")
            result = _handle(encoder, request)
        except Exception as exc:
            result = {
                "error": {"type": type(exc).__name__, "message": "request failed"}
            }
        sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
