"""Evaluation helpers for the paired Malt shortlist experiment."""

import numpy as np


def rerank_shortlist(ids, query, matrix, positions):
    """Rank exactly the supplied IDs by cosine with deterministic ID ties."""
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs")
    query = np.asarray(query, dtype=np.float64)
    vectors = np.asarray(matrix[[positions[x] for x in ids]], dtype=np.float64)
    if not ids:
        return []
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("vectors must be finite and nonzero")
    scores = (vectors @ query) / norms
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return sorted(zip(ids, scores.tolist()), key=lambda item: (-item[1], item[0]))
