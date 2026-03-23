"""
Embedding matching utilities for re-identification.
"""

import numpy as np
from typing import List, Tuple


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    Assumes embeddings are already L2-normalized (fast-reid does this).

    Args:
        emb1: 1D embedding (2048,)
        emb2: 1D embedding (2048,)

    Returns:
        Similarity in [0, 1] (1 = identical)
    """
    emb1 = np.asarray(emb1, dtype=np.float32).flatten()
    emb2 = np.asarray(emb2, dtype=np.float32).flatten()

    # If not normalized, normalize
    n1 = np.linalg.norm(emb1)
    n2 = np.linalg.norm(emb2)
    if n1 > 0:
        emb1 = emb1 / n1
    if n2 > 0:
        emb2 = emb2 / n2

    sim = np.dot(emb1, emb2)
    # Cosine similarity is in [-1, 1]; re-id typically uses [0, 1] after normalization
    return float(np.clip(sim, 0.0, 1.0))


def is_match(
    ref_embedding: np.ndarray,
    query_embedding: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Determine if query embedding matches reference.

    Args:
        ref_embedding: Reference person embedding
        query_embedding: Candidate person embedding
        threshold: Minimum cosine similarity to consider a match

    Returns:
        (is_match: bool, similarity: float)
    """
    sim = cosine_similarity(ref_embedding, query_embedding)
    return (sim >= threshold, sim)


def match_batch(
    ref_embedding: np.ndarray,
    query_embeddings: np.ndarray,
    threshold: float = 0.5,
) -> List[Tuple[int, float]]:
    """
    Find which query embeddings match the reference.

    Args:
        ref_embedding: (2048,) reference
        query_embeddings: (N, 2048) candidate embeddings
        threshold: Match threshold

    Returns:
        List of (index, similarity) for matches above threshold
    """
    ref = np.asarray(ref_embedding, dtype=np.float32).flatten()
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    queries = np.asarray(query_embeddings, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    # Normalize
    ref_norm = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(q_norm, ref_norm.T).flatten()
    sims = np.clip(sims, 0.0, 1.0)

    matches = []
    for i, s in enumerate(sims):
        if s >= threshold:
            matches.append((i, float(s)))
    return matches
