"""
Re-Identification module using fast-reid.
Extracts person embeddings and matches across videos.
"""

from .embedding_extractor import ReIDEmbeddingExtractor
from .matcher import cosine_similarity, is_match

__all__ = [
    "ReIDEmbeddingExtractor",
    "cosine_similarity",
    "is_match",
]
