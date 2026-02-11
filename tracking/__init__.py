"""Person tracking and re-identification module."""

from .feature_extractor import (
    FeatureExtractor,
    get_embedding,
    compare_embeddings,
    interpret_similarity,
    extract_crop_features
)

__all__ = [
    'FeatureExtractor',
    'get_embedding',
    'compare_embeddings',
    'interpret_similarity',
    'extract_crop_features'
]
