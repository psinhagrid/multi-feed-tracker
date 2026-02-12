"""LLM integration for image description and labeling."""

from .image_describer import describe_image, describe_image_batch

__all__ = ['describe_image', 'describe_image_batch']
