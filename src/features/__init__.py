"""Feature extraction module with factory pattern for experiments."""

from .base import BaseFeatureExtractor
from .factory import FeatureExtractorFactory, get_feature_extractor

__all__ = [
    'BaseFeatureExtractor',
    'FeatureExtractorFactory',
    'get_feature_extractor',
]
