"""Feature extractor factory for experiments."""

from typing import Dict, Any, Type
import logging

from .base import BaseFeatureExtractor
from .individual import IndividualFeatureExtractor
from .cross_sectional import CrossSectionalFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractorFactory:
    """
    Factory for creating feature extractors.

    Supports registration of new extractors for experimentation.
    """

    _extractors: Dict[str, Type[BaseFeatureExtractor]] = {
        'individual': IndividualFeatureExtractor,
        'cross_sectional': CrossSectionalFeatureExtractor,
    }

    @classmethod
    def register(cls, name: str, extractor_class: Type[BaseFeatureExtractor]):
        """
        Register a new feature extractor.

        Args:
            name: Unique identifier for this extractor
            extractor_class: Class that inherits from BaseFeatureExtractor
        """
        if not issubclass(extractor_class, BaseFeatureExtractor):
            raise ValueError(f"{extractor_class} must inherit from BaseFeatureExtractor")

        cls._extractors[name] = extractor_class
        logger.info(f"Registered feature extractor: {name}")

    @classmethod
    def create(
        cls,
        extractor_type: str,
        db_path: str,
        config: Dict[str, Any] = None
    ) -> BaseFeatureExtractor:
        """
        Create a feature extractor instance.

        Args:
            extractor_type: Type of extractor ('individual', 'cross_sectional', etc.)
            db_path: Path to the database
            config: Configuration dictionary for the extractor

        Returns:
            Configured feature extractor instance
        """
        if extractor_type not in cls._extractors:
            available = ', '.join(cls._extractors.keys())
            raise ValueError(
                f"Unknown extractor type: {extractor_type}. "
                f"Available: {available}"
            )

        extractor_class = cls._extractors[extractor_type]
        return extractor_class(db_path, config)

    @classmethod
    def list_extractors(cls) -> Dict[str, str]:
        """
        List all available extractors with descriptions.

        Returns:
            Dictionary of extractor names to descriptions
        """
        return {
            name: extractor.description
            for name, extractor in cls._extractors.items()
        }


def get_feature_extractor(
    extractor_type: str,
    db_path: str,
    config: Dict[str, Any] = None
) -> BaseFeatureExtractor:
    """
    Convenience function to get a feature extractor.

    Args:
        extractor_type: Type of extractor
        db_path: Path to the database
        config: Configuration dictionary

    Returns:
        Configured feature extractor instance
    """
    return FeatureExtractorFactory.create(extractor_type, db_path, config)
