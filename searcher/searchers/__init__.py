"""
Searchers package for different search implementations.
"""

from enum import Enum

from .base import BaseSearcher


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    BM25 = "bm25"
    FAISS = "faiss"
    REASONIR = "reasonir"
    CUSTOM = "custom"

    def __init__(self, cli_name):
        self.cli_name = cli_name

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name, importing only when needed."""
        if cli_name == cls.BM25.cli_name:
            from .bm25_searcher import BM25Searcher
            return BM25Searcher
        elif cli_name == cls.FAISS.cli_name:
            from .faiss_searcher import FaissSearcher
            return FaissSearcher
        elif cli_name == cls.REASONIR.cli_name:
            from .faiss_searcher import ReasonIrSearcher
            return ReasonIrSearcher
        elif cli_name == cls.CUSTOM.cli_name:
            from .custom_searcher import CustomSearcher
            return CustomSearcher
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
