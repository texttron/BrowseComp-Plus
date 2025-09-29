"""
Abstract base class for rerank implementations.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseReranker(ABC):
    """Abstract base class for all rerank implementations."""

    @classmethod
    @abstractmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add reranker-specific arguments to the argument parser."""
        pass

    @abstractmethod
    def __init__(self, args):
        """Initialize the reranker with parsed arguments."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        query_id: str | None = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform reranking and return results.

        Args:
            query: Reranking query string
            retrieved_documents: A list of retrieved documents to be reranked
            query_id: The id of the reranking query
            k: Number of results to return after reranking

        Returns:
            List of rerank results with format: {"docid": str, "score": float, "snippet": str}
        """
        pass

    @property
    @abstractmethod
    def rerank_type(self) -> str:
        """Return the type of rerank (e.g., 'Pointwise', 'Listwise', 'BatchListwise)."""
        pass
