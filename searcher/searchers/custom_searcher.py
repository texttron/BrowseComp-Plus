"""
Placeholder for custom searcher. Implement your own searcher here.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseSearcher

logger = logging.getLogger(__name__)


class CustomSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        # Parse arguments from CLI that you will need to instantiate the searcher
        pass

    def __init__(self, reranker, args):
        super().__init__(reranker)
        # Initialize the searcher with the arguments
        self.args = args
        # ...

    def _retrieve(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        # Implement your retrieval logic here
        # ...
        return [
            {
                "docid": "place-holder-docid",  # Required
                "score": 0.0,  # Optional
                "text": "place-holder-text",  # Required
            },
            # ...
        ]

    def retrieve_batch(
        self, queries: List[str], qids: List[str], k: int = 10
    ) -> Dict[str, list[dict[str, Any]]]:
        # Implement your retrieval batch  logic here
        return {
            "qid": [
                {
                    "docid": "place-holder-docid",  # Required
                    "score": 0.0,  # Optional
                    "text": "place-holder-text",  # Required
                },
                # ...
            ]
        }

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        # Implement your logic for getting a full document by id
        # ...
        return {
            "docid": docid,
            "text": "place-holder-text",
        }

    @property
    def search_type(self) -> str:
        return "custom"

    # Optionally, you can also override the description of your tools to be used in the LLM:
    def search_description(self, k: int = 10) -> str:
        return "Perform a search on a custom knowledge source. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."

    def get_document_description(self) -> str:
        return "Retrieve a full document by its docid."
