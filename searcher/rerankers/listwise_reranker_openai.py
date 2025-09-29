import os
from typing import Any, Dict, List
from importlib.resources import files

from rank_llm.data import Request, Query, Candidate, Result
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import SafeOpenai

from .base import BaseReranker


class ListwiseRerankerOpenAI(BaseReranker):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--reranker-model",
            type=str,
            required=True,
            help="The model name used for reranking (required).",
        )
        parser.add_argument(
            "--base-url",
            type=str,
            default="https://openrouter.ai/api/v1/",
            help="The base URL used for inference, by default 'https://openrouter.ai/api/v1/' for open router.",
        )
        parser.add_argument(
            "--context-size",
            type=int,
            default=16384,
            help="The maximum context size (default: 16384).",
        )
        parser.add_argument(
            "--prompt-template-path",
            type=str,
            default=None,
            help="Optional path to a prompt template file, if None is provided, by default the rank_zephyr_template.yaml template will be used.",
        )
        parser.add_argument(
            "--window-size",
            type=int,
            default=20,
            help="Window size for the sliding window algorithm (default: 20).",
        )
        parser.add_argument(
            "--stride",
            type=int,
            default=10,
            help="Stride for the sliding window algorithm (default: 10).",
        )
        parser.add_argument(
            "--first-stage-k",
            type=int,
            default=100,
            help="The number of first stage candidates retrieved for reranking (default: 100)",
        )

    def __init__(self, args):
        api_key = os.getenv("RERANKER_API_KEY")
        if not api_key:
            raise RuntimeError("RERANKER_API_KEY is not set in environment")
        prompt_template_path = args.prompt_template_path
        if not prompt_template_path:
            prompt_template_path = (
                files("rank_llm.rerank.prompt_templates") / "rank_zephyr_template.yaml"
            )
        model_coordinator = SafeOpenai(
            model=args.reranker_model,
            context_size=args.context_size,
            prompt_template_path=prompt_template_path,
            window_size=args.window_size,
            stride=args.stride,
            keys=api_key,
            base_url=args.base_url,
        )
        self.reranker = Reranker(model_coordinator)
        self.first_stage_k = args.first_stage_k
        print("reranker successfully created!")

    def _create_request(
        self, query: str, retrieved_documents: List[Dict[str, Any]]
    ) -> Request:
        candidates = []
        for result in retrieved_documents:
            candidates.append(
                Candidate(
                    docid=result["docid"],
                    score=result.get("score", 0),
                    doc={"text": result["text"]},
                )
            )
        query = Query(text=query, qid=0)
        return Request(query=query, candidates=candidates)

    def _process_rerank_result(
        self, rerank_result: Result, k: int = 10
    ) -> List[Dict[str, Any]]:
        processed_candidates = []
        for candidate in rerank_result.candidates[:k]:
            processed_candidates.append(
                {
                    "docid": candidate.docid,
                    "score": candidate.score,
                    "text": candidate.doc["text"],
                }
            )
        return processed_candidates

    def rerank(
        self, query: str, retrieved_documents: List[Dict[str, Any]], k: int = 10
    ) -> List[Dict[str, Any]]:
        request = self._create_request(query, retrieved_documents)
        print(f"heeey in 105")
        rerank_result = self.reranker.rerank_batch(
            [request], rank_start=0, rank_end=self.first_stage_k
        )[0]
        print(f"heeey in 108")
        temp = self._process_rerank_result(rerank_result)
        print(f"heey in 110")
        print(f"{[doc['docid'] for doc in temp]}")
        return temp

    def rerank_batch(
        self,
        queries: List[str],
        retrieved_documents: List[List[Dict[str, Any]]],
        k: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        requests = [
            self._create_request(query, candidates)
            for query, candidates in zip(queries, retrieved_documents)
        ]
        rerank_results = self.reranker.rerank_batch(
            requests, rank_start=0, rank_end=self.first_stage_k
        )
        return [self._process_rerank_result(result) for result in rerank_results]

    @property
    def rerank_type(self) -> str:
        return "Listwise"
