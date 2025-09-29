import os
from datetime import datetime
from typing import Any, Dict, List
from importlib.resources import files

from rank_llm.data import DataWriter, Request, Query, Candidate, Result
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM

from .base import BaseReranker


class ListwiseRerankerVLLM(BaseReranker):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--reranker-model",
            type=str,
            required=True,
            help="The model name used for reranking (required).",
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
        parser.add_argument(
            "--reasoning-token-budget",
            type=int,
            default=4 * 4096
            - 100,  # 100 reserved for the actual sorted array as the final answer
            help="The max number tokens used for reasoning",
        )
        parser.add_argument(
            "--reranker-base-url",
            type=str,
            default="http://localhost:18000/v1",
            help="The url for the vllm server used by the reranker for inference calls.",
        )

    def __init__(self, args):
        prompt_template_path = args.prompt_template_path
        if not prompt_template_path:
            prompt_template_path = (
                files("rank_llm.rerank.prompt_templates") / "rank_zephyr_template.yaml"
            )
        model_coordinator = RankListwiseOSLLM(
            model=args.reranker_model,
            context_size=args.context_size,
            prompt_template_path=prompt_template_path,
            window_size=args.window_size,
            stride=args.stride,
            is_thinking=True,
            base_url=args.reranker_base_url,
            reasoning_token_budget=args.reasoning_token_budget,
        )
        self.reranker = Reranker(model_coordinator)
        self.first_stage_k = args.first_stage_k
        print("reranker successfully created!")

    def _create_request(
        self, query: str, retrieved_documents: List[Dict[str, Any]], qid=0
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
        query = Query(text=query, qid=qid)
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
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        history_file_name: str,
        query_id: str | None = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        request = self._create_request(query, retrieved_documents, query_id)
        kwargs = {"populate_invocations_history": True}
        rerank_results = self.reranker.rerank_batch(
            [request], rank_start=0, rank_end=self.first_stage_k, **kwargs
        )
        writer = DataWriter(rerank_results, append=True)
        writer.write_inference_invocations_history(history_file_name)
        return self._process_rerank_result(rerank_results[0])

    def rerank_batch(
        self,
        queries: Dict[str, str],
        retrieved_results: Dict[str, list[dict[str, Any]]],
        history_file_name: str,
        k: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        requests = []
        for qid, candidates in retrieved_results.items():
            requests.append(self._create_request(queries[qid], candidates, qid))
        kwargs = {"populate_invocations_history": True}
        rerank_results = self.reranker.rerank_batch(
            requests, rank_start=0, rank_end=k, **kwargs
        )
        writer = DataWriter(rerank_results)
        writer.write_inference_invocations_history(history_file_name)
        return [self._process_rerank_result(result) for result in rerank_results]

    @property
    def rerank_type(self) -> str:
        return "listwise_vllm"
