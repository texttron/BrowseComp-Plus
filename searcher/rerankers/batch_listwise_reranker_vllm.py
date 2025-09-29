import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from importlib.resources import files

from rank_llm.data import DataWriter, Request, Query, Candidate, Result
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import RankListwiseOSLLM

# NEW imports
import atexit
import threading
import queue
import time
from concurrent.futures import Future

from .base import BaseReranker


class BatchListwiseRerankerVLLM(BaseReranker):
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
            default=4096 - 100,
            help="The max number tokens used for reasoning",
        )
        parser.add_argument(
            "--reranker-base-url",
            type=str,
            default="http://localhost:18000/v1",
            help="The url for the vllm server used by the reranker for inference calls.",
        )
        # NEW: batching controls
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="Max number of rerank requests to batch together before issuing a rerank_batch call.",
        )
        parser.add_argument(
            "--batch-timeout-ms",
            type=int,
            default=500,
            help="Max time to wait (in ms) for a batch to fill before issuing rerank_batch.",
        )
        parser.add_argument(
            "--invocation-history-dir",
            type=str,
            help="Where to write the invocation history",
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

        # --- NEW: store history dir; may be None ---
        self._invocation_history_dir: Optional[str] = args.invocation_history_dir
        if self._invocation_history_dir:
            os.makedirs(self._invocation_history_dir, exist_ok=True)

        # --- NEW: batching infra ---
        self._batch_size = max(1, int(args.batch_size))
        self._batch_timeout_s = max(0.0, int(args.batch_timeout_ms) / 1000.0)
        self._q: "queue.Queue[tuple[Optional[Request], int, Optional[Future]]]" = (
            queue.Queue()
        )
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name="ListwiseRerankerVLLMBatcher", daemon=True
        )
        self._worker.start()
        # ---------------------------

        # --- NEW: async writer infra ---
        # Queue holds (results, filename) to persist history out-of-band.
        self._write_q: "queue.Queue[tuple[Optional[List[Result]], Optional[str]]]" = (
            queue.Queue()
        )
        self._writer_stop = threading.Event()
        self._writer: Optional[threading.Thread] = None
        if self._invocation_history_dir:
            self._writer = threading.Thread(
                target=self._write_worker_loop,
                name="InvocationHistoryWriter",
                daemon=True,
            )
            self._writer.start()
        # --------------------------------

        atexit.register(self.shutdown)

        print(
            "reranker successfully created (batching enabled, async history writing)!"
        )

    # --- NEW: graceful shutdown ---
    def shutdown(self):
        # stop batcher
        if not self._stop.is_set():
            self._stop.set()
            self._q.put_nowait((None, 0, None))  # sentinel
            self._worker.join(timeout=2.0)
        # stop writer
        if self._writer and not self._writer_stop.is_set():
            self._writer_stop.set()
            self._write_q.put_nowait((None, None))  # sentinel
            self._writer.join(timeout=2.0)

    # --- NEW: batch worker loop ---
    def _worker_loop(self):
        while not self._stop.is_set():
            batch: List[Request] = []
            metas: List[Tuple[int, Future]] = []

            try:
                # wait for the first item up to the full timeout
                item = self._q.get(
                    timeout=self._batch_timeout_s if self._batch_timeout_s > 0 else None
                )
            except queue.Empty:
                continue

            if item[0] is None:
                continue  # sentinel

            req, k, fut = item
            batch.append(req)  # type: ignore[arg-type]
            metas.append((k, fut))  # type: ignore[arg-type]

            # Start a coalescing window from *now*
            t0 = time.time()
            deadline = t0 + (self._batch_timeout_s)

            # Keep collecting until batch_size or timeout since first item
            while len(batch) < self._batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    # Wait a little for more items (bounded by remaining)
                    item2 = self._q.get(timeout=remaining)
                    if item2[0] is None:
                        # put back the sentinel for shutdown path and stop adding
                        self._q.put_nowait((None, 0, None))
                        break
                    req2, k2, fut2 = item2
                    batch.append(req2)  # type: ignore[arg-type]
                    metas.append((k2, fut2))  # type: ignore[arg-type]
                except queue.Empty:
                    continue

            # Execute the batch
            try:
                kwargs = {"populate_invocations_history": True}
                print(f"sending {len(batch)} requests in batch")
                results: List[Result] = self.reranker.rerank_batch(
                    batch, rank_start=0, rank_end=self.first_stage_k, **kwargs
                )

                # --- NEW: enqueue async write instead of writing here ---
                if self._invocation_history_dir and self._writer:
                    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                    history_file_name = os.path.join(
                        self._invocation_history_dir, f"invocation_history_{ts}.json"
                    )
                    # Do not block the batcher; hand off to writer thread.
                    try:
                        self._write_q.put_nowait((results, history_file_name))
                    except queue.Full:
                        # Fallback: drop on floor with a warning rather than blocking.
                        print(
                            "WARNING: write queue full; dropping invocation history for this batch."
                        )

                # Fan-out each result to its waiting caller immediately
                for res, (k_i, fut_i) in zip(results, metas):
                    try:
                        processed = self._process_rerank_result(res, k=k_i)
                        fut_i.set_result(processed)
                    except Exception as e:
                        fut_i.set_exception(e)

            except Exception as e:
                print(e)
                # Fail all futures in this batch
                for _, fut_i in metas:
                    try:
                        fut_i.set_exception(e)
                    except Exception:
                        pass

    # --- NEW: writer thread loop ---
    def _write_worker_loop(self):
        """
        Dedicated background writer. Consumes (results, filename) tasks and writes
        invocation history to disk. This keeps IO out of the hot path.
        """
        while not self._writer_stop.is_set():
            try:
                item = self._write_q.get(timeout=0.25)
            except queue.Empty:
                continue

            results, filename = item
            if results is None and filename is None:
                continue  # sentinel

            try:
                writer = DataWriter(results, append=True)
                writer.write_inference_invocations_history(filename)  # IO-bound
            except Exception as e:
                # Don't crash the writer thread; just log.
                print(f"ERROR writing invocation history to {filename}: {e}")

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

    # PUBLIC API remains the same, but now enqueues and waits on a Future
    def rerank(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        query_id: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        request = self._create_request(query, retrieved_documents, query_id)
        fut: Future = Future()
        # Store k per-request so each caller can choose different top-k
        self._q.put((request, k, fut))
        # Block the calling thread until its batch returns (history writing is async)
        return fut.result()

    @property
    def rerank_type(self) -> str:
        return "batch_listwise_vllm"
