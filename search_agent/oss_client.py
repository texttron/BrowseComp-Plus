from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai
from prompts import format_query
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from searcher.rerankers import RerankerType
from searcher.searchers import SearcherType
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType


class SearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "local_knowledge_base_retrieval":
            return self._search(arguments["user_query"], arguments["query_id"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "name": "local_knowledge_base_retrieval",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["user_query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )

        return tools

    def _search(self, query: str, query_id: str | None):
        candidates = self.searcher.search(query, query_id, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


def run_conversation_with_tools(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    query_id: str | None = None,
    max_iterations: int = 100,
    verbose: bool = False,
):
    tool_usage = {}

    messages = initial_request["input"]

    iteration = 1

    while iteration <= max_iterations:
        try:
            request = initial_request.copy()
            request["input"] = messages
            response = client.responses.create(
                **request,
            )
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
                rprint(f"Request: {request}")
            iteration += 1
            continue

        response_dict = response.model_dump(mode="python")

        messages.extend(response_dict["output"])

        if (
            len(response_dict["output"]) >= 1
            and response_dict["output"][-1]["type"] == "reasoning"
        ):
            messages.pop()
            continue

        function_calls = [
            item for item in response_dict["output"] if item["type"] == "function_call"
        ]

        if not function_calls:
            return messages, tool_usage, "completed"

        new_messages = messages.copy()

        for tool_call in function_calls:
            try:
                arguments = json.loads(tool_call["arguments"])
                arguments["query_id"] = query_id
                result = tool_handler.execute_tool(tool_call["name"], arguments)
                tool_usage[tool_call["name"]] = tool_usage.get(tool_call["name"], 0) + 1

                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": result,
                    }
                )

            except Exception as e:
                error_msg = f"Error executing {tool_call['name']}: {str(e)}"
                new_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": error_msg,
                    }
                )
        messages = new_messages
        iteration += 1

    return messages, tool_usage, "incomplete"


def _persist_response(
    out_dir: str,
    initial_request: dict,
    messages: list,
    tool_usage: dict,
    status: str,
    *,
    query_id: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    call_output_by_id: dict[str, str | dict | None] = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_output_by_id[item.get("call_id")] = item.get("output")

    normalized_results: list[dict] = []

    for item in messages or []:
        if not isinstance(item, dict):
            continue

        itype = item.get("type")

        if itype == "function_call":
            call_id = item.get("call_id")
            normalized_results.append(
                {
                    "type": "tool_call",
                    "tool_name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "output": call_output_by_id.get(call_id),
                }
            )

        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list) and len(summary) > 0:
                reasoning_output = summary
            else:
                reasoning_output = []
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") in {
                        "reasoning_text",
                        "output_text",
                        "text",
                    }:
                        text_val = str(part.get("text", "")).strip()
                        if text_val:
                            reasoning_output.append(text_val)
            normalized_results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": reasoning_output,
                }
            )

        elif itype == "message":
            parts = item.get("content", []) or []
            text_chunks: list[str] = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_chunks.append(str(part.get("text", "")))
            text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if text:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": text,
                    }
                )

    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        normalized_name = (
            "search"
            if isinstance(tool_name, str) and ("retrieval" in tool_name.lower())
            else tool_name
        )
        normalized_tool_counts[normalized_name] = normalized_tool_counts.get(
            normalized_name, 0
        ) + int(count or 0)

    normalized_record = {
        "metadata": {
            "model": initial_request.get("model"),
            "reasoning": initial_request.get("reasoning"),
            "output_dir": str(out_dir),
        },
        "query_id": query_id,
        "tool_call_counts": normalized_tool_counts,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts)


def _process_tsv_dataset(
    tsv_path: str, client: openai.OpenAI, args, tool_handler: SearchToolHandler
):
    """Process a TSV file of (id \t query) pairs sequentially and persist responses."""

    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue  # ignore corrupt files

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    import threading

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        """Build request, send and persist response for one query."""

        initial_request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "input": [
                {"role": "user", "content": format_query(qtext, args.query_template)}
            ],
            "tools": tool_handler.get_tool_definitions(),
            "truncation": "auto",
            "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
        }

        try:
            messages, tool_usage, status = run_conversation_with_tools(
                client,
                initial_request,
                tool_handler,
                qid,
                args.max_iterations,
                args.verbose,
            )

            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                out_dir, initial_request, messages, tool_usage, status, query_id=qid
            )

        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(_handle_single_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]

            for _ in as_completed(futures):
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Call vLLM OpenAI Responses API with native function calling and local search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query *or* TSV file path",
    )
    parser.add_argument(
        "--model", default="openai/gpt-oss-20b", help="Model name served by vLLM"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="max_output_tokens for Responses API",
    )
    parser.add_argument(
        "--output-dir", default="runs_vllm", help="Directory to store run JSON files"
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="Parallel threads for dataset mode"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max conversation rounds with function calls",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "--model-url", default="http://localhost:8000/v1", help="Model URL"
    )

    # Searcher selection and shared tool options --------------------------
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--reranker-type",
        choices=RerankerType.get_choices(),
        default=None,
        help=f"Type of reranker to use: None, {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models",
    )
    parser.add_argument(
        "--hf-home", type=str, help="Hugging Face home directory for caching"
    )

    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Tokens per document snippet",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Top-k search results to return for each query"
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="Also register the get_document tool",
    )

    temp_args, _ = parser.parse_known_args()
    reranker = None
    if temp_args.reranker_type:
        reranker_class = RerankerType.get_reranker_class(temp_args.reranker_type)
        reranker_class.parse_args(parser)
        rerank_args, _ = parser.parse_known_args()
        reranker = reranker_class(rerank_args)
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    client = openai.OpenAI(
        base_url=args.model_url,
        api_key="EMPTY",
    )

    searcher = searcher_class(reranker, args)
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    print("Processing TSV dataset", potential_path)
                    _process_tsv_dataset(
                        str(potential_path), client, args, tool_handler
                    )
                    return
            except OSError:
                pass

    print("Processing single query", args.query)

    args.query = format_query(args.query, args.query_template)

    messages = [{"role": "user", "content": args.query}]

    initial_request = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "input": messages,
        "tools": tool_handler.get_tool_definitions(),
        "truncation": "auto",
        "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
    }

    messages, tool_usage, status = run_conversation_with_tools(
        client, initial_request, tool_handler, None, args.max_iterations, args.verbose
    )

    _persist_response(
        args.output_dir, initial_request, messages, tool_usage, status, query_id=None
    )

    rprint(messages)


if __name__ == "__main__":
    main()
