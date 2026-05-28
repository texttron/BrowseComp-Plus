import os
import json
from datetime import datetime
import argparse
from pathlib import Path
import csv
import sys
from typing import Any, Optional

from rich import print

from openai import OpenAI

import sys as _sys
_sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from searcher.searchers import SearcherType
from utils import extract_retrieved_docids_from_result
from transformers import AutoTokenizer
from prompts import format_query


class SearchToolHandler:
    def __init__(self, searcher, snippet_max_tokens: int | None = None, k: int = 5, include_get_document: bool = True):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query string"}
                        },
                        "required": ["query"]
                    },
                },
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_document",
                        "description": self.searcher.get_document_description(),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "docid": {"type": "string", "description": "Document ID to retrieve"}
                            },
                            "required": ["docid"]
                        },
                    },
                }
            )

        return tools

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
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
                results.append({"docid": cand["docid"], "score": cand["score"], "snippet": cand["snippet"]})

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


def run_conversation_with_tools(
    client: OpenAI,
    *,
    query: str,
    model: str,
    max_tokens: int,
    tool_handler: SearchToolHandler,
    system_prompt: str | None = None,
    query_template: str | None = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_iterations: int = 100,
):
    tools = tool_handler.get_tool_definitions()

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    formatted_query = format_query(query, query_template)
    messages.append({"role": "user", "content": formatted_query})

    cumulative_usage = {
        "prompt_tokens": 0,
        "prompt_tokens_cached": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }

    normalized_results: list[dict[str, Any]] = []

    finish_reason: Optional[str] = None

    # Treat max_tokens as a global output budget across the entire conversation.
    global_max_tokens = max_tokens

    for _ in range(max_iterations):
        remaining_tokens = global_max_tokens - cumulative_usage["completion_tokens"]
        if remaining_tokens <= 0:
            print(f"Warning: Reached global max_tokens output budget ({global_max_tokens})")
            break
        create_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": remaining_tokens,
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        completion = client.chat.completions.create(**create_kwargs)

        choice = completion.choices[0]
        finish_reason = choice.finish_reason

        usage = getattr(completion, "usage", None)
        if usage is not None:
            cumulative_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            cumulative_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            cumulative_usage["total_tokens"] += getattr(usage, "total_tokens", 0)

            comp_details = getattr(usage, "completion_tokens_details", None)
            if comp_details is not None:
                cumulative_usage["reasoning_tokens"] += getattr(comp_details, "reasoning_tokens", 0) or 0

            cached_this = 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None and getattr(prompt_details, "cached_tokens", None) is not None:
                try:
                    cached_this = int(getattr(prompt_details, "cached_tokens", 0) or 0)
                except Exception:
                    cached_this = 0
            else:
                try:
                    cached_this = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
                except Exception:
                    cached_this = 0

            cumulative_usage["prompt_tokens_cached"] += cached_this

        assistant_msg = choice.message.model_dump()

        # Log thinking trace separately; DeepSeek-V4-thinking/flash expects the assistant
        # message to retain `reasoning_content` across multi-turn/tool rounds (do not strip it).
        reasoning_text = assistant_msg.get("reasoning_content")
        if reasoning_text:
            normalized_results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": reasoning_text,
                }
            )

        messages.append(assistant_msg)

        if assistant_msg['content'] is not None and assistant_msg['content'].strip():
            normalized_results.append({
                "type": "output_text",
                "tool_name": None,
                "arguments": None,
                "output": assistant_msg["content"],
            })

        if finish_reason != "tool_calls":
            break

        # Execute tool calls
        for tool_call in choice.message.tool_calls:
            tname = tool_call.function.name
            targs = json.loads(tool_call.function.arguments)

            normalized_results.append({
                "type": "tool_call",
                "tool_name": tname,
                "arguments": tool_call.function.arguments,
                "output": None,
            })

            try:
                output = tool_handler.execute_tool(tname, targs)
                # Patch the last tool_call in normalized_results with the output
                normalized_results[-1]["output"] = output

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tname,
                    "content": output,
                })
            except Exception as e:
                error_msg = f"Error executing {tname}: {str(e)}"
                print(error_msg)
                normalized_results[-1]["output"] = error_msg

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tname,
                    "content": error_msg,
                })

    if finish_reason is None:
        print(f"Warning: Conversation hit max iterations ({max_iterations}) without final response")

    return normalized_results, cumulative_usage, finish_reason


def _persist_response(out_dir: str, *, model: str, query_id: str | None, system_prompt: str | None, max_tokens: int, normalized_results: list[dict[str, Any]], cumulative_usage: dict, finish_reason: Optional[str]):
    os.makedirs(out_dir, exist_ok=True)

    tool_call_counts: dict[str, int] = {}
    for item in normalized_results:
        if item.get("type") == "tool_call" and item.get("tool_name"):
            name = item["tool_name"]
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

    normalized_usage = {
        "input_tokens": cumulative_usage.get("prompt_tokens", 0),
        "input_tokens_cached": cumulative_usage.get("prompt_tokens_cached", 0),
        "output_tokens": cumulative_usage.get("completion_tokens", 0) + cumulative_usage.get("reasoning_tokens", 0),
        "included_reasoning_tokens": cumulative_usage.get("reasoning_tokens", 0),
        "total_tokens": cumulative_usage.get("total_tokens", 0),
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(out_dir, f"run_{ts}.json")

    status = finish_reason
    if status == "stop":
        status = "completed"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": model,
                "output_dir": str(out_dir),
                "max_tokens": max_tokens,
            },
            "query_id": query_id,
            "tool_call_counts": tool_call_counts,
            "usage": normalized_usage,
            "status": status,
            "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
            "result": normalized_results,
        }, f, indent=2, default=str)

    print("Saved response to", filename, "| tool call counts:", tool_call_counts)


def _process_tsv_dataset(tsv_path: str, client: OpenAI, args, tool_handler: SearchToolHandler):
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
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    from tqdm import tqdm
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        try:
            normalized_results, cumulative_usage, finish_reason = run_conversation_with_tools(
                client,
                query=qtext,
                model=args.model,
                max_tokens=args.max_tokens,
                tool_handler=tool_handler,
                system_prompt=args.system,
                query_template=args.query_template,
                temperature=args.temperature,
                top_p=args.top_p,
                max_iterations=args.max_iterations,
            )

            with completed_lock:
                completed_count[0] += 1
                if pbar:
                    pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                args.output_dir,
                model=args.model,
                query_id=qid,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                normalized_results=normalized_results,
                cumulative_usage=cumulative_usage,
                finish_reason=finish_reason,
            )
        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            futures = [executor.submit(_handle_single_query, qid, qtext, pbar) for qid, qtext in remaining]
            for _ in as_completed(futures):
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Call DeepSeek (OpenAI-compatible Chat Completions API) with native function calling and local search."
    )
    parser.add_argument("--query", default="topics-qrels/queries.tsv", help="User query text or path to TSV. Wrap in quotes if contains spaces.")
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek API model id (default: %(default)s). Override for e.g. flash/reasoning variants per DeepSeek docs.",
    )
    parser.add_argument("--max_tokens", type=int, default=20000, help="Max tokens to generate (default: %(default)s)")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument(
        "--output-dir",
        default="runs/bm25/deepseek",
        help="Directory to store logs (default: %(default)s)",
    )
    parser.add_argument("--query-template", choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT", "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION"], default="QUERY_TEMPLATE_NO_GET_DOCUMENT", help="Specify the query template to use")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for the model (default: use model defaults)")
    parser.add_argument("--top_p", type=float, default=None, help="Top P for the model (default: use model defaults)")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (default: DEEPSEEK_BASE_URL env or https://api.deepseek.com)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for dataset processing (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of conversation rounds with function calls (default: %(default)s)",
    )

    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )

    # Server configuration arguments for search tool
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 512).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed number of search results to return for all queries in this session (default: 5).",
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="If set, register both the search tool and the get_document tool.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models. If not provided, will use environment variables or CLI login.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Hugging Face home directory for caching models and datasets. If not provided, will use environment variables or default.",
    )

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.hf_token:
        print(f"[DEBUG] Setting HF token from CLI argument: {args.hf_token[:10]}...")
        os.environ['HF_TOKEN'] = args.hf_token
        os.environ['HUGGINGFACE_HUB_TOKEN'] = args.hf_token

    if args.hf_home:
        print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
        os.environ['HF_HOME'] = args.hf_home

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment")

    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    client = OpenAI(api_key=api_key, base_url=base_url)

    searcher = searcher_class(args)

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")
    tools_str = ", ".join(tools_registered)

    print(
        f"Search agent started with {searcher.search_type} search (snippet_max_tokens={args.snippet_max_tokens}, k={args.k})"
    )
    print(f"Registered tools: {tools_str}")

    # If --query looks like a TSV path, process dataset
    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    _process_tsv_dataset(str(potential_path), client, args, tool_handler)
                    return
            except OSError:
                pass

    print("Sending request to DeepSeek with function calling...")
    normalized_results, cumulative_usage, finish_reason = run_conversation_with_tools(
        client,
        query=args.query,
        model=args.model,
        max_tokens=args.max_tokens,
        tool_handler=tool_handler,
        system_prompt=args.system,
        query_template=args.query_template,
        temperature=args.temperature,
        top_p=args.top_p,
        max_iterations=args.max_iterations,
    )

    _persist_response(
        args.output_dir,
        model=args.model,
        query_id=None,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        normalized_results=normalized_results,
        cumulative_usage=cumulative_usage,
        finish_reason=finish_reason,
    )

    # Print final output text if present
    final_texts = [item["output"] for item in normalized_results if item.get("type") == "output_text"]
    if final_texts:
        print(final_texts[-1])


if __name__ == "__main__":
    main()


