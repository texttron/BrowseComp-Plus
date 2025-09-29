import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from searcher.rerankers import RerankerType
from searcher.searchers import BaseSearcher, SearcherType


def load_queries(tsv_path: str) -> List[Tuple[str, str]]:
    """Loads queries from a TSV file."""
    if not tsv_path.strip().lower().endswith(".tsv"):
        raise ValueError(f"Invalid query file format, expected .tsv: {tsv_path}")
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV query file not found: {tsv_path}")
    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))
    return queries


def process_queries(
    searcher: BaseSearcher, queries: List[Tuple[str, str]], args: argparse.Namespace
):
    """
    Processes queries in batches using retrieve_batch (dict[qid -> candidates]),
    optionally reranks per batch, and streams results to file to avoid OOM.
    """
    batch_size = args.batch_size

    if getattr(args, "reranker_type", None):
        model_name = (
            "bm25" if args.searcher_type == "bm25" else args.model_name.split("/")[-1]
        )
        filename = (
            f"retrieve_{model_name}_k_{args.k}_"
            f"rerank_{args.reranker_model.split('/')[-1]}_k_{args.first_stage_k}.trec"
        )
    else:
        filename = f"retrieve_{args.searcher_type}_k_{args.k}.trec"

    os.makedirs(args.output_dir, exist_ok=True)
    filepath = os.path.join(args.output_dir, filename)
    reranker = None
    if getattr(args, "reranker_type", None):
        reranker_class = RerankerType.get_reranker_class(args.reranker_type)
        reranker = reranker_class(args)

    print(
        f"Retrieving{' + reranking' if reranker else ''} in batches of {batch_size}..."
    )
    with open(filepath, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(queries), batch_size)):
            end = min(start + batch_size, len(queries))
            batch = queries[start:end]
            batch_qids = [qid for qid, _ in batch]
            batch_qtexts = [qtext for _, qtext in batch]
            retrieved = searcher.retrieve_batch(batch_qtexts, batch_qids, args.k)
            if reranker is not None:
                Path(f"{args.output_dir}/invocaton_history").mkdir(
                    parents=True, exist_ok=True
                )
                history_file_name = f"{args.output_dir}/invocaton_history/{filename[:-5]}_{start}_{end}.json"
                batch_queries_dict = {qid: qtext for qid, qtext in batch}
                rerank_results = reranker.rerank_batch(
                    batch_queries_dict, retrieved, history_file_name, args.first_stage_k
                )
                retrieved = {}
                for i, qid in enumerate(batch_qids):
                    retrieved[qid] = rerank_results[i]
            for qid, _ in batch:
                candidates = retrieved.get(qid, [])
                for rank, cand in enumerate(candidates, start=1):
                    f.write(
                        f'{qid} Q0 {cand["docid"]} {rank} {cand["score"]} {args.searcher_type}\n'
                    )
            del retrieved

    print(f"Done. Wrote results to: {filepath}")


def main():
    """Main function to handle argument parsing and query processing."""
    parser = argparse.ArgumentParser(description="Retrieval with optional reranking")
    parser.add_argument(
        "--query-file",
        default="topics-qrels/queries.tsv",
        help="The .tsv file containing queries.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to store retrieved results."
    )
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}.",
    )
    parser.add_argument(
        "--reranker-type",
        choices=RerankerType.get_choices(),
        default=None,
        help=f"Type of reranker to use: None, {', '.join(RerankerType.get_choices())}. Default: None.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed number of search results to return for all queries (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The batch size used for retreival and optionally reranking. (default: 64).",
    )
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)
    if temp_args.reranker_type:
        reranker_class = RerankerType.get_reranker_class(temp_args.reranker_type)
        reranker_class.parse_args(parser)
    args = parser.parse_args()
    searcher = searcher_class(
        reranker=None, args=args
    )  # process queries will batch rerank after batch retrieval, don't pass the reranker here.
    queries = load_queries(args.query_file)
    process_queries(searcher, queries, args)


if __name__ == "__main__":
    main()
