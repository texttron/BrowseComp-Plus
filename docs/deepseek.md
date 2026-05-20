# Evaluating models on DeepSeek API

Follow the main [README.md](../README.md) to decrypt the dataset, set up Python dependencies, and download indexes.

## Environment

Set your API key:

```bash
export DEEPSEEK_API_KEY="..."
```

Optional: override the OpenAI-compatible base URL (defaults to `https://api.deepseek.com`):

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

Or pass `--base-url` on the CLI.

Confirm the **`--model`** id with DeepSeek’s documentation (e.g. chat vs reasoning vs flash variants).

### Thinking / flash models (`deepseek-v4-flash`, etc.)

Some DeepSeek endpoints return **`reasoning_content`** and require that field on assistant turns to be sent back verbatim on subsequent API calls (e.g. after tool results). **`search_agent/deepseek_client.py`** keeps `reasoning_content` on the message history for that reason. If you ever see `The reasoning_content in the thinking mode must be passed back to the API`, you are hitting this contract—the client must not strip reasoning before the next completion request.

### Using a conda environment named `bcp`

If you install project dependencies into conda env `bcp`, activate it and point `uv` at that env so packages land in conda instead of a repo-local `.venv`:

```bash
conda activate bcp
cd /path/to/BrowseComp-Plus
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
uv sync
```

Installing everything (especially **vLLM**, **torch**, optional **flash-attn**) can be large and slow — run `uv sync` when you are ready.

## BM25

```bash
python search_agent/deepseek_client.py \
  --model deepseek-chat \
  --output-dir runs/bm25/deepseek/ \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --num-threads 10
```

Adjust `--model`, `--output-dir`, `--num-threads`, and `--max_tokens` as needed.

Evaluate trajectories (requires a **GPU** for the local Qwen3 judge):

```bash
python scripts_evaluation/evaluate_run.py --input_dir runs/bm25/deepseek --tensor_parallel_size 1
```

Replace `--tensor_parallel_size` with your GPU layout.

## Qwen3-Embedding

```bash
python search_agent/deepseek_client.py \
  --model deepseek-chat \
  --output-dir runs/qwen3-8/deepseek/ \
  --searcher-type faiss \
  --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize \
  --num-threads 10
```

You can swap `--model-name` and `--index-path` for other Qwen3-Embedding sizes.

```bash
python scripts_evaluation/evaluate_run.py --input_dir runs/qwen3-8/deepseek --tensor_parallel_size 1
```

## Smoke test (single query)

With indexes and `DEEPSEEK_API_KEY` set, run one query string (no TSV):

```bash
python search_agent/deepseek_client.py \
  --model deepseek-chat \
  --query "Your short test question here" \
  --output-dir runs/smoke/deepseek \
  --searcher-type bm25 \
  --index-path indexes/bm25/
```

This calls the DeepSeek API — use a minimal question if you want to limit spend.
