# llm-entity-reranker

Entity matching pipeline that combines ANN blocking, fuzzy scoring (Monge-Elkan), sentence-transformer embedding reranking, and optional LLM binary classification via a local Ollama instance.

Built as an extension of [large_scale_entity_matching](https://github.com/jannisru/large_scale_entity_matching), adding semantic embedding and LLM reranking stages on top of the candidate scoring step.

## Pipeline

```
Input CSVs / Parquet
    └─ Preprocessing      normalize keys, tokenize, extract blocking features
    └─ Exact matching     normalized-key equality
    └─ ANN blocking       HNSW index per group (faiss), top-k candidates
    └─ Fuzzy scoring      Monge-Elkan token similarity, threshold filter
    └─ Embedding reranking  sentence-transformers cosine similarity
    └─ LLM reranking      (optional) Ollama binary yes/no classifier
Final matches (Parquet / CSV)
```

## Installation

```bash
pip install llm-entity-reranker
```

## Quick start

```python
from llm_entity_reranker import run_pipeline, MatchingConfig

result = run_pipeline(
    left_input_file="dblp.csv",
    right_input_file="acm.csv",
    left_id_col="id",
    right_id_col="id",
    left_key_cols=["title"],
    right_key_cols=["title"],
    work_dir="/tmp/matching_work",
    config=MatchingConfig(threshold=0.5),
)
```

Add embedding reranking on top of the fuzzy candidates:

```python
from llm_entity_reranker import embed_candidates, rerank_by_embedding

embed_candidates(
    candidates_parquet=result["fuzzy_best_output_parquet"],
    output_parquet="/tmp/embedded.parquet",
)
rerank_by_embedding(
    candidates_parquet="/tmp/embedded.parquet",
    output_parquet="/tmp/reranked.parquet",
    fuzzy_weight=0.3,
    embedding_weight=0.7,
)
```

## Benchmark (DBLP-ACM)

```python
from llm_entity_reranker import run_benchmark, format_results_table

results = run_benchmark()
print(format_results_table(results))
```

| Method | Precision | Recall | F1 | Predicted Pairs |
| --- | --- | --- | --- | --- |
| Fuzzy only (Monge-Elkan) | — | — | — | — |
| Fuzzy + Embedding | — | — | — | — |

> Results will be filled in after running the benchmark.

## Configuration

Key parameters in `MatchingConfig`:

| Parameter | Default | Description |
| --- | --- | --- |
| `model_name` | `all-MiniLM-L6-v2` | Sentence-transformer model for blocking and reranking |
| `threshold` | `0.88` | Minimum fuzzy score to keep a candidate |
| `top_k` | `20` | ANN candidates per left entity |
| `group_strategy` | `last_token` | Blocking group key (`last_token`, `first_token`, `none`) |
| `hnsw_m` | `32` | HNSW graph degree |
| `memory_limit` | `12GB` | DuckDB memory cap |
