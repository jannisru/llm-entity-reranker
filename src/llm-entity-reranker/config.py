from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchingConfig:
    group_strategy: str = "last_token"   # "last_token" | "first_token" | "none"
    prepare_chunk_size: int = 200_000

    
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None

    top_k: int = 20
    left_query_chunk_size: int = 100_000

    right_encode_batch_size: int = 1024
    left_encode_batch_size: int = 1024

    max_abs_len_diff: int = 2
    max_token_diff: int = 1

    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 128

    normalize_embeddings: bool = True

    
    num_candidate_partitions: int = 256


    threshold: float = 0.88
    max_rel_len_diff: float = 0.15


    memory_limit: str = "12GB"
    temp_directory: str = "/tmp/duckdb_tmp"
    threads: int = 2

    progress_every: int = 10