from .config import MatchingConfig
from .pipeline import run_pipeline, run_pipeline_only_result

from .preprocessing import (
    normalize_key,
    tokenize_normalized_name,
    convert_file_to_parquet,
    convert_inputs_to_parquet,
    create_id_key_file,
    prepare_input_file,
    prepare_blocking_features,
)

from .blocking import (
    write_exact_matches,
    write_candidate_pairs_ann_blocking_by_group,
    split_candidates_into_partitions,
)

from .scoring import (
    load_candidate_partition,
    score_candidate_batch_optimized,
    score_candidate_partitions,
)

from .merging import (
    keep_best_ties_from_parts,
    merge_exact_and_fuzzy,
)

from .embedder import embed_candidates

from .reranker import (
    rerank_by_embedding,
    rerank_with_ollama,
)

from .benchmark import (
    download_dblp_acm,
    compute_metrics,
    run_benchmark,
    format_results_table,
)