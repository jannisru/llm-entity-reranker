import os

from .config import MatchingConfig
from .preprocessing import (
    prepare_input_file,
    prepare_blocking_features,
)
from .blocking import (
    write_exact_matches,
    write_candidate_pairs_ann_blocking_by_group,
    split_candidates_into_partitions,
)
from .scoring import score_candidate_partitions
from .merging import (
    keep_best_ties_from_parts,
    merge_exact_and_fuzzy,
)


def run_pipeline(
    left_input_file,
    right_input_file,
    left_id_col,
    right_id_col,
    left_key_cols,
    right_key_cols,
    work_dir,
    config: MatchingConfig = MatchingConfig(),
    left_parquet_filename="left_input.parquet",
    right_parquet_filename="right_input.parquet",
    left_id_key_filename="left_id_key.parquet",
    right_id_key_filename="right_id_key.parquet",
    final_output_parquet=None,
    final_output_csv=None,
):
    """
    Gesamte generische Matching-Pipeline.

    Ablauf:
    1. Inputs in Parquet umwandeln + id/key Datei erzeugen
    2. generische Blocking-Features erzeugen
    3. Exact Matches auf normalized_key
    4. ANN Candidate Generation innerhalb von group_value
    5. Candidate Partitioning
    6. Fuzzy Scoring
    7. Beste fuzzy Matches pro left_id behalten
    8. Exact + Fuzzy mergen
    """

    os.makedirs(work_dir, exist_ok=True)

    left_work_dir = os.path.join(work_dir, "left")
    right_work_dir = os.path.join(work_dir, "right")

    left_prepared_dir = os.path.join(left_work_dir, "prepared")
    right_prepared_dir = os.path.join(right_work_dir, "prepared")

    exact_output_parquet = os.path.join(work_dir, "exact_matches.parquet")
    ann_candidates_output_parquet = os.path.join(work_dir, "ann_candidates.parquet")
    ann_work_dir = os.path.join(work_dir, "ann_candidate_parts_raw")
    candidate_parts_dir = os.path.join(work_dir, "candidate_parts")
    fuzzy_parts_dir = os.path.join(work_dir, "fuzzy_parts")
    fuzzy_best_output_parquet = os.path.join(work_dir, "fuzzy_best.parquet")

    if final_output_parquet is None:
        final_output_parquet = os.path.join(work_dir, "final_matches.parquet")

    left_info = prepare_input_file(
        input_file=left_input_file,
        id_col=left_id_col,
        key_cols=left_key_cols,
        output_dir=left_work_dir,
        parquet_filename=left_parquet_filename,
        id_key_filename=left_id_key_filename,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    right_info = prepare_input_file(
        input_file=right_input_file,
        id_col=right_id_col,
        key_cols=right_key_cols,
        output_dir=right_work_dir,
        parquet_filename=right_parquet_filename,
        id_key_filename=right_id_key_filename,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    prepare_blocking_features(
        id_key_parquet=left_info["id_key_file"],
        output_dir=left_prepared_dir,
        chunk_size=config.prepare_chunk_size,
        group_strategy=config.group_strategy,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    prepare_blocking_features(
        id_key_parquet=right_info["id_key_file"],
        output_dir=right_prepared_dir,
        chunk_size=config.prepare_chunk_size,
        group_strategy=config.group_strategy,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    write_exact_matches(
        left_prepared_dir=left_prepared_dir,
        right_prepared_dir=right_prepared_dir,
        output_parquet=exact_output_parquet,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    write_candidate_pairs_ann_blocking_by_group(
        left_prepared_dir=left_prepared_dir,
        right_prepared_dir=right_prepared_dir,
        output_parquet=ann_candidates_output_parquet,
        work_dir=ann_work_dir,
        model_name=config.model_name,
        device=config.device,
        right_encode_batch_size=config.right_encode_batch_size,
        left_encode_batch_size=config.left_encode_batch_size,
        left_query_chunk_size=config.left_query_chunk_size,
        top_k=config.top_k,
        max_abs_len_diff=config.max_abs_len_diff,
        max_token_diff=config.max_token_diff,
        hnsw_m=config.hnsw_m,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
        normalize_embeddings=config.normalize_embeddings,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    split_candidates_into_partitions(
        candidates_parquet=ann_candidates_output_parquet,
        output_dir=candidate_parts_dir,
        num_partitions=config.num_candidate_partitions,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    score_info = score_candidate_partitions(
        candidate_parts_dir=candidate_parts_dir,
        left_prepared_dir=left_prepared_dir,
        right_prepared_dir=right_prepared_dir,
        output_parts_dir=fuzzy_parts_dir,
        threshold=config.threshold,
        max_rel_len_diff=config.max_rel_len_diff,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
        progress_every=config.progress_every,
    )

    keep_best_ties_from_parts(
        fuzzy_parts_dir=fuzzy_parts_dir,
        fuzzy_best_output_parquet=fuzzy_best_output_parquet,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    merge_exact_and_fuzzy(
        exact_parquet=exact_output_parquet,
        fuzzy_best_parquet=fuzzy_best_output_parquet,
        final_output_parquet=final_output_parquet,
        final_output_csv=final_output_csv,
        memory_limit=config.memory_limit,
        temp_directory=config.temp_directory,
        threads=config.threads,
    )

    return {
        "left_input_info": left_info,
        "right_input_info": right_info,
        "left_prepared_dir": left_prepared_dir,
        "right_prepared_dir": right_prepared_dir,
        "exact_output_parquet": exact_output_parquet,
        "ann_candidates_output_parquet": ann_candidates_output_parquet,
        "ann_work_dir": ann_work_dir,
        "candidate_parts_dir": candidate_parts_dir,
        "fuzzy_parts_dir": fuzzy_parts_dir,
        "fuzzy_best_output_parquet": fuzzy_best_output_parquet,
        "final_output_parquet": final_output_parquet,
        "final_output_csv": final_output_csv,
        "score_info": score_info,
        "group_strategy": config.group_strategy,
        "config": config,
    }


def run_pipeline_only_result(
    left_input_file,
    right_input_file,
    left_id_col,
    right_id_col,
    left_key_cols,
    right_key_cols,
    work_dir,
    config: MatchingConfig = MatchingConfig(),
    left_parquet_filename="left_input.parquet",
    right_parquet_filename="right_input.parquet",
    left_id_key_filename="left_id_key.parquet",
    right_id_key_filename="right_id_key.parquet",
    final_output_parquet=None,
    final_output_csv=None,
):
    """
    Führt die gesamte Pipeline aus und gibt nur den finalen Output-Pfad zurück.
    """

    result = run_pipeline(
        left_input_file=left_input_file,
        right_input_file=right_input_file,
        left_id_col=left_id_col,
        right_id_col=right_id_col,
        left_key_cols=left_key_cols,
        right_key_cols=right_key_cols,
        work_dir=work_dir,
        config=config,
        left_parquet_filename=left_parquet_filename,
        right_parquet_filename=right_parquet_filename,
        left_id_key_filename=left_id_key_filename,
        right_id_key_filename=right_id_key_filename,
        final_output_parquet=final_output_parquet,
        final_output_csv=final_output_csv,
    )

    if final_output_csv is not None:
        return result["final_output_csv"]

    return result["final_output_parquet"]