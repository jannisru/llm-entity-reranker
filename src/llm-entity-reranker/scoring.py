import gc
from difflib import SequenceMatcher

import pandas as pd

from .preprocessing import tokenize_normalized_name
from .utils import make_duckdb_connection, parquet_glob, reset_dir


def load_candidate_partition(
    con,
    candidate_partition_parquet,
    left_prepared_dir,
    right_prepared_dir,
):
    query = f"""
    SELECT
        c.left_id,
        c.right_id,
        c.blocking_rule,

        l.key AS key_left,
        r.key AS key_right,

        l.match_value AS match_value_left,
        r.match_value AS match_value_right,

        l.value_length AS value_length_left,
        r.value_length AS value_length_right,

        l.token_count AS token_count_left,
        r.token_count AS token_count_right,

        l.token_first AS token_first_left,
        r.token_first AS token_first_right,

        l.token_last AS token_last_left,
        r.token_last AS token_last_right
    FROM read_parquet('{candidate_partition_parquet}') c
    JOIN read_parquet('{parquet_glob(left_prepared_dir)}') l
      ON c.left_id = l.id
    JOIN read_parquet('{parquet_glob(right_prepared_dir)}') r
      ON c.right_id = r.id
    """
    return con.execute(query).fetchdf()


def token_similarity_cached(a, b, sim_cache):
    if not a or not b:
        return 0.0

    key = (a, b) if a <= b else (b, a)
    if key in sim_cache:
        return sim_cache[key]

    score = SequenceMatcher(None, a, b).ratio()
    sim_cache[key] = score
    return score


def monge_elkan_similarity_cached(tokens1, tokens2, sim_cache):
    if not tokens1 or not tokens2:
        return 0.0

    best_scores = []
    for t1 in tokens1:
        best = max(token_similarity_cached(t1, t2, sim_cache) for t2 in tokens2)
        best_scores.append(best)

    return sum(best_scores) / len(best_scores)


def symmetric_monge_elkan_similarity_cached(tokens1, tokens2, sim_cache):
    if not tokens1 or not tokens2:
        return 0.0

    s12 = monge_elkan_similarity_cached(tokens1, tokens2, sim_cache)
    s21 = monge_elkan_similarity_cached(tokens2, tokens1, sim_cache)
    return (s12 + s21) / 2.0


def score_candidate_batch_optimized(
    df,
    threshold=0.88,
    max_rel_len_diff=0.15,
):
    if df.empty:
        return pd.DataFrame(columns=[
            "left_id", "right_id", "key_left", "key_right", "score", "match_type"
        ])

    unique_match_values = pd.unique(
        pd.concat(
            [df["match_value_left"], df["match_value_right"]],
            ignore_index=True,
        )
    )

    token_cache = {
        s: tokenize_normalized_name(s)
        for s in unique_match_values
    }

    token_sim_cache = {}
    pair_score_cache = {}
    rows = []

    for row in df.itertuples(index=False):
        max_len = max(row.value_length_left, row.value_length_right)
        if max_len <= 0:
            continue

        rel_len_diff = abs(row.value_length_left - row.value_length_right) / max_len
        if rel_len_diff > max_rel_len_diff:
            continue

        if abs(row.token_count_left - row.token_count_right) > 1:
            continue

        if row.token_first_left and row.token_first_right:
            if row.token_first_left[0] != row.token_first_right[0]:
                if row.token_last_left and row.token_last_right:
                    if row.token_last_left[0] != row.token_last_right[0]:
                        continue

        a = row.match_value_left
        b = row.match_value_right
        pair_key = (a, b) if a <= b else (b, a)

        if pair_key in pair_score_cache:
            score = pair_score_cache[pair_key]
        else:
            score = symmetric_monge_elkan_similarity_cached(
                token_cache[a],
                token_cache[b],
                token_sim_cache,
            )
            pair_score_cache[pair_key] = score

        if score >= threshold:
            rows.append({
                "left_id": row.left_id,
                "right_id": row.right_id,
                "key_left": row.key_left,
                "key_right": row.key_right,
                "score": score,
                "match_type": row.blocking_rule,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "left_id", "right_id", "key_left", "key_right", "score", "match_type"
        ])

    return pd.DataFrame(rows)


def score_candidate_partitions(
    candidate_parts_dir,
    left_prepared_dir,
    right_prepared_dir,
    output_parts_dir,
    threshold=0.88,
    max_rel_len_diff=0.15,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
    progress_every=10,
):
    reset_dir(output_parts_dir)

    part_files = sorted([
        f"{candidate_parts_dir}/{f}"
        for f in __import__("os").listdir(candidate_parts_dir)
        if f.endswith(".parquet")
    ])

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    rows_written = 0
    result_files = 0
    candidate_pairs = 0

    for i, part_path in enumerate(part_files, start=1):
        part_count = con.execute(
            f"SELECT count(*) FROM read_parquet('{part_path}')"
        ).fetchone()[0]
        candidate_pairs += int(part_count)

        if part_count == 0:
            if i % progress_every == 0 or i == len(part_files):
                print(
                    f"Scored partitions {i}/{len(part_files)}, "
                    f"candidate_pairs={candidate_pairs}, "
                    f"rows_written={rows_written}, result_files={result_files}"
                )
            continue

        cand_df = load_candidate_partition(
            con=con,
            candidate_partition_parquet=part_path,
            left_prepared_dir=left_prepared_dir,
            right_prepared_dir=right_prepared_dir,
        )

        scored_df = score_candidate_batch_optimized(
            cand_df,
            threshold=threshold,
            max_rel_len_diff=max_rel_len_diff,
        )

        if not scored_df.empty:
            out_path = f"{output_parts_dir}/scored_{i:03d}.parquet"
            scored_df.to_parquet(out_path, index=False, compression="zstd")
            rows_written += len(scored_df)
            result_files += 1

        del cand_df, scored_df
        gc.collect()

        if i % progress_every == 0 or i == len(part_files):
            print(
                f"Scored partitions {i}/{len(part_files)}, "
                f"candidate_pairs={candidate_pairs}, "
                f"rows_written={rows_written}, result_files={result_files}"
            )

    con.close()

    return {
        "candidate_pairs": candidate_pairs,
        "rows_written": rows_written,
        "result_files": result_files,
    }