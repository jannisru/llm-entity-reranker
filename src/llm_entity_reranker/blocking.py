import gc
import os

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from .utils import make_duckdb_connection, parquet_glob, reset_dir


def write_exact_matches(
    left_prepared_dir,
    right_prepared_dir,
    output_parquet,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    if not os.path.exists(left_prepared_dir):
        raise FileNotFoundError(f"Ordner nicht gefunden: {left_prepared_dir}")

    if not os.path.exists(right_prepared_dir):
        raise FileNotFoundError(f"Ordner nicht gefunden: {right_prepared_dir}")

    if os.path.exists(output_parquet):
        os.remove(output_parquet)

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    query = f"""
    COPY (
        SELECT
            l.id AS left_id,
            r.id AS right_id,
            l.key AS key_left,
            r.key AS key_right,
            1.0 AS score,
            'exact' AS match_type
        FROM read_parquet('{parquet_glob(left_prepared_dir)}') l
        JOIN read_parquet('{parquet_glob(right_prepared_dir)}') r
          ON l.normalized_key = r.normalized_key
    )
    TO '{output_parquet}' (
        FORMAT PARQUET,
        COMPRESSION 'zstd'
    );
    """

    con.execute(query)
    con.close()

    return output_parquet


def get_distinct_group_values(
    prepared_dir,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    values = con.execute(f"""
        SELECT DISTINCT group_value
        FROM read_parquet('{parquet_glob(prepared_dir)}')
        WHERE group_value IS NOT NULL
          AND group_value <> ''
        ORDER BY group_value
    """).fetchdf()["group_value"].astype(str).tolist()

    con.close()
    return values


def count_rows_for_group_value(
    prepared_dir,
    group_value,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    n = con.execute(f"""
        SELECT count(*)
        FROM read_parquet('{parquet_glob(prepared_dir)}')
        WHERE group_value = ?
    """, [group_value]).fetchone()[0]

    con.close()
    return int(n)


def load_group_df(
    prepared_dir,
    group_value,
    limit=None,
    offset=0,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    limit_clause = ""
    if limit is not None:
        limit_clause = f"LIMIT {int(limit)} OFFSET {int(offset)}"

    query = f"""
    SELECT
        id,
        key,
        normalized_key,
        group_value,
        match_value,
        token_first,
        token_last,
        token_count,
        value_length
    FROM read_parquet('{parquet_glob(prepared_dir)}')
    WHERE group_value = ?
    {limit_clause}
    """

    df = con.execute(query, [group_value]).fetchdf()
    con.close()
    return df


def build_record_text_df(df):
    token_first = df["token_first"].fillna("").astype(str)
    token_last = df["token_last"].fillna("").astype(str)
    match_value = df["match_value"].fillna("").astype(str).str.replace("_", " ", regex=False)
    group_value = df["group_value"].fillna("").astype(str)
    token_count = df["token_count"].fillna(0).astype(int).astype(str)

    text = (
        "value " + match_value
        + " | first " + token_first
        + " | last " + token_last
        + " | group " + group_value
        + " | tokens " + token_count
    )

    return text.tolist()


def encode_texts(
    model,
    texts,
    batch_size=512,
    normalize_embeddings=True,
):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return emb.astype("float32")


def build_faiss_hnsw_index(
    embeddings,
    m=32,
    ef_construction=200,
    ef_search=128,
):
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)
    return index


def write_candidate_pairs_ann_blocking_by_group(
    left_prepared_dir,
    right_prepared_dir,
    output_parquet,
    work_dir,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=None,
    right_encode_batch_size=1024,
    left_encode_batch_size=1024,
    left_query_chunk_size=100_000,
    top_k=20,
    max_abs_len_diff=2,
    max_token_diff=1,
    hnsw_m=32,
    ef_construction=200,
    ef_search=128,
    normalize_embeddings=True,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    if os.path.exists(output_parquet):
        os.remove(output_parquet)

    reset_dir(work_dir)

    model = SentenceTransformer(model_name, device=device)

    left_groups = set(get_distinct_group_values(
        left_prepared_dir, memory_limit, temp_directory, threads
    ))
    right_groups = set(get_distinct_group_values(
        right_prepared_dir, memory_limit, temp_directory, threads
    ))
    groups = sorted(left_groups.intersection(right_groups))

    part_no = 0
    total_rows_written = 0

    for group_value in groups:
        right_count = count_rows_for_group_value(
            right_prepared_dir, group_value, memory_limit, temp_directory, threads
        )
        left_count = count_rows_for_group_value(
            left_prepared_dir, group_value, memory_limit, temp_directory, threads
        )

        if right_count == 0 or left_count == 0:
            continue

        right_df = load_group_df(
            right_prepared_dir,
            group_value,
            limit=None,
            offset=0,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
            threads=threads,
        )

        if right_df.empty:
            del right_df
            gc.collect()
            continue

        right_texts = build_record_text_df(right_df)
        right_emb = encode_texts(
            model=model,
            texts=right_texts,
            batch_size=right_encode_batch_size,
            normalize_embeddings=normalize_embeddings,
        )

        index = build_faiss_hnsw_index(
            right_emb,
            m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )

        right_ids = right_df["id"].to_numpy()
        right_value_length = right_df["value_length"].to_numpy()
        right_token_count = right_df["token_count"].to_numpy()
        right_normalized_key = right_df["normalized_key"].astype(str).to_numpy()

        del right_texts
        gc.collect()

        for offset in range(0, left_count, left_query_chunk_size):
            left_df = load_group_df(
                left_prepared_dir,
                group_value,
                limit=left_query_chunk_size,
                offset=offset,
                memory_limit=memory_limit,
                temp_directory=temp_directory,
                threads=threads,
            )

            if left_df.empty:
                continue

            left_texts = build_record_text_df(left_df)
            left_emb = encode_texts(
                model=model,
                texts=left_texts,
                batch_size=left_encode_batch_size,
                normalize_embeddings=normalize_embeddings,
            )

            D, I = index.search(left_emb, top_k)

            rows = []
            left_ids = left_df["id"].to_numpy()
            left_value_length = left_df["value_length"].to_numpy()
            left_token_count = left_df["token_count"].to_numpy()
            left_normalized_key = left_df["normalized_key"].astype(str).to_numpy()

            for i in range(len(left_df)):
                l_id = left_ids[i]
                l_len = int(left_value_length[i])
                l_tok = int(left_token_count[i])
                l_norm = left_normalized_key[i]

                seen = set()

                for rank in range(top_k):
                    j = int(I[i, rank])
                    if j < 0:
                        continue

                    r_id = right_ids[j]
                    if r_id in seen:
                        continue
                    seen.add(r_id)

                    if abs(l_len - int(right_value_length[j])) > max_abs_len_diff:
                        continue

                    if abs(l_tok - int(right_token_count[j])) > max_token_diff:
                        continue

                    if l_norm == right_normalized_key[j]:
                        continue

                    rows.append({
                        "left_id": l_id,
                        "right_id": r_id,
                        "blocking_rule": "ann_dense",
                        "ann_rank": int(rank + 1),
                        "ann_distance": float(D[i, rank]),
                        "group_value": str(group_value),
                    })

            if rows:
                cand_df = pd.DataFrame(rows).drop_duplicates(subset=["left_id", "right_id"])
                out_path = os.path.join(work_dir, f"cand_{part_no:06d}.parquet")
                cand_df.to_parquet(out_path, index=False, compression="zstd")
                total_rows_written += len(cand_df)
                part_no += 1

            del left_df, left_texts, left_emb
            gc.collect()

        del right_df, right_emb, index
        gc.collect()

    part_glob = os.path.join(work_dir, "*.parquet")
    part_files_exist = any(fn.endswith(".parquet") for fn in os.listdir(work_dir))

    if not part_files_exist:
        empty_df = pd.DataFrame(columns=[
            "left_id", "right_id", "blocking_rule", "ann_rank", "ann_distance", "group_value"
        ])
        empty_df.to_parquet(output_parquet, index=False, compression="zstd")
        return output_parquet

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    con.execute(f"""
    COPY (
        SELECT
            left_id,
            right_id,
            min(blocking_rule) AS blocking_rule,
            min(ann_rank) AS ann_rank,
            min(ann_distance) AS ann_distance,
            min(group_value) AS group_value
        FROM read_parquet('{part_glob}')
        GROUP BY left_id, right_id
    )
    TO '{output_parquet}' (
        FORMAT PARQUET,
        COMPRESSION 'zstd'
    )
    """)

    con.close()
    return output_parquet


def split_candidates_into_partitions(
    candidates_parquet,
    output_dir,
    num_partitions=256,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    reset_dir(output_dir)

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    for p in range(num_partitions):
        out_path = os.path.join(output_dir, f"part_{p:03d}.parquet")
        con.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet('{candidates_parquet}')
            WHERE abs(hash(left_id)) % {int(num_partitions)} = {int(p)}
        )
        TO '{out_path}' (
            FORMAT PARQUET,
            COMPRESSION 'zstd'
        );
        """)

    con.close()
    return output_dir