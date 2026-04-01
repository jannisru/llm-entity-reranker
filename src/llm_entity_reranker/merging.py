import os
import pandas as pd

from .utils import make_duckdb_connection


def keep_best_ties_from_parts(
    fuzzy_parts_dir,
    fuzzy_best_output_parquet,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    if os.path.exists(fuzzy_best_output_parquet):
        os.remove(fuzzy_best_output_parquet)

    parquet_files = [
        os.path.join(fuzzy_parts_dir, f)
        for f in os.listdir(fuzzy_parts_dir)
        if f.endswith(".parquet")
    ]

    if not parquet_files:
        empty_df = pd.DataFrame(columns=[
            "left_id", "right_id", "key_left", "key_right", "score", "match_type"
        ])
        empty_df.to_parquet(fuzzy_best_output_parquet, index=False, compression="zstd")
        return fuzzy_best_output_parquet

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    query = f"""
    COPY (
        WITH all_scored AS (
            SELECT *
            FROM read_parquet('{os.path.join(fuzzy_parts_dir, "*.parquet")}')
        ),
        best_per_left AS (
            SELECT
                left_id,
                max(score) AS best_score
            FROM all_scored
            GROUP BY left_id
        )
        SELECT
            a.left_id,
            a.right_id,
            a.key_left,
            a.key_right,
            a.score,
            a.match_type
        FROM all_scored a
        JOIN best_per_left b
          ON a.left_id = b.left_id
         AND a.score = b.best_score
    )
    TO '{fuzzy_best_output_parquet}' (
        FORMAT PARQUET,
        COMPRESSION 'zstd'
    );
    """

    con.execute(query)
    con.close()

    return fuzzy_best_output_parquet


def merge_exact_and_fuzzy(
    exact_parquet,
    fuzzy_best_parquet,
    final_output_parquet,
    final_output_csv=None,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    if not os.path.exists(exact_parquet):
        raise FileNotFoundError(f"Datei nicht gefunden: {exact_parquet}")

    if not os.path.exists(fuzzy_best_parquet):
        raise FileNotFoundError(f"Datei nicht gefunden: {fuzzy_best_parquet}")

    if os.path.exists(final_output_parquet):
        os.remove(final_output_parquet)

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('{exact_parquet}')
        UNION ALL
        SELECT * FROM read_parquet('{fuzzy_best_parquet}')
    )
    TO '{final_output_parquet}' (
        FORMAT PARQUET,
        COMPRESSION 'zstd'
    );
    """)

    if final_output_csv is not None:
        if os.path.exists(final_output_csv):
            os.remove(final_output_csv)

        con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{final_output_parquet}')
        )
        TO '{final_output_csv}' (
            HEADER,
            DELIMITER ','
        );
        """)

    con.close()
    return final_output_parquet