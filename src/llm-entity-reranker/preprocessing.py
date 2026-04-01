import os
import re
import pandas as pd

from .utils import make_duckdb_connection


_multi_us_regex = re.compile(r"_+")


def normalize_key(s):
    if s is None or pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = _multi_us_regex.sub("_", s)
    s = s.strip("_")
    return s


def tokenize_normalized_name(s):
    if s is None or pd.isna(s):
        return []
    s = str(s).strip("_")
    if not s:
        return []
    return [tok for tok in s.split("_") if tok]


def recognize_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return "csv"
    if ext in [".xls", ".xlsx"]:
        return "excel"
    if ext == ".parquet":
        return "parquet"

    raise ValueError(f"Unsupported file type: {file_path}")


def convert_file_to_parquet(input_path, output_path=None):
    file_type = recognize_file_type(input_path)

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.parquet"

    if file_type == "parquet":
        return input_path

    if file_type == "csv":
        df = pd.read_csv(input_path)
    elif file_type == "excel":
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path}")

    df.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def convert_inputs_to_parquet(file_a, file_b, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_a = os.path.join(output_dir, "left_input.parquet")
        out_b = os.path.join(output_dir, "right_input.parquet")
    else:
        out_a = None
        out_b = None

    parquet_a = convert_file_to_parquet(file_a, out_a)
    parquet_b = convert_file_to_parquet(file_b, out_b)

    return parquet_a, parquet_b


def create_id_key_file(
    parquet_file,
    id_col,
    list_of_cols,
    output_parquet=None,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Datei nicht gefunden: {parquet_file}")

    if not list_of_cols:
        raise ValueError("list_of_cols darf nicht leer sein")

    if output_parquet is None:
        base, _ = os.path.splitext(parquet_file)
        output_parquet = f"{base}_id_key.parquet"

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    existing_cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_file}')"
    ).fetchdf()["column_name"].tolist()

    required_cols = [id_col] + list_of_cols
    missing_cols = [col for col in required_cols if col not in existing_cols]
    if missing_cols:
        con.close()
        raise ValueError(f"Fehlende Spalten in {parquet_file}: {missing_cols}")

    raw_key_expr = ", ".join([
        f"lower(trim(coalesce(cast(\"{col}\" as varchar), '')))"
        for col in list_of_cols
    ])

    query = f"""
    COPY (
        WITH src AS (
            SELECT
                "{id_col}" AS id,
                concat_ws('_', {raw_key_expr}) AS raw_key
            FROM read_parquet('{parquet_file}')
        ),
        norm1 AS (
            SELECT
                id,
                lower(trim(raw_key)) AS k0
            FROM src
        ),
        norm2 AS (
            SELECT
                id,
                regexp_replace(k0, '[^a-z0-9]+', '_', 'g') AS k1
            FROM norm1
        ),
        norm3 AS (
            SELECT
                id,
                regexp_replace(k1, '_+', '_', 'g') AS k2
            FROM norm2
        )
        SELECT
            id,
            trim(both '_' from k2) AS key
        FROM norm3
    )
    TO '{output_parquet}' (
        FORMAT PARQUET,
        COMPRESSION 'zstd'
    )
    """

    con.execute(query)
    con.close()

    return output_parquet


def prepare_input_file(
    input_file,
    id_col,
    key_cols,
    output_dir=None,
    parquet_filename=None,
    id_key_filename=None,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    if not key_cols:
        raise ValueError("key_cols must not be empty")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    file_type = recognize_file_type(input_file)

    if file_type == "parquet":
        parquet_file = input_file
    else:
        if output_dir is not None:
            if parquet_filename is None:
                parquet_filename = "input.parquet"
            parquet_output_path = os.path.join(output_dir, parquet_filename)
        else:
            base, _ = os.path.splitext(input_file)
            parquet_output_path = f"{base}.parquet"

        parquet_file = convert_file_to_parquet(
            input_path=input_file,
            output_path=parquet_output_path,
        )

    if output_dir is not None:
        if id_key_filename is None:
            id_key_filename = "id_key.parquet"
        id_key_output_path = os.path.join(output_dir, id_key_filename)
    else:
        base, _ = os.path.splitext(parquet_file)
        id_key_output_path = f"{base}_id_key.parquet"

    id_key_file = create_id_key_file(
        parquet_file=parquet_file,
        id_col=id_col,
        list_of_cols=key_cols,
        output_parquet=id_key_output_path,
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    return {
        "input_file": input_file,
        "parquet_file": parquet_file,
        "id_key_file": id_key_file,
    }



def prepare_blocking_features(
    id_key_parquet,
    output_dir,
    chunk_size=200_000,
    group_strategy="last_token",
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    
    if not os.path.exists(id_key_parquet):
        raise FileNotFoundError(f"File not found: {id_key_parquet}")

    if group_strategy not in {"last_token", "first_token", "none"}:
        raise ValueError(
            "group_strategy must be in {'last_token', 'first_token', 'none'}"
        )

    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".parquet"):
            os.remove(os.path.join(output_dir, f))

    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )

    existing_cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{id_key_parquet}')"
    ).fetchdf()["column_name"].tolist()

    required_cols = ["id", "key"]
    missing_cols = [col for col in required_cols if col not in existing_cols]
    if missing_cols:
        con.close()
        raise ValueError(f"Missing coulumn in {id_key_parquet}: {missing_cols}")

    total_rows = con.execute(f"""
        SELECT count(*) FROM read_parquet('{id_key_parquet}')
    """).fetchone()[0]

    offset = 0
    part = 0

    while offset < total_rows:
        if group_strategy == "last_token":
            group_value_expr = "regexp_extract(normalized_key, '([^_]+)$', 1)"
            match_value_expr = "regexp_replace(normalized_key, '_?([^_]+)$', '', 'g')"
        elif group_strategy == "first_token":
            group_value_expr = "regexp_extract(normalized_key, '^([^_]+)', 1)"
            match_value_expr = "regexp_replace(normalized_key, '^([^_]+)_?', '', 'g')"
        else:
            group_value_expr = "''"
            match_value_expr = "normalized_key"

        query = f"""
        SELECT
            id,
            key,
            normalized_key,
            group_value,
            match_value,
            key_length,
            value_length,
            token_count,
            token_first,
            token_last,

            substr(match_value, 1, 1) AS prefix_1,
            substr(match_value, 1, 2) AS prefix_2,
            substr(match_value, 1, 3) AS prefix_3,

            right(match_value, 1) AS suffix_1,
            right(match_value, 2) AS suffix_2,
            right(match_value, 3) AS suffix_3,

            substr(token_first, 1, 1) AS first_prefix_1,
            substr(token_first, 1, 2) AS first_prefix_2,
            substr(token_first, 1, 3) AS first_prefix_3,

            substr(token_last, 1, 1) AS last_prefix_1,
            substr(token_last, 1, 2) AS last_prefix_2,
            substr(token_last, 1, 3) AS last_prefix_3,

            right(token_last, 1) AS last_suffix_1,
            right(token_last, 2) AS last_suffix_2,
            right(token_last, 3) AS last_suffix_3,

            CASE
                WHEN length(match_value) < 8 THEN 'L1'
                WHEN length(match_value) < 12 THEN 'L2'
                WHEN length(match_value) < 16 THEN 'L3'
                WHEN length(match_value) < 22 THEN 'L4'
                ELSE 'L5'
            END AS length_bucket
        FROM (
            WITH src AS (
                SELECT
                    id,
                    cast(key AS varchar) AS key
                FROM read_parquet('{id_key_parquet}')
                WHERE key IS NOT NULL
                LIMIT {int(chunk_size)}
                OFFSET {int(offset)}
            ),
            norm1 AS (
                SELECT
                    id,
                    key,
                    lower(trim(key)) AS k0
                FROM src
            ),
            norm2 AS (
                SELECT
                    id,
                    key,
                    regexp_replace(k0, '[^a-z0-9]+', '_', 'g') AS k1
                FROM norm1
            ),
            norm3 AS (
                SELECT
                    id,
                    key,
                    regexp_replace(k1, '_+', '_', 'g') AS k2
                FROM norm2
            ),
            norm4 AS (
                SELECT
                    id,
                    key,
                    trim(both '_' from k2) AS normalized_key
                FROM norm3
            ),
            parsed AS (
                SELECT
                    id,
                    key,
                    normalized_key,
                    {group_value_expr} AS group_value,
                    trim(both '_' from {match_value_expr}) AS match_value,
                    length(normalized_key) AS key_length
                FROM norm4
                WHERE normalized_key <> ''
            ),
            feats AS (
                SELECT
                    id,
                    key,
                    normalized_key,
                    group_value,
                    match_value,
                    key_length,
                    length(match_value) AS value_length
                FROM parsed
                WHERE match_value IS NOT NULL
                  AND trim(both '_' from match_value) <> ''
            ),
            tok AS (
                SELECT
                    *,
                    regexp_extract(match_value, '^([^_]+)', 1) AS token_first,
                    regexp_extract(match_value, '([^_]+)$', 1) AS token_last,
                    array_length(string_split(match_value, '_')) AS token_count
                FROM feats
            )
            SELECT *
            FROM tok
        ) t
        """

        df = con.execute(query).fetchdf()

        if df.empty:
            break

        out_path = os.path.join(output_dir, f"part_{part:05d}.parquet")
        df.to_parquet(out_path, index=False, compression="zstd")

        offset += chunk_size
        part += 1

    con.close()
    return output_dir