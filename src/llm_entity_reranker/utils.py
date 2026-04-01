import os
import shutil
import duckdb

def make_duckdb_connection(memory_limit="12GB",
                           temp_directory="/tmp/duckdb_tmp",
                           threads=2):
    os.makedirs(temp_directory, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{temp_directory}'")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    con.execute("PRAGMA disable_progress_bar")
    return con


def reset_dir(path):
    os.makedirs(path, exist_ok=True)
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp):
            os.remove(fp)
        else:
            shutil.rmtree(fp)


def parquet_glob(path_or_dir):
    if os.path.isfile(path_or_dir):
        return path_or_dir
    return os.path.join(path_or_dir, "*.parquet")


def count_rows_parquet(
    parquet_path,
    memory_limit="12GB",
    temp_directory="/tmp/duckdb_tmp",
    threads=2,
):
    con = make_duckdb_connection(
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        threads=threads,
    )
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]
    con.close()
    return int(n)