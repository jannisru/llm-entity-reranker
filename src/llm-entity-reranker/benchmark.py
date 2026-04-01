"""
Benchmark: runs fuzzy-only and fuzzy+embedding matching over DBLP-ACM and
computes Precision, Recall, F1 for both variants.

DBLP-ACM dataset (Magellan / DeepMatcher format):
  tableA.csv  — DBLP records   (id, title, authors, venue, year)
  tableB.csv  — ACM  records   (id, title, authors, venue, year)
  matches.csv — ground truth   (ltable_id, rtable_id)

Download URLs are provided as defaults; pass explicit file paths to skip download.
"""
import os
import urllib.request

import pandas as pd

from .config import MatchingConfig
from .pipeline import run_pipeline
from .embedder import embed_candidates
from .reranker import rerank_by_embedding


# Public URLs for the DBLP-ACM structured dataset (DeepMatcher repository)
DBLP_ACM_URLS = {
    "dblp": (
        "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/"
        "master/Datasets/Structured/DBLP-ACM/tableA.csv"
    ),
    "acm": (
        "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/"
        "master/Datasets/Structured/DBLP-ACM/tableB.csv"
    ),
    "matches": (
        "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/"
        "master/Datasets/Structured/DBLP-ACM/matches.csv"
    ),
}


def download_dblp_acm(data_dir):
    """
    Downloads the DBLP-ACM dataset into data_dir if not already present.

    Returns
    -------
    dict
        {'dblp': path, 'acm': path, 'matches': path}
    """
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for name, url in DBLP_ACM_URLS.items():
        dest = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(dest):
            print(f"  Downloading {name}.csv ...")
            urllib.request.urlretrieve(url, dest)
        paths[name] = dest
    return paths


def compute_metrics(predicted_pairs, true_pairs):
    """
    Computes Precision, Recall, and F1 for a set of predicted entity pairs.

    Parameters
    ----------
    predicted_pairs : set of (left_id, right_id)
    true_pairs      : set of (left_id, right_id)

    Returns
    -------
    dict with keys: precision, recall, f1, tp, fp, fn
    """
    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def run_benchmark(
    data_dir="/tmp/dblp_acm_data",
    work_dir="/tmp/dblp_acm_work",
    dblp_file=None,
    acm_file=None,
    matches_file=None,
    left_id_col="id",
    right_id_col="id",
    left_key_cols=None,
    right_key_cols=None,
    left_match_id_col="ltable_id",
    right_match_id_col="rtable_id",
    config=None,
    fuzzy_weight=0.3,
    embedding_weight=0.7,
):
    """
    Runs the full benchmark and returns metrics for both variants.

    The pipeline is executed once; embedding and reranking are then applied
    on top of the fuzzy results without re-running the expensive blocking stage.

    Parameters
    ----------
    data_dir : str
        Directory where the downloaded DBLP-ACM CSV files are cached.
    work_dir : str
        Working directory for intermediate pipeline files.
    dblp_file, acm_file, matches_file : str or None
        Local file paths. If None, files are downloaded automatically.
    left_id_col, right_id_col : str
        Primary key column names in the input CSVs.
    left_key_cols, right_key_cols : list of str
        Columns used to build the matching key (default: ['title']).
    left_match_id_col, right_match_id_col : str
        Column names in matches_file that refer to left/right IDs.
    config : MatchingConfig or None
        Pipeline configuration. Defaults to MatchingConfig(threshold=0.5).
    fuzzy_weight : float
        Weight for fuzzy score in combined reranking score.
    embedding_weight : float
        Weight for embedding score in combined reranking score.

    Returns
    -------
    dict
        {
          'fuzzy_only':           {precision, recall, f1, tp, fp, fn},
          'fuzzy_plus_embedding': {precision, recall, f1, tp, fp, fn},
          'ground_truth_size':    int,
          'fuzzy_predictions':    int,
          'embedding_predictions': int,
        }
    """
    if left_key_cols is None:
        left_key_cols = ["title"]
    if right_key_cols is None:
        right_key_cols = ["title"]
    if config is None:
        config = MatchingConfig(threshold=0.5)

    # Resolve input files
    if any(p is None for p in (dblp_file, acm_file, matches_file)):
        print("Downloading DBLP-ACM dataset ...")
        paths = download_dblp_acm(data_dir)
        dblp_file    = dblp_file    or paths["dblp"]
        acm_file     = acm_file     or paths["acm"]
        matches_file = matches_file or paths["matches"]

    # Load ground truth
    gt_df = pd.read_csv(matches_file)
    true_pairs = set(
        zip(
            gt_df[left_match_id_col].astype(str),
            gt_df[right_match_id_col].astype(str),
        )
    )
    print(f"Ground truth: {len(true_pairs)} matching pairs")

    # --- Run pipeline (blocking + fuzzy scoring) ---
    os.makedirs(work_dir, exist_ok=True)
    print("Running matching pipeline ...")
    pipeline_result = run_pipeline(
        left_input_file=dblp_file,
        right_input_file=acm_file,
        left_id_col=left_id_col,
        right_id_col=right_id_col,
        left_key_cols=left_key_cols,
        right_key_cols=right_key_cols,
        work_dir=work_dir,
        config=config,
    )

    # --- Evaluate fuzzy-only ---
    fuzzy_df = pd.read_parquet(pipeline_result["final_output_parquet"])
    fuzzy_pairs = set(
        zip(fuzzy_df["left_id"].astype(str), fuzzy_df["right_id"].astype(str))
    )
    fuzzy_metrics = compute_metrics(fuzzy_pairs, true_pairs)

    # --- Embed fuzzy candidates and rerank ---
    print("Embedding candidates ...")
    embedded_parquet = os.path.join(work_dir, "embedded_candidates.parquet")
    embed_candidates(
        candidates_parquet=pipeline_result["fuzzy_best_output_parquet"],
        output_parquet=embedded_parquet,
        model_name=config.model_name,
        device=config.device,
    )

    print("Reranking ...")
    reranked_parquet = os.path.join(work_dir, "reranked_candidates.parquet")
    rerank_by_embedding(
        candidates_parquet=embedded_parquet,
        output_parquet=reranked_parquet,
        fuzzy_weight=fuzzy_weight,
        embedding_weight=embedding_weight,
    )

    # Combine reranked fuzzy with exact matches (exact matches are always correct)
    exact_df    = pd.read_parquet(pipeline_result["exact_output_parquet"])
    reranked_df = pd.read_parquet(reranked_parquet)
    combined_df = pd.concat([exact_df, reranked_df], ignore_index=True)
    combined_pairs = set(
        zip(combined_df["left_id"].astype(str), combined_df["right_id"].astype(str))
    )
    embedding_metrics = compute_metrics(combined_pairs, true_pairs)

    return {
        "fuzzy_only":            fuzzy_metrics,
        "fuzzy_plus_embedding":  embedding_metrics,
        "ground_truth_size":     len(true_pairs),
        "fuzzy_predictions":     len(fuzzy_pairs),
        "embedding_predictions": len(combined_pairs),
    }


def format_results_table(results):
    """
    Formats benchmark results as a GitHub-flavoured Markdown table.

    Parameters
    ----------
    results : dict
        Return value of run_benchmark().

    Returns
    -------
    str
        Markdown table string.
    """
    header = "| Method | Precision | Recall | F1 | Predicted Pairs |"
    sep    = "| --- | --- | --- | --- | --- |"

    def row(label, metrics, n_pred):
        return (
            f"| {label} "
            f"| {metrics['precision']:.3f} "
            f"| {metrics['recall']:.3f} "
            f"| {metrics['f1']:.3f} "
            f"| {n_pred} |"
        )

    lines = [
        header,
        sep,
        row("Fuzzy only (Monge-Elkan)", results["fuzzy_only"],           results["fuzzy_predictions"]),
        row("Fuzzy + Embedding",         results["fuzzy_plus_embedding"], results["embedding_predictions"]),
    ]
    return "\n".join(lines)
