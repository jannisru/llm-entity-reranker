"""
CLI script: compare Fuzzy-only vs Fuzzy+Embedding entity matching on DBLP-ACM.

Usage
-----
    # run as module (recommended when package is installed)
    python -m llm_entity_reranker.compare

    # with options
    python -m llm_entity_reranker.compare \\
        --threshold 0.6 \\
        --key-cols title authors \\
        --work-dir /tmp/my_benchmark

    # with local files (skip download)
    python -m llm_entity_reranker.compare \\
        --dblp-file /data/tableA.csv \\
        --acm-file  /data/tableB.csv \\
        --matches-file /data/matches.csv
"""
import argparse
import sys

from .benchmark import run_benchmark, format_results_table
from .config import MatchingConfig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare Fuzzy-only vs Fuzzy+Embedding entity matching on DBLP-ACM "
            "and print Precision / Recall / F1."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data locations
    parser.add_argument(
        "--data-dir", default="/tmp/dblp_acm_data",
        help="Directory where downloaded DBLP-ACM CSV files are cached.",
    )
    parser.add_argument(
        "--work-dir", default="/tmp/dblp_acm_work",
        help="Working directory for pipeline intermediate files.",
    )
    parser.add_argument("--dblp-file",    default=None, help="Local path to DBLP CSV (skips download).")
    parser.add_argument("--acm-file",     default=None, help="Local path to ACM CSV (skips download).")
    parser.add_argument("--matches-file", default=None, help="Local path to ground truth CSV (skips download).")

    # Ground truth column names
    parser.add_argument(
        "--left-match-id-col", default="ltable_id",
        help="Column in matches CSV referring to left (DBLP) record IDs.",
    )
    parser.add_argument(
        "--right-match-id-col", default="rtable_id",
        help="Column in matches CSV referring to right (ACM) record IDs.",
    )

    # Matching options
    parser.add_argument(
        "--key-cols", nargs="+", default=["title"],
        help="Column(s) used to build the matching key (applied to both sides).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Fuzzy match threshold passed to MatchingConfig.",
    )

    # Reranking weights
    parser.add_argument(
        "--fuzzy-weight", type=float, default=0.3,
        help="Weight for Monge-Elkan fuzzy score in combined reranking score.",
    )
    parser.add_argument(
        "--embedding-weight", type=float, default=0.7,
        help="Weight for cosine embedding score in combined reranking score.",
    )

    args = parser.parse_args(argv)

    config = MatchingConfig(threshold=args.threshold)

    print("=" * 60)
    print("  llm-entity-reranker  —  DBLP-ACM benchmark")
    print("=" * 60)
    print(f"  Key columns  : {args.key_cols}")
    print(f"  Threshold    : {args.threshold}")
    print(f"  Fuzzy weight : {args.fuzzy_weight}")
    print(f"  Embed weight : {args.embedding_weight}")
    print()

    results = run_benchmark(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        dblp_file=args.dblp_file,
        acm_file=args.acm_file,
        matches_file=args.matches_file,
        left_key_cols=args.key_cols,
        right_key_cols=args.key_cols,
        left_match_id_col=args.left_match_id_col,
        right_match_id_col=args.right_match_id_col,
        config=config,
        fuzzy_weight=args.fuzzy_weight,
        embedding_weight=args.embedding_weight,
    )

    print()
    print("Results")
    print("-" * 60)
    print(format_results_table(results))
    print()

    for variant, label in [
        ("fuzzy_only",           "Fuzzy only           "),
        ("fuzzy_plus_embedding", "Fuzzy + Embedding    "),
    ]:
        m = results[variant]
        print(f"  {label}  TP={m['tp']:5d}  FP={m['fp']:5d}  FN={m['fn']:5d}")

    print()
    print(f"  Ground truth pairs : {results['ground_truth_size']}")
    print("=" * 60)


if __name__ == "__main__":
    # Allow direct execution: python compare.py
    # Relative imports won't work, so adjust sys.path
    import os
    import importlib
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    main()
