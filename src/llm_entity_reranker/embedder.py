"""
Embeds candidate pairs using sentence-transformers to compute semantic similarity.
Takes the output of the fuzzy-scoring stage and adds an 'embedding_score' column
(cosine similarity between the two encoded key strings).
"""
import gc

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def embed_candidates(
    candidates_parquet,
    output_parquet,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=None,
    batch_size=512,
    normalize_embeddings=True,
    text_col_left="key_left",
    text_col_right="key_right",
):
    """
    Reads candidate pairs from a Parquet file, encodes both sides with a
    sentence-transformer model, computes cosine similarity, and writes the
    result (with an added 'embedding_score' column) to output_parquet.

    Parameters
    ----------
    candidates_parquet : str
        Path to Parquet file with at least columns text_col_left and text_col_right.
    output_parquet : str
        Path where the enriched Parquet file is written.
    model_name : str
        Sentence-transformer model identifier.
    device : str or None
        PyTorch device ('cpu', 'cuda', etc.). None = auto.
    batch_size : int
        Encoding batch size.
    normalize_embeddings : bool
        If True, embeddings are L2-normalised so dot product equals cosine similarity.
    text_col_left : str
        Column in candidates_parquet containing the left entity text.
    text_col_right : str
        Column in candidates_parquet containing the right entity text.

    Returns
    -------
    str
        Path to the written output Parquet file.
    """
    df = pd.read_parquet(candidates_parquet)

    if df.empty:
        df["embedding_score"] = pd.Series(dtype="float32")
        df.to_parquet(output_parquet, index=False, compression="zstd")
        return output_parquet

    # Replace underscores with spaces for better embedding quality
    left_texts = (
        df[text_col_left].fillna("").astype(str).str.replace("_", " ", regex=False).tolist()
    )
    right_texts = (
        df[text_col_right].fillna("").astype(str).str.replace("_", " ", regex=False).tolist()
    )

    # Deduplicate to avoid redundant encoding
    unique_texts = list(dict.fromkeys(left_texts + right_texts))

    model = SentenceTransformer(model_name, device=device)

    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32")

    text_to_emb = {t: embeddings[i] for i, t in enumerate(unique_texts)}

    del model, embeddings
    gc.collect()

    left_embs = np.stack([text_to_emb[t] for t in left_texts])
    right_embs = np.stack([text_to_emb[t] for t in right_texts])

    if normalize_embeddings:
        # Dot product of L2-normalised vectors equals cosine similarity
        scores = (left_embs * right_embs).sum(axis=1)
    else:
        norms_l = np.linalg.norm(left_embs, axis=1, keepdims=True)
        norms_r = np.linalg.norm(right_embs, axis=1, keepdims=True)
        norms_l = np.where(norms_l == 0, 1.0, norms_l)
        norms_r = np.where(norms_r == 0, 1.0, norms_r)
        scores = ((left_embs / norms_l) * (right_embs / norms_r)).sum(axis=1)

    df["embedding_score"] = scores.astype("float32")
    df.to_parquet(output_parquet, index=False, compression="zstd")

    return output_parquet
