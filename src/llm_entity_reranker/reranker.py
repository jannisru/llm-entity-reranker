"""
Reranks entity matching candidates by combining fuzzy score (Monge-Elkan)
with semantic embedding score, and optionally with an LLM binary classifier
queried via a local Ollama instance.
"""
import json
import urllib.request
import urllib.error

import pandas as pd


def rerank_by_embedding(
    candidates_parquet,
    output_parquet,
    fuzzy_weight=0.3,
    embedding_weight=0.7,
):
    """
    Combines the Monge-Elkan fuzzy score and the cosine embedding score into
    a single combined_score, then keeps the best-scoring match per left_id.

    Expects columns: left_id, right_id, key_left, key_right, score, embedding_score.
    Writes columns:  left_id, right_id, key_left, key_right, fuzzy_score,
                     embedding_score, score  (where score = combined_score).

    Parameters
    ----------
    candidates_parquet : str
        Path to Parquet produced by embed_candidates().
    output_parquet : str
        Path where the reranked Parquet is written.
    fuzzy_weight : float
        Weight applied to the original Monge-Elkan score (0..1).
    embedding_weight : float
        Weight applied to the cosine embedding score (0..1).

    Returns
    -------
    str
        Path to the written output Parquet file.
    """
    df = pd.read_parquet(candidates_parquet)

    if df.empty:
        df.to_parquet(output_parquet, index=False, compression="zstd")
        return output_parquet

    df["combined_score"] = (
        fuzzy_weight * df["score"].astype(float)
        + embedding_weight * df["embedding_score"].astype(float)
    )

    best_idx = df.groupby("left_id")["combined_score"].idxmax()
    result = df.loc[best_idx].copy()
    result = result.rename(columns={"score": "fuzzy_score", "combined_score": "score"})
    result.to_parquet(output_parquet, index=False, compression="zstd")

    return output_parquet


def _call_ollama(prompt, model, host, timeout):
    url = f"{host}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


def rerank_with_ollama(
    candidates_parquet,
    output_parquet,
    ollama_model="llama3.2",
    ollama_host="http://localhost:11434",
    fuzzy_weight=0.2,
    embedding_weight=0.4,
    llm_weight=0.4,
    timeout=30,
):
    """
    Extends rerank_by_embedding() with a binary LLM score from Ollama.

    For each candidate pair the model is asked "Are these the same entity?
    yes/no". The answer is mapped to 1.0 (yes) or 0.0 (no) and blended with
    the fuzzy and embedding scores. On connection error the LLM score falls
    back to 0.5 (neutral) so the other signals still drive ranking.

    Requires embedding_score to already be present (run embed_candidates first).

    Parameters
    ----------
    candidates_parquet : str
        Path to Parquet produced by embed_candidates().
    output_parquet : str
        Path where the reranked Parquet is written.
    ollama_model : str
        Ollama model tag to use (e.g. 'llama3.2', 'mistral').
    ollama_host : str
        Base URL of the Ollama API (default http://localhost:11434).
    fuzzy_weight : float
        Weight for Monge-Elkan score.
    embedding_weight : float
        Weight for cosine embedding score.
    llm_weight : float
        Weight for LLM binary score.
    timeout : int
        HTTP timeout in seconds per Ollama request.

    Returns
    -------
    str
        Path to the written output Parquet file.
    """
    df = pd.read_parquet(candidates_parquet)

    if df.empty:
        df.to_parquet(output_parquet, index=False, compression="zstd")
        return output_parquet

    llm_scores = []
    for row in df.itertuples(index=False):
        a = str(row.key_left).replace("_", " ")
        b = str(row.key_right).replace("_", " ")
        prompt = (
            "Are these two entries referring to the same real-world entity? "
            "Answer only 'yes' or 'no'.\n\n"
            f"Entry A: {a}\n"
            f"Entry B: {b}"
        )
        try:
            response = _call_ollama(prompt, model=ollama_model, host=ollama_host, timeout=timeout)
            llm_scores.append(1.0 if response.lower().startswith("yes") else 0.0)
        except Exception:
            llm_scores.append(0.5)

    llm_series = pd.Series(llm_scores, index=df.index, dtype="float32")
    df["llm_score"] = llm_series
    df["combined_score"] = (
        fuzzy_weight * df["score"].astype(float)
        + embedding_weight * df["embedding_score"].astype(float)
        + llm_weight * llm_series
    )

    best_idx = df.groupby("left_id")["combined_score"].idxmax()
    result = df.loc[best_idx].copy()
    result = result.rename(columns={"score": "fuzzy_score", "combined_score": "score"})
    result.to_parquet(output_parquet, index=False, compression="zstd")

    return output_parquet
