import numpy as np
from typing import Dict, List
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def average_probability_trials(
    probabilities: List[float],
) -> Dict:
    """
    Average probability results across trials

    Args:
        * probabilities: List of probability floats

    Returns:
        * Dict with mean, std, min, max
    """
    arr = np.array(probabilities)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def average_ranking_trials(rankings: List[int]) -> Dict:
    """
    Average ranking results across trials using mode

    Args:
        * rankings: List of ranking ints (1 or 2)

    Returns:
        * Dict with mode, counts, agreement_ratio
    """
    counter = Counter(rankings)
    mode_val = counter.most_common(1)[0][0]
    total = len(rankings)
    counts = {str(k): v for k, v in counter.items()}
    agreement = counter[mode_val] / total if total > 0 else 0.0

    return {
        "mode": mode_val,
        "counts": counts,
        "agreement_ratio": round(agreement, 4),
    }


def _semantic_consensus(
    trial_cleaned: List[List[str]],
    trial_per_function_embeddings: List[List[List[float]]],
    consensus_threshold: float,
) -> tuple:
    """
    Build consensus using semantic clustering of per-function
    embeddings across trials

    Groups semantically similar functions (e.g. "motor control" and "motor
    function") into clusters, then ranks clusters by trial coverage (how many
    distinct trials mention the concept). Each cluster is represented by the
    most frequent function name

    Args:
        * trial_cleaned: List of cleaned function lists per trial
        * trial_per_function_embeddings: Per-function embedding vectors per
            trial (shape: [n_trials][n_functions][dims])
        * consensus_threshold: Cosine similarity threshold for merging
            functions into the same cluster (0.0-1.0)

    Returns:
        * (consensus_functions, function_frequencies) tuple
    """
    # Flatten: collect (function_name, embedding, trial_index)
    items = []
    for trial_idx, (funcs, embs) in enumerate(
        zip(trial_cleaned, trial_per_function_embeddings)
    ):
        for func, emb_vec in zip(funcs, embs):
            items.append((func, emb_vec, trial_idx))

    all_embs = np.array([item[1] for item in items])
    n = len(items)

    # Single item edge case
    if n == 1:
        return [items[0][0]], {items[0][0]: 1}

    # Compute pairwise cosine distance
    sim_matrix = cosine_similarity(all_embs)
    distance_matrix = np.clip(1 - sim_matrix, 0, 2)

    # Agglomerative clustering with distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - consensus_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    # Group items by cluster label
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(items[idx])

    # Rank clusters: trial coverage (desc), then size (desc)
    ranked = []
    for label, members in clusters.items():
        trial_coverage = len(set(m[2] for m in members))
        # Representative: most frequent string in cluster
        rep = Counter(m[0] for m in members).most_common(1)[0][0]
        ranked.append((
            trial_coverage,
            len(members),
            rep, members,
        ))

    ranked.sort(key=lambda x: (-x[0], -x[1]))
    consensus = [r[2] for r in ranked[:5]]

    # Build frequency dict from clusters
    frequencies = {}
    for _, _, rep, members in ranked:
        frequencies[rep] = len(members)

    return consensus, frequencies


def average_function_trials(
    trial_cleaned: List[List[str]],
    trial_embeddings: List[List[float]],
    trial_per_function_embeddings: List[List[List[float]]],
    consensus_threshold: float = 0.80,
) -> Dict:
    """
    Average function results across trials. Uses semantic clustering to group
    synonymous functions across trials (e.g. "motor control" and "motor
    function" merge into one cluster)

    Returns:
        * consensus_functions: top 5 functions by semantic consensus
        * mean_embedding: mean of combined embedding vectors
        * consistency_score: pairwise cosine similarity of embeddings
        * function_frequencies: cluster-based or string-based counts

    Args:
        * trial_cleaned: List of cleaned function lists per trial
        * trial_embeddings: List of combined embedding vectors
        * trial_per_function_embeddings: Optional per-function embedding
            vectors per trial
        * consensus_threshold: Cosine similarity threshold for semantic
            clustering (default: 0.80)
    """
    # Consensus: semantic clustering of per-function embeddings across trials
    consensus, frequencies = _semantic_consensus(
        trial_cleaned=trial_cleaned,
        trial_per_function_embeddings=trial_per_function_embeddings,
        consensus_threshold=consensus_threshold,
    )

    # Mean embedding vector across trials
    emb_array = np.array(trial_embeddings)
    mean_embedding = emb_array.mean(axis=0).tolist()

    # Consistency: mean pairwise cosine similarity
    n = len(trial_embeddings)
    if n > 1:
        sim_matrix = cosine_similarity(emb_array)
        upper = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper]
        consistency = float(pairwise_sims.mean())
    else:
        consistency = 1.0

    return {
        "consensus_functions": consensus,
        "mean_embedding": mean_embedding,
        "consistency_score": round(consistency, 4),
        "function_frequencies": frequencies,
    }
