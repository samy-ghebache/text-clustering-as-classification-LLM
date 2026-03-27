import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Small and fast — runs on CPU, ~80MB
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embedding_merge(
    labels: list[str],
    seed_labels: list[str] | None = None,
    threshold: float = 0.85,
) -> list[str]:
    """Merge similar labels using embedding cosine similarity + agglomerative clustering.

    For each cluster of similar labels, picks a representative:
      1. Seed label if present (closest to ground truth)
      2. Otherwise the shortest label (simpler = better for classification)
    """
    if len(labels) <= 1:
        return labels

    seed_set = {l.lower().strip() for l in (seed_labels or [])}

    model = _get_model()
    embeddings = model.encode(labels, normalize_embeddings=True)

    # Agglomerative clustering with cosine distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - threshold,  # cosine distance = 1 - similarity
    )
    cluster_ids = clustering.fit_predict(embeddings)

    # Group labels by cluster
    clusters: dict[int, list[str]] = {}
    for label, cid in zip(labels, cluster_ids):
        clusters.setdefault(cid, []).append(label)

    # Pick representative for each cluster
    merged = []
    for group in clusters.values():
        # Priority 1: seed label
        seed_match = [l for l in group if l.lower().strip() in seed_set]
        if seed_match:
            merged.append(seed_match[0])
        else:
            # Priority 2: shortest label
            merged.append(min(group, key=len))

    return sorted(merged)
