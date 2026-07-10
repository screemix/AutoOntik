"""
Entity Name Canonicalization via FAISS Nearest-Neighbor Search + LLM Verification
==================================================================================

Merges entity surface forms ("Nolan", "Christopher Nolan", "C. Nolan")
into canonical entities.

Entity identity is (name, type), not just name. Each triplet mention
produces a compound label "name [type]" that flows through the entire
pipeline. This prevents merging homonymous entities with different types
(e.g. "Paris [city]" vs "Paris [person]").

Uses FAISS to retrieve top-k nearest neighbors per entity across the full
index (no hard partitioning), then union-find over similar pairs to form
candidate clusters for LLM verification.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import faiss

from src.ontodisco.utils.dedup_base import (
    CanonicalItem,
    ContrieverEmbedder,
    DeduplicationResult,
    deduplicate,
    normalize_label,
)

logger = logging.getLogger(__name__)

_COMPOUND_RE = re.compile(r'^(.+?)\s*\[(.+?)\]$')


def _make_compound(name: str, type_: str) -> str:
    return f"{name} [{type_}]"


def _parse_compound(label: str) -> tuple[str, str]:
    m = _COMPOUND_RE.match(label)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return label.strip(), ""


def _normalize_compound(label: str) -> str:
    name, type_ = _parse_compound(label)
    norm_name = normalize_label(name)
    if type_:
        norm_type = normalize_label(type_)
        return f"{norm_name} [{norm_type}]"
    return norm_name


# ═══════════════════════════════════════════════════════════════════════════════
#  FAISS Nearest-Neighbor Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def _union_find_from_pairs(
    n: int,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """
    Build connected components from a sparse list of (i, j) pairs
    using union-find with path compression and union by rank.

    Returns array of integer cluster labels, shape (N,).
    """
    if n <= 1:
        return np.zeros(n, dtype=int)

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    for i, j in pairs:
        union(i, j)

    labels = np.array([find(i) for i in range(n)])
    unique_roots = {r: idx for idx, r in enumerate(sorted(set(labels)))}
    return np.array([unique_roots[labels[i]] for i in range(n)])


def _cluster_faiss_nn(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    top_k: int = 50,
) -> np.ndarray:
    """
    Scalable clustering via FAISS nearest-neighbor search + union-find.

    For each entity, retrieves top_k nearest neighbors from the full index
    (no hard partitioning). Pairs above the cosine similarity threshold are
    connected via union-find to form clusters.

    Embeddings must be L2-normalised (so inner product = cosine similarity).

    Memory: O(N × top_k).  Time: O(N × top_k × log(N)) for index search.
    """
    n = len(embeddings)

    if n <= 1:
        return np.zeros(n, dtype=int)

    k = min(top_k, n)
    emb = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    similarities, indices = index.search(emb, k)

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for rank_pos in range(k):
            j = int(indices[i, rank_pos])
            sim = float(similarities[i, rank_pos])
            if j != i and sim >= threshold:
                pairs.append((i, j))

    labels = _union_find_from_pairs(n, pairs)
    n_clusters = int(labels.max()) + 1

    logger.info(
        "FAISS NN clustering: %d entities, top_k=%d, threshold=%.2f → "
        "%d similar pairs → %d clusters",
        n, k, threshold, len(pairs), n_clusters,
    )

    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalEntity(CanonicalItem):
    """One canonical entity after deduplication."""
    type_labels: set[str] = field(default_factory=set)

    @property
    def entity_id(self) -> str:
        return self.item_id


@dataclass
class EntityDeduplicationResult(DeduplicationResult):
    """Full output of entity name canonicalization."""

    @property
    def entities(self) -> dict[str, CanonicalEntity]:
        return self.items

    @property
    def surface_to_entity_id(self) -> dict[str, str]:
        return self.surface_to_id

    @property
    def num_raw_entities(self) -> int:
        return self.num_raw

    @property
    def num_canonical_entities(self) -> int:
        return self.num_canonical


# ═══════════════════════════════════════════════════════════════════════════════
#  Surface Form Collection
# ═══════════════════════════════════════════════════════════════════════════════

def collect_entity_surface_forms(
    triplets: list[dict],
) -> list[str]:
    """
    Collect all entity surface forms from triplets as compound labels.

    Each triplet mention with a non-empty type produces a "name [type]"
    compound label. Mentions without a type are skipped.

    Returns all compound labels (with duplicates).
    """
    all_compound_labels: list[str] = []

    for triplet in triplets:
        for name_key, type_key in [
            ("subject", "subject_type"),
            ("object", "object_type"),
        ]:
            raw_name = triplet.get(name_key, "").strip()
            raw_type = triplet.get(type_key, "").strip()
            if not raw_name or not raw_type:
                continue

            all_compound_labels.append(_make_compound(raw_name, raw_type))

    logger.info(
        "Collected %d compound entity mentions from %d triplets",
        len(all_compound_labels), len(triplets),
    )

    return all_compound_labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_entities(
    triplets: list[dict],
    llm_extractor,
    *,
    contriever_model: str = "facebook/contriever",
    similarity_threshold: float = 0.85,
    faiss_top_k: int = 50,
    embed_batch_size: int = 64,
    device: str = None,
    prompt_path: str = None,
) -> EntityDeduplicationResult:
    """
    Full entity name deduplication with type-aware compound labels.

    Uses FAISS nearest-neighbor search for scalable candidate generation:
    each entity is compared against its top-k nearest neighbors across the
    full index (no hard partitioning), then union-find groups similar
    entities into clusters for LLM verification.

    Entity mentions are represented as compound "name [type]" labels.
    This ensures that homonymous entities with different types (e.g.
    "Paris [city]" vs "Paris [person]") are never merged.

    Args:
        triplets:             List of triplet dicts from extraction.
        llm_extractor:        LLMTripletExtractor instance.
        contriever_model:     HuggingFace model ID for Contriever.
        similarity_threshold: Cosine similarity threshold for merging (default 0.85).
        faiss_top_k:          Number of nearest neighbors to retrieve per entity (default 50).
        embed_batch_size:     Batch size for Contriever encoding.
        device:               "cuda", "cpu", or None (auto).
        prompt_path:          Path to entity_cluster_verify.txt. If None, uses default.

    Returns:
        EntityDeduplicationResult with canonical entities and mappings.
    """
    all_compound_labels = collect_entity_surface_forms(triplets)

    def _entity_embedding_text(compound_label: str) -> str:
        name, type_ = _parse_compound(compound_label)
        return f"{name} {type_}" if type_ else name


    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    def _build_entity(item_id, canonical_label, surface_forms, mention_count,
                      surface_form_counts):
        all_types: set[str] = set()
        for sf in surface_forms:
            _, type_ = _parse_compound(sf)
            if type_:
                all_types.add(type_)
        return CanonicalEntity(
            item_id=item_id,
            canonical_label=canonical_label,
            surface_forms=surface_forms,
            mention_count=mention_count,
            surface_form_counts=surface_form_counts,
            type_labels=all_types,
        )

    def _cluster_fn(embeddings: np.ndarray) -> np.ndarray:
        return _cluster_faiss_nn(
            embeddings,
            threshold=similarity_threshold,
            top_k=faiss_top_k,
        )

    base_result = deduplicate(
        all_labels=all_compound_labels,
        llm_verifier=llm_extractor,
        embedder=embedder,
        normalizer=_normalize_compound,
        embedding_text_fn=_entity_embedding_text,
        cluster_fn=_cluster_fn,
        embed_batch_size=embed_batch_size,
        id_prefix="ent",
        id_width=5,
        build_item=_build_entity,
    )

    logger.info(
        "Entity deduplication complete: %d → %d canonical entities (%.1f%% reduction)",
        base_result.num_raw, base_result.num_canonical, base_result.reduction_pct,
    )

    return EntityDeduplicationResult(
        items=base_result.items,
        surface_to_id=base_result.surface_to_id,
        num_raw=base_result.num_raw,
        num_canonical=base_result.num_canonical,
        reduction_pct=base_result.reduction_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Bridge: entity_type_map
# ═══════════════════════════════════════════════════════════════════════════════

def build_entity_type_map(
    entity_result: EntityDeduplicationResult,
    type_result: DeduplicationResult,
    triplets: list[dict],
) -> dict[str, list[str]]:
    """
    Build canonical_entity_id → [type_ids] mapping.

    Looks up entities by compound key "name [type]" and types by
    normalised type label.
    """
    entity_types: dict[str, set[str]] = defaultdict(set)

    for triplet in triplets:
        for name_key, type_key in [
            ("subject", "subject_type"),
            ("object", "object_type"),
        ]:
            raw_name = triplet.get(name_key, "").strip()
            raw_type = triplet.get(type_key, "").strip()
            if not raw_name or not raw_type:
                continue

            compound = _make_compound(raw_name, raw_type)
            norm_compound = _normalize_compound(compound)
            entity_id = entity_result.surface_to_id.get(norm_compound)
            if not entity_id:
                entity_id = entity_result.surface_to_id.get(compound)
            if not entity_id:
                continue

            norm_type = normalize_label(raw_type)
            type_id = type_result.surface_to_id.get(norm_type)
            if not type_id:
                type_id = type_result.surface_to_id.get(raw_type)
            if not type_id:
                continue

            entity_types[entity_id].add(type_id)

    result = {eid: sorted(tids) for eid, tids in entity_types.items()}
    logger.info(
        "Built entity_type_map: %d entities with type assignments",
        len(result),
    )
    return result
