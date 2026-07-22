from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.ontodisco.utils.dedup_base import (
    CanonicalItem,
    ContrieverEmbedder,
    DeduplicationResult,
    cluster_hdbscan,
    deduplicate,
    normalize_label,
)

logger = logging.getLogger(__name__)

# Candidate generation used to be FAISS nearest-neighbor search + union-find
# (chosen for scalability over a full O(N^2) HAC pass). Replaced with
# dedup_base.cluster_hdbscan(), which clusters the full pairwise cosine
# distance matrix directly -- this reintroduces O(N^2) memory (this corpus's
# relation vocabulary is ~10k raw surface forms, i.e. a ~400MB float32
# distance matrix: workable here, but a real trade-off relative to FAISS's
# O(N x top_k) footprint for much larger vocabularies).

# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalRelation(CanonicalItem):
    """One canonical relation after deduplication."""
    subject_types: set[str] = field(default_factory=set)
    object_types: set[str] = field(default_factory=set)

    @property
    def relation_id(self) -> str:
        return self.item_id


@dataclass
class RelationDeduplicationResult(DeduplicationResult):
    """Full output of entity name canonicalization."""

    @property
    def relations(self) -> dict[str, CanonicalRelation]:
        return self.items

    @property
    def surface_to_relation_id(self) -> dict[str, str]:
        return self.surface_to_id

    @property
    def num_raw_relations(self) -> int:
        return self.num_raw

    @property
    def num_canonical_relations(self) -> int:
        return self.num_canonical


# ═══════════════════════════════════════════════════════════════════════════════
#  Surface Form Collection
# ═══════════════════════════════════════════════════════════════════════════════

def collect_relation_surface_forms(
    triplets: list[dict],
) -> list[str]:
    """
    Collect all relation surface forms from triplets as compound labels.

    Each triplet mention with a non-empty type produces a "name [type]"
    compound label. Mentions without a type are skipped.

    Returns all compound labels (with duplicates).
    """
    all_relation_surface_forms: list[str] = []

    for triplet in triplets:
        raw_relation = triplet.get("relation", "").strip()
        if not raw_relation:
            continue
        all_relation_surface_forms.append(raw_relation)

    logger.info(
        "Collected %d relation mentions from %d triplets",
        len(all_relation_surface_forms), len(triplets),
    )

    return all_relation_surface_forms


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_relations(
    triplets: list[dict],
    llm_extractor,
    *,
    contriever_model: str = "facebook/contriever",
    similarity_threshold: float = 0.85,
    embed_batch_size: int = 64,
    device: str = None,
) -> RelationDeduplicationResult:
    """
    Full entity name deduplication with type-aware compound labels.

    Candidate generation is HDBSCAN over the full pairwise cosine-distance
    matrix (dedup_base.cluster_hdbscan()): similarity_threshold sets HDBSCAN's
    cluster_selection_epsilon, and within that similarity band, variable-
    density substructure can form separate clusters rather than one flat cut.

    Relation mentions are represented as compound "name [type]" labels.
    This ensures that homonymous entities with different types (e.g.
    "Paris [city]" vs "Paris [person]") are never merged.

    Args:
        triplets:             List of triplet dicts from extraction.
        llm_extractor:        LLMTripletExtractor instance.
        contriever_model:     HuggingFace model ID for Contriever.
        similarity_threshold: Cosine similarity threshold for merging (default 0.85).
        embed_batch_size:     Batch size for Contriever encoding.
        device:               "cuda", "cpu", or None (auto).

    Returns:
        RelationDeduplicationResult with canonical entities and mappings.
    """
    relation_2_subject_types = defaultdict(set)
    relation_2_object_types = defaultdict(set)
    for triplet in triplets:
        relation = triplet.get("relation", "").strip()
        if not relation:
            continue
        relation_2_subject_types[relation].add(triplet.get("subject_type", "").strip())
        relation_2_object_types[relation].add(triplet.get("object_type", "").strip())

    logger.info(
        "Collected %d relation subject types and %d relation object types from %d triplets",
        len(relation_2_subject_types), len(relation_2_object_types), len(triplets),
    )

    all_relation_surface_forms = collect_relation_surface_forms(triplets)


    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    def _build_relation(item_id, canonical_label, surface_forms, count_per_normalized, 
                      surface_form_counts):
        subject_types = relation_2_subject_types[canonical_label]
        object_types = relation_2_object_types[canonical_label]
        return CanonicalRelation(
            item_id=item_id,
            canonical_label=canonical_label,
            surface_forms=surface_forms,
            count_per_normalized=count_per_normalized,
            surface_form_counts=surface_form_counts,
            subject_types=subject_types,
            object_types=object_types,
        )

    def _cluster_fn(embeddings: np.ndarray) -> np.ndarray:
        return cluster_hdbscan(embeddings, threshold=similarity_threshold)

    base_result = deduplicate(
        all_labels=all_relation_surface_forms,
        llm_verifier=llm_extractor,
        surface_form_type='relation',
        embedder=embedder,
        normalizer=normalize_label,
        cluster_fn=_cluster_fn,
        embed_batch_size=embed_batch_size,
        id_prefix="rel",
        id_width=5,
        build_item=_build_relation,
    )

    logger.info(
        "Relation deduplication complete: %d → %d canonical entities (%.1f%% reduction)",
        base_result.num_raw, base_result.num_canonical, base_result.reduction_pct,
    )

    return RelationDeduplicationResult(
        items=base_result.items,
        surface_to_id=base_result.surface_to_id,
        num_raw=base_result.num_raw,
        num_canonical=base_result.num_canonical,
        reduction_pct=base_result.reduction_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Bridge: relation_type_map
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_type_ids(raw_types: set[str], type_result: DeduplicationResult) -> set[str]:
    resolved: set[str] = set()
    for raw_type in raw_types:
        norm_type = normalize_label(raw_type)
        type_id = type_result.surface_to_id.get(norm_type)
        if not type_id:
            type_id = type_result.surface_to_id.get(raw_type)
        if type_id:
            resolved.add(type_id)
    return resolved


def update_relation_type_map(
    relation_result: RelationDeduplicationResult,
    type_result: DeduplicationResult,
) -> RelationDeduplicationResult:
    """
    Resolve each relation's subject/object types to canonical type IDs.

    Returns relation_result, mutated in place, for convenience.
    """
    for relation in relation_result.relations.values():
        relation.subject_types = _resolve_type_ids(relation.subject_types, type_result)
        relation.object_types = _resolve_type_ids(relation.object_types, type_result)

    logger.info(
        "Updated relation_type_map: resolved subject/object type ids for %d relations",
        len(relation_result.relations),
    )
    return relation_result
