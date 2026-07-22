"""
Entity Name Canonicalization via FAISS Nearest-Neighbor Search + LLM Verification
==================================================================================

Merges entity surface forms ("Nolan", "Christopher Nolan", "C. Nolan") into
canonical entities, and assigns each canonical entity its set of canonical
type_ids as a byproduct of clustering -- no separate "class assignment" pass
is needed.

Entity identity is (name, type), not just name: each triplet mention produces
a compound label "name [canonical type label]" that flows through embedding
and clustering. This prevents merging homonymous entities with different
types (e.g. "Paris [city]" vs "Paris [person]").

Must run AFTER type_dedup.py and hierarchy_induction.py (unlike relation/type
dedup, which only need each other):
  - Using the already-canonicalized type in the compound label is what makes
    class assignment fall out for free during clustering.
  - The induced TypeHierarchy gates candidate merges by immediate-parent
    membership (see _cluster_hdbscan_with_parent_gate): two mentions are
    only clusterable if their types are identical, or siblings under the
    same parent. Nothing else counts -- not grandparent/grandchild, not "any
    shared ancestor". This is a hard, symbolic gate applied by PARTITIONING
    candidates by parent key BEFORE clustering, not an upfront embedding-space
    partition -- so it doesn't reintroduce the recall loss of the KMeans
    hard-blocking approach this module used before (see git history): that
    approach silently dropped true synonym pairs that happened to fall into
    different embedding-space blocks. Gating on type-hierarchy parentage is
    exact and symbolic rather than an approximate embedding-space partition,
    so genuine name-synonym pairs are never dropped for a reason unrelated to
    their type, and partitioning first also means HDBSCAN only ever compares
    candidates that could possibly merge, rather than the whole vocabulary at
    once.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.ontodisco.utils.dedup_base import (
    CanonicalItem,
    ContrieverEmbedder,
    DeduplicationResult,
    cluster_hdbscan,
    deduplicate,
    normalize_label,
)

if TYPE_CHECKING:
    from src.ontodisco.type_dedup import TypeDeduplicationResult
    from src.ontodisco.hierarchy_induction import TypeHierarchy

logger = logging.getLogger(__name__)

_COMPOUND_RE = re.compile(r'^(.+?)\s*\[(.+?)\]$')


def _make_compound(name: str, type_label: str) -> str:
    return f"{name} [{type_label}]"


def _parse_compound(label: str) -> tuple[str, str]:
    m = _COMPOUND_RE.match(label)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return label.strip(), ""


def _normalize_compound(label: str) -> str:
    name, type_label = _parse_compound(label)
    norm_name = normalize_label(name)
    if type_label:
        norm_type = normalize_label(type_label)
        return f"{norm_name} [{norm_type}]"
    return norm_name


# ═══════════════════════════════════════════════════════════════════════════════
#  HDBSCAN Clustering, Gated by Shared Immediate Parent
# ═══════════════════════════════════════════════════════════════════════════════

def _parent_key(type_id: str, hierarchy: "TypeHierarchy") -> tuple:
    """A key identifying type_id's "sibling group": ("parent", parent_id)
    for non-root types, ("self", type_id) for roots.

    Two types are only eligible to cluster together if this key agrees for
    both -- i.e. they are literally the same type, or siblings under the
    same parent. No other relatedness counts -- in particular, a root and
    one of its own children must NOT match: using type_id itself as a root's
    key (with no "self"/"parent" tag) would make a root's key collide with
    its child's parent_id, incorrectly treating parent/child as siblings.
    """
    parents = hierarchy.parents.get(type_id)
    if parents:
        return ("parent", parents[0])
    return ("self", type_id)


def _cluster_hdbscan_with_parent_gate(
    embeddings: np.ndarray,
    type_ids: list[str],
    hierarchy: "TypeHierarchy",
    *,
    threshold: float = 0.85,
) -> np.ndarray:
    """
    Partition candidates by _parent_key FIRST (the hard, symbolic gate:
    identical type, or siblings under the same immediate parent), then run
    HDBSCAN (dedup_base.cluster_hdbscan) independently WITHIN each partition.

    Two candidates in different partitions can never end up in the same
    cluster, by construction -- equivalent to generating all candidate pairs
    and gating them post-hoc, but cheaper (HDBSCAN only ever compares
    candidates that could possibly merge) and avoids materialising one
    global N x N distance matrix across the whole entity vocabulary.

    type_ids must be aligned index-for-index with embeddings.
    """
    n = len(embeddings)

    if n <= 1:
        return np.zeros(n, dtype=int)

    partitions: dict[tuple, list[int]] = defaultdict(list)
    for i, type_id in enumerate(type_ids):
        partitions[_parent_key(type_id, hierarchy)].append(i)

    labels = np.zeros(n, dtype=int)
    next_cluster_id = 0
    for indices in partitions.values():
        sub_embeddings = embeddings[indices]
        sub_labels = cluster_hdbscan(sub_embeddings, threshold=threshold)
        remap: dict[int, int] = {}
        for local_idx, global_idx in enumerate(indices):
            local_label = int(sub_labels[local_idx])
            if local_label not in remap:
                remap[local_label] = next_cluster_id
                next_cluster_id += 1
            labels[global_idx] = remap[local_label]

    logger.info(
        "HDBSCAN clustering (parent-gated): %d entities, threshold=%.2f, "
        "%d parent-partitions -> %d clusters",
        n, threshold, len(partitions), next_cluster_id,
    )

    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalEntity(CanonicalItem):
    """One canonical entity after deduplication, with its class assignment."""
    type_ids: set[str] = field(default_factory=set)

    @property
    def entity_id(self) -> str:
        return self.item_id


@dataclass
class EntityDeduplicationResult(DeduplicationResult):
    """Full output of entity name canonicalization + class assignment."""

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

def _resolve_type_id(raw_type: str, type_vocab: "TypeDeduplicationResult") -> Optional[str]:
    norm_type = normalize_label(raw_type)
    return type_vocab.surface_to_id.get(norm_type) or type_vocab.surface_to_id.get(raw_type)


def collect_entity_surface_forms(
    triplets: list[dict],
    type_vocab: "TypeDeduplicationResult",
) -> tuple[list[str], dict[str, str]]:
    """
    Collect all entity surface forms from triplets as compound labels
    "name [canonical type label]", using the ALREADY-CANONICALIZED type
    (type_dedup must run first). Mentions whose type doesn't resolve to a
    known type_id (e.g. discarded as noise by type_dedup's min_type_freq)
    are skipped -- the entity may still be canonicalized via its other,
    resolvable mentions.

    Returns (all_compound_labels, type_id_by_normalized_compound): the
    second dict maps each unique normalized compound label to the type_id
    used to build it (built inline, since a given normalized compound
    string is only ever produced from one (name, type_id) pair).
    """
    all_compound_labels: list[str] = []
    type_id_by_normalized_compound: dict[str, str] = {}
    skipped = 0

    for triplet in triplets:
        for name_key, type_key in [
            ("subject", "subject_type"),
            ("object", "object_type"),
        ]:
            raw_name = triplet.get(name_key, "").strip()
            raw_type = triplet.get(type_key, "").strip()
            if not raw_name or not raw_type:
                continue

            type_id = _resolve_type_id(raw_type, type_vocab)
            if type_id is None:
                skipped += 1
                continue

            canonical_type_label = type_vocab.types[type_id].canonical_label
            compound = _make_compound(raw_name, canonical_type_label)
            all_compound_labels.append(compound)
            type_id_by_normalized_compound[_normalize_compound(compound)] = type_id

    logger.info(
        "Collected %d compound entity mentions (%d unique) from %d triplets "
        "(%d mentions skipped: type not resolvable in TypeVocabulary)",
        len(all_compound_labels), len(type_id_by_normalized_compound),
        len(triplets), skipped,
    )

    return all_compound_labels, type_id_by_normalized_compound


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_entities(
    triplets: list[dict],
    type_vocab: "TypeDeduplicationResult",
    hierarchy: "TypeHierarchy",
    llm_extractor,
    *,
    contriever_model: str = "facebook/contriever",
    similarity_threshold: float = 0.85,
    embed_batch_size: int = 64,
    device: str = None,
) -> EntityDeduplicationResult:
    """
    Full entity name deduplication with type-aware compound labels and
    class assignment.

    Must run AFTER type_dedup.py (needs canonical type_ids) and
    hierarchy_induction.py (needs the induced TypeHierarchy to gate
    candidate merges -- see module docstring).

    Args:
        triplets:             List of triplet dicts from extraction.
        type_vocab:           TypeDeduplicationResult from type_dedup.py.
        hierarchy:            TypeHierarchy from hierarchy_induction.py.
        llm_extractor:        LLMTripletExtractor instance.
        contriever_model:     HuggingFace model ID for Contriever.
        similarity_threshold: Cosine similarity threshold for merging (default 0.85).
        embed_batch_size:     Batch size for Contriever encoding.
        device:               "cuda", "cpu", or None (auto).

    Returns:
        EntityDeduplicationResult with canonical entities (each carrying its
        assigned type_ids) and surface form mappings.
    """
    all_compound_labels, type_id_by_normalized_compound = collect_entity_surface_forms(
        triplets, type_vocab,
    )

    # Aligned index-for-index with the embeddings deduplicate() computes
    # internally (its own unique_labels = sorted(normalized_to_raws.keys())
    # over the exact same all_compound_labels + normalizer) -- this is the
    # only way to get per-item type metadata into cluster_fn, since that
    # callback's signature is (embeddings) -> labels with no side channel.
    unique_labels_ordered = sorted(type_id_by_normalized_compound.keys())
    type_ids_per_item = [type_id_by_normalized_compound[u] for u in unique_labels_ordered]

    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    def _entity_embedding_text(compound_label: str) -> str:
        name, type_label = _parse_compound(compound_label)
        return f"{name} {type_label}" if type_label else name

    def _build_entity(item_id, canonical_label, surface_forms, count_per_normalized,
                      surface_form_counts):
        assigned_type_ids: set[str] = set()
        for sf in surface_forms:
            type_id = type_id_by_normalized_compound.get(_normalize_compound(sf))
            if type_id:
                assigned_type_ids.add(type_id)
        return CanonicalEntity(
            item_id=item_id,
            canonical_label=canonical_label,
            surface_forms=surface_forms,
            count_per_normalized=count_per_normalized,
            surface_form_counts=surface_form_counts,
            type_ids=assigned_type_ids,
        )

    def _cluster_fn(embeddings: np.ndarray) -> np.ndarray:
        return _cluster_hdbscan_with_parent_gate(
            embeddings, type_ids_per_item, hierarchy,
            threshold=similarity_threshold,
        )

    base_result = deduplicate(
        all_labels=all_compound_labels,
        llm_verifier=llm_extractor,
        surface_form_type='entity',
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
        "Entity deduplication complete: %d -> %d canonical entities (%.1f%% reduction)",
        base_result.num_raw, base_result.num_canonical, base_result.reduction_pct,
    )

    return EntityDeduplicationResult(
        items=base_result.items,
        surface_to_id=base_result.surface_to_id,
        num_raw=base_result.num_raw,
        num_canonical=base_result.num_canonical,
        reduction_pct=base_result.reduction_pct,
    )
