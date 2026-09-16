"""
Type Deduplication via HAC + Contriever + LLM Verification
==========================================================

Merges surface-form variants of the same conceptual type
("film director", "movie director", "filmmaker") into a single canonical type.

This is a thin wrapper around dedup_base.deduplicate_with_rounds().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.ontodisco.utils.dedup_base import (
    CanonicalItem,
    ContrieverEmbedder,
    DeduplicationResult,
    cluster_hdbscan,
    deduplicate_with_rounds,
    normalize_label,
    verify_clusters_with_llm,
)
from src.ontodisco.relation_context import (
    build_relation_counts,
    build_type_relation_profiles,
    describe_relation_context,
)

if TYPE_CHECKING:
    from src.ontodisco.relation_dedup import RelationDeduplicationResult

logger = logging.getLogger(__name__)

# ── Backward-compatible re-exports ────────────────────────────────────────────
normalize_type_label = normalize_label
cluster_types_hdbscan = cluster_hdbscan


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalType(CanonicalItem):
    """One entry in the flat type vocabulary T*."""

    @property
    def type_id(self) -> str:
        return self.item_id

    @property
    def type_count(self) -> int:
        return self.count_per_normalized


@dataclass
class TypeDeduplicationResult(DeduplicationResult):
    """Full output of the type deduplication step."""

    @property
    def types(self) -> dict[str, CanonicalType]:
        return self.items

    @property
    def surface_to_type_id(self) -> dict[str, str]:
        return self.surface_to_id

    @property
    def num_raw_types(self) -> int:
        return self.num_raw

    @property
    def num_canonical_types(self) -> int:
        return self.num_canonical


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_types(
    raw_type_labels: list[str],
    llm_extractor,
    *,
    contriever_model: str = "facebook/contriever",
    hac_threshold: float = 0.85,
    embed_batch_size: int = 64,
    device: str = None,
    relation_result: "RelationDeduplicationResult | None" = None,
    relation_context_top_k: int = 5,
    max_merge_rounds: int = 5,
    max_parallel_workers: int = 8,
    max_cluster_size: int = 40,
) -> TypeDeduplicationResult:
    """
    Full type deduplication pipeline: normalise → embed → HDBSCAN → LLM verify.

    Args:
        raw_type_labels:   All type labels from extracted triplets (may contain
                           duplicates).
        llm_extractor:     LLMTripletExtractor instance with verify_cluster_with_llm().
        contriever_model:  HuggingFace model ID for Contriever.
        hac_threshold:     Cosine similarity threshold for HDBSCAN candidate
                           clustering (see dedup_base.cluster_hdbscan() for how
                           this maps to cluster_selection_epsilon). Kept under
                           this name for config/call-site compatibility.
        embed_batch_size:  Batch size for Contriever encoding.
        device:            Device for Contriever ("cuda", "cpu", or None for auto).
        relation_result:   Optional RelationDeduplicationResult from relation
                           dedup (relation_dedup.deduplicate_relations()). When
                           given, each candidate shown to the LLM verifier is
                           annotated with a short relational-context
                           description (e.g. "film (context: often appears as
                           object of: directed, starred in)"), giving the LLM
                           concrete evidence for merge/split decisions.
                           Relation-signature evidence is deliberately NOT used
                           to auto-union HDBSCAN clusters before LLM
                           verification here any more -- that job now belongs
                           to hierarchy induction's Weeds-precision pairwise
                           track (Step 3), which catches the same
                           relationally-similar-but-lexically-distant pairs
                           via genuine pairwise LLM verification (including
                           same_concept) rather than a blind cosine-threshold
                           cluster merge. When relation_result is None,
                           behavior is identical to before this parameter
                           existed.
        relation_context_top_k: How many top relation dimensions to
                           render per label in the LLM-facing context string.
        max_merge_rounds:  Max embed -> cluster -> LLM-verify passes
                           (dedup_base.deduplicate_with_rounds) before
                           stopping, once a round produces no further
                           reduction in the canonical type count. Mirrors
                           entity_dedup's per-partition round loop; 1
                           behaves identically to a plain single-pass
                           deduplicate() call.
        max_parallel_workers: Thread pool size for concurrent LLM cluster
                           verification calls within each round.
        max_cluster_size:  Size cap before a cluster is split into
                           sub-batches + stitched back together (see
                           dedup_base.verify_clusters_with_llm).

    Returns:
        TypeDeduplicationResult with the canonical type vocabulary.
    """
    logger.info("Starting type deduplication with %d raw labels", len(raw_type_labels))

    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    def _build_type(item_id, canonical_label, surface_forms, count_per_normalized,
                    surface_form_counts):
        return CanonicalType(
            item_id=item_id,
            canonical_label=canonical_label,
            surface_forms=surface_forms,
            count_per_normalized=count_per_normalized,
            surface_form_counts=surface_form_counts,
        )

    member_context = None
    if relation_result is not None:
        relation_counts = build_relation_counts(relation_result)
        ppmi_profiles = build_type_relation_profiles(relation_counts, weighting="ppmi")

        member_context = {
            label: describe_relation_context(
                label, ppmi_profiles, relation_result, top_k=relation_context_top_k,
            )
            for label in relation_counts
        }
        member_context = {label: ctx for label, ctx in member_context.items() if ctx}

    base_result = deduplicate_with_rounds(
        all_labels=raw_type_labels,
        llm_verifier=llm_extractor,
        embedder=embedder,
        hac_threshold=hac_threshold,
        embed_batch_size=embed_batch_size,
        id_prefix="type",
        id_width=4,
        build_item=_build_type,
        member_context=member_context,
        max_rounds=max_merge_rounds,
        max_workers=max_parallel_workers,
        max_cluster_size=max_cluster_size,
    )

    return TypeDeduplicationResult(
        items=base_result.items,
        surface_to_id=base_result.surface_to_id,
        num_raw=base_result.num_raw,
        num_canonical=base_result.num_canonical,
        reduction_pct=base_result.reduction_pct,
    )