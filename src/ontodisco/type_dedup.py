"""
Type Deduplication via HAC + Contriever + LLM Verification
==========================================================

Merges surface-form variants of the same conceptual type
("film director", "movie director", "filmmaker") into a single canonical type.

This is a thin wrapper around dedup_base.deduplicate().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.ontodisco.utils.dedup_base import (
    CanonicalItem,
    ContrieverEmbedder,
    DeduplicationResult,
    cluster_hac,
    deduplicate,
    normalize_label,
    verify_clusters_with_llm,
)

logger = logging.getLogger(__name__)

# ── Backward-compatible re-exports ────────────────────────────────────────────
normalize_type_label = normalize_label
cluster_types_hac = cluster_hac


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
    hac_linkage: str = "average",
    embed_batch_size: int = 64,
    device: str = None,
) -> TypeDeduplicationResult:
    """
    Full type deduplication pipeline: normalise → embed → HAC → LLM verify.

    Args:
        raw_type_labels:   All type labels from extracted triplets (may contain
                           duplicates).
        llm_extractor:     LLMTripletExtractor instance with verify_cluster_with_llm().
        contriever_model:  HuggingFace model ID for Contriever.
        hac_threshold:     Cosine similarity threshold for HAC.
        hac_linkage:       HAC linkage method.
        embed_batch_size:  Batch size for Contriever encoding.
        device:            Device for Contriever ("cuda", "cpu", or None for auto).

    Returns:
        TypeDeduplicationResult with the canonical type vocabulary.
    """
    logger.info("Starting type deduplication with %d raw labels", len(raw_type_labels))

    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    def _build_type(item_id, canonical_label, surface_forms, mention_count,
                    surface_form_counts):
        return CanonicalType(
            item_id=item_id,
            canonical_label=canonical_label,
            surface_forms=surface_forms,
            mention_count=mention_count,
            surface_form_counts=surface_form_counts,
        )

    base_result = deduplicate(
        all_labels=raw_type_labels,
        llm_verifier=llm_extractor,
        embedder=embedder,
        hac_threshold=hac_threshold,
        hac_linkage=hac_linkage,
        embed_batch_size=embed_batch_size,
        id_prefix="type",
        id_width=4,
        build_item=_build_type,
    )

    return TypeDeduplicationResult(
        items=base_result.items,
        surface_to_id=base_result.surface_to_id,
        num_raw=base_result.num_raw,
        num_canonical=base_result.num_canonical,
        reduction_pct=base_result.reduction_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Cluster Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def format_cluster_table(
    result: TypeDeduplicationResult,
    *,
    sort_by: str = "type_count",
    max_surface_forms: int = 10,
) -> str:
    """Format the deduplication result as a human-readable text table."""
    types = list(result.types.values())

    if sort_by == "type_count":
        types.sort(key=lambda t: t.type_count, reverse=True)
    elif sort_by == "label":
        types.sort(key=lambda t: t.canonical_label)
    elif sort_by == "cluster_size":
        types.sort(key=lambda t: len(t.surface_forms), reverse=True)

    id_width = max((len(t.type_id) for t in types), default=7)
    label_width = max((len(t.canonical_label) for t in types), default=15)
    id_width = max(id_width, 7)
    label_width = max(label_width, 15)
    count_width = 5

    header = (
        f"{'Type ID':<{id_width}}  "
        f"{'Canonical Label':<{label_width}}  "
        f"{'Count':>{count_width}}  "
        f"Surface Forms"
    )
    separator = "─" * len(header)

    lines = [
        f"Type Deduplication Results: {result.num_raw_types} raw → "
        f"{result.num_canonical_types} canonical ({result.reduction_pct:.1f}% reduction)",
        "",
        header,
        separator,
    ]

    for ct in types:
        forms = sorted(ct.surface_forms)
        if len(forms) > max_surface_forms:
            displayed = forms[:max_surface_forms]
            suffix = f" … +{len(forms) - max_surface_forms} more"
        else:
            displayed = forms
            suffix = ""

        forms_str = ", ".join(f'"{f}"' for f in displayed) + suffix

        lines.append(
            f"{ct.type_id:<{id_width}}  "
            f"{ct.canonical_label:<{label_width}}  "
            f"{ct.type_count:>{count_width}}  "
            f"{forms_str}"
        )

    return "\n".join(lines)


def print_clusters(
    result: TypeDeduplicationResult,
    *,
    sort_by: str = "type_count",
    max_surface_forms: int = 10,
) -> None:
    """Print the deduplication clusters to stdout."""
    print(format_cluster_table(result, sort_by=sort_by, max_surface_forms=max_surface_forms))
