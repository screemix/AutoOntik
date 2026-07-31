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
    membership (see _parent_key): two mentions are only clusterable if their
    types are identical, or siblings under the same parent. Nothing else
    counts -- not grandparent/grandchild, not "any shared ancestor". This is
    a hard, symbolic gate applied by PARTITIONING candidates by parent key
    BEFORE clustering, not an upfront embedding-space partition -- so it
    doesn't reintroduce the recall loss of the KMeans hard-blocking approach
    this module used before (see git history): that approach silently
    dropped true synonym pairs that happened to fall into different
    embedding-space blocks. Gating on type-hierarchy parentage is exact and
    symbolic rather than an approximate embedding-space partition, so
    genuine name-synonym pairs are never dropped for a reason unrelated to
    their type, and partitioning first also means HDBSCAN only ever compares
    candidates that could possibly merge, rather than the whole vocabulary at
    once.

Within each parent partition, merging runs for SEVERAL ROUNDS
(_run_partition_merge_rounds) rather than a single embed -> cluster -> LLM
verify pass: a merge in round N can pull a canonical entity's embedding
close enough to a fourth surface form to only become clusterable in round
N+1 (e.g. "Chris Nolan" + "C. Nolan" merge into "Christopher Nolan" in round
1, which is then a closer embedding match to "Christopher J. Nolan" than
either original form was alone), and a single HDBSCAN cut can simply fail to
chain 3+ near-duplicates together that a second pass over the now-smaller
pool catches. Each round re-embeds the CURRENT pool (already-merged entities
plus untouched singletons), re-clusters, and re-verifies with the LLM;
rounds stop as soon as one produces no further reduction in pool size (nothing
left to merge), or after `max_merge_rounds` rounds, whichever comes first.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.ontodisco.utils.dedup_base import (
    CanonicalItem,
    ContrieverEmbedder,
    DeduplicationResult,
    cluster_hdbscan,
    normalize_label,
    verify_clusters_with_llm,
)
from src.ontodisco.hierarchy_induction import lowest_common_ancestor

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


@dataclass
class _EntityPoolItem:
    """One node in a parent-partition's merge pool: a (possibly already
    merged) entity, tracked by its current display name plus a
    representative type label (used ONLY to build embedding text and the
    compound string shown to the LLM -- the entity's final type_ids come
    from `normalized_members`, not from this single label) and the set of
    original normalized compound labels folded into it so far."""
    label: str
    type_label: str
    normalized_members: set[str] = field(default_factory=set)


def _run_partition_merge_rounds(
    unique_compound_labels: list[str],
    embedder: ContrieverEmbedder,
    llm_extractor,
    *,
    similarity_threshold: float,
    embed_batch_size: int,
    max_merge_rounds: int,
    partition_type_label: str,
    max_workers: int = 8,
    max_cluster_size: int = 40,
) -> list[tuple[str, set[str]]]:
    """
    Repeatedly embed -> HDBSCAN-cluster -> LLM-verify the current pool of
    entities within ONE parent partition (all candidates here already share
    the hard parent-key gate -- see module docstring), folding merged groups
    back into the pool each round, until a round produces no further
    reduction in pool size or `max_merge_rounds` is reached.

    partition_type_label: the representative type label for this WHOLE
    partition -- the parent's canonical_label for a ("parent", parent_id)
    partition, or the type's own canonical_label for a ("self", type_id)
    partition (see _parent_key / the caller). Every item merged in this
    partition is either literally that one type already, or a sibling under
    that exact parent, so this is always the correct "earliest common
    parent" for any subset merged here -- no separate LCA walk needed.

    Returns a list of (canonical_label, {normalized compound members}) --
    one entry per final entity surviving in this partition.
    """
    pool: list[_EntityPoolItem] = []
    for compound in unique_compound_labels:
        name, type_label = _parse_compound(compound)
        pool.append(_EntityPoolItem(label=name, type_label=type_label,
                                     normalized_members={compound}))

    for round_num in range(1, max_merge_rounds + 1):
        if len(pool) < 2:
            break

        texts = [f"{item.label} {item.type_label}" for item in pool]
        embeddings = embedder.embed(texts, batch_size=embed_batch_size)
        cluster_labels = cluster_hdbscan(embeddings, threshold=similarity_threshold)

        clusters_by_str: dict[int, list[str]] = defaultdict(list)
        str_to_index: dict[str, int] = {}
        for idx, (item, cl) in enumerate(zip(pool, cluster_labels)):
            compound_str = f"{item.label} [{item.type_label}]"
            clusters_by_str[int(cl)].append(compound_str)
            str_to_index[compound_str] = idx
        clusters_by_str = dict(clusters_by_str)

        if all(len(members) == 1 for members in clusters_by_str.values()):
            logger.info("Partition merge round %d: no candidate clusters, stopping", round_num)
            break

        verified_groups = verify_clusters_with_llm(
            clusters_by_str, llm_extractor, surface_form_type="entity",
            max_workers=max_workers, max_cluster_size=max_cluster_size,
        )

        next_pool: list[_EntityPoolItem] = []
        for canonical_label, member_strs in verified_groups:
            member_indices = [str_to_index[s] for s in member_strs if s in str_to_index]
            if not member_indices:
                continue
            if len(member_indices) == 1:
                # No merge happened for this member -- carry the pool item
                # over UNCHANGED rather than trusting the echoed
                # canonical_label: verify_clusters_with_llm never calls the
                # LLM for singleton clusters, so canonical_label there is
                # just the input compound string as-is (brackets included),
                # not a real naming decision.
                next_pool.append(pool[member_indices[0]])
                continue
            normalized_members: set[str] = set()
            for idx in member_indices:
                normalized_members |= pool[idx].normalized_members
            # Keep the true label when every merged item already agrees (no
            # ambiguity); only fall back to the partition's shared parent
            # when merging genuine siblings with different type labels --
            # never an arbitrary alphabetical pick (see docstring above).
            merged_type_labels = {pool[idx].type_label for idx in member_indices}
            representative_type = (
                merged_type_labels.pop() if len(merged_type_labels) == 1 else partition_type_label
            )
            next_pool.append(_EntityPoolItem(
                label=canonical_label.strip(),
                type_label=representative_type,
                normalized_members=normalized_members,
            ))

        progressed = len(next_pool) < len(pool)
        logger.info(
            "Partition merge round %d: %d -> %d entities", round_num, len(pool), len(next_pool),
        )
        pool = next_pool
        if not progressed:
            break
    else:
        logger.info("Partition merge reached max_merge_rounds=%d, stopping", max_merge_rounds)

    return [(item.label, item.normalized_members) for item in pool]


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalEntity(CanonicalItem):
    """One canonical entity after deduplication, with its class assignment."""
    type_ids: set[str] = field(default_factory=set)
    primary_type_id: Optional[str] = None
    # Single-representative-type convenience field on top of type_ids: the
    # lowest common ancestor of type_ids in the induced TypeHierarchy (None
    # if type_ids is empty, or if the observed types don't share one --
    # disconnected trees of the hierarchy forest). type_ids stays the
    # authoritative class assignment; this is derived from it, not a
    # replacement for it.

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
    max_merge_rounds: int = 5,
    max_parallel_workers: int = 8,
    max_cluster_size: int = 40,
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
        max_merge_rounds:     Max embed -> cluster -> LLM-verify passes run
                              PER parent partition before moving on (see
                              _run_partition_merge_rounds / module docstring).
        max_parallel_workers: Thread pool size for concurrent LLM cluster
                              verification calls within each partition's rounds.
        max_cluster_size:     Size cap before a cluster is split into
                              sub-batches + stitched back together (see
                              dedup_base.verify_clusters_with_llm).

    Returns:
        EntityDeduplicationResult with canonical entities (each carrying its
        assigned type_ids) and surface form mappings.
    """
    all_compound_labels, type_id_by_normalized_compound = collect_entity_surface_forms(
        triplets, type_vocab,
    )

    # ── Normalisation stats (mirrors dedup_base.deduplicate()'s Step 1) ──────
    normalized_to_raws: dict[str, set[str]] = defaultdict(set)
    surface_form_counts: dict[str, int] = defaultdict(int)
    count_per_normalized: dict[str, int] = defaultdict(int)
    for label in all_compound_labels:
        norm = _normalize_compound(label)
        normalized_to_raws[norm].add(label)
        surface_form_counts[label] += 1
        count_per_normalized[norm] += 1

    unique_labels = sorted(normalized_to_raws.keys())
    logger.info("After normalisation: %d unique compound entity labels", len(unique_labels))

    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    # ── Partition by parent-key (hard gate), then run merge rounds
    #    independently WITHIN each partition ─────────────────────────────────
    partitions: dict[tuple, list[str]] = defaultdict(list)
    for norm_label in unique_labels:
        type_id = type_id_by_normalized_compound[norm_label]
        partitions[_parent_key(type_id, hierarchy)].append(norm_label)

    all_verified_groups: list[tuple[str, list[str]]] = []
    for partition_key, members in partitions.items():
        if len(members) == 1:
            all_verified_groups.append((members[0], members))
            continue
        # ("parent", parent_id) -> the shared parent's own label; ("self",
        # type_id) -> that one type's own label (trivial -- every item in a
        # self-partition already shares that exact type). Either way, this
        # is the correct representative type for ANY subset merged within
        # this partition (see _run_partition_merge_rounds docstring).
        _, partition_type_id = partition_key
        partition_type = type_vocab.types.get(partition_type_id)
        partition_type_label = partition_type.canonical_label if partition_type else partition_type_id
        partition_groups = _run_partition_merge_rounds(
            members, embedder, llm_extractor,
            similarity_threshold=similarity_threshold,
            embed_batch_size=embed_batch_size,
            max_merge_rounds=max_merge_rounds,
            partition_type_label=partition_type_label,
            max_workers=max_parallel_workers,
            max_cluster_size=max_cluster_size,
        )
        for canonical_label, normalized_members in partition_groups:
            all_verified_groups.append((canonical_label, sorted(normalized_members)))

    logger.info(
        "Entity merge rounds complete across %d parent-partitions: %d unique labels -> %d groups",
        len(partitions), len(unique_labels), len(all_verified_groups),
    )

    # ── Build result (mirrors dedup_base.deduplicate()'s Step 5) ────────────
    def _build_entity(item_id, canonical_label, surface_forms, count_per_normalized,
                      surface_form_counts):
        assigned_type_ids: set[str] = set()
        for sf in surface_forms:
            type_id = type_id_by_normalized_compound.get(_normalize_compound(sf))
            if type_id:
                assigned_type_ids.add(type_id)
        primary_type_id = (
            lowest_common_ancestor(assigned_type_ids, hierarchy) if assigned_type_ids else None
        )
        return CanonicalEntity(
            item_id=item_id,
            canonical_label=canonical_label,
            surface_forms=surface_forms,
            count_per_normalized=count_per_normalized,
            surface_form_counts=surface_form_counts,
            type_ids=assigned_type_ids,
            primary_type_id=primary_type_id,
        )

    items: dict[str, CanonicalEntity] = {}
    surface_to_id: dict[str, str] = {}
    for idx, (canonical_label, members) in enumerate(all_verified_groups):
        item_id = f"ent_{idx:05d}"

        all_surface_forms: set[str] = set()
        total_count = 0
        for member in members:
            all_surface_forms.update(normalized_to_raws.get(member, {member}))
            all_surface_forms.add(member)
            total_count += count_per_normalized.get(member, 0)

        sorted_forms = sorted(all_surface_forms)
        sf_counts = {sf: surface_form_counts.get(sf, 0) for sf in sorted_forms}

        item = _build_entity(item_id, canonical_label, sorted_forms, total_count, sf_counts)
        items[item_id] = item

        for sf in all_surface_forms:
            surface_to_id[sf] = item_id
        surface_to_id[canonical_label] = item_id

    num_canonical = len(items)
    reduction = (1 - num_canonical / max(len(unique_labels), 1)) * 100

    logger.info(
        "Entity deduplication complete: %d -> %d canonical entities (%.1f%% reduction)",
        len(unique_labels), num_canonical, reduction,
    )

    return EntityDeduplicationResult(
        items=items,
        surface_to_id=surface_to_id,
        num_raw=len(unique_labels),
        num_canonical=num_canonical,
        reduction_pct=reduction,
        normalized_to_raws=dict(normalized_to_raws),
        surface_form_counts=dict(surface_form_counts),
    )
