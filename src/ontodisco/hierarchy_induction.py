"""
Hierarchy Induction via Recursive Fused-Similarity Clustering + LLM Labeling
============================================================================

Induces a subClassOf type hierarchy over the flat type vocabulary T* with a single recursive mechanism:

  1. Combine label-embedding similarity with relation-argument-signature
     similarity (cosine over the canonical RelationVocabulary's per-type
     profiles -- see relation_context.py) into one fused similarity score.
  2. Cluster the CURRENT pool of active nodes (leaf types, plus any
     already-formed parent surrogates from earlier rounds) via HDBSCAN on the
     fused score, with cluster_selection_epsilon derived from
     hac_threshold_floor (a similarity FLOOR nodes must clear to ever land in
     the same cluster) and allow_single_cluster=True (required -- see
     _hdbscan_round_cut). Unlike a single flat percentile-based HAC cut,
     HDBSCAN's stability-based selection can accept a tighter cluster in one
     part of the pool and a looser one in another within the SAME round,
     adapting to whatever density each branch of the vocabulary happens to
     have without needing an explicit per-round cut threshold at all.
  3. Every multi-member cluster is sent to the LLM (prompts/hierarchy_cluster_action.txt),
     which decides -- per sub-group -- "merge" (name a parent, preferring an
     existing member's label over inventing one), "no_parent" (decline;
     members stay siblings and are re-tried in a later round), or
     "same_concept" (these aren't parent/child at all -- they're independently-
     formed abstractions that turned out to be the identical concept under
     different wording, e.g. two unrelated batches each inventing "entity";
     collapsed into one survivor node via _collapse_duplicate_nodes rather
     than given a fake hierarchy edge). same_concept is restricted to
     synthesized nodes -- an original T* leaf type can never be a member of
     one, since Step 3 already made that identity call.
  4. The parent -- reused or invented -- replaces its children in the pool for
     the next round, with its relation profile rolled up (summed) from its
     children. This is what lets a later round correctly attach a leftover
     sibling (e.g. "elephant") to an already-formed parent (e.g. "mammal")
     once the parent's aggregated profile is broad enough: the surrogate
     simply re-enters the same pool and is re-clustered next round using
     real, richer evidence -- no separate nearest-neighbor safety net needed.
  5. Stop when a round's natural cut produces no new multi-member clusters, or
     no merges are accepted, or the pool drops below 2, or the required cut
     similarity falls below a floor, or a max-round budget is hit. Whatever
     remains in the pool becomes forest roots (TypeHierarchy is a forest, not
     a tree -- forcing convergence to one root tends to produce a
     semantically empty top node and is not attempted here).

Because a parent is only ever formed from nodes already in the PREVIOUS
round's pool, the result is acyclic and non-transitive by construction.

Precondition: relation_vocab's CanonicalRelation.subject_types / object_types
must already hold canonical type_ids rather than raw surface forms, i.e.
relation_dedup.update_relation_type_map() must have been run first.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.cluster import hierarchy as scipy_hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from src.ontodisco.utils.dedup_base import (
    ContrieverEmbedder,
    DeduplicationResult,
    normalize_label,
    union_find_from_pairs,
)
from src.ontodisco.relation_context import (
    build_relation_counts,
    build_type_relation_profiles,
    cosine_sim_sparse,
    describe_relation_context,
    weeds_precision,
)

if TYPE_CHECKING:
    from src.ontodisco.relation_dedup import RelationDeduplicationResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HierarchyEdge:
    """A single directed subClassOf edge. Direction: child IS-A parent."""
    child_type_id: str
    parent_type_id: str
    relation_signature_score: Optional[float]   # Weeds precision(child, parent)
    llm_score: Optional[float]                  # LLM's own confidence for this merge
    ensemble_score: float                       # kept for Step 7 compatibility; == llm_score here
    is_direct: bool = True                      # always True: no transitive shortcuts are possible
                                                 # by construction (see module docstring)


@dataclass
class TypeHierarchy:
    """The type hierarchy as a directed acyclic forest."""
    edges: list[HierarchyEdge]
    children: dict[str, list[str]]   # parent_type_id -> [child_type_ids]
    parents: dict[str, list[str]]    # child_type_id -> [parent_type_ids] (always length <= 1 here)
    roots: list[str]


@dataclass
class SynthesizedType:
    """
    A new abstract type minted during hierarchy induction, i.e. the LLM
    invented a canonical label rather than promoting an existing member as
    the parent. The caller is expected to fold these into the TypeVocabulary
    (T*) alongside the types produced by Step 3.
    """
    type_id: str
    canonical_label: str
    definition: str
    child_type_ids: list[str]


@dataclass
class HierarchyConfig:
    # -- Fused similarity (signal for grouping only, never for direction) --
    fusion_weight_embedding: float = 0.5     # weight on label-embedding cosine vs relation-profile cosine
    relation_signature_weighting: str = "ppmi"  # "ppmi" | "tfidf" | "raw" -- see relation_context.py

    # -- Recursive clustering --
    hac_threshold_floor: float = 0.35        # similarity floor: nodes farther apart than this can never
                                              # land in the same HDBSCAN cluster (cluster_selection_epsilon
                                              # = 1 - this); within the floor, HDBSCAN picks cluster
                                              # boundaries itself instead of using one flat per-round cut
    max_depth: int = 8                       # safety cap on number of rounds
    max_batch_size: int = 15                 # max cluster size sent to the LLM at once (re-split if exceeded)
    min_batch_size: int = 2                  # minimum cluster size worth sending to the LLM

    # -- Embedding --
    embed_batch_size: int = 64

    # -- LLM prompting --
    context_top_k: int = 5                   # top relation dimensions shown per candidate
    col_num_votes: int = 1                   # independent LLM calls per batch; 1 = no voting
    col_vote_agreement: int = 1              # min votes required to accept a merge (only if col_num_votes > 1)

    # -- Duplicate-label reconciliation (see _reconcile_duplicate_labels) --
    # Independent LLM batches routinely reinvent the same abstract parent label
    # (e.g. "entity") with no shared context, so every round we look for
    # SYNTHESIZED nodes (never original T* leaf types -- that split is Step 3's
    # decision to make, not this step's) that share a normalized label and
    # reconcile them.
    duplicate_label_profile_merge_floor: float = 0.3   # cosine sim >= this -> auto-merge, no LLM call
    duplicate_label_llm_batch_size: int = 30            # max ambiguous groups per disambiguation call

    # -- Stopping rule --
    # A round producing zero accepted merges doesn't necessarily mean the pool
    # is exhausted -- HDBSCAN can legitimately find no cluster clearing
    # hac_threshold_floor for one round's particular pool composition while a
    # later round (or the duplicate-label pass, which runs regardless of the
    # HDBSCAN cut) still has genuine progress to make. Only stop after this
    # many CONSECUTIVE rounds with no progress from either mechanism.
    max_consecutive_stall_rounds: int = 2

    # -- Final root-stitching pass --
    # After the main loop stops, the leftover pool (`TypeHierarchy.roots`) is
    # small (hundreds, not thousands) and already maximally-generalized, so
    # it's cheap to run a few more clustering rounds scoped to just the roots,
    # with a lower similarity floor than the main loop uses -- this is what
    # catches near-duplicate roots that differ in wording (e.g. "geographic
    # entity" vs. "geographical entity") rather than being exact-label repeats.
    root_stitch_threshold_floor: float = 0.15
    root_stitch_max_rounds: int = 3


@dataclass
class HierarchyInductionResult:
    hierarchy: TypeHierarchy
    synthesized_types: list[SynthesizedType] = field(default_factory=list)


@dataclass
class _Node:
    """One entry in the current active pool: either an original T* leaf type
    or a parent surrogate formed in an earlier round of this same process."""
    type_id: str
    label: str
    profile: dict[str, float]
    is_leaf: bool
    definition: str = ""

    def embed_text(self) -> str:
        if self.definition:
            return f"{self.label}. Definition: {self.definition}"
        return self.label


@dataclass
class _ResolvedGroup:
    member_type_ids: list[str]           # children only (parent excluded if reused)
    parent_label: str
    parent_is_new: bool
    parent_type_id: Optional[str] = None  # set only when parent_is_new is False
    parent_definition: str = ""
    confidence: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Relation-signature profiles (type_id -> {relation_id: weight})
# ═══════════════════════════════════════════════════════════════════════════════

def _initial_pool(
    type_vocab: DeduplicationResult,
    relation_vocab: "RelationDeduplicationResult",
    config: HierarchyConfig,
) -> dict[str, _Node]:
    # build_relation_counts() is generic over whatever subject_types/
    # object_types currently hold; by the time hierarchy induction runs,
    # relation_dedup.update_relation_type_map() has already resolved them to
    # canonical type_ids, so this yields type_id -> {relation_id: count}
    # directly, with no label -> type_id rollup step needed.
    type_relation_counts = build_relation_counts(relation_vocab)
    profiles = build_type_relation_profiles(
        type_relation_counts, weighting=config.relation_signature_weighting,
    )

    pool: dict[str, _Node] = {}
    for type_id, canonical_type in type_vocab.items.items():
        pool[type_id] = _Node(
            type_id=type_id,
            label=canonical_type.canonical_label,
            profile=profiles.get(type_id, {}),
            is_leaf=True,
        )

    n_with_profile = sum(1 for n in pool.values() if n.profile)
    logger.info(
        "Hierarchy induction: initial pool of %d types (%d with a non-empty "
        "relation-signature profile)",
        len(pool), n_with_profile,
    )
    return pool


def _merge_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = defaultdict(float)
    for profile in profiles:
        for dim, weight in profile.items():
            merged[dim] += weight
    return dict(merged)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fused similarity + adaptive round cut
# ═══════════════════════════════════════════════════════════════════════════════

def _fused_similarity_matrix(
    embeddings: np.ndarray,
    profiles: list[dict[str, float]],
    w_embedding: float,
) -> np.ndarray:
    """
    Combine label-embedding cosine similarity with relation-signature cosine
    similarity into one N x N matrix. 
    
    Falls back to pure embedding similarity for any pair where either node
    has an empty relation profile (too little relation evidence to say
    anything), rather than penalising it.
    """
    emb_sim = 1.0 - cosine_distances(embeddings)
    n = len(profiles)
    fused = emb_sim.copy()

    for i in range(n):
        if not profiles[i]:
            continue
        for j in range(i + 1, n):
            if not profiles[j]:
                continue
            rel_sim = cosine_sim_sparse(profiles[i], profiles[j])
            combined = w_embedding * emb_sim[i, j] + (1 - w_embedding) * rel_sim
            fused[i, j] = fused[j, i] = combined

    np.fill_diagonal(fused, 1.0)
    return fused


def _hdbscan_round_cut(
    fused_sim: np.ndarray,
    config: HierarchyConfig,
    *,
    threshold_floor: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Cluster this round's pool with HDBSCAN over the fused-similarity distance
    matrix, replacing the earlier fixed-percentile HAC cut: instead of
    picking one single cut distance for the whole pool, HDBSCAN's stability-
    based selection can accept a tighter cluster in one part of the pool and
    a looser one in another, within the same round.

    threshold_floor (a cosine SIMILARITY, overriding config.hac_threshold_floor
    -- used by the final root-stitching pass, which deliberately runs with a
    lower floor than the main loop, see HierarchyConfig.root_stitch_threshold_floor)
    is converted to cluster_selection_epsilon = 1 - floor: nodes farther apart
    than this floor can never land in the same cluster, but within it HDBSCAN
    decides cluster boundaries itself.

    allow_single_cluster=True is required: HDBSCAN's default rejects a pool
    that has no viable sub-split at the very top of its internal hierarchy
    and marks the ENTIRE pool as noise instead of calling it one cluster --
    verified empirically to otherwise silently noise-out an entire
    well-formed group of mutually-similar types that obviously belong
    together.

    Returns raw HDBSCAN labels (including -1 for noise), or None if the pool
    is too small to cluster at all.
    """
    floor = config.hac_threshold_floor if threshold_floor is None else threshold_floor
    n = fused_sim.shape[0]
    if n < 2:
        return None

    distance_matrix = np.clip(1.0 - fused_sim, 0.0, None)
    np.fill_diagonal(distance_matrix, 0.0)

    clusterer = HDBSCAN(
        metric="precomputed",
        min_cluster_size=max(2, config.min_batch_size),
        min_samples=1,
        cluster_selection_epsilon=max(0.0, 1.0 - floor),
        allow_single_cluster=True,
    )
    return clusterer.fit_predict(distance_matrix)


def _split_oversized(
    indices: list[int],
    fused_sim: np.ndarray,
    max_size: int,
) -> list[list[int]]:
    """If a cluster exceeds max_size, subdivide it via HAC into the fewest
    sub-clusters that each fit, using the same fused-similarity matrix."""
    if len(indices) <= max_size:
        return [indices]

    sub_sim = fused_sim[np.ix_(indices, indices)]
    distance_matrix = np.clip(1.0 - sub_sim, 0.0, None)
    np.fill_diagonal(distance_matrix, 0.0)
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = scipy_hierarchy.linkage(condensed, method="average")

    # how many groups we need to get groups with sizes <= max_size
    n_sub = math.ceil(len(indices) / max_size)
    sub_labels = scipy_hierarchy.fcluster(linkage_matrix, t=n_sub, criterion="maxclust")

    groups: dict[int, list[int]] = defaultdict(list)
    for local_idx, lbl in enumerate(sub_labels):
        groups[int(lbl)].append(indices[local_idx])
    return list(groups.values())


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM cluster resolution
# ═══════════════════════════════════════════════════════════════════════════════

def _reconcile_votes(votes: list[list[dict]], min_agreement: int) -> list[dict]:
    """Lightweight majority voting across independent LLM calls on the SAME
    batch: a merge or same_concept group is accepted only if the same
    (action, parent, member-set) triple appears in at least min_agreement of
    the votes. Declined ("no_parent") groups are never voted on -- members
    simply stay ungrouped by default."""
    tally: dict[tuple, int] = defaultdict(int)
    example: dict[tuple, dict] = {}

    for vote in votes:
        seen_this_vote = set()
        for group in vote:
            action = group.get("action")
            if action not in ("merge", "same_concept"):
                continue
            key = (
                action,
                str(group.get("parent", "")).strip().lower(),
                frozenset(str(m).strip().lower() for m in group.get("members", [])),
            )
            if key in seen_this_vote:
                continue
            seen_this_vote.add(key)
            tally[key] += 1
            example.setdefault(key, group)

    return [example[key] for key, count in tally.items() if count >= min_agreement]


def _resolve_cluster_with_llm(
    batch_nodes: list[_Node],
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    config: HierarchyConfig,
) -> tuple[list[_ResolvedGroup], list[list[str]]]:
    id_to_node = {node.type_id: node for node in batch_nodes}
    label_to_id: dict[str, str] = {}
    for node in batch_nodes:
        if node.label in label_to_id:
            logger.warning(
                "Duplicate label %r in hierarchy batch; keeping first occurrence", node.label,
            )
            continue
        label_to_id[node.label] = node.type_id

    # describe_relation_context() is generic over key space; batch_nodes are
    # keyed by type_id here, so build an ad-hoc type_id -> profile lookup
    # scoped to this batch rather than a corpus-wide one.
    batch_profiles = {node.type_id: node.profile for node in batch_nodes}
    member_context = {
        node.label: describe_relation_context(
            node.type_id, batch_profiles, relation_vocab, top_k=config.context_top_k,
        )
        for node in batch_nodes
    }
    member_context = {label: ctx for label, ctx in member_context.items() if ctx}

    votes: list[list[dict]] = []
    for _ in range(max(1, config.col_num_votes)):
        try:
            raw_groups = llm_extractor.resolve_hierarchy_cluster(
                members=list(label_to_id.keys()), member_context=member_context,
            )
        except Exception:
            logger.exception("Hierarchy cluster LLM call failed; treating batch as unresolved")
            raw_groups = []
        votes.append(raw_groups)

    groups = votes[0] if config.col_num_votes <= 1 else _reconcile_votes(votes, config.col_vote_agreement)

    resolved: list[_ResolvedGroup] = []
    same_concept_groups: list[list[str]] = []
    assigned: set[str] = set()
    # Tracks every type_id already claimed (as a parent OR a member) by an
    # earlier group in THIS SAME LLM response. A single response can be
    # internally inconsistent -- e.g. label X is listed as a member being
    # merged under Y in one group, then reused as the parent of a different
    # group later in the same response. Without this guard, the first group
    # pops X from the pool and a later group's _materialize_parent() call
    # crashes with a KeyError looking X back up. Reject (not just skip) any
    # group that touches an already-claimed type_id, rather than partially
    # applying it, since a group with a stale parent or child is not
    # trustworthy as a whole.
    claimed: set[str] = set()

    for group in groups:
        action = group.get("action")
        if action == "same_concept":
            raw_members = group.get("members", [])
            member_ids = [label_to_id[m] for m in raw_members if m in label_to_id]
            if len(member_ids) < 2:
                continue
            if claimed.intersection(member_ids):
                logger.warning(
                    "Hierarchy LLM response reused type_id(s) across groups "
                    "(same_concept members=%r); skipping this conflicting group",
                    raw_members,
                )
                continue
            # Never collapse original T* leaf types on this signal -- that
            # split (or non-split) was Step 3's deliberate decision to make
            # (see homonym-split tests), not this step's to relitigate. Only
            # independently-synthesized abstractions formed earlier in this
            # same recursive process are fair game here.
            if any(id_to_node[tid].is_leaf for tid in member_ids):
                logger.warning(
                    "Hierarchy LLM proposed same_concept collapse touching an "
                    "original leaf type (members=%r); skipping -- leaf identity "
                    "is Step 3's decision, not hierarchy induction's", raw_members,
                )
                continue
            same_concept_groups.append(member_ids)
            claimed.update(member_ids)
            assigned.update(member_ids)
            continue

        if action != "merge":
            continue

        raw_members = group.get("members", [])
        member_ids = [label_to_id[m] for m in raw_members if m in label_to_id]
        if not member_ids:
            continue

        parent_label = str(group.get("parent", "")).strip()
        parent_is_new = bool(group.get("parent_is_new", False))
        try:
            confidence = float(group.get("confidence", 0.5))
        except (TypeError, ValueError):
            logger.warning(
                "Hierarchy LLM returned a non-numeric confidence %r; defaulting to 0.5",
                group.get("confidence"),
            )
            confidence = 0.5

        if not parent_label:
            logger.warning("Hierarchy LLM proposed merge with no parent label; skipping group %r", raw_members)
            continue

        if not parent_is_new and parent_label in label_to_id:
            parent_id = label_to_id[parent_label]
            children = [tid for tid in member_ids if tid != parent_id]
            if not children:
                continue
            if parent_id in claimed or claimed.intersection(children):
                logger.warning(
                    "Hierarchy LLM response reused type_id(s) across groups "
                    "(parent=%r, members=%r); skipping this conflicting group",
                    parent_label, raw_members,
                )
                continue
            resolved.append(_ResolvedGroup(
                member_type_ids=children, parent_label=parent_label,
                parent_is_new=False, parent_type_id=parent_id, confidence=confidence,
            ))
            assigned.update(children)
            claimed.add(parent_id)
            claimed.update(children)
        else:
            if claimed.intersection(member_ids):
                logger.warning(
                    "Hierarchy LLM response reused type_id(s) across groups "
                    "(parent=%r, members=%r); skipping this conflicting group",
                    parent_label, raw_members,
                )
                continue
            resolved.append(_ResolvedGroup(
                member_type_ids=member_ids, parent_label=parent_label,
                parent_is_new=True, parent_definition=group.get("parent_definition", "") or "",
                confidence=confidence,
            ))
            assigned.update(member_ids)
            claimed.update(member_ids)

    dropped = [m for m, tid in label_to_id.items() if tid not in assigned]
    if dropped:
        logger.debug(
            "Hierarchy LLM left %d member(s) ungrouped this round: %s", len(dropped), dropped,
        )

    return resolved, same_concept_groups


def _materialize_parent(
    group: _ResolvedGroup,
    pool: dict[str, _Node],
    synthetic_counter: list[int],
) -> Optional[_Node]:
    children_profiles = [pool[tid].profile for tid in group.member_type_ids if tid in pool]

    if not group.parent_is_new and group.parent_type_id is not None:
        # Belt-and-suspenders: _resolve_cluster_with_llm already rejects
        # groups that reuse a type_id claimed elsewhere in the same LLM
        # response, but this stays defensive against any other path that
        # could hand back a parent_type_id no longer in the pool, rather
        # than crashing the whole run on a KeyError.
        parent_node = pool.get(group.parent_type_id)
        if parent_node is None:
            logger.warning(
                "Hierarchy: parent type_id %s (%r) no longer in pool; skipping group",
                group.parent_type_id, group.parent_label,
            )
            return None
        parent_node.profile = _merge_profiles([parent_node.profile] + children_profiles)
        return parent_node

    synthetic_id = f"type_h{synthetic_counter[0]:04d}"
    synthetic_counter[0] += 1
    return _Node(
        type_id=synthetic_id,
        label=group.parent_label,
        profile=_merge_profiles(children_profiles),
        is_leaf=False,
        definition=group.parent_definition,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Duplicate-label reconciliation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Large clusters get split into <= max_batch_size groups and resolved by
# INDEPENDENT LLM calls with no shared context (see _split_oversized /
# induce_hierarchy's batch loop), so the same abstract parent concept
# routinely gets invented more than once under the identical label (e.g.
# five different batches each proposing "entity" as a new parent). Left
# alone, these never reconcile with each other unless they happen to land in
# the same round's clustering pool AND HDBSCAN accepts them into one cluster.
#
# This reconciliation pass runs every round, scoped ONLY to SYNTHESIZED
# nodes currently in the pool -- never to original T* leaf types. Two
# original T* types can share a label too (this corpus has 64 such pairs),
# but that split was Step 3's (type canonicalization's) deliberate decision
# -- see its homonym-split tests -- and is not this step's to relitigate.
#
# A same-label pair of synthesized nodes is merged automatically only when
# their relation-argument profiles also overlap strongly (the same kind of
# symbolic gate CLAUDE.md already uses for relations via
# arg_sig_split_threshold, applied here to types): label match alone is not
# proof of same concept, and this corpus is proof of that (see the 64 T*
# pairs above).
#
# Deliberately ONE-DIRECTIONAL: high cosine similarity confirms a merge, but
# low/zero similarity does NOT confirm a split. Verified empirically on this
# corpus -- rolled-up profiles for large abstract nodes (e.g. "entity") are
# thin (a handful of PPMI-surviving dimensions each, out of thousands of
# possible relation_ids), so two genuinely-identical duplicate nodes from
# different subtrees routinely come back cosine=0.0 purely from sparsity,
# not real semantic difference. Treating that as confirmed-different would
# silently leave the vast majority of true duplicates split forever -- the
# opposite of the point of this pass. So anything short of a confident merge
# (empty profile on either side, OR both present but below the floor) is
# left ambiguous and batched into one LLM disambiguation call, rather than
# being silently force-merged OR silently accepted as a real homonym split.

def _duplicate_label_groups(pool: dict[str, "_Node"]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for type_id, node in pool.items():
        if node.is_leaf:
            continue
        groups[normalize_label(node.label)].append(type_id)
    return {label: ids for label, ids in groups.items() if len(ids) > 1}


def _resolve_label_ambiguity_with_llm(
    ambiguous: list[tuple[str, list[str]]],
    pool: dict[str, "_Node"],
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    config: HierarchyConfig,
) -> list[list[str]]:
    """
    For each (label, type_ids) group where profile evidence couldn't confirm
    or deny same-concept, ask the LLM once (batched, config.duplicate_label_
    llm_batch_size groups per call) whether the group's members are the same
    concept or genuinely distinct senses of the same wording. Returns the
    list of type_id clusters to collapse (only groups confirmed same-concept).
    """
    to_merge: list[list[str]] = []
    for start in range(0, len(ambiguous), config.duplicate_label_llm_batch_size):
        batch = ambiguous[start:start + config.duplicate_label_llm_batch_size]
        # Individual children are already consumed out of `pool` by the time
        # they're rolled into a parent's profile, so their labels aren't
        # retrievable here -- relation-context evidence (which relations this
        # node's aggregated profile fires on) is the best available proxy for
        # "what does this node cover" beyond the (often-empty) definition.
        batch_profiles = {tid: pool[tid].profile for _, ids in batch for tid in ids}
        groups_payload = [
            {
                "label": label,
                "candidates": [
                    {
                        "id": tid,
                        "definition": pool[tid].definition,
                        "relation_context": describe_relation_context(
                            tid, batch_profiles, relation_vocab, top_k=config.context_top_k,
                        ),
                    }
                    for tid in ids
                ],
            }
            for label, ids in batch
        ]
        try:
            decisions = llm_extractor.disambiguate_duplicate_labels(groups_payload)
        except Exception:
            logger.exception(
                "Duplicate-label disambiguation LLM call failed; leaving %d group(s) split",
                len(batch),
            )
            continue

        for label, ids in batch:
            same_concept_ids = decisions.get(label, [])
            # A decision may itself split the group into more than one
            # same-concept cluster (e.g. 3 nodes labeled "type" turn out to
            # be 2 + 1 distinct senses) -- decisions.get(label) is a list of
            # clusters, each a list of type_ids confirmed as one concept.
            for cluster in same_concept_ids:
                cluster = [tid for tid in cluster if tid in ids]
                if len(cluster) > 1:
                    to_merge.append(cluster)
    return to_merge


def _collapse_duplicate_nodes(
    cluster_ids: list[str],
    pool: dict[str, "_Node"],
    edges: list[HierarchyEdge],
    synthesized: list[SynthesizedType],
) -> None:
    """Collapse a cluster of duplicate synthesized nodes into one survivor
    (the earliest-created, i.e. lowest type_hNNNN -- deterministic and stable
    across runs given the same LLM outputs). Every edge and SynthesizedType
    record pointing at a loser is rewritten to point at the survivor."""
    survivor_id = min(cluster_ids)
    loser_ids = [tid for tid in cluster_ids if tid != survivor_id]

    survivor = pool[survivor_id]
    survivor.profile = _merge_profiles([pool[tid].profile for tid in cluster_ids])

    synth_by_id = {s.type_id: s for s in synthesized}
    survivor_synth = synth_by_id.get(survivor_id)

    for loser_id in loser_ids:
        loser_synth = synth_by_id.get(loser_id)
        if survivor_synth is not None and loser_synth is not None:
            survivor_synth.child_type_ids.extend(loser_synth.child_type_ids)
        if loser_synth is not None:
            synthesized.remove(loser_synth)

        for edge in edges:
            if edge.parent_type_id == loser_id:
                edge.parent_type_id = survivor_id

        pool.pop(loser_id, None)

    logger.debug(
        "Duplicate-label reconciliation: collapsed %r into %s", cluster_ids, survivor_id,
    )


def _reconcile_duplicate_labels(
    pool: dict[str, "_Node"],
    edges: list[HierarchyEdge],
    synthesized: list[SynthesizedType],
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    config: HierarchyConfig,
) -> bool:
    """Find and collapse duplicate-labeled SYNTHESIZED nodes in the current
    pool. Returns True iff at least one collapse happened (counts as round
    progress for the stall-based stopping rule)."""
    dup_groups = _duplicate_label_groups(pool)
    if not dup_groups:
        return False

    to_merge: list[list[str]] = []
    ambiguous: list[tuple[str, list[str]]] = []

    for label, ids in dup_groups.items():
        pairs: list[tuple[int, int]] = []
        undecided = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                profile_i, profile_j = pool[ids[i]].profile, pool[ids[j]].profile
                # Cosine similarity over these rolled-up profiles is only a
                # trustworthy POSITIVE signal: two duplicate-labeled nodes
                # from different subtrees genuinely can accumulate only a
                # handful of relation dimensions each (PPMI zeroes out most),
                # so two truly-identical nodes routinely land on cosine=0.0
                # just from sparsity, not real semantic difference (verified
                # empirically on this corpus's "entity"/"concept"/"status"
                # duplicates -- every sampled pair came back 0.000 despite
                # almost certainly being the same accidental-duplication
                # failure mode, not real homonyms). So low/zero similarity is
                # NOT treated as confirmed-different -- only high similarity
                # confirms merge; everything else is left ambiguous for the
                # LLM rather than silently accepted as a real split.
                if profile_i and profile_j and cosine_sim_sparse(profile_i, profile_j) >= config.duplicate_label_profile_merge_floor:
                    pairs.append((i, j))
                else:
                    undecided = True

        cluster_labels = union_find_from_pairs(len(ids), pairs)
        clusters: dict[int, list[str]] = defaultdict(list)
        for idx, tid in enumerate(ids):
            clusters[int(cluster_labels[idx])].append(tid)

        for cluster_ids in clusters.values():
            if len(cluster_ids) > 1:
                to_merge.append(cluster_ids)

        # Only nodes that profile evidence left as their own singleton
        # cluster are genuinely undecided -- anything already folded into a
        # confirmed merge above doesn't need the LLM.
        if undecided:
            leftover = [tid for tid in ids if len(clusters[int(cluster_labels[ids.index(tid)])]) == 1]
            if len(leftover) > 1:
                ambiguous.append((label, leftover))

    if ambiguous:
        to_merge.extend(
            _resolve_label_ambiguity_with_llm(ambiguous, pool, relation_vocab, llm_extractor, config)
        )

    if not to_merge:
        return False

    for cluster_ids in to_merge:
        # A node may already have been folded into a survivor by an earlier
        # cluster in this same pass (shouldn't happen given groups are keyed
        # by distinct labels, but stay defensive rather than KeyError).
        live_ids = [tid for tid in cluster_ids if tid in pool]
        if len(live_ids) > 1:
            _collapse_duplicate_nodes(live_ids, pool, edges, synthesized)

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  One clustering round (shared by the main loop and the final root-stitching pass)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_clustering_round(
    round_idx: int,
    pool: dict[str, _Node],
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    config: HierarchyConfig,
    embedder: ContrieverEmbedder,
    edges: list[HierarchyEdge],
    synthesized: list[SynthesizedType],
    synthetic_counter: list[int],
    *,
    threshold_floor: Optional[float] = None,
    desc_prefix: str = "Round",
) -> bool:
    """
    Run one HDBSCAN + LLM clustering round over the current `pool`, mutating
    pool/edges/synthesized in place. Returns True iff at least one merge was
    accepted.

    threshold_floor overrides config.hac_threshold_floor for this round only
    -- used by the final root-stitching pass (see HierarchyConfig.
    root_stitch_threshold_floor).
    """
    if len(pool) < config.min_batch_size:
        return False

    node_ids = list(pool.keys())
    nodes = [pool[tid] for tid in node_ids]

    embeddings = embedder.embed(
        [n.embed_text() for n in nodes], batch_size=config.embed_batch_size,
    )
    fused_sim = _fused_similarity_matrix(
        embeddings, [n.profile for n in nodes], config.fusion_weight_embedding,
    )

    flat_labels = _hdbscan_round_cut(fused_sim, config, threshold_floor=threshold_floor)
    if flat_labels is None:
        logger.info(
            "Hierarchy induction: %s %d found nothing (pool too small)",
            desc_prefix, round_idx,
        )
        return False

    floor_used = config.hac_threshold_floor if threshold_floor is None else threshold_floor
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cl in enumerate(flat_labels):
        if cl == -1:
            continue  # HDBSCAN noise -- no confident merge partner this round, stays in pool
        clusters[int(cl)].append(idx)

    multi_clusters = {cid: idxs for cid, idxs in clusters.items() if len(idxs) >= config.min_batch_size}
    if not multi_clusters:
        logger.info(
            "Hierarchy induction: %s %d fixed point (similarity floor=%.2f, "
            "pool=%d) -- nothing left to merge", desc_prefix, round_idx, floor_used, len(pool),
        )
        return False

    logger.info(
        "Hierarchy %s %d: pool=%d, similarity floor=%.2f, %d candidate cluster(s) (HDBSCAN)",
        desc_prefix, round_idx, len(pool), floor_used, len(multi_clusters),
    )

    batches = [
        sub_idxs
        for idxs in multi_clusters.values()
        for sub_idxs in _split_oversized(idxs, fused_sim, config.max_batch_size)
        if len(sub_idxs) >= config.min_batch_size
    ]

    any_merge = False
    for sub_idxs in tqdm(batches, desc=f"{desc_prefix} {round_idx} (LLM batches)"):
        batch_nodes = [nodes[i] for i in sub_idxs]
        resolved_groups, same_concept_groups = _resolve_cluster_with_llm(
            batch_nodes, relation_vocab, llm_extractor, config,
        )

        for cluster_ids in same_concept_groups:
            live_ids = [tid for tid in cluster_ids if tid in pool]
            if len(live_ids) > 1:
                _collapse_duplicate_nodes(live_ids, pool, edges, synthesized)
                any_merge = True

        for group in resolved_groups:
            parent_node = _materialize_parent(group, pool, synthetic_counter)
            if parent_node is None:
                continue

            if group.parent_is_new:
                pool[parent_node.type_id] = parent_node
                synthesized.append(SynthesizedType(
                    type_id=parent_node.type_id,
                    canonical_label=group.parent_label,
                    definition=group.parent_definition,
                    child_type_ids=list(group.member_type_ids),
                ))

            for child_id in group.member_type_ids:
                if child_id not in pool:
                    logger.warning(
                        "Hierarchy %s %d: %s already resolved this round, "
                        "skipping duplicate assignment", desc_prefix, round_idx, child_id,
                    )
                    continue

                child_profile = pool[child_id].profile
                rel_score = (
                    weeds_precision(child_profile, parent_node.profile)
                    if child_profile and parent_node.profile else None
                )

                edges.append(HierarchyEdge(
                    child_type_id=child_id,
                    parent_type_id=parent_node.type_id,
                    relation_signature_score=rel_score,
                    llm_score=group.confidence,
                    ensemble_score=group.confidence,
                    is_direct=True,
                ))
                pool.pop(child_id, None)

            any_merge = True

    return any_merge


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def induce_hierarchy(
    type_vocab: DeduplicationResult,
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    *,
    contriever_model: str = "facebook/contriever",
    device: str = None,
    config: Optional[HierarchyConfig] = None,
) -> HierarchyInductionResult:
    """
    Induce a subClassOf hierarchy over the flat type vocabulary T*.

    Requires relation_vocab's CanonicalRelation.subject_types / object_types
    to already hold canonical type_ids (relation_dedup.update_relation_type_map()
    must have been called first) so relation-signature profiles can be built
    directly at the type_id level.
    """
    config = config or HierarchyConfig()
    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    pool = _initial_pool(type_vocab, relation_vocab, config)
    edges: list[HierarchyEdge] = []
    synthesized: list[SynthesizedType] = []
    synthetic_counter = [0]

    round_idx = 0
    stall_rounds = 0
    while len(pool) >= config.min_batch_size and round_idx < config.max_depth:
        round_idx += 1
        cluster_progress = _run_clustering_round(
            round_idx, pool, relation_vocab, llm_extractor, config, embedder,
            edges, synthesized, synthetic_counter,
        )
        # Runs regardless of whether the HDBSCAN round found anything: independent
        # LLM batches routinely reinvent the same abstract label (see module
        # docstring), and reconciling that is cheap enough to check every
        # round rather than waiting for a final pass.
        label_progress = _reconcile_duplicate_labels(
            pool, edges, synthesized, relation_vocab, llm_extractor, config,
        )

        if cluster_progress or label_progress:
            stall_rounds = 0
        else:
            stall_rounds += 1
            if stall_rounds >= config.max_consecutive_stall_rounds:
                logger.info(
                    "Hierarchy induction: stopping at round %d after %d consecutive "
                    "stall round(s)", round_idx, stall_rounds,
                )
                break

    logger.info(
        "Hierarchy induction: main loop complete after %d round(s), %d edges, "
        "%d synthesized types, %d root(s) -- starting root-stitching pass",
        round_idx, len(edges), len(synthesized), len(pool),
    )

    # Final root-stitching: the leftover pool is now small (roots only) and
    # already maximally-generalized, so a few more rounds with a relaxed
    # similarity floor are cheap and catch near-duplicate roots that differ
    # in wording (e.g. "geographic entity" vs. "geographical entity") --
    # something exact-label reconciliation above can't catch on its own.
    stitch_round = 0
    stitch_stall = 0
    while (
        stitch_round < config.root_stitch_max_rounds
        and len(pool) >= config.min_batch_size
        and stitch_stall < config.max_consecutive_stall_rounds
    ):
        stitch_round += 1
        cluster_progress = _run_clustering_round(
            stitch_round, pool, relation_vocab, llm_extractor, config, embedder,
            edges, synthesized, synthetic_counter,
            threshold_floor=config.root_stitch_threshold_floor,
            desc_prefix="Root-stitch round",
        )
        label_progress = _reconcile_duplicate_labels(
            pool, edges, synthesized, relation_vocab, llm_extractor, config,
        )
        stitch_stall = 0 if (cluster_progress or label_progress) else stitch_stall + 1

    logger.info(
        "Hierarchy induction complete: %d main round(s) + %d root-stitch round(s), "
        "%d edges, %d synthesized types, %d root(s)",
        round_idx, stitch_round, len(edges), len(synthesized), len(pool),
    )

    hierarchy = _build_type_hierarchy(edges, roots=list(pool.keys()))
    return HierarchyInductionResult(hierarchy=hierarchy, synthesized_types=synthesized)


def _build_type_hierarchy(edges: list[HierarchyEdge], roots: list[str]) -> TypeHierarchy:
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        children[edge.parent_type_id].append(edge.child_type_id)
        parents[edge.child_type_id].append(edge.parent_type_id)

    return TypeHierarchy(
        edges=edges,
        children=dict(children),
        parents=dict(parents),
        roots=sorted(roots),
    )
