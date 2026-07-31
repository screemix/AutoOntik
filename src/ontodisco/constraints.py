"""
Domain/Range Constraint Induction
==================================

Mines a domain (expected subject type) and range (expected object type) for
every canonical relation.

Three stages, each addressing a specific problem surfaced while designing
this step (see CLAUDE.md Step 6 for the full writeup):

  0. Re-resolve raw triplets to (relation_id, subject_type_id, object_type_id).
     Nothing upstream kept this joint pairing -- CanonicalRelation.subject_types
     / object_types (relation_dedup.py) are marginal sets, not paired counts --
     so it has to be rebuilt here from the raw triplet dicts.

  1. Relation-direction resolution. Relation canonicalization (relation_dedup.py)
     merges surface forms role-blind, so a canonical relation can silently pool
     a predicate with its grammatical inverse (e.g. "directed" / "was directed
     by"). Left uncorrected this fragments domain/range support into two
     competing, individually-weak orientations. Direction can't be recovered
     from type statistics in general (subject/object type sets always share
     *some* common ancestor eventually, and plenty of genuinely directional
     relations have identical domain/range types to begin with) -- it comes
     from the literal surface form via one small LLM call per multi-surface-
     form relation (see prompts/relation_direction.txt). This does not catch
     genuinely order-free relations (e.g. "collaborated with") where a
     single surface form gets extracted with subject/object swapped across
     sentences purely by chance -- domain and range are still walked
     independently for those, which is a known simplification, not
     something this module tries to detect.

  2. Hierarchy generalization via exact LCA (no threshold, no pruning). Every
     observed type, however rare, is folded into its lowest common ancestor
     with the rest of its role's observed types (TypeHierarchy is a
     single-parent forest, so this is a deterministic ancestor-path
     intersection). If the observed types don't all share one ancestor, they
     are partitioned by which root-tree they descend from and generalized
     independently per partition, producing more than one signature type
     (and hence more than one RelationConstraint) for that relation.

Confidence (PCA-style) is scored once, at the very end, on the resulting
direction-corrected, hierarchy-generalized (domain, range) pairs, using the
same three thresholds (hard/soft/hint) as the original design -- the only
tunables anywhere in this module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from src.ontodisco.utils.dedup_base import normalize_label
from src.ontodisco.hierarchy_induction import ancestor_path, lowest_common_ancestor

if TYPE_CHECKING:
    from src.ontodisco.hierarchy_induction import TypeHierarchy
    from src.ontodisco.relation_dedup import RelationDeduplicationResult
    from src.ontodisco.type_dedup import TypeDeduplicationResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintStrength(Enum):
    HARD = "hard"    # pca_confidence >= hard_threshold
    SOFT = "soft"     # pca_confidence >= soft_threshold
    HINT = "hint"     # pca_confidence >= hint_threshold


@dataclass
class RelationConstraint:
    """One domain/range constraint for a canonical relation. A relation can
    have several of these (one per disconnected hierarchy partition, or a
    genuinely disjunctive domain/range)."""
    relation_id: str
    domain_type_id: str
    range_type_id: str
    support: int              # joint triple count backing this (domain, range) pair
    total: int                # total resolved+direction-corrected triples for this relation
    pca_confidence: float     # support / total
    strength: ConstraintStrength


@dataclass
class ConstraintConfig:
    hard_threshold: float = 0.90
    soft_threshold: float = 0.50
    hint_threshold: float = 0.20


@dataclass
class _ResolvedTriplet:
    relation_id: str
    subject_type_id: str
    object_type_id: str
    surface_relation: str   # raw predicate string, needed to look up its direction label


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 0 — Re-resolve raw triplets to canonical ids
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_triplets(
    triplets: list[dict],
    type_vocab: "TypeDeduplicationResult",
    relation_vocab: "RelationDeduplicationResult",
) -> list[_ResolvedTriplet]:
    """
    Resolve each raw triplet dict's relation/subject_type/object_type surface
    forms to canonical ids. Triplets that don't resolve (empty fields, or a
    surface form that was never seen by relation_dedup.py/type_dedup.py) are
    dropped and counted, not raised on -- extraction noise is expected.
    """
    resolved: list[_ResolvedTriplet] = []
    n_dropped = 0

    for triplet in triplets:
        raw_relation = triplet.get("relation", "").strip()
        raw_subject_type = triplet.get("subject_type", "").strip()
        raw_object_type = triplet.get("object_type", "").strip()
        if not raw_relation or not raw_subject_type or not raw_object_type:
            n_dropped += 1
            continue

        relation_id = relation_vocab.surface_to_id.get(raw_relation) or \
            relation_vocab.surface_to_id.get(normalize_label(raw_relation))
        subject_type_id = type_vocab.surface_to_id.get(normalize_label(raw_subject_type)) or \
            type_vocab.surface_to_id.get(raw_subject_type)
        object_type_id = type_vocab.surface_to_id.get(normalize_label(raw_object_type)) or \
            type_vocab.surface_to_id.get(raw_object_type)

        if not (relation_id and subject_type_id and object_type_id):
            n_dropped += 1
            continue

        resolved.append(_ResolvedTriplet(
            relation_id=relation_id,
            subject_type_id=subject_type_id,
            object_type_id=object_type_id,
            surface_relation=raw_relation,
        ))

    logger.info(
        "Resolved %d/%d triplets to (relation_id, subject_type_id, object_type_id) "
        "(%d dropped: unresolved relation/type surface form)",
        len(resolved), len(triplets), n_dropped,
    )
    return resolved


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Relation-direction resolution
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_relation_directions(
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    relation_ids: set[str],
) -> dict[str, dict[str, str]]:
    """
    relation_id -> {normalized_surface_form: "forward" | "inverted"}

    Only classifies relations in `relation_ids` (the ones that actually have
    resolved triplets) to avoid wasted LLM calls. Relations with a single
    surface form skip the LLM call entirely -- nothing to disambiguate, and
    that surface form is trivially "forward".
    """
    direction_by_relation: dict[str, dict[str, str]] = {}

    for relation_id in relation_ids:
        relation = relation_vocab.relations[relation_id]
        surface_forms = relation.surface_forms

        if len(surface_forms) <= 1:
            labels = {sf: "forward" for sf in surface_forms}
        else:
            labels = llm_extractor.classify_relation_direction(
                canonical_label=relation.canonical_label,
                surface_forms=surface_forms,
            )

        direction_by_relation[relation_id] = {
            normalize_label(sf): direction for sf, direction in labels.items()
        }

    n_with_inversion = sum(
        1 for labels in direction_by_relation.values()
        if any(v == "inverted" for v in labels.values())
    )
    logger.info(
        "Direction classification: %d relations classified, %d with an inverted surface form",
        len(direction_by_relation), n_with_inversion,
    )
    return direction_by_relation


def _apply_direction(
    resolved: list[_ResolvedTriplet],
    direction_by_relation: dict[str, dict[str, str]],
) -> list[_ResolvedTriplet]:
    """
    Swap subject/object on triplets whose surface form was classified
    "inverted", so every triplet for a relation shares one consistent
    orientation.
    """
    corrected: list[_ResolvedTriplet] = []

    for rt in resolved:
        direction = direction_by_relation.get(rt.relation_id, {}).get(
            normalize_label(rt.surface_relation), "forward"
        )
        if direction == "inverted":
            corrected.append(_ResolvedTriplet(
                relation_id=rt.relation_id,
                subject_type_id=rt.object_type_id,
                object_type_id=rt.subject_type_id,
                surface_relation=rt.surface_relation,
            ))
        else:
            corrected.append(rt)

    return corrected


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — Hierarchy generalization (exact LCA, no threshold)
# ═══════════════════════════════════════════════════════════════════════════════

# ancestor_path / lowest_common_ancestor live in hierarchy_induction.py (that's
# where TypeHierarchy itself is defined) -- entity_dedup.py needs the same LCA
# walk for its primary_type_id field, so it's a shared helper rather than a
# constraints.py-private one.


def _root_of(type_id: str, hierarchy: "TypeHierarchy") -> str:
    return ancestor_path(type_id, hierarchy)[-1]


def _generalize_to_signature_map(
    observed_types: set[str],
    hierarchy: "TypeHierarchy",
) -> dict[str, str]:
    """
    Map every type in observed_types to its signature type (the most
    specific common ancestor). If they all share one, every type maps to
    that single node. If they don't, types are partitioned by root-tree and
    each partition gets its own signature -- one relation's domain (or
    range) can end up with more than one signature type. No thresholding:
    every observed type, however rare, is included in some partition.
    """
    if not observed_types:
        return {}

    lca = lowest_common_ancestor(observed_types, hierarchy)
    if lca is not None:
        return {t: lca for t in observed_types}

    by_root: dict[str, set[str]] = defaultdict(set)
    for type_id in observed_types:
        by_root[_root_of(type_id, hierarchy)].add(type_id)

    signature_map: dict[str, str] = {}
    for members in by_root.values():
        partition_lca = lowest_common_ancestor(members, hierarchy)  # always resolves: members share their root
        for t in members:
            signature_map[t] = partition_lca
    return signature_map


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Confidence scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_strength(
    pca_confidence: float, config: ConstraintConfig,
) -> Optional[ConstraintStrength]:
    if pca_confidence >= config.hard_threshold:
        return ConstraintStrength.HARD
    if pca_confidence >= config.soft_threshold:
        return ConstraintStrength.SOFT
    if pca_confidence >= config.hint_threshold:
        return ConstraintStrength.HINT
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def induce_constraints(
    triplets: list[dict],
    type_vocab: "TypeDeduplicationResult",
    relation_vocab: "RelationDeduplicationResult",
    hierarchy: "TypeHierarchy",
    llm_extractor,
    config: Optional[ConstraintConfig] = None,
) -> list[RelationConstraint]:
    """
    Mine domain/range constraints for every canonical relation with at least
    one resolvable triplet.

    Args:
        triplets:      Raw triplet dicts (same input pipeline.py loads).
        type_vocab:    Output of type_dedup.deduplicate_types().
        relation_vocab: Output of relation_dedup.deduplicate_relations(),
                       AFTER relation_dedup.update_relation_type_map() has
                       run (subject_types/object_types must already be
                       type_ids -- not required here directly, but the
                       caller's pipeline order should guarantee it).
        hierarchy:     TypeHierarchy from hierarchy_induction.induce_hierarchy().
        llm_extractor: LLMTripletExtractor instance (classify_relation_direction()).
        config:        Confidence thresholds (see ConstraintConfig).

    Returns:
        RelationConstraints, one per (relation_id, domain, range) triple that
        cleared hint_threshold. Multiple constraints per relation are allowed.
    """
    if config is None:
        config = ConstraintConfig()

    resolved = _resolve_triplets(triplets, type_vocab, relation_vocab)
    if not resolved:
        logger.warning("induce_constraints: no triplets resolved, returning no constraints")
        return []

    relevant_relation_ids = {rt.relation_id for rt in resolved}
    direction_by_relation = _classify_relation_directions(
        relation_vocab, llm_extractor, relevant_relation_ids,
    )
    corrected = _apply_direction(resolved, direction_by_relation)

    by_relation: dict[str, list[_ResolvedTriplet]] = defaultdict(list)
    for rt in corrected:
        by_relation[rt.relation_id].append(rt)

    constraints: list[RelationConstraint] = []
    n_discarded = 0

    for relation_id, rts in by_relation.items():
        total = len(rts)

        subject_types = {rt.subject_type_id for rt in rts}
        object_types = {rt.object_type_id for rt in rts}
        domain_map = _generalize_to_signature_map(subject_types, hierarchy)
        range_map = _generalize_to_signature_map(object_types, hierarchy)

        # Regroup by resolved (domain_signature, range_signature) using the
        # per-triplet mapping -- NOT a cross-product of the marginal
        # signature sets, which would fabricate (domain, range) pairs that
        # never actually co-occurred.
        joint_support: dict[tuple[str, str], int] = defaultdict(int)
        for rt in rts:
            domain_sig = domain_map[rt.subject_type_id]
            range_sig = range_map[rt.object_type_id]
            joint_support[(domain_sig, range_sig)] += 1

        for (domain_type_id, range_type_id), support in joint_support.items():
            pca_confidence = support / total
            strength = _assign_strength(pca_confidence, config)
            if strength is None:
                n_discarded += 1
                logger.debug(
                    "Discarding constraint: relation=%s domain=%s range=%s "
                    "support=%d/%d (pca_confidence=%.2f below hint_threshold=%.2f)",
                    relation_id, domain_type_id, range_type_id, support, total,
                    pca_confidence, config.hint_threshold,
                )
                continue
            constraints.append(RelationConstraint(
                relation_id=relation_id,
                domain_type_id=domain_type_id,
                range_type_id=range_type_id,
                support=support,
                total=total,
                pca_confidence=pca_confidence,
                strength=strength,
            ))

    logger.info(
        "Constraint induction complete: %d constraints emitted across %d relations "
        "(%d discarded below hint_threshold)",
        len(constraints), len(by_relation), n_discarded,
    )
    return constraints
