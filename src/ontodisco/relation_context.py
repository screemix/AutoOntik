"""
Relation-Signature Context for Entity Types
============================================

Turns the output of relation deduplication (relation_dedup.py) into a
per-type-label distributional signature: a sparse vector over canonical
relation dimensions, describing which relations a type tends to participate
in (as either subject or object — the role is deliberately not part of the
dimension, since synonymous/inverse relations can swap which argument slot a
type fills, e.g. "type" vs. "type of").

This is the distributional inclusion hypothesis (Weeds & Weir 2003) applied to
relation-argument slots instead of window-based word co-occurrence: two type
labels that fill the same argument slots of the same relations are evidence of
being the same or closely related concepts. It requires no entity-level
deduplication and no corpus re-scanning — only the (small) relation vocabulary
already produced by relation_dedup.deduplicate_relations().

Two consumers:
  - type_dedup.py:  cosine similarity between profiles -> folded into the
                     Contriever embedding text (symmetric "are these the same
                     type?" signal).
  - hierarchy.py:    Weeds Precision (directional containment) between
                     profiles -> subClassOf edge direction ("which one is
                     more general?").
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

from src.ontodisco.utils.dedup_base import (
    DeduplicationResult,
    normalize_label,
    union_find_from_pairs,
)

if TYPE_CHECKING:
    from src.ontodisco.relation_dedup import RelationDeduplicationResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step A — Raw relation counts
# ═══════════════════════════════════════════════════════════════════════════════

def build_relation_counts(
    relation_result: RelationDeduplicationResult,
) -> dict[str, dict[str, int]]:
    """
    Invert relation -> type into type -> relation.

    CanonicalRelation.subject_types / object_types are sets of distinct types
    observed in that argument slot (not per-mention frequency counts), so
    "count" here is binary presence: a type contributes 1 to a relation's
    dimension for each of the subject/object sets it appears in. Subject-slot
    and object-slot occurrences are summed into the same dimension: the
    argument role is not tracked, only which canonical relation the type
    participated in. This keeps e.g. a synonym/inverse pair like "type"
    (subject-heavy) and "type of" (object-heavy) from looking distributionally
    different just because they favour opposite argument slots.

    Works whether subject_types/object_types currently hold raw type labels
    (pre type-dedup, e.g. when called from type_dedup.py's Step 3) or
    canonical type_ids (post relation_dedup.update_relation_type_map(), e.g.
    when called from hierarchy_induction.py's Step 5) -- the keys are passed
    through as-is either way.

    Returns: type_label_or_type_id -> {"<relation_id>": count}
    """
    relation_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for relation in relation_result.relations.values():
        for subj_type in relation.subject_types:
            relation_counts[subj_type][relation.relation_id] += 1
        for obj_type in relation.object_types:
            relation_counts[obj_type][relation.relation_id] += 1

    result = {t: dict(dims) for t, dims in relation_counts.items()}
    logger.info(
        "Built relation counts for %d distinct types from %d relations",
        len(result), len(relation_result.relations),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Step B — Weighting
# ═══════════════════════════════════════════════════════════════════════════════

def build_type_relation_profiles(
    relation_counts: dict[str, dict[str, int]],
    *,
    weighting: str = "ppmi",
) -> dict[str, dict[str, float]]:
    """
    Weight each (type, dimension) cell so that relations shared by nearly every
    type (uninformative) don't drown out discriminative ones.

    weighting:
        "ppmi" — Positive PMI: max(0, log(count(t,d) * N / (count(t,*) * count(*,d))))
        "tfidf" — count(t,d) * log(num_types / (1 + df(d)))
        "raw"   — count(t,d) unchanged
    """
    if weighting not in ("ppmi", "tfidf", "raw"):
        raise ValueError(f"Unknown weighting scheme: {weighting!r}")

    if not relation_counts:
        return {}

    if weighting == "raw":
        return {t: dict(dims) for t, dims in relation_counts.items()}

    # relation_counts is a sparse type x rel co-occurrence matrix,
    # where a "rel" r is a canonical relation_id like "rel_0008"
    # c = count(t, r)
    # is how many times type t participated in that relation (as subject or
    # object, summed). Both weighting schemes below need the matrix's marginals:
    #
    #   count_t_star[t]  = count(t, *)  = row sum   = how often t appears at all
    #   count_star_r[r]  = count(*, r)  = column sum = how often relation r fires,
    #                      across every type (i.e. how "common"/uninformative r is)
    #   df_r[r]                        = number of *distinct* types that ever hit
    #                      relation r at all (used only by TF-IDF's IDF term)
    count_t_star: dict[str, float] = {t: sum(dims.values()) for t, dims in relation_counts.items()}
    count_star_r: dict[str, float] = defaultdict(float)
    df_r: dict[str, int] = defaultdict(int)
    for rels in relation_counts.values():
        for r, c in rels.items():
            count_star_r[r] += c
            df_r[r] += 1

    num_types = len(relation_counts)          # N in the TF-IDF IDF term
    grand_total = sum(count_t_star.values())  # count(*, *) = N in the PPMI term

    profiles: dict[str, dict[str, float]] = {}
    for t, rels in relation_counts.items():
        weighted: dict[str, float] = {}
        for r, c in rels.items():
            if weighting == "tfidf":
                # Classic inverse-document-frequency down-weighting:
                #   idf(r) = log( num_types / (1 + df(r)) )
                # r appearing under almost every type (large df(r)) -> idf ~ 0,
                # so it contributes ~nothing regardless of how large c is.
                # r that is rare across types but frequent for this t -> large
                # idf, so it dominates the profile. weight = c * idf(r).
                idf = math.log(num_types / (1 + df_r[r]))
                weight = c * idf
            else:  # ppmi
                # Pointwise Mutual Information between type t and relation r:
                #   PMI(t,r) = log( P(t,r) / (P(t) * P(r)) )
                # Expanding P(t,r)=c/N, P(t)=count_t_star[t]/N, P(r)=count_star_r[r]/N
                # and cancelling the N's:
                #   PMI(t,r) = log( c * N / (count_t_star[t] * count_star_r[r]) )
                # where N = grand_total. PMI > 0 means t and r co-occur more than
                # chance would predict (informative/discriminative pairing);
                # PMI < 0 means they co-occur less than chance (t actively avoids
                # r). "Positive PMI" clamps the uninformative/negative case to 0
                # instead of keeping a negative weight, since for this profile we
                # only care about *positive* association evidence.
                denom = count_t_star[t] * count_star_r[r]
                weight = max(0.0, math.log((c * grand_total) / denom)) if denom > 0 else 0.0
            # Dimensions that end up with zero (or negative, pre-clamp) weight
            # carry no signal, so they're dropped to keep the profile sparse.
            if weight > 0:
                weighted[r] = weight
        profiles[t] = weighted

    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
#  Vector math
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_sim_sparse(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (dicts). 0.0 if either is empty."""
    if not vec_a or not vec_b:
        return 0.0

    # Iterate the smaller dict for the dot product.
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(w * vec_b.get(d, 0.0) for d, w in vec_a.items())

    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def weeds_precision(vec_child: dict[str, float], vec_parent: dict[str, float]) -> float:
    """
    Directional distributional-inclusion measure (Weeds & Weir 2003):
    what fraction of vec_child's weight falls on dimensions vec_parent also has.

    weeds_precision(child, parent) == 1.0  -> every dimension child touches, parent also touches.
    weeds_precision(child, parent) == 0.0  -> no evidence, or no overlap at all.
    """
    total = sum(vec_child.values())
    if total <= 0.0:
        return 0.0

    overlap = sum(w for d, w in vec_child.items() if d in vec_parent)
    return overlap / total


# ═══════════════════════════════════════════════════════════════════════════════
#  Human-readable evidence string for LLM verification prompts
# ═══════════════════════════════════════════════════════════════════════════════

def describe_relation_context(
    key: str,
    profiles: dict[str, dict[str, float]],
    relation_result: RelationDeduplicationResult,
    top_k: int = 5,
) -> str:
    """
    Render a short human-readable string describing which canonical relations
    a type label or type_id most strongly participates in (by weight in
    `profiles`), for use as extra LLM-facing evidence during cluster
    verification (type_dedup.py's member_context, hierarchy_induction.py's
    per-candidate context).

    `key` looks up `profiles` directly, so it must be whatever key space
    `profiles` was built over -- a raw (normalised) type label if `profiles`
    came from build_relation_counts() called pre type-dedup, or a canonical
    type_id if called post relation_dedup.update_relation_type_map().

    Returns "" if the key has no profile entries (no evidence to show) --
    callers should omit rather than pad candidates with empty context.
    """
    profile = profiles.get(key)
    if not profile:
        return ""

    top_dims = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    relation_labels = []
    for relation_id, _ in top_dims:
        relation = relation_result.relations.get(relation_id)
        if relation:
            relation_labels.append(relation.canonical_label)

    if not relation_labels:
        return ""

    return "often appears with: " + ", ".join(relation_labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-LLM candidate expansion — merge clusters whose relation signatures agree
# ═══════════════════════════════════════════════════════════════════════════════

def _sum_profiles(
    labels: list[str],
    profiles: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-label profiles into one profile for a whole cluster."""
    summed: dict[str, float] = defaultdict(float)
    for label in labels:
        for dim, weight in profiles.get(label, {}).items():
            summed[dim] += weight
    return dict(summed)


def merge_clusters_by_relation_signature(
    clusters: dict[int, list[str]],
    profiles: dict[str, dict[str, float]],
    *,
    threshold: float = 0.8,
) -> dict[int, list[str]]:
    """
    Additionally union HAC clusters whose *cluster-level* relation-signature
    cosine similarity is >= threshold, even when their label embeddings never
    put them in the same cluster to begin with. Intended to run as
    dedup_base.deduplicate()'s cluster_postprocess_fn, right before the
    (possibly now-larger) clusters go to LLM verification.

    `profiles` is expected to be TF-IDF-weighted (build_type_relation_profiles(
    ..., weighting="tfidf")) rather than PPMI-weighted: TF-IDF gives a simpler,
    more conservative similarity for a hard threshold decision than PPMI's
    log-odds scale.

    Clusters with no relation-context evidence at all are left untouched
    (never merged on this signal alone) — this can only coarsen clusters
    further, never split them.
    """
    cluster_ids = list(clusters.keys())
    cluster_profiles = {cid: _sum_profiles(clusters[cid], profiles) for cid in cluster_ids}

    n = len(cluster_ids)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        profile_i = cluster_profiles[cluster_ids[i]]
        if not profile_i:
            continue
        for j in range(i + 1, n):
            profile_j = cluster_profiles[cluster_ids[j]]
            if not profile_j:
                continue
            if cosine_sim_sparse(profile_i, profile_j) >= threshold:
                pairs.append((i, j))

    merge_labels = union_find_from_pairs(n, pairs)

    merged: dict[int, list[str]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        merged[int(merge_labels[idx])].extend(clusters[cid])

    if pairs:
        logger.info(
            "Relation-signature merge: %d clusters -> %d clusters (%d cluster-pairs merged, threshold=%.2f)",
            n, len(merged), len(pairs), threshold,
        )

    return dict(merged)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 rollup — label-level profiles -> canonical type_id-level profiles
# ═══════════════════════════════════════════════════════════════════════════════

def rollup_profiles_to_types(
    type_vocab: DeduplicationResult,
    label_profiles: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Sum label-level relation-context profiles over each CanonicalType's
    surface forms, producing a type_id -> profile mapping for use once type
    dedup has produced canonical type_ids (hierarchy induction operates on
    type_ids, not raw labels).
    """
    type_profiles: dict[str, dict[str, float]] = {}

    for type_id, canonical_type in type_vocab.items.items():
        normalized_forms = {normalize_label(sf) for sf in canonical_type.surface_forms}
        summed: dict[str, float] = defaultdict(float)
        for norm_form in normalized_forms:
            for dim, weight in label_profiles.get(norm_form, {}).items():
                summed[dim] += weight
        type_profiles[type_id] = dict(summed)

    return type_profiles
