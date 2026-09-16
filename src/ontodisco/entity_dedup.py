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
from src.ontodisco.hierarchy_induction import ancestor_path, lowest_common_ancestor

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


def _run_two_stage_partition(
    members: list[str],
    type_id_by_normalized: dict[str, str],
    type_label_of,
    embedder: "ContrieverEmbedder",
    llm_extractor,
    *,
    partition_type_label: str,
    **round_kwargs,
) -> list[tuple[str, set[str]]]:
    """Split a parent-partition by the entity's OWN type before merging, then
    reconcile across types in a second pass.

    A ("parent", P) partition holds every entity whose type is any child of P.
    When P has many children that is most of the corpus in one pool -- measured
    on MINE run_12, the ("parent", "cultural concept") partition held 1403 of
    2988 entities (48%). Feeding that to one embed->cluster->verify loop means
    HDBSCAN routinely groups co-hyponyms from unrelated child types, and every
    such cluster is a chance for the verifier to merge things that merely share
    a parent.

    Stage A runs the ordinary merge rounds within each exact type -- the safest
    possible pool, since every candidate is literally the same type (this is the
    `("self", type_id)` case the gate already trusts most). Stage B then runs one
    more pass over Stage A's SURVIVORS, so genuine cross-sibling duplicates
    ("Chris Nolan [film director]" / "Chris Nolan [filmmaker]") are still
    reachable -- but over a pool already reduced by Stage A rather than the raw
    partition.

    Stage B is skipped when the partition only has one type in it (nothing to
    reconcile) -- that case is just Stage A, identical to the old behaviour.
    """
    by_type: dict[str, list[str]] = defaultdict(list)
    for norm_label in members:
        by_type[type_id_by_normalized.get(norm_label, "")].append(norm_label)

    if len(by_type) <= 1:
        return _run_partition_merge_rounds(
            members, embedder, llm_extractor,
            partition_type_label=partition_type_label, **round_kwargs,
        )

    # ── Stage A: within each exact type ────────────────────────────────────
    stage_a: list[tuple[str, str, set[str]]] = []   # (name, own type label, members)
    for type_id, subset in by_type.items():
        own_label = type_label_of(type_id) or partition_type_label
        if len(subset) == 1:
            name, _ = _parse_compound(subset[0])
            stage_a.append((name, own_label, {subset[0]}))
            continue
        for name, norm_members in _run_partition_merge_rounds(
            subset, embedder, llm_extractor,
            partition_type_label=own_label, **round_kwargs,
        ):
            stage_a.append((name, own_label, norm_members))
    logger.info(
        "Partition stage A: %d entities across %d type(s) -> %d survivors",
        len(members), len(by_type), len(stage_a),
    )
    if len(stage_a) < 2:
        return [(name, members) for name, _, members in stage_a]

    # ── Stage B: reconcile survivors across the sibling types ──────────────
    # Re-enter the same machinery on compound labels rebuilt from Stage A's
    # canonical names, each keeping its OWN type label -- not the parent's.
    # Two same-name survivors from different sibling types ("Paris [city]" /
    # "Paris [town]") must stay DISTINCT strings so the verifier decides on
    # them; collapsing them onto one parent-typed string would union them by
    # dict-key collision with no LLM call at all, which is precisely the
    # unverified merge this module forbids (see _parse_cluster_response).
    # Stage B's returned members are these rebuilt labels, mapped back below
    # to the real normalized members each stands for.
    stage_b_input: list[str] = []
    back: dict[str, set[str]] = {}
    for name, own_label, norm_members in stage_a:
        compound = f"{name} [{own_label}]"
        suffix = 0
        while compound in back:   # identical name AND type: disambiguate, never union
            suffix += 1
            compound = f"{name} [{own_label}]#{suffix}"
        back[compound] = set(norm_members)
        stage_b_input.append(compound)
    if len(stage_b_input) < 2:
        return [(_parse_compound(c)[0], m) for c, m in back.items()]

    merged: list[tuple[str, set[str]]] = []
    for name, group_members in _run_partition_merge_rounds(
        stage_b_input, embedder, llm_extractor,
        partition_type_label=partition_type_label, **round_kwargs,
    ):
        real: set[str] = set()
        for synthetic in group_members:
            real |= back.get(synthetic, set())
        if real:
            merged.append((name, real))
    logger.info("Partition stage B: %d survivors -> %d final entities", len(stage_a), len(merged))
    return merged


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
            # The prompt asks for the bare entity name, but the LLM regularly
            # echoes back the compound "name [type]" form it was shown. Left
            # in, that string becomes the entity's canonical_label and then a
            # GRAPH NODE LABEL, so "alexander fleming [human]" no longer
            # embeds like "alexander fleming" and retrieval degrades badly --
            # measured on MINE, the leak rate rose 3.5% -> 17.3% once entity
            # dedup actually started merging (3.4% -> 25.6% reduction), taking
            # judge accuracy down with it. _parse_compound strips a trailing
            # bracket and is a no-op on an already-bare name; it also collapses
            # the doubled "[century] [century]" form that repeated rounds produce.
            next_pool.append(_EntityPoolItem(
                label=_parse_compound(canonical_label.strip())[0],
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
    *,
    include_qualifiers: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """
    Collect all entity surface forms from triplets as compound labels
    "name [canonical type label]", using the ALREADY-CANONICALIZED type
    (type_dedup must run first). Mentions whose type doesn't resolve to a
    known type_id (e.g. discarded as noise by type_dedup's min_type_freq)
    are skipped -- the entity may still be canonicalized via its other,
    resolvable mentions.

    include_qualifiers also treats each qualifier's (object, object_type) as
    an entity mention. A qualifier value denotes the same real-world thing
    whether it fills a statement slot or qualifies one: measured on MINE,
    27% of distinct qualifier objects ALREADY occur as a main subject/object
    (1969, 1970s, 15th century), and without this they exist twice -- once as
    a canonical entity, once as a raw string nobody canonicalizes. Requires
    the extraction prompt to emit qualifier `object_type` (it does) and the
    qualifier naming rules that keep those values atomic; on a corpus
    extracted before those rules, qualifier objects are phrases like
    "create sense of urgency" and are better left out.

    Note this changes the entity VOCABULARY only, not graph topology --
    build_kg_graph folds qualifier values into the enriched predicate string
    either way, so no new nodes appear in the exported graph.

    Returns (all_compound_labels, type_id_by_normalized_compound): the
    second dict maps each unique normalized compound label to the type_id
    used to build it (built inline, since a given normalized compound
    string is only ever produced from one (name, type_id) pair).
    """
    all_compound_labels: list[str] = []
    type_id_by_normalized_compound: dict[str, str] = {}
    skipped = 0
    num_qualifier_mentions = 0

    def _add(raw_name: str, raw_type: str) -> bool:
        """Returns True if the mention resolved and was recorded."""
        if not raw_name or not raw_type:
            return False
        type_id = _resolve_type_id(raw_type, type_vocab)
        if type_id is None:
            return False
        canonical_type_label = type_vocab.types[type_id].canonical_label
        compound = _make_compound(raw_name, canonical_type_label)
        all_compound_labels.append(compound)
        type_id_by_normalized_compound[_normalize_compound(compound)] = type_id
        return True

    for triplet in triplets:
        for name_key, type_key in [
            ("subject", "subject_type"),
            ("object", "object_type"),
        ]:
            raw_name = (triplet.get(name_key) or "").strip()
            raw_type = (triplet.get(type_key) or "").strip()
            if not raw_name or not raw_type:
                continue
            if not _add(raw_name, raw_type):
                skipped += 1

        if not include_qualifiers:
            continue
        for qualifier in (triplet.get("qualifiers") or []):
            if not isinstance(qualifier, dict):
                continue
            q_name = (qualifier.get("object") or "").strip()
            q_type = (qualifier.get("object_type") or "").strip()
            if not q_name or not q_type:
                continue
            if _add(q_name, q_type):
                num_qualifier_mentions += 1
            else:
                skipped += 1

    logger.info(
        "Collected %d compound entity mentions (%d unique) from %d triplets "
        "(%d from qualifier values; %d mentions skipped: type not resolvable in TypeVocabulary)",
        len(all_compound_labels), len(type_id_by_normalized_compound),
        len(triplets), num_qualifier_mentions, skipped,
    )

    return all_compound_labels, type_id_by_normalized_compound


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage C: cross-type reconciliation (CLAUDE.md sec.7)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Stage A/B (above) partition strictly by _parent_key -- exact same type, or
# siblings under the exact same immediate parent. That is deliberately narrow
# and catches the bulk of real duplicates safely, but it permanently misses a
# real, measured failure mode: the SAME entity extracted under two DIFFERENT,
# non-sibling types (e.g. "Britain" resolved once as [country], once as
# [state]; "water" as [resource]/[substance]/[liquid]). These survive Stage
# A/B as separate canonical entities forever, since the partition gate never
# lets them be compared at all.
#
# Stage C widens the boundary to "share SOME common ancestor anywhere in the
# hierarchy forest" -- entities whose types are in fully DISCONNECTED trees
# are still never reconciled (a deliberate, accepted cost: on a real MINE
# run, 104 of 225 cross-type name collisions fall in this disconnected
# category, and some of those are plausibly genuine merges too -- but
# widening the gate further reopens exactly the risk the narrow gate exists
# to bound, and the connected-component boundary already reaches every
# concretely-verified case, including non-same-name pairs).
#
# A connected component can span most of one root-tree, so naively comparing
# everything within it would reintroduce the giant-partition problem
# (measured: a single component held 1403 of 2988 entities on an earlier
# run). _recursive_hierarchy_partition splits an oversized component along
# the SAME axis the tree itself provides -- which child of the current
# anchor a member's type descends through -- recursing deeper only into
# buckets still oversized, rather than chunking alphabetically. Each small
# bucket is deduplicated with the SAME embed -> cluster -> LLM-verify
# machinery Stage A/B already uses (_run_partition_merge_rounds via
# _reconcile_via_synthetic_labels, the same synthetic-compound-label pattern
# _run_two_stage_partition's own Stage B already established), then ONE
# stitching pass reconciles across sibling buckets -- the same "the split
# was a capacity decision, not a semantic one, so check across it once more"
# idea used everywhere else a size cap forces a split (dedup_base's
# oversized-cluster stitching, _run_two_stage_partition's Stage A -> Stage B,
# hierarchy_induction's regroup).
#
# No new prompt or LLM method: cluster_entity_names.txt already discriminates
# this exact task well when tested directly (8/10 on a real control set
# spanning both directions -- 'water'/'trust'/'soviet union' correctly merge
# across types, 'Paris'/'egg' correctly stay split), so the merge/split
# judgment is unchanged. "Type re-assignment" falls out for free: once a
# cross-type group merges, the existing lowest_common_ancestor(type_ids,
# hierarchy) call in _build_entity naturally reassigns primary_type_id to
# the (now more general) shared ancestor -- no separate type-choice call
# needed, and type_ids keeps the full union so every original typing survives
# as provenance.

def _reconcile_via_synthetic_labels(
    groups: list[tuple[str, list[str], str]],  # (canonical_label, normalized_members, representative_type_label)
    embedder: "ContrieverEmbedder",
    llm_extractor,
    *,
    partition_type_label: str,
    **round_kwargs,
) -> list[tuple[str, set[str]]]:
    """Re-enter the merge machinery on synthetic compound labels built from
    already-merged group names, each tagged with its OWN representative
    type -- identical in shape to _run_two_stage_partition's Stage B,
    generalized to run on arbitrary group lists.

    TWO PHASES, not one, because embedding-based candidate generation is
    the wrong tool for exactly the case Stage C exists for. The embedding
    text is f"{name} {type_label}" -- appending each item's own, DIFFERENT
    type word pushes same-named items APART in embedding space rather than
    together: measured directly, cosine("trust concept", "trust abstract
    concept") = 0.74, well under the 0.85 threshold, so HDBSCAN marks them
    as separate clusters and the LLM is never even asked -- reproducing
    this pipeline's own Stage A/B gate failure one level up, not fixing
    it. Phase 1 sidesteps this with a symbolic, high-precision override:
    any items sharing the EXACT same normalized name are forced into one
    cluster and sent straight to verify_clusters_with_llm, bypassing
    HDBSCAN entirely for them. Phase 2 runs the embedding-driven pass
    (unchanged) on whatever Phase 1 didn't touch -- which is what actually
    earns the "embedding-driven, not name-match-only" design this was
    built to be, catching a genuine different-name cross-type synonym
    (e.g. "USA [country]" / "United States [nation]") that no exact-name
    check could ever reach."""
    def bare(label: str) -> str:
        # Defensive: a group's own label should already be bare (Stage A/B's
        # singleton passthrough strips it at the source too), but never
        # trust that from two layers away -- re-wrapping an already-
        # bracketed label produces a double bracket, not a parse error.
        return _parse_compound(label)[0]

    by_name: dict[str, list[tuple[str, list[str], str]]] = defaultdict(list)
    for g in groups:
        by_name[normalize_label(bare(g[0]))].append(g)

    phase1_survivors: list[tuple[str, list[str], str]] = []
    clusters_by_str: dict[int, list[str]] = {}
    compound_to_group: dict[str, tuple[str, list[str], str]] = {}
    cluster_idx = 0
    for name, same_name_groups in by_name.items():
        if len(same_name_groups) < 2:
            phase1_survivors.extend(same_name_groups)
            continue
        compounds = []
        for label, members, type_label in same_name_groups:
            compound = _make_compound(bare(label), type_label or "")
            suffix = 0
            while compound in compound_to_group:
                suffix += 1
                compound = f"{_make_compound(bare(label), type_label or '')}#{suffix}"
            compound_to_group[compound] = (label, members, type_label)
            compounds.append(compound)
        clusters_by_str[cluster_idx] = compounds
        cluster_idx += 1

    if clusters_by_str:
        verified = verify_clusters_with_llm(
            clusters_by_str, llm_extractor, surface_form_type="entity",
            max_workers=round_kwargs.get("max_workers", 8),
            max_cluster_size=round_kwargs.get("max_cluster_size", 40),
        )
        for canonical_label, member_compounds in verified:
            merged_members: set[str] = set()
            merged_types: set[str] = set()
            for c in member_compounds:
                if c in compound_to_group:
                    _, orig_members, orig_type = compound_to_group[c]
                    merged_members |= set(orig_members)
                    if orig_type:
                        merged_types.add(orig_type)
            if merged_members:
                # Representative type going forward: keep it if every
                # survivor already agreed, otherwise fall back to the
                # bucket's own shared label -- same rule
                # _run_partition_merge_rounds already uses for this exact
                # situation (never an arbitrary alphabetical pick).
                rep = merged_types.pop() if len(merged_types) == 1 else partition_type_label
                phase1_survivors.append((canonical_label, sorted(merged_members), rep))

    # Phase 2: embedding-driven pass over whatever Phase 1 left untouched.
    synthetic_input: list[str] = []
    back: dict[str, set[str]] = {}
    for label, members, type_label in phase1_survivors:
        compound = _make_compound(bare(label), type_label or "")
        suffix = 0
        while compound in back:
            suffix += 1
            compound = f"{_make_compound(bare(label), type_label or '')}#{suffix}"
        back[compound] = set(members)
        synthetic_input.append(compound)

    if len(synthetic_input) < 2:
        return [(_parse_compound(c)[0], m) for c, m in back.items()]

    merged: list[tuple[str, set[str]]] = []
    for label, group_members in _run_partition_merge_rounds(
        synthetic_input, embedder, llm_extractor,
        partition_type_label=partition_type_label, **round_kwargs,
    ):
        real: set[str] = set()
        for synthetic in group_members:
            real |= back.get(synthetic, set())
        if real:
            merged.append((label, real))
    return merged


def _hierarchy_bucket(type_id: str, hierarchy: "TypeHierarchy", anchor: Optional[str]) -> Optional[str]:
    """Which direct child of `anchor` (or which root, if anchor is None)
    type_id's ancestor path passes through -- the split axis
    _recursive_hierarchy_partition uses instead of arbitrary chunking.
    None if type_id IS anchor itself (nothing left to descend into)."""
    path = ancestor_path(type_id, hierarchy)  # leaf-to-root, type_id first
    if anchor is None:
        return path[-1]  # the root
    if anchor not in path:
        return None
    idx = path.index(anchor)
    return path[idx - 1] if idx > 0 else None


def _recursive_hierarchy_partition(
    groups: list[tuple[str, list[str], str]],  # (label, members, representative_type_id)
    hierarchy: "TypeHierarchy",
    max_size: int,
    anchor: Optional[str] = None,
) -> list[list[tuple[str, list[str], str]]]:
    """Split an oversized hierarchy-connected component into buckets no
    larger than max_size, using the tree itself as the splitting axis
    (which child of `anchor` each member's type descends through) rather
    than an arbitrary chunk boundary -- the same principle
    hierarchy_induction's regroup applies to an overflowing tree level,
    applied here to an overflowing entity partition."""
    if len(groups) <= max_size:
        return [groups]
    buckets: dict[Optional[str], list] = defaultdict(list)
    for g in groups:
        _, _, rep_type = g
        buckets[_hierarchy_bucket(rep_type, hierarchy, anchor)].append(g)
    if len(buckets) <= 1:
        # Every member sits at the exact same depth relative to anchor --
        # the hierarchy can't split this further. Let the LLM-verify step's
        # own max_cluster_size chunking (dedup_base.verify_clusters_with_llm)
        # be the last-resort fallback rather than looping forever.
        return [groups]
    result: list[list[tuple[str, list[str], str]]] = []
    for child_anchor, bucket in buckets.items():
        if len(bucket) > max_size and child_anchor is not None:
            result.extend(_recursive_hierarchy_partition(bucket, hierarchy, max_size, child_anchor))
        else:
            result.append(bucket)
    return result


def _reconcile_cross_type_entities(
    verified_groups: list[tuple[str, list[str]]],
    type_id_by_normalized_compound: dict[str, str],
    type_vocab: "TypeDeduplicationResult",
    hierarchy: "TypeHierarchy",
    embedder: "ContrieverEmbedder",
    llm_extractor,
    *,
    similarity_threshold: float,
    embed_batch_size: int,
    max_merge_rounds: int,
    max_parallel_workers: int,
    max_cluster_size: int,
) -> list[tuple[str, list[str]]]:
    """Stage C entry point -- see module comment above. Runs on Stage A/B's
    OUTPUT (already-merged groups, a much smaller pool than raw mentions),
    grouped into hierarchy-connected components (by shared root -- entities
    in disconnected trees are never even considered together, satisfying
    the "no common parent at all" gate structurally rather than via a
    pairwise check), recursively sub-partitioned if oversized, deduplicated
    per-bucket, then stitched once across buckets."""
    def rep_type_of(members: list[str]) -> Optional[str]:
        for m in members:
            tid = type_id_by_normalized_compound.get(m)
            if tid:
                return tid
        return None

    enriched = []
    untyped: list[tuple[str, list[str]]] = []
    for label, members in verified_groups:
        rep_type = rep_type_of(members)
        if rep_type is None:
            untyped.append((label, members))
            continue
        root = ancestor_path(rep_type, hierarchy)[-1]
        enriched.append((root, label, members, rep_type))

    by_root: dict[str, list[tuple[str, list[str], str]]] = defaultdict(list)
    for root, label, members, rep_type in enriched:
        by_root[root].append((label, members, rep_type))

    round_kwargs = dict(
        similarity_threshold=similarity_threshold, embed_batch_size=embed_batch_size,
        max_merge_rounds=max_merge_rounds, max_workers=max_parallel_workers,
        max_cluster_size=max_cluster_size,
    )

    final_groups: list[tuple[str, list[str]]] = list(untyped)
    for root, groups in by_root.items():
        if len(groups) == 1:
            label, members, _ = groups[0]
            final_groups.append((label, members))
            continue

        root_label = type_vocab.types[root].canonical_label if root in type_vocab.types else root
        buckets = _recursive_hierarchy_partition(groups, hierarchy, max_cluster_size, anchor=None)
        logger.info(
            "Stage C: connected component under %r (%d group(s)) split into %d hierarchy bucket(s)",
            root_label, len(groups), len(buckets),
        )

        def label_of(rep_tid: str) -> str:
            t = type_vocab.types.get(rep_tid)
            return t.canonical_label if t else root_label

        bucket_survivors: list[tuple[str, list[str], str]] = []
        for bucket in buckets:
            if len(bucket) < 2:
                bucket_survivors.extend(bucket)
                continue
            rep_types_in_bucket = {rep for _, _, rep in bucket if rep}
            bucket_anchor = lowest_common_ancestor(rep_types_in_bucket, hierarchy) if rep_types_in_bucket else None
            bucket_label = label_of(bucket_anchor) if bucket_anchor else root_label
            synth_input = [(label, members, label_of(rep_tid)) for label, members, rep_tid in bucket]
            for label, members in _reconcile_via_synthetic_labels(
                synth_input, embedder, llm_extractor, partition_type_label=bucket_label, **round_kwargs,
            ):
                bucket_survivors.append((label, sorted(members), rep_type_of(list(members)) or bucket[0][2]))

        if len(bucket_survivors) > 1:
            stitch_input = [(label, members, label_of(rep_tid)) for label, members, rep_tid in bucket_survivors]
            for label, members in _reconcile_via_synthetic_labels(
                stitch_input, embedder, llm_extractor, partition_type_label=root_label, **round_kwargs,
            ):
                final_groups.append((label, sorted(members)))
        else:
            final_groups.extend((label, members) for label, members, _ in bucket_survivors)

    return final_groups


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
    include_qualifiers: bool = True,
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
        triplets, type_vocab, include_qualifiers=include_qualifiers,
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
            # A singleton "group" never goes through _run_partition_merge_rounds
            # (nothing to verify), so nothing ever strips the [type] bracket
            # off it the way a real merge's canonical_label naturally would --
            # bare name, not the raw compound string, or the bracket leaks all
            # the way to the final CanonicalEntity (measured: 84 entities on a
            # real MINE run). _parse_compound is a no-op on an already-bare
            # name, so this is safe regardless of how members[0] is shaped.
            all_verified_groups.append((_parse_compound(members[0])[0], members))
            continue
        # ("parent", parent_id) -> the shared parent's own label; ("self",
        # type_id) -> that one type's own label (trivial -- every item in a
        # self-partition already shares that exact type). Either way, this
        # is the correct representative type for ANY subset merged within
        # this partition (see _run_partition_merge_rounds docstring).
        _, partition_type_id = partition_key
        partition_type = type_vocab.types.get(partition_type_id)
        partition_type_label = partition_type.canonical_label if partition_type else partition_type_id
        partition_groups = _run_two_stage_partition(
            members, type_id_by_normalized_compound,
            lambda tid: (type_vocab.types.get(tid).canonical_label
                         if type_vocab.types.get(tid) else None),
            embedder, llm_extractor,
            partition_type_label=partition_type_label,
            similarity_threshold=similarity_threshold,
            embed_batch_size=embed_batch_size,
            max_merge_rounds=max_merge_rounds,
            max_workers=max_parallel_workers,
            max_cluster_size=max_cluster_size,
        )
        for canonical_label, normalized_members in partition_groups:
            all_verified_groups.append((canonical_label, sorted(normalized_members)))

    logger.info(
        "Entity merge rounds complete across %d parent-partitions: %d unique labels -> %d groups",
        len(partitions), len(unique_labels), len(all_verified_groups),
    )

    before_stage_c = len(all_verified_groups)
    all_verified_groups = _reconcile_cross_type_entities(
        all_verified_groups, type_id_by_normalized_compound, type_vocab, hierarchy,
        embedder, llm_extractor,
        similarity_threshold=similarity_threshold, embed_batch_size=embed_batch_size,
        max_merge_rounds=max_merge_rounds, max_parallel_workers=max_parallel_workers,
        max_cluster_size=max_cluster_size,
    )
    logger.info(
        "Stage C (cross-type reconciliation): %d -> %d groups",
        before_stage_c, len(all_verified_groups),
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
