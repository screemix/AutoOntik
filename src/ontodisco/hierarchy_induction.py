"""
Top-Down, Priority-Ordered Hierarchy Induction
=================================================

ONE PRIORITY QUEUE, SEEDED OR NOT. Every canonical type
in T* starts as a leaf `_Node` carrying a relation-argument profile. A single
pass processes nodes from most-general to most-specific, in "bands" of equal
Weeds-precision in-degree (how many other nodes look like they fit inside a
given node -- a structural breadth proxy, computed once over the untouched
pool, not recomputed per round the way the old phase 1 did). External seed
roots (`config.seed_roots`) simply pre-populate the top of the queue as
pinned candidates that are never themselves placed; without seeds, the top
band of the same in-degree ranking IS root discovery -- there is no separate
root-synthesis algorithm. Seeds may also nest (e.g. DOLCE's Endurant ->
Physical/Non-Physical Endurant): a seed entry can carry its own `children`,
connected to their seed parent by a trusted, non-LLM HierarchyEdge
(`is_seed=True`). Only root-level seeds are pinned top-level candidates --
nested seed children surface the ordinary way, by descending into their
seed parent as the current anchor (see `_apply_seed_roots`).

TWO DIRECTIONAL QUESTIONS PER LEVEL, NOT ONE FIVE-WAY
CHOICE: at each level, a PARENT CHECK asks whether the focal node belongs
under any current candidate (yes -> descend into its children; a
"same_concept" flag means collapse instead). Only if that finds nothing does
a separate CHILD CHECK ask whether focal subsumes any current candidates
(yes -> retroactively insert focal above them, correcting an earlier
too-shallow attachment; focal settles here). Splitting these into two
single-purpose questions, instead of one prompt offering parent/child/
same_concept/same_class/none at once, measurably fixed the model's
positional bias toward answering "the candidate is my parent" regardless of
true direction (framing-experiment accuracy on directional pairs: ~31-69%
for the old five-way prompt vs. 94-100% for a plain yes/no framing of the
same question) -- see CLAUDE.md §6.

CAPACITY-TRIGGERED REGROUPING, NOT UNBOUNDED SYNTHESIS:
there is no general "these seem related, invent a shared parent" action
available on every comparison (the old `same_class`) -- inventing new
abstract nodes on a subjective judgment, with no bound on how often it
fires, is what let one bad batch (bulk-claiming `historical period` over 20
candidates) turn into 270 synthesized nodes with only 185 distinct labels
(`abstract concept` minted 20 separate times). Instead, a level is shown to
a focal node WHOLE, always <= `candidate_batch_size`: the moment it would
exceed that, `_regroup_level` consolidates it first by grouping true
siblings under an invented (or, via `_check_collision`, REUSED) shared
parent. Regrouping is a capacity-management response to a structural fact
(too many candidates to show), not a semantic judgment offered on every
comparison -- which is also what makes candidate visibility unconditional:
a focal node always sees every current candidate at its level, so no
similarity-ranking or escalation-batching machinery is needed to decide
what it gets shown.

DEPTH AS A DEFERRED CUT: once a node's placement would
exceed `max_depth`, it is NOT discarded and NOT silently attached as if it
were a confirmed sibling relationship -- it's recorded in
`HierarchyInductionResult.unresolved_children`, a structurally separate
mapping from `TypeHierarchy.edges`, precisely so a downstream consumer can
never mistake a deferred placement for an LLM-confirmed one. Resuming later
just means seeding a fresh pool from exactly that bucket and re-running this
same module on it -- no new algorithm needed.

PARALLEL PLACEMENT: band 1 (the root band) always runs
sequentially -- every node's first comparison is against the top-level pool
(current_anchor=None), so band 1 is one shared anchor regardless, and
threading it would just add overhead for no gain. From band 2 onward,
placement runs on a ThreadPoolExecutor, one worker per node, serialized only
where two nodes are actually comparing against the SAME anchor's candidates
(a `threading.Lock` per anchor, held across that node's whole step --
candidate read, LLM call, decision apply -- so a less-similar batch can never
interleave with a more-similar one for the same anchor). Nodes anchored
elsewhere run fully concurrently, including their LLM calls. Note this same
root-anchor bottleneck recurs at the START of every node's walk in every
band, not just band 1 -- concurrency only kicks in once nodes have each made
their own first "parent" descent into different branches. This is exactly
why `priority_band_tolerance` (below) matters for more than correctness: a
wider band keeps the thread pool's queue full of ready work while the root
lock drains, instead of workers idling for lack of anything else to start.
"""

from __future__ import annotations

import math
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from src.ontodisco.utils.dedup_base import (
    ContrieverEmbedder,
    DeduplicationResult,
    normalize_label,
)
from src.ontodisco.relation_context import (
    build_relation_counts,
    build_type_relation_profiles,
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
    is_direct: bool = True
    is_seed: bool = False                       # True for an edge given directly by an external
                                                 # seed hierarchy (config.seed_roots), never an LLM
                                                 # call -- relation_signature_score/llm_score are None
                                                 # for these (no LLM confidence exists to report);
                                                 # ensemble_score is a fixed 1.0 placeholder, not a
                                                 # real confidence value. Every other edge in this
                                                 # module still means "an LLM confirmed this."


@dataclass
class TypeHierarchy:
    """The type hierarchy as a directed acyclic forest."""
    edges: list[HierarchyEdge]
    children: dict[str, list[str]]   # parent_type_id -> [child_type_ids]
    parents: dict[str, list[str]]    # child_type_id -> [parent_type_ids] (always length <= 1 here)
    roots: list[str]


@dataclass
class SynthesizedType:
    """A new abstract type minted during induction, rather than promoting an
    existing member as the parent. Callers must fold these into the type
    vocabulary alongside Step 2's output."""
    type_id: str
    canonical_label: str
    definition: str
    child_type_ids: list[str]


@dataclass
class HierarchyConfig:
    relation_signature_weighting: str = "ppmi"  # "ppmi" | "tfidf" | "raw" -- see relation_context.py

    # -- Priority queue / seeding --
    seed_roots: list = field(default_factory=list)   # external seed hierarchy; each entry is either a
                                                       # plain label (str) -- a root with no seed parent
                                                       # -- or a dict {"label": ..., "children": [...]}
                                                       # nesting further seed labels underneath it (e.g.
                                                       # DOLCE's Endurant -> Physical/Non-Physical
                                                       # Endurant). Every label is matched against T* by
                                                       # normalized label, or synthesized fresh if no
                                                       # match exists. Nested (non-root) entries are
                                                       # connected to their seed parent by a trusted,
                                                       # non-LLM HierarchyEdge (is_seed=True) and are
                                                       # never exposed as top-level candidates -- they
                                                       # only surface once a node has already descended
                                                       # into their seed parent as its anchor. 
    seed_roots_path: Optional[str] = None       # path to a separate YAML file holding the seed
                                                 # hierarchy (a `seed_roots:` key, same shape as above,
                                                 # or a bare top-level list), so a reusable seed ontology
                                                 # (e.g. configs/seeds/dolce.yaml) doesn't have to be
                                                 # copy-pasted into every pipeline config.
                                                 # `seed_roots` and has no knowledge of this field.
    weeds_containment_threshold: float = 0.75   # gates both in-degree computation (which pairs count
                                                 # toward a node's breadth score) and is passed through
                                                 # as informational context only -- the LLM call is what
                                                 # actually decides each relation, this only decides
                                                 # which pairs are worth flagging as evidence at all.
    priority_band_tolerance: int = 10           # a band includes every node within this many in-degree
                                                 # units of the band's own top score, not just exact
                                                 # ties -- widens the tie tolerance already accepted for
                                                 # in-degree as a frequency proxy,
                                                 # and keeps each band's node count large enough to
                                                 # actually saturate the parallel placement thread pool
                                                 # (see max_parallel_workers below) rather than leaving
                                                 # it starved of independent work.

    # -- Level capacity / regrouping --
    candidate_batch_size: int = 30      # the cap on how many candidates a focal node is ever shown AT
                                         # ONCE for a level (the live root set, or one anchor's current
                                         # children). Enforced PROACTIVELY: the moment a level would
                                         # exceed this, _regroup_level consolidates it back under the
                                         # cap BEFORE any focal node compares against it -- so a focal
                                         # node never needs escalation/ranking machinery to see "enough"
                                         # of a level, it just sees all of it, always <= this size.
    max_regroup_input: int = 90         # safety cap on one regroup CALL's own input, in case heavy
                                         # concurrency lets a level balloon past candidate_batch_size
                                         # before any thread notices (should be rare in practice, since
                                         # any single thread's own overflow check fires immediately).
    max_regroup_rounds: int = 3         # convergence cap on _regroup_level's own loop, mirroring
                                         # dedup_base.deduplicate_with_rounds' pattern: keep regrouping
                                         # survivors until the level is back under cap or nothing
                                         # further consolidates, whichever comes first.
    label_collision_threshold: float = 0.90   # embedding cosine similarity above which a newly
                                         # proposed synthesized label is treated as the SAME concept as
                                         # an existing one (real T* type, or already synthesized this
                                         # run) rather than minted as a duplicate. Exact normalized-label
                                         # match is always checked first (free); this catches near-miss
                                         # variants an exact match wouldn't (e.g. "biological entity" vs
                                         # "biological entity type" minted independently for the same
                                         # underlying grouping -- measured on MINE run_13's real data).
    max_child_claim_share: float = 0.6  # PATHOLOGY BACKSTOP ONLY, not the primary defense (that's the
                                         # decomposed parent/child question itself, which measured no
                                         # bulk-claiming tendency even at 35 real candidates on MINE).
                                         # A "child" decision is rejected only if it claims more than
                                         # this share of the batch AND more than 3 absolute candidates.
    min_type_support: int = 1           # types with count_per_normalized BELOW this are held out of the
                                         # initial pool and queued in deferred_low_support. 1 keeps
                                         # everything (old behaviour). On MINE run_13, 32% of T* is seen
                                         # exactly once and 50% at most twice.

    max_depth: Optional[int] = 15               # bounds routing cost to at most this many "descend a
                                                 # level" LLM calls per node; anything that would go
                                                 # deeper is deferred into unresolved_children, not lost.
                                                 # None means uncapped (normalized to np.inf below) --
                                                 # every node is routed as deep as the LLM will take it,
                                                 # so unresolved_children is guaranteed to stay empty.

    def __post_init__(self) -> None:
        if self.max_depth is None:
            self.max_depth = np.inf

    # -- Embedding --
    embed_batch_size: int = 64

    # -- LLM prompting: node context is built from real corpus EXAMPLES and
    #    already-placed SUBCLASSES, never relation-profile text (CLAUDE.md
    #    §6/§16 -- PPMI relation context was measured to CAUSE the model to
    #    read co-occurrence as subsumption, e.g. reproducing 'historical
    #    period' bulk-claiming 20/20 candidates; the same inputs with context
    #    stripped, or replaced with instance examples, did not). --
    context_max_examples: int = 10      # real corpus entity mentions shown per type, shortest first
    context_max_subclasses: int = 5     # already-placed children shown per type (empty for a leaf
                                         # with nothing under it yet, or a synthesized node whose
                                         # own children haven't been decided)

    # -- Parallel placement (CLAUDE.md §16.2.4) --
    max_parallel_workers: int = 8                # ThreadPoolExecutor size for band 2+ (band 1 always
                                                  # runs sequentially -- see induce_hierarchy). Keep
                                                  # this within whatever the LLM provider's own rate
                                                  # limit tolerates; nothing here throttles beneath it.


@dataclass
class HierarchyInductionResult:
    hierarchy: TypeHierarchy
    synthesized_types: list[SynthesizedType] = field(default_factory=list)
    unresolved_children: dict[str, list[str]] = field(default_factory=dict)

    deferred_low_support: list[str] = field(default_factory=list)
    # type_ids held back by min_type_support: QUEUED for a later construction
    # stage, not discarded. Same "defer, never destroy" contract as
    # unresolved_children above -- a rarely-attested type is weak EVIDENCE,
    # not a non-existent type, and placing it against the main pool is what
    # let 'item' (ONE corpus mention) acquire 836 descendants.


@dataclass
class _Node:
    """One entry in the current active pool (not yet attached anywhere) or
    the never-shrinking all_nodes registry (every node that has ever
    existed, attached or not): either an original T* leaf type, a parent
    surrogate synthesized during induction, or a pinned external seed root."""
    type_id: str
    label: str
    profile: dict[str, float]
    is_leaf: bool
    definition: str = ""

    def embed_text(self) -> str:
        if self.definition:
            return f"{self.label}. Definition: {self.definition}"
        return self.label


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared pool / profile helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _initial_pool(
    type_vocab: DeduplicationResult,
    relation_vocab: "RelationDeduplicationResult",
    config: HierarchyConfig,
) -> tuple[dict[str, _Node], list[str]]:
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
    deferred: list[str] = []
    for type_id, canonical_type in type_vocab.items.items():
        if config.min_type_support > 1 and canonical_type.count_per_normalized < config.min_type_support:
            deferred.append(type_id)
            continue
        pool[type_id] = _Node(
            type_id=type_id,
            label=canonical_type.canonical_label,
            profile=profiles.get(type_id, {}),
            is_leaf=True,
        )

    n_with_profile = sum(1 for n in pool.values() if n.profile)
    if deferred:
        logger.info(
            "Hierarchy induction: %d type(s) below min_type_support=%d QUEUED for a later stage "
            "(not discarded); e.g. %s",
            len(deferred), config.min_type_support,
            [type_vocab.items[t].canonical_label for t in deferred[:8]],
        )
    logger.info(
        "Hierarchy induction: initial pool of %d types (%d with a non-empty "
        "relation-signature profile)",
        len(pool), n_with_profile,
    )
    return pool, deferred


def _merge_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = defaultdict(float)
    for profile in profiles:
        for dim, weight in profile.items():
            merged[dim] += weight
    return dict(merged)


def _make_edge(child_id: str, parent_id: str, child_profile: dict, parent_profile: dict,
               confidence: float) -> HierarchyEdge:
    rel_score = (
        weeds_precision(child_profile, parent_profile)
        if child_profile and parent_profile else None
    )
    return HierarchyEdge(
        child_type_id=child_id, parent_type_id=parent_id,
        relation_signature_score=rel_score, llm_score=confidence,
        ensemble_score=confidence, is_direct=True,
    )


def _collapse_duplicate_nodes(
    cluster_ids: list[str],
    pool: dict[str, "_Node"],
    all_nodes: dict[str, "_Node"],
    edges: list[HierarchyEdge],
    synthesized: list[SynthesizedType],
) -> str:
    """Collapse a cluster of duplicate nodes into one survivor (the
    earliest-created, i.e. lowest type_id -- deterministic and stable across
    runs given the same LLM outputs). Every edge and SynthesizedType record
    pointing at a loser as PARENT is rewritten to point at the survivor.
    Returns the survivor's type_id.

    Unlike the earlier design, a same_concept match can now involve an
    ALREADY-ATTACHED node (encountered as a descent-level candidate, not a
    `pool` member) -- descent keeps already-placed nodes visible precisely
    so this can happen (see the module docstring's Gap 1). So this also
    handles the loser (or the survivor, if it's the already-attached side
    that happens to have the lower type_id) already having its own parent
    edge: that edge is retargeted to the survivor rather than assuming it
    can't exist. If BOTH sides already have a (necessarily different)
    parent edge, the survivor's own edge wins and the loser's is simply
    dropped -- a genuine conflict, logged rather than silently resolved
    either way, and not otherwise handled specially here."""
    survivor_id = min(cluster_ids)
    loser_ids = [tid for tid in cluster_ids if tid != survivor_id]

    survivor = all_nodes[survivor_id]
    survivor.profile = _merge_profiles([all_nodes[tid].profile for tid in cluster_ids])

    synth_by_id = {s.type_id: s for s in synthesized}
    survivor_synth = synth_by_id.get(survivor_id)

    survivor_has_parent = any(e.child_type_id == survivor_id for e in edges)

    for loser_id in loser_ids:
        loser_synth = synth_by_id.get(loser_id)
        if survivor_synth is not None and loser_synth is not None:
            survivor_synth.child_type_ids.extend(loser_synth.child_type_ids)
        if loser_synth is not None:
            synthesized.remove(loser_synth)

        # Loser as PARENT side: every child pointing at the loser now points
        # at the survivor instead.
        for edge in edges:
            if edge.parent_type_id == loser_id:
                edge.parent_type_id = survivor_id

        # Loser as CHILD side: if the loser was already attached somewhere,
        # that attachment is real information the survivor should inherit --
        # unless the survivor already has its own (necessarily conflicting)
        # parent edge, in which case the loser's is dropped, not merged.
        loser_parent_edges = [e for e in edges if e.child_type_id == loser_id]
        for edge in loser_parent_edges:
            if survivor_has_parent:
                logger.warning(
                    "same_concept collapse: both %s and %s already had a parent edge; "
                    "keeping %s's and dropping %s's (%s)",
                    survivor_id, loser_id, survivor_id, loser_id, edge.parent_type_id,
                )
                edges.remove(edge)
            else:
                edge.child_type_id = survivor_id
                survivor_has_parent = True

        pool.pop(loser_id, None)
        # Deliberately NOT removed from all_nodes: an unresolved_children
        # bucket recorded before this collapse could still reference
        # loser_id as its anchor key, and all_nodes is meant to answer any
        # historical lookup, not just currently-live ones. survivor's
        # profile already absorbed loser's, so loser's entry is inert.

    if survivor_has_parent:
        pool.pop(survivor_id, None)

    logger.debug("Duplicate collapse: %r into %s", cluster_ids, survivor_id)
    return survivor_id


def _collect_type_examples(
    triplets: list[dict], type_vocab: DeduplicationResult, max_examples: int,
) -> dict[str, list[str]]:
    """Real corpus entity mentions per canonical type_id, shortest names
    first (avoids surfacing long noisy extraction artifacts ahead of clean
    short names) and capped at max_examples. This is the ONLY source of
    hierarchy-placement context now -- see HierarchyConfig.context_max_examples
    for why relation-profile text was removed rather than kept alongside it."""
    raw: dict[str, set[str]] = defaultdict(set)
    for triplet in triplets:
        for name_key, type_key in (("subject", "subject_type"), ("object", "object_type")):
            name, raw_type = triplet.get(name_key), triplet.get(type_key)
            if not name or not raw_type:
                continue
            type_id = type_vocab.surface_to_id.get(normalize_label(str(raw_type)))
            if type_id:
                raw[type_id].add(str(name).strip())
    return {
        type_id: sorted(names, key=lambda s: (len(s), s))[:max_examples]
        for type_id, names in raw.items()
    }


def _subclass_labels(
    type_id: str, children_map: dict[str, list[str]], all_nodes: dict[str, _Node], max_k: int,
) -> list[str]:
    """Labels of up to max_k of type_id's already-placed children, per the
    CURRENT (partially built) tree -- empty for a leaf with nothing under it
    yet, or a synthesized node whose own children haven't been decided."""
    return [all_nodes[cid].label for cid in children_map.get(type_id, [])[:max_k] if cid in all_nodes]


def _node_context(
    node: _Node, examples_by_type: dict[str, list[str]], subclass_labels: list[str],
) -> str:
    """Everything worth telling the LLM about one node beyond its bare
    label: its definition (a seed's "description", or a synthesized type's
    own definition -- otherwise ""), real corpus EXAMPLES of that type, and
    a few of its already-placed SUBCLASSES. No relation-profile text --
    measured to actively mislead the model into reading co-occurrence as
    subsumption (CLAUDE.md §6): the same 20-candidate 'historical period'
    batch that bulk-claimed food/currency/language/etc. with relation context
    claimed only century/civilization/time once that context was stripped.
    A type with almost no examples (e.g. a single-mention noise label) is
    weak evidence, and the prompts tell the model to treat it that way rather
    than guess."""
    parts = []
    if node.definition:
        parts.append(node.definition)
    examples = examples_by_type.get(node.type_id, [])
    if examples:
        parts.append("for example: " + ", ".join(examples))
    if subclass_labels:
        parts.append("known subclasses: " + ", ".join(subclass_labels))
    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Priority queue construction (CLAUDE.md §16.2.1)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_inverted_index(pool: dict[str, _Node]) -> dict[str, set[str]]:
    """relation_id -> set of type_ids with nonzero weight on that dimension."""
    index: dict[str, set[str]] = defaultdict(set)
    for type_id, node in pool.items():
        for dim in node.profile:
            index[dim].add(type_id)
    return index


def _candidate_pairs(pool: dict[str, _Node], index: dict[str, set[str]]) -> list[tuple[str, str]]:
    """All pairs of pool nodes sharing at least one relation dimension --
    disjoint profiles can't have containment, so this is a cheap prefilter
    before any Weeds-precision computation."""
    seen: set[frozenset] = set()
    pairs: list[tuple[str, str]] = []
    for type_id, node in pool.items():
        if not node.profile:
            continue
        neighbors: set[str] = set()
        for dim in node.profile:
            neighbors |= index[dim]
        neighbors.discard(type_id)
        for other in neighbors:
            key = frozenset((type_id, other))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(tuple(sorted((type_id, other))))
    pairs.sort()
    return pairs


def _compute_indegree(pool: dict[str, _Node], config: HierarchyConfig) -> dict[str, int]:
    """How many other pool nodes currently look like they fit inside a given
    node, by Weeds-precision containment -- a one-shot structural breadth
    signal over the WHOLE untouched pool (unlike the old phase 1, which
    recomputed this per round over a shrinking pool). Nodes with an empty
    relation profile simply never accrue in-degree via this signal and fall
    to the lowest band by default -- they still get placed in due course,
    just later, since this signal only controls PROCESSING ORDER, not
    whether a node ever reaches placement."""
    index = _build_inverted_index(pool)
    pairs = _candidate_pairs(pool, index)
    in_degree: dict[str, int] = defaultdict(int)
    for a, b in pairs:
        wp_ab = weeds_precision(pool[a].profile, pool[b].profile)
        wp_ba = weeds_precision(pool[b].profile, pool[a].profile)
        if max(wp_ab, wp_ba) < config.weeds_containment_threshold:
            continue
        if wp_ab > wp_ba:
            in_degree[b] += 1
        elif wp_ba > wp_ab:
            in_degree[a] += 1
        else:
            in_degree[a] += 1
            in_degree[b] += 1
    return dict(in_degree)


def _priority_bands(
    pool: dict[str, _Node], pinned_roots: set[str], config: HierarchyConfig,
) -> list[list[str]]:
    """Highest in-degree first. Pinned (seeded) roots are excluded entirely
    from the returned bands -- they are never themselves placed, only ever
    offered as candidates.

    A band groups every node within `priority_band_tolerance` in-degree
    units of the band's OWN top score (fixed at the first, highest-scoring
    member -- not a rolling window, so one band's total spread never drifts
    past the configured tolerance via a chain of small gaps), not just exact
    ties. This widens the tie tolerance already accepted for in-degree as a
    frequency proxy, not a validated total order (CLAUDE.md §16.2.1's
    caveat, §16.4) -- tolerance=0 recovers the original exact-tie behavior.
    It also matters for parallel placement (CLAUDE.md §16.2.4): a corpus
    with a wide, mostly-distinct in-degree spread would otherwise produce
    long runs of singleton bands, each with nothing to parallelize and each
    paying the full sequential "resolve one band before starting the next"
    cost individually."""
    in_degree = _compute_indegree(pool, config)
    placeable = [tid for tid in pool if tid not in pinned_roots]
    placeable.sort(key=lambda tid: -in_degree.get(tid, 0))

    bands: list[list[str]] = []
    current_band: list[str] = []
    band_ceiling: Optional[int] = None
    for tid in placeable:
        score = in_degree.get(tid, 0)
        if band_ceiling is None or band_ceiling - score <= config.priority_band_tolerance:
            current_band.append(tid)
            if band_ceiling is None:
                band_ceiling = score
        else:
            bands.append(current_band)
            current_band = [tid]
            band_ceiling = score
    if current_band:
        bands.append(current_band)
    return bands


def _apply_seed_roots(
    seed_items: list,
    pool: dict[str, _Node],
    all_nodes: dict[str, _Node],
    edges: list[HierarchyEdge],
    synthetic_counter: list[int],
) -> set[str]:
    """Promote existing T* types matching a seed label (by normalized label)
    to pinned status, or synthesize a fresh node for seed labels with no
    match in T* (e.g. a DOLCE category like "Endurant" that may not appear
    verbatim in this corpus's own type vocabulary).

    `seed_items` may nest: each entry is either a plain label (str) -- a
    root with no seed parent -- or a dict {"label": ..., "children": [...]}
    whose children are recursively applied one level down, each connected
    to its seed parent by a trusted (non-LLM) HierarchyEdge (is_seed=True).
    An optional "description" key on a dict entry becomes that node's
    `_Node.definition` -- the same field `SynthesizedType.definition`
    already plays during ordinary LLM-driven placement -- but ONLY when the
    label is freshly SYNTHESIZED (no match in T*). A label that matches an
    existing T* type keeps that type's own (empty, today) definition rather
    than being overwritten by a generic external gloss, since that would
    silently change an already-established corpus type's embedding/prompt
    context as a side effect of seeding. Consumed downstream by both
    `embed_text()` (embedding/similarity ranking) and `_node_context()` (the
    "(context: ...)" shown to the LLM per candidate during placement) --
    without it, a synthesized seed node reaches the LLM as a bare label with
    no context at all, since it starts with an empty relation profile.

    Only ROOT-level seed nodes are added to `pool` -- they're the only ones
    ever offered as top-level (current_anchor=None) candidates. Nested seed
    children live only in `all_nodes`: they surface exactly the way any
    other node's children do, by descending into their seed parent as the
    current anchor and reading `_build_children_map(edges)` -- no special-
    casing needed in `_place_node` for the nested case. A seed label that
    happens to match an EXISTING pool member and is used as a nested
    (non-root) entry is popped out of `pool` here, so it isn't also
    independently routed through a priority band as if it were an
    unplaced leaf -- its seed edge is already authoritative.

    Returns the set of ROOT pinned type_ids (i.e. exactly `pool`'s pinned
    members) -- what the priority-band loop excludes from placement."""
    pinned_roots: set[str] = set()
    assigned_parent: set[str] = set()
    label_to_id = {normalize_label(n.label): tid for tid, n in pool.items()}

    def _resolve_one(label: str, description: str = "") -> str:
        norm = normalize_label(label)
        existing_id = label_to_id.get(norm)
        if existing_id is not None:
            return existing_id
        new_id = f"type_seed{synthetic_counter[0]:04d}"
        synthetic_counter[0] += 1
        new_node = _Node(type_id=new_id, label=label.strip(), profile={}, is_leaf=False, definition=description.strip())
        all_nodes[new_id] = new_node
        label_to_id[norm] = new_id
        return new_id

    def _walk(items: list, parent_id: Optional[str]) -> None:
        for item in items:
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                description = str(item.get("description", "") or "").strip()
                children = item.get("children") or []
            else:
                label = str(item).strip()
                description = ""
                children = []
            if not label:
                logger.warning("Skipping seed entry with no label: %r", item)
                continue

            node_id = _resolve_one(label, description)
            if parent_id is None:
                pool[node_id] = all_nodes[node_id]
                pinned_roots.add(node_id)
            elif node_id in assigned_parent:
                logger.warning(
                    "Seed label %r already has a seed parent; ignoring duplicate "
                    "occurrence under %r", label, all_nodes[parent_id].label,
                )
            else:
                pool.pop(node_id, None)
                edges.append(HierarchyEdge(
                    child_type_id=node_id, parent_type_id=parent_id,
                    relation_signature_score=None, llm_score=None,
                    ensemble_score=1.0, is_direct=True, is_seed=True,
                ))
                assigned_parent.add(node_id)

            _walk(children, node_id)

    _walk(seed_items, None)
    return pinned_roots


# ═══════════════════════════════════════════════════════════════════════════════
#  Placement: two directional questions per level, capacity-triggered
#  regrouping instead of unbounded synthesis (CLAUDE.md §16.2.2, §16.2.7,
#  §16.2.4)
# ═══════════════════════════════════════════════════════════════════════════════


class _AnchorLocks:
    """Lazily-created per-anchor lock registry, scoped to one band and
    rebuilt fresh for the next (CLAUDE.md §16.2.4: "resolve one band fully
    before starting the next" -- an anchor's candidate set from one band has
    no bearing on the next). `None` (the top-level pool) is a valid, and
    typically the most-contended, key: every node's FIRST comparison in
    EVERY band is against the top-level pool, so band 1 is effectively one
    giant `None`-keyed critical section regardless -- see the module
    docstring's "PARALLEL PLACEMENT" note for why priority_band_tolerance
    matters alongside this."""
    def __init__(self) -> None:
        self._locks: dict[Optional[str], threading.Lock] = {}
        self._guard = threading.Lock()

    def acquire(self, anchor_key: Optional[str]) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(anchor_key)
            if lock is None:
                lock = threading.Lock()
                self._locks[anchor_key] = lock
            return lock


def _next_synthetic_id(prefix: str, synthetic_counter: list[int], counter_lock: threading.Lock) -> str:
    """synthetic_counter[0] += 1 is a read-then-write over two bytecode
    ops, not atomic under threads (unlike a single dict/list mutation,
    which CPython's GIL already makes safe) -- two concurrently-running
    "same_class" decisions under DIFFERENT anchors (hence different anchor
    locks) could otherwise both read the same counter value and mint the
    same type_id for two different synthesized types. This is the one piece
    of shared state _place_node touches that isn't already scoped to a
    single anchor's lock, so it gets its own dedicated lock."""
    with counter_lock:
        n = synthetic_counter[0]
        synthetic_counter[0] += 1
    return f"{prefix}{n:04d}"


def _build_children_map(edges: list[HierarchyEdge]) -> dict[str, list[str]]:
    children_map: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        children_map[e.parent_type_id].append(e.child_type_id)
    return dict(children_map)


def _is_ancestor_or_self(candidate_id: str, descendant_id: str, edges: list[HierarchyEdge]) -> bool:
    """True if candidate_id IS descendant_id, or is already its ancestor in
    the CURRENT edge set. Reparenting descendant_id under candidate_id when
    this holds would create a cycle -- either a direct self-loop
    (candidate_id == descendant_id) or a longer one (candidate_id is
    downstream of descendant_id today, e.g. via a synthesized node minted
    in between, so pointing descendant_id at it loops back).

    Both retroactive-insertion sites (_place_node's child-check absorption,
    _regroup_level's member reparenting) can propose exactly this: neither
    a "does focal subsume this candidate" LLM judgment nor a regroup
    label-collision reuse has any way to know the CURRENT tree shape, so
    checking it structurally here is the only guard. Measured without this
    guard on MINE (gpt-oss, 765 real types): 244 self-loops and at least one
    3-cycle (`property -> abstract attribute -> attribute -> property`)."""
    if candidate_id == descendant_id:
        return True
    parents = {e.child_type_id: e.parent_type_id for e in edges}
    seen: set[str] = set()
    cur = descendant_id
    while cur in parents:
        cur = parents[cur]
        if cur == candidate_id:
            return True
        if cur in seen:
            break
        seen.add(cur)
    return False


def _reparent(
    child_id: str, new_parent_id: str,
    edges: list[HierarchyEdge], all_nodes: dict[str, _Node], confidence: float,
) -> None:
    """Retroactively insert new_parent_id ABOVE child_id: remove any
    existing parent edge child_id had (none, if child_id was still an
    unattached pool member) and replace it with one pointing at
    new_parent_id. This is the concrete mechanism behind the "parent of an
    existing sibling" outcome -- CLAUDE.md §16.2.2c, the fix for the
    permanently-too-shallow-attachment gap."""
    for i, e in enumerate(edges):
        if e.child_type_id == child_id:
            edges.pop(i)
            break
    child_node = all_nodes[child_id]
    parent_node = all_nodes[new_parent_id]
    parent_node.profile = _merge_profiles([parent_node.profile, child_node.profile])
    edges.append(_make_edge(child_id, new_parent_id, child_node.profile, parent_node.profile, confidence))


def _embed_inline(
    type_id: str, node: _Node, embedder, embedding_by_id: dict, embed_lock,
) -> None:
    """Embed a node minted by _regroup_level, so a later _check_collision
    call has an embedding to compare against. `embedding_by_id` is built
    once per band from `pool`, and a regroup-minted node does not exist yet
    at that point. Best-effort -- a failure costs collision-check recall for
    one node, never the placement itself."""
    if embedder is None or embedding_by_id is None or type_id in embedding_by_id:
        return
    try:
        vec = embedder.embed([node.embed_text()], batch_size=1)[0]
    except Exception:   # noqa: BLE001 -- ranking degrades, placement continues
        logger.debug("inline embed failed for %s; candidate ranking falls back to Weeds/alphabetical", type_id)
        return
    if embed_lock is not None:
        with embed_lock:
            embedding_by_id[type_id] = vec
    else:
        embedding_by_id[type_id] = vec


def _check_collision(
    label: str,
    known_labels: dict[str, str],
    known_lock: threading.Lock,
    embedding_by_id: dict[str, np.ndarray],
    embedder,
    threshold: float,
) -> Optional[str]:
    """Before minting a synthesized parent, check whether `label` is really
    the SAME concept as something that already exists -- a real T* type, or
    a label a DIFFERENT regroup call already synthesized elsewhere in this
    run. Two independent regroup calls have zero visibility into each
    other's decisions, so this is the only thing standing between them and
    reinventing the same abstraction twice (measured on MINE: two disjoint
    batches of biological-ish orphans independently minted 'biological
    entity' and 'biological concept' for overlapping content).

    Exact normalized-label match first (free, no embedding call). Then an
    embedding-similarity check against every known label, since exact match
    alone misses near-miss variants ('biological entity' vs 'biological
    entity type' for the identical member set, also measured). Returns the
    existing type_id to reuse, or None if this is genuinely new."""
    norm = normalize_label(label)
    with known_lock:
        existing = known_labels.get(norm)
        if existing is not None:
            return existing
        snapshot = dict(known_labels)

    if embedder is None or not snapshot:
        return None
    try:
        candidate_vec = embedder.embed([label], batch_size=1)[0]
    except Exception:  # noqa: BLE001 -- collision check is best-effort, never blocks minting
        logger.debug("collision-check embed failed for %r; skipping near-duplicate check", label)
        return None

    known_ids = [tid for tid in snapshot.values() if tid in embedding_by_id]
    if not known_ids:
        return None
    known_matrix = np.stack([embedding_by_id[tid] for tid in known_ids])
    sims = 1.0 - cosine_distances(candidate_vec.reshape(1, -1), known_matrix)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return known_ids[best_idx]
    return None


def _regroup_level(
    candidate_ids: list[str],
    current_anchor: Optional[str],
    pool: dict[str, _Node],
    all_nodes: dict[str, _Node],
    edges: list[HierarchyEdge],
    synthesized: list[SynthesizedType],
    synthetic_counter: list[int],
    counter_lock: threading.Lock,
    known_labels: dict[str, str],
    known_lock: threading.Lock,
    embedding_by_id: dict[str, np.ndarray],
    embedder,
    embed_lock: threading.Lock,
    llm_extractor,
    config: HierarchyConfig,
    roots: set[str],
) -> list[str]:
    """Consolidate an overflowing level down to <= config.candidate_batch_size
    by grouping true siblings under an invented (or REUSED -- see
    _check_collision) shared parent. Every consumed member is reparented
    under its group's parent, which takes the members' place at this same
    level -- no respawn/re-placement needed, since regroup is only ever
    answering "what capacity-managing structure does THIS level need", not a
    general "are these related" judgment (that's what made the old
    unbounded same_class mechanism dangerous: CLAUDE.md §6/§14 L2).

    Runs under the caller's anchor lock already -- see _place_node."""
    remaining = list(candidate_ids)
    for _round in range(config.max_regroup_rounds):
        if len(remaining) <= config.candidate_batch_size:
            break
        chunk = remaining[: config.max_regroup_input]
        rest = remaining[config.max_regroup_input:]
        labels_by_id = {tid: all_nodes[tid].label for tid in chunk}
        label_to_id = {label: tid for tid, label in labels_by_id.items()}
        try:
            groups = llm_extractor.regroup_hierarchy_siblings(list(labels_by_id.values()))
        except Exception:
            logger.exception("regroup_hierarchy_siblings failed; leaving this level unconsolidated")
            groups = []

        consumed: set[str] = set()
        new_ids: list[str] = []
        for g in groups:
            member_ids = [label_to_id[m] for m in (g.get("members") or []) if m in label_to_id]
            member_ids = [m for m in member_ids if m not in consumed]
            if len(member_ids) < 2:
                continue
            new_label = str(g.get("label") or "").strip()
            if not new_label:
                continue

            existing_id = _check_collision(
                new_label, known_labels, known_lock, embedding_by_id, embedder,
                config.label_collision_threshold,
            )
            if existing_id is not None and existing_id in all_nodes:
                new_id = existing_id
            else:
                new_id = None  # decided below, once we know at least one member can safely attach

            # A candidate member that is ALREADY an ancestor of new_id (or,
            # in the reuse case, literally new_id itself -- e.g. the group's
            # own proposed label collided with one of its OWN members'
            # labels) must not be reparented under it: that would loop the
            # tree back on itself. Measured without this guard: 244
            # self-loops on a real MINE run.
            safe_member_ids = [
                m for m in member_ids
                if not (new_id is not None and _is_ancestor_or_self(m, new_id, edges))
            ]
            dropped = [m for m in member_ids if m not in safe_member_ids]
            if dropped:
                logger.warning(
                    "regroup: dropping member(s) %r from group %r -- would create a cycle",
                    [labels_by_id.get(m, m) for m in dropped], new_label,
                )
            if len(safe_member_ids) < 2 and existing_id is None:
                continue  # nothing safe left to justify minting a brand-new node
            member_ids = safe_member_ids
            if not member_ids:
                continue

            if new_id is None:
                new_id = _next_synthetic_id("type_h", synthetic_counter, counter_lock)
                new_node = _Node(
                    type_id=new_id, label=new_label, profile={}, is_leaf=False,
                    definition=str(g.get("definition") or ""),
                )
                all_nodes[new_id] = new_node
                synthesized.append(SynthesizedType(
                    type_id=new_id, canonical_label=new_label, definition=new_node.definition,
                    child_type_ids=list(member_ids),
                ))
                _embed_inline(new_id, new_node, embedder, embedding_by_id, embed_lock)
                with known_lock:
                    known_labels[normalize_label(new_label)] = new_id
            else:
                new_node = all_nodes[new_id]
                logger.info(
                    "regroup: reusing existing label %r (%s) instead of minting a duplicate for %r",
                    new_node.label, new_id, [labels_by_id.get(m, m) for m in member_ids],
                )

            for member_id in member_ids:
                _reparent(member_id, new_id, edges, all_nodes, 0.5)
                pool.pop(member_id, None)
                roots.discard(member_id)
                consumed.add(member_id)

            if new_id != existing_id:
                if current_anchor is not None:
                    anchor_node = all_nodes[current_anchor]
                    edges.append(_make_edge(new_id, current_anchor, new_node.profile, anchor_node.profile, 0.5))
                else:
                    roots.add(new_id)
            new_ids.append(new_id)

        if not consumed:
            logger.warning(
                "regroup made no progress on an oversized level (%d candidates); "
                "leaving it over the %d-candidate cap", len(remaining), config.candidate_batch_size,
            )
            break
        survivors = list(dict.fromkeys([tid for tid in chunk if tid not in consumed] + new_ids))
        remaining = survivors + rest

    return remaining


def _place_node(
    focal_id: str,
    pool: dict[str, _Node],
    all_nodes: dict[str, _Node],
    edges: list[HierarchyEdge],
    synthesized: list[SynthesizedType],
    synthetic_counter: list[int],
    llm_extractor,
    embedding_by_id: dict[str, np.ndarray],
    config: HierarchyConfig,
    unresolved_children: dict[str, list[str]],
    anchor_locks: "_AnchorLocks",
    counter_lock: threading.Lock,
    roots: set[str],
    known_labels: dict[str, str],
    known_lock: threading.Lock,
    examples_by_type: dict[str, list[str]],
    embedder=None,
    embed_lock: Optional[threading.Lock] = None,
    start_anchor: Optional[str] = None,
    start_depth: int = 0,
) -> None:
    """Route one focal node through the tree via two directional questions
    per level, asked separately rather than as one five-way choice:

      1. PARENT CHECK -- does focal belong under any of the level's current
         candidates? If yes, descend into that candidate's children next
         (depth += 1). A "same_concept" flag on this answer means focal and
         the match are the identical concept -- collapse rather than
         descend.
      2. CHILD CHECK -- only asked if (1) found nothing -- does focal
         subsume any of the level's current candidates? If yes, retroactively
         insert focal above them (_reparent) and focal settles at this level.
      3. Neither: focal settles at this level with no relation asserted (a
         new root, or a plain sibling under the current anchor).

    Before EITHER question is asked, the level currently in front of focal
    (the live root set, or current_anchor's children) is proactively
    consolidated via _regroup_level if it exceeds config.candidate_batch_size
    -- so a level is never larger than one call can show, and no
    escalation/ranking machinery is needed to decide what a focal node gets
    to see."""
    depth = start_depth
    current_anchor: Optional[str] = start_anchor

    while depth < config.max_depth:
        with anchor_locks.acquire(current_anchor):
            if focal_id not in pool:
                # Consumed by a concurrently-placed sibling under a
                # DIFFERENT anchor's lock before this node's own turn came
                # up -- nothing left to place.
                return

            children_map = _build_children_map(edges)
            if current_anchor is None:
                candidate_ids = [tid for tid in roots if tid != focal_id and tid in all_nodes]
            else:
                candidate_ids = [
                    cid for cid in children_map.get(current_anchor, [])
                    if cid in all_nodes and cid != focal_id
                ]

            if len(candidate_ids) > config.candidate_batch_size:
                candidate_ids = _regroup_level(
                    candidate_ids, current_anchor, pool, all_nodes, edges, synthesized,
                    synthetic_counter, counter_lock, known_labels, known_lock,
                    embedding_by_id, embedder, embed_lock, llm_extractor, config, roots,
                )
                children_map = _build_children_map(edges)
                if current_anchor is None:
                    candidate_ids = [tid for tid in roots if tid != focal_id and tid in all_nodes]
                else:
                    candidate_ids = [
                        cid for cid in children_map.get(current_anchor, [])
                        if cid in all_nodes and cid != focal_id
                    ]

            if not candidate_ids:
                break

            focal_node = all_nodes[focal_id]
            candidate_nodes = [all_nodes[cid] for cid in candidate_ids]
            focal_ctx = _node_context(
                focal_node, examples_by_type,
                _subclass_labels(focal_id, children_map, all_nodes, config.context_max_subclasses),
            )
            candidate_ctx = {
                n.label: _node_context(
                    n, examples_by_type,
                    _subclass_labels(n.type_id, children_map, all_nodes, config.context_max_subclasses),
                )
                for n in candidate_nodes
            }
            candidate_labels = [n.label for n in candidate_nodes]
            label_to_id = {n.label: n.type_id for n in candidate_nodes}

            try:
                parent_result = llm_extractor.check_hierarchy_parent(
                    focal_label=focal_node.label, focal_context=focal_ctx,
                    candidate_labels=candidate_labels, candidate_context=candidate_ctx,
                )
            except Exception:
                logger.exception("check_hierarchy_parent failed for %r; treating as no match", focal_node.label)
                parent_result = {"parent": None}

            parent_label = parent_result.get("parent") if isinstance(parent_result, dict) else None
            matched_id = label_to_id.get(parent_label) if parent_label else None
            if matched_id is not None:
                if parent_result.get("same_concept"):
                    merged_ids = [focal_id, matched_id]
                    survivor_id = _collapse_duplicate_nodes(merged_ids, pool, all_nodes, edges, synthesized)
                    for tid in merged_ids:
                        if tid != survivor_id:
                            roots.discard(tid)
                    if survivor_id in pool:
                        if current_anchor is None:
                            roots.add(survivor_id)
                    else:
                        roots.discard(survivor_id)
                    return
                current_anchor = matched_id
                depth += 1
                continue

            try:
                child_result = llm_extractor.check_hierarchy_children(
                    focal_label=focal_node.label, focal_context=focal_ctx,
                    candidate_labels=candidate_labels, candidate_context=candidate_ctx,
                )
            except Exception:
                logger.exception("check_hierarchy_children failed for %r; treating as no match", focal_node.label)
                child_result = {"children": []}

            child_labels = (child_result.get("children") or []) if isinstance(child_result, dict) else []
            matched_ids = [label_to_id[c] for c in child_labels if c in label_to_id]
            cyclic = [mid for mid in matched_ids if _is_ancestor_or_self(mid, focal_id, edges)]
            if cyclic:
                logger.warning(
                    "Rejecting child claim(s) by %r: %s would create a cycle (already an ancestor)",
                    focal_node.label, [all_nodes[m].label for m in cyclic],
                )
                matched_ids = [mid for mid in matched_ids if mid not in cyclic]
            if matched_ids and len(matched_ids) > 3 and len(matched_ids) > config.max_child_claim_share * len(candidate_ids):
                logger.warning(
                    "Rejecting 'child' claim by %r: claimed %d of %d candidates -- "
                    "pathology backstop, treating as no match", focal_node.label, len(matched_ids), len(candidate_ids),
                )
                matched_ids = []
            if matched_ids:
                try:
                    child_confidence = float(child_result.get("confidence", 0.8))
                except (TypeError, ValueError):
                    child_confidence = 0.8
                for mid in matched_ids:
                    _reparent(mid, focal_id, edges, all_nodes, child_confidence)
                    pool.pop(mid, None)
                    roots.discard(mid)
                break

            break  # neither check matched anything -- settle here

    with anchor_locks.acquire(current_anchor):
        if depth >= config.max_depth:
            bucket_key = current_anchor or ""
            unresolved_children.setdefault(bucket_key, []).append(focal_id)
            pool.pop(focal_id, None)
            return

        if current_anchor is not None and focal_id in pool:
            focal_node = pool[focal_id]
            anchor_node = all_nodes[current_anchor]
            edges.append(_make_edge(focal_id, current_anchor, focal_node.profile, anchor_node.profile, 0.5))
            pool.pop(focal_id, None)
        elif focal_id in pool:
            roots.add(focal_id)



# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def induce_hierarchy(
    type_vocab: DeduplicationResult,
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    triplets: Optional[list[dict]] = None,
    *,
    contriever_model: str = "facebook/contriever",
    device: str = None,
    config: Optional[HierarchyConfig] = None,
) -> HierarchyInductionResult:
    """
    Induce a subClassOf hierarchy over the flat type vocabulary T* via a
    single priority-ordered pass.

    Requires relation_vocab's CanonicalRelation.subject_types / object_types
    to already hold canonical type_ids (relation_dedup.update_relation_type_map()
    must have been called first) so priority-band in-degree can be computed
    directly at the type_id level.

    `triplets` (the same raw list pipeline.load_triplets() produces) supplies
    real corpus EXAMPLES per type for placement context -- see
    HierarchyConfig.context_max_examples. None/empty means every node's
    context is definition + subclasses only, no examples (a degraded but
    still-functional mode, e.g. for a caller that only has vocabularies).
    """
    config = config or HierarchyConfig()
    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    pool, deferred_low_support = _initial_pool(type_vocab, relation_vocab, config)
    all_nodes: dict[str, _Node] = dict(pool)
    edges: list[HierarchyEdge] = []
    synthesized: list[SynthesizedType] = []
    synthetic_counter = [0]
    counter_lock = threading.Lock()   # protects synthetic_counter across concurrently-placed bands 2+
    unresolved_children: dict[str, list[str]] = defaultdict(list)
    examples_by_type = _collect_type_examples(triplets or [], type_vocab, config.context_max_examples)

    # The tree's live top level. Distinct from `pool`, which also holds
    # nodes that simply have not been routed yet -- conflating the two is
    # what let unplaced nodes act as top-level candidates.
    roots: set[str] = set()
    embed_lock = threading.Lock()

    # Every label currently in play (real T* types + anything synthesized so
    # far this run), normalized -> type_id. The ONLY defense against two
    # independent regroup calls reinventing the same abstraction under
    # different names -- see _check_collision.
    known_labels: dict[str, str] = {
        normalize_label(node.label): tid for tid, node in all_nodes.items()
    }
    known_lock = threading.Lock()

    pinned_roots: set[str] = set()
    if config.seed_roots:
        pinned_roots = _apply_seed_roots(config.seed_roots, pool, all_nodes, edges, synthetic_counter)
        num_seed_edges = sum(1 for e in edges if e.is_seed)
        logger.info(
            "Seeded with %d pinned root(s): %s (%d nested seed edge(s) below them)",
            len(pinned_roots), sorted(pool[r].label for r in pinned_roots), num_seed_edges,
        )
        roots |= pinned_roots

    bands = _priority_bands(pool, pinned_roots, config)
    logger.info(
        "Hierarchy induction: %d priority band(s) covering %d node(s) (%d pinned root(s) excluded)",
        len(bands), sum(len(b) for b in bands), len(pinned_roots),
    )

    # Embeddings are cached by type_id across the whole run, not recomputed
    # per band: embed_text() depends only on a node's label/definition, and
    # neither is ever mutated after the node is created (only .profile is),
    # so a given type_id's embedding is stable for its entire lifetime. Only
    # newly-appeared pool members (nodes synthesized by a "same_class"
    # decision in an earlier band) ever need a fresh embed() call.
    embedding_cache: dict[str, np.ndarray] = {}

    for band_idx, band in enumerate(bands, start=1):
        if pool:
            uncached_ids = [tid for tid in pool if tid not in embedding_cache]
            if uncached_ids:
                embeddings = embedder.embed(
                    [pool[tid].embed_text() for tid in uncached_ids], batch_size=config.embed_batch_size,
                )
                for tid, emb in zip(uncached_ids, embeddings):
                    embedding_cache[tid] = emb
            embedding_by_id = {tid: embedding_cache[tid] for tid in pool}
        else:
            embedding_by_id = {}

        logger.info("Priority band %d/%d: %d node(s) to place, pool=%d", band_idx, len(bands), len(band), len(pool))
        anchor_locks = _AnchorLocks()   # fresh per band -- see _AnchorLocks docstring

        if band_idx == 1:
            # Root band: every node's FIRST comparison is against the
            # top-level pool (current_anchor=None), so band 1 is one shared
            # anchor regardless of how many worker threads we throw at it --
            # run it as a plain sequential loop and skip the executor
            # bookkeeping entirely (CLAUDE.md §16.2.4).
            for focal_id in tqdm(band, desc=f"Priority band {band_idx}/{len(bands)}"):
                if focal_id not in pool:
                    continue  # consumed earlier this band (e.g. absorbed as someone's child, or merged away)
                _place_node(
                    focal_id, pool, all_nodes, edges, synthesized, synthetic_counter,
                    llm_extractor, embedding_by_id, config, unresolved_children,
                    anchor_locks, counter_lock, roots, known_labels, known_lock, examples_by_type,
                    embedder=embedder, embed_lock=embed_lock,
                )
        else:
            # Bands 2+: nodes may already be anchored under different,
            # disjoint parents by the time they diverge past their own
            # first step -- safe to place concurrently, serialized only
            # where two nodes actually contend for the same anchor's
            # candidates (CLAUDE.md §16.2.4; see _AnchorLocks).
            with ThreadPoolExecutor(max_workers=config.max_parallel_workers) as executor:
                futures = {
                    executor.submit(
                        _place_node, focal_id, pool, all_nodes, edges, synthesized, synthetic_counter,
                        llm_extractor, embedding_by_id, config, unresolved_children,
                        anchor_locks, counter_lock, roots, known_labels, known_lock, examples_by_type,
                        embedder=embedder, embed_lock=embed_lock,
                    ): focal_id
                    for focal_id in band if focal_id in pool
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Priority band {band_idx}/{len(bands)}"):
                    future.result()  # re-raise any worker exception instead of silently swallowing it

    # Registered roots, plus two defensive catches: a node that parents
    # something but was never itself given a parent (e.g. a regroup-minted
    # node whose own settle step hit max_depth) would otherwise leave its
    # whole subtree unreachable from hierarchy.roots; and anything still
    # sitting in `pool` unrouted. Minus, in all cases, anything that does have a
    # parent edge.
    parented = {e.child_type_id for e in edges}
    subtree_tops = {e.parent_type_id for e in edges}
    final_roots = (roots | subtree_tops | set(pool)) - parented
    stray = final_roots - roots
    if stray:
        logger.warning(
            "%d root(s) recovered defensively (parentless but unregistered): %s",
            len(stray), sorted(all_nodes[t].label for t in stray if t in all_nodes)[:10],
        )
    hierarchy = _build_type_hierarchy(edges, roots=sorted(final_roots))
    num_unresolved = sum(len(v) for v in unresolved_children.values())
    logger.info(
        "Hierarchy induction complete: %d root(s), %d edge(s), %d synthesized type(s), "
        "%d node(s) deferred past max_depth across %d bucket(s)",
        len(hierarchy.roots), len(edges), len(synthesized), num_unresolved, len(unresolved_children),
    )

    return HierarchyInductionResult(
        hierarchy=hierarchy, synthesized_types=synthesized,
        unresolved_children=dict(unresolved_children),
        deferred_low_support=deferred_low_support,
    )


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Ancestor-path / LCA helpers (shared by constraints.py and entity_dedup.py)
# ═══════════════════════════════════════════════════════════════════════════════

def ancestor_path(type_id: str, hierarchy: "TypeHierarchy") -> list[str]:
    """Leaf-to-root ancestor chain for type_id (type_id itself first),
    following hierarchy.parents (single parent per type -- a forest)."""
    path = [type_id]
    seen = {type_id}
    current = type_id
    while current in hierarchy.parents:
        parents = hierarchy.parents[current]
        if not parents:
            break
        current = parents[0]
        if current in seen:
            break  # defensive only: the hierarchy is acyclic by construction
        seen.add(current)
        path.append(current)
    return path


def lowest_common_ancestor(type_ids: set[str], hierarchy: "TypeHierarchy") -> Optional[str]:
    """Most specific ancestor common to every type in type_ids, or None if
    they don't all share one (disconnected trees of the hierarchy forest)."""
    if not type_ids:
        return None
    if len(type_ids) == 1:
        return next(iter(type_ids))

    paths = [ancestor_path(t, hierarchy) for t in type_ids]
    common = set(paths[0])
    for path in paths[1:]:
        common &= set(path)
    if not common:
        return None

    for node in paths[0]:  # leaf-to-root order -> first hit is the deepest
        if node in common:
            return node
    return None
