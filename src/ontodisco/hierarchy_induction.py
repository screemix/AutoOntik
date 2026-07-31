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

FOUR-WAY DECISION PER LEVEL (CLAUDE.md §16.2.2), replacing the old
`_descend_tree()`'s binary parent-vs-stop check: at each level, an incoming
node is compared against the current candidate set and can be told it's a
synonym (merge -- this is what fixes the lost-synonym gap: the old code saw
this candidate too, it just discarded a same_concept answer), a child of one
of them (descend further), a PARENT of one OR MORE of them (insert above all
of them at once, in the same call, retroactively correcting an earlier
too-shallow attachment -- the concrete fix for the sparse-forest gap; the LLM
names every candidate it subsumes rather than just one, so a later call isn't
relied on to catch the rest), or none of the above (settle here).

CANDIDATE BATCHING WITH ESCALATION (CLAUDE.md §16.2.3): candidates are shown
in similarity-ranked batches (`candidate_batch_size`) rather than a small
fixed top-k. Only a "none" result escalates to the next, less-similar batch;
every other outcome is conclusive and stops the search immediately, since
later batches can't produce a *better* match than an already-conclusive one.

DEPTH AS A DEFERRED CUT (CLAUDE.md §16.2.7): once a node's placement would
exceed `max_depth`, it is NOT discarded and NOT silently attached as if it
were a confirmed sibling relationship -- it's recorded in
`HierarchyInductionResult.unresolved_children`, a structurally separate
mapping from `TypeHierarchy.edges`, precisely so a downstream consumer can
never mistake a deferred placement for an LLM-confirmed one. Resuming later
just means seeding a fresh pool from exactly that bucket and re-running this
same module on it -- no new algorithm needed.

PARALLEL PLACEMENT (CLAUDE.md §16.2.4): band 1 (the root band) always runs
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

NOT IMPLEMENTED YET (see CLAUDE.md §16.4 for the full list): confidence-tiered
model routing (§16.2.6, explicitly deferred), and backtracking across
branches once a node has committed to one.
"""

from __future__ import annotations

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
    """A single directed subClassOf edge. Direction: child IS-A parent.

    Unlike the earlier design, an edge is not necessarily permanent for the
    edge's lifetime of the run: a "parent of an existing sibling" decision
    (CLAUDE.md §16.2.2c) removes an existing child's edge and replaces it
    with one pointing at the newly-inserted intermediate node. `is_direct`
    remains always True regardless -- every edge that exists at any given
    moment still connects a direct parent/child pair; what changed is that
    edges can now be retroactively replaced rather than being immutable from
    the moment they're created."""
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

    # -- Priority queue / seeding (CLAUDE.md §16.2.1) --
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
                                                       # into their seed parent as its anchor. No pinned
                                                       # node (root or not) is ever itself placed by the
                                                       # priority-band loop -- only ever offered as a
                                                       # candidate. A dict entry may also carry an
                                                       # optional "description" -- becomes the node's
                                                       # definition (_Node.definition), but only when the
                                                       # label is freshly synthesized (no T* match); an
                                                       # existing T* type's own (currently always empty)
                                                       # definition is never overwritten by a seed's
                                                       # description. Feeds BOTH embed_text() (embedding/
                                                       # similarity ranking) AND the "(context: ...)"
                                                       # shown to the LLM during placement (_node_context)
                                                       # -- the latter matters most for a synthesized seed
                                                       # node, which otherwise reaches the LLM as a bare
                                                       # label with zero context (no relation evidence
                                                       # exists yet for a node nothing has attached under).
    seed_roots_path: Optional[str] = None       # path to a separate YAML file holding the seed
                                                 # hierarchy (a `seed_roots:` key, same shape as above,
                                                 # or a bare top-level list), so a reusable seed ontology
                                                 # (e.g. configs/seeds/dolce.yaml) doesn't have to be
                                                 # copy-pasted into every pipeline config. Resolved by
                                                 # pipeline.run_pipeline() at run time (mirrors how
                                                 # config.input_path is only read when the pipeline
                                                 # actually runs, not at load_config() time) -- this
                                                 # dataclass itself does no file I/O. If both this and
                                                 # `seed_roots` are set, the file wins and a warning is
                                                 # logged; induce_hierarchy() itself only ever looks at
                                                 # `seed_roots` and has no knowledge of this field.
    weeds_containment_threshold: float = 0.75   # gates both in-degree computation (which pairs count
                                                 # toward a node's breadth score) and is passed through
                                                 # as informational context only -- the LLM call is what
                                                 # actually decides each relation, this only decides
                                                 # which pairs are worth flagging as evidence at all.
    priority_band_tolerance: int = 10           # a band includes every node within this many in-degree
                                                 # units of the band's own top score, not just exact
                                                 # ties -- widens the tie tolerance already accepted for
                                                 # in-degree as a frequency proxy (CLAUDE.md §16.2.1),
                                                 # and keeps each band's node count large enough to
                                                 # actually saturate the parallel placement thread pool
                                                 # (see max_parallel_workers below) rather than leaving
                                                 # it starved of independent work.

    # -- Candidate batching with escalation (CLAUDE.md §16.2.3) --
    candidate_batch_size: int = 50              # candidates shown per LLM call
    max_escalation_batches: int = 5             # safety cap on how many less-similar batches one node
                                                 # will escalate through before giving up (proposed
                                                 # value, not validated -- see CLAUDE.md §16.4)

    # -- Depth (CLAUDE.md §16.2.7) --
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

    # -- LLM prompting --
    context_top_k: int = 5                      # top relation dimensions shown per candidate

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
    # parent_type_id (or "" for the top level) -> [type_ids deferred past
    # max_depth]. Deliberately NOT part of `hierarchy.edges` -- see the
    # module docstring and CLAUDE.md §16.2.7 for why keeping this
    # structurally separate matters.


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


def _node_context(
    node: _Node, batch_profiles: dict[str, dict[str, float]],
    relation_vocab: "RelationDeduplicationResult", top_k: int,
) -> str:
    """Everything worth telling the LLM about one node beyond its bare
    label, combined into the single "(context: ...)" parenthetical
    resolve_hierarchy_relation already renders per focal/candidate
    (openai_utils.py:306-316) -- no prompt-format change needed on that
    side. Two independent sources, joined when both exist:
      - node.definition: a seed's "description" (_apply_seed_roots), or an
        LLM-synthesized SynthesizedType's own definition -- otherwise "".
        This is the ONLY way seed-node descriptions ever reach the LLM's
        placement decision; embed_text() (used for embedding/similarity
        ranking) is a separate consumer of the same field, not this one.
      - the PPMI relation-signature summary (describe_relation_context) --
        "" for a freshly-synthesized seed node, which has no relation
        evidence at all (profile={}) until something attaches under it.
    Definition matters most for exactly that case: without it, a synthesized
    seed node like "conceptual entity" would reach the LLM as a bare label
    with NO context whatsoever, since describe_relation_context alone
    returns "" for it."""
    parts = []
    if node.definition:
        parts.append(node.definition)
    rel_ctx = describe_relation_context(node.type_id, batch_profiles, relation_vocab, top_k=top_k)
    if rel_ctx:
        parts.append(rel_ctx)
    return "; ".join(parts)


def _resolve_relation_with_llm(
    focal_node: _Node,
    candidate_nodes: list[_Node],
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    config: HierarchyConfig,
) -> dict:
    """One pairwise/multi-candidate LLM call: focal vs. up to
    config.candidate_batch_size candidates. Never raises -- falls back to
    {"relation": "none"} on any failure, since a no-op is always the safe
    outcome (this node just doesn't resolve this call, and either escalates
    to the next batch or settles at its current position)."""
    batch_profiles = {n.type_id: n.profile for n in [focal_node] + candidate_nodes}
    focal_context = _node_context(focal_node, batch_profiles, relation_vocab, config.context_top_k)
    candidate_context = {
        n.label: _node_context(n, batch_profiles, relation_vocab, config.context_top_k)
        for n in candidate_nodes
    }
    candidate_context = {label: ctx for label, ctx in candidate_context.items() if ctx}

    try:
        return llm_extractor.resolve_hierarchy_relation(
            focal_label=focal_node.label,
            candidate_labels=[n.label for n in candidate_nodes],
            focal_context=focal_context,
            candidate_context=candidate_context,
        )
    except Exception:
        logger.exception("resolve_hierarchy_relation call failed; treating as no relation")
        return {"relation": "none"}


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
    to the lowest band by default -- they still get placed, just via the
    embedding-similarity fallback in candidate ranking (see
    _resolve_with_batched_escalation)."""
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
#  Candidate batching with escalation (CLAUDE.md §16.2.3)
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_by_similarity(focal_id: str, candidate_ids: list[str], embedding_by_id: dict[str, np.ndarray]) -> list[str]:
    focal_emb = embedding_by_id[focal_id].reshape(1, -1)
    other_embs = np.stack([embedding_by_id[cid] for cid in candidate_ids])
    sims = 1.0 - cosine_distances(focal_emb, other_embs)[0]
    ranked = sorted(zip(candidate_ids, sims), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked]


def _match_candidate_labels(
    labels: list[str], batch_ids: list[str], batch_nodes: list[_Node],
) -> tuple[list[str], list[str]]:
    """Resolve each candidate label the LLM named back to its type_id within
    this batch. Returns (matched_ids, unmatched_labels) -- a PARTIAL match
    (e.g. a "child" decision naming two labels when only one exists in this
    batch under that exact string) still proceeds with whatever DID match,
    logged by the caller rather than silently dropped or treated as a full
    escalation."""
    label_to_id: dict[str, str] = {}
    for cid, n in zip(batch_ids, batch_nodes):
        label_to_id.setdefault(n.label, cid)
    matched_ids: list[str] = []
    unmatched: list[str] = []
    for label in labels:
        cid = label_to_id.get(label)
        if cid is not None:
            matched_ids.append(cid)
        else:
            unmatched.append(label)
    return matched_ids, unmatched


def _resolve_with_batched_escalation(
    focal_id: str,
    candidate_ids: list[str],
    all_nodes: dict[str, _Node],
    embedding_by_id: dict[str, np.ndarray],
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    config: HierarchyConfig,
) -> tuple[Optional[dict], Optional[list[str]]]:
    """Show candidates in similarity-ranked batches, escalating to the next
    (less-similar) batch ONLY on "none" -- any other outcome is conclusive
    and stops immediately, since a less-similar batch can't produce a
    BETTER "child of"/"parent of"/"synonym" match than an already-conclusive
    one from a more-similar batch. Capped at max_escalation_batches.

    Falls back to a stable (sorted-by-id) order, skipping similarity
    ranking, when embeddings aren't available for every candidate -- this
    happens for descent-level candidates (a node's existing children, which
    were never part of any band's precomputed pool embeddings). Descent-
    level candidate lists are typically small enough that ranking order
    matters far less than at the top level, where the candidate set can be
    the whole remaining pool.

    The returned matched-ids list has more than one entry only for a
    "child" decision (CLAUDE.md §16.2.2's multi-candidate requirement --
    see resolve_hierarchy_relation's docstring); every other relation
    always resolves to exactly one id here."""
    if not candidate_ids:
        return None, None

    if embedding_by_id and focal_id in embedding_by_id and all(cid in embedding_by_id for cid in candidate_ids):
        ranked_ids = _rank_by_similarity(focal_id, candidate_ids, embedding_by_id)
    else:
        ranked_ids = sorted(candidate_ids)

    focal_node = all_nodes[focal_id]
    batch_size = max(1, config.candidate_batch_size)
    for batch_num in range(config.max_escalation_batches):
        start = batch_num * batch_size
        if start >= len(ranked_ids):
            break
        batch_ids = ranked_ids[start: start + batch_size]
        batch_nodes = [all_nodes[cid] for cid in batch_ids]

        decision = _resolve_relation_with_llm(focal_node, batch_nodes, relation_vocab, llm_extractor, config)
        relation = decision.get("relation")
        if relation and relation != "none":
            if relation == "child":
                # Accept either the normalized plural "candidates" (what
                # openai_utils.LLMTripletExtractor.resolve_hierarchy_relation
                # always produces) or a bare singular "candidate" -- don't
                # assume every llm_extractor implementation (e.g. a test
                # double) applies that normalization itself.
                labels = decision.get("candidates")
                if labels is None:
                    single = decision.get("candidate")
                    labels = [single] if single else []
                elif not isinstance(labels, list):
                    labels = [labels]
            else:
                single = decision.get("candidate")
                labels = [single] if single else []
            matched_ids, unmatched = _match_candidate_labels(labels, batch_ids, batch_nodes)
            if unmatched:
                logger.warning(
                    "Batched decision for %r (relation=%s) named unmatched candidate "
                    "label(s) %r; proceeding with whatever matched", focal_node.label, relation, unmatched,
                )
            if matched_ids:
                return decision, matched_ids
            logger.warning(
                "Batched decision for %r (relation=%s) matched no candidates in this "
                "batch; treating as inconclusive and escalating", focal_node.label, relation,
            )
        # "none", or no matched candidates at all -> escalate to the next batch

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  Placement (CLAUDE.md §16.2.2, §16.2.7, §16.2.4)
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


def _place_node(
    focal_id: str,
    pool: dict[str, _Node],
    all_nodes: dict[str, _Node],
    edges: list[HierarchyEdge],
    synthesized: list[SynthesizedType],
    synthetic_counter: list[int],
    relation_vocab: "RelationDeduplicationResult",
    llm_extractor,
    embedding_by_id: dict[str, np.ndarray],
    config: HierarchyConfig,
    unresolved_children: dict[str, list[str]],
    anchor_locks: _AnchorLocks,
    counter_lock: threading.Lock,
) -> None:
    """Route one focal node from the top of the current pool down through
    the tree, applying the four-way decision at each level. Terminates via
    exactly one of: merged away (same_concept), replaced by a freshly
    synthesized shared parent (same_class), attached with a normal
    confirmed edge (parent chain bottoming out, or a "child" absorption
    settling focal at its current level), deferred into
    unresolved_children (max_depth reached), or left untouched in `pool` as
    a root (no candidates, or every batch came back "none", at the top
    level).

    Each iteration's candidate-read/LLM-call/decision-apply sequence runs
    under `anchor_locks.acquire(current_anchor)` (CLAUDE.md §16.2.4): two
    nodes currently comparing against the SAME anchor's candidates can never
    have their steps interleaved, since acting on a stale read of that
    anchor's children (e.g. two nodes independently claiming the same
    existing child, or independently inventing two different same_class
    parents for it) would otherwise be a real race. Nodes at DIFFERENT
    anchors hold different locks and run their steps -- including the LLM
    call itself -- fully concurrently. When called with `anchor_locks`
    single-threaded (band 1), this still works correctly; the locking is
    just uncontended overhead."""
    depth = 0
    current_anchor: Optional[str] = None   # None == top level, candidates drawn from `pool`

    while depth < config.max_depth:
        with anchor_locks.acquire(current_anchor):
            if focal_id not in pool:
                # Consumed by a concurrently-placed sibling (e.g. claimed as
                # a "child", or merged via same_concept) under a DIFFERENT
                # anchor's lock before this node's own turn came up --
                # nothing left to place. Cheap efficiency/determinism
                # guard, not a correctness requirement: _reparent() and
                # _collapse_duplicate_nodes() already re-scan `edges` fresh
                # rather than trust a stale snapshot, so a race that slips
                # past this check still can't produce a double-parented
                # node -- it would just waste the LLM call this check
                # avoids.
                return

            if current_anchor is None:
                candidate_ids = [tid for tid in pool if tid != focal_id]
            else:
                children_map = _build_children_map(edges)
                candidate_ids = [cid for cid in children_map.get(current_anchor, []) if cid in all_nodes]

            if not candidate_ids:
                break

            decision, matched_ids = _resolve_with_batched_escalation(
                focal_id, candidate_ids, all_nodes, embedding_by_id, relation_vocab, llm_extractor, config,
            )
            if decision is None:
                break  # exhausted escalation without a conclusive answer

            relation = decision["relation"]
            try:
                confidence = float(decision.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5

            if relation == "parent":
                # focal IS-A matched_ids[0]: descend into its own children
                # next iteration, rather than attaching yet -- an edge only
                # gets created once descent bottoms out (see the "settle"
                # step below), so a deeper, more specific attachment never
                # has to be undone. "parent" only ever resolves to one
                # candidate (CLAUDE.md §16.2.2c) -- a single-parent forest
                # can't have focal descend into two branches at once.
                current_anchor = matched_ids[0]
                depth += 1
                continue

            if relation == "child":
                # Every id in matched_ids IS-A focal: retroactively insert
                # focal above ALL of them in this one call (CLAUDE.md
                # §16.2.2's multi-candidate requirement -- claiming only
                # one per call and hoping a later call catches the rest
                # would leave the tree in a partially-corrected state,
                # since nothing guarantees a later call ever re-examines
                # the ones left behind). Each _reparent() call further
                # enriches focal's absorbed profile, so order within the
                # loop doesn't matter. Once focal absorbs its matched
                # children, it settles at the current level rather than
                # continuing to search for an even-more-general position
                # in the same call -- a later pass (e.g. a future
                # extension of a resumed unresolved_children bucket) can
                # still discover that focal itself belongs even higher.
                for matched_id in matched_ids:
                    _reparent(matched_id, focal_id, edges, all_nodes, confidence)
                    pool.pop(matched_id, None)
                break

            if relation == "same_concept":
                _collapse_duplicate_nodes([focal_id, matched_ids[0]], pool, all_nodes, edges, synthesized)
                return

            if relation == "same_class":
                matched_id = matched_ids[0]
                new_label = str(decision.get("new_parent_label", "")).strip()
                if not new_label:
                    logger.warning("same_class proposed with no new_parent_label; treating focal %s as unresolved", focal_id)
                    break
                new_id = _next_synthetic_id("type_h", synthetic_counter, counter_lock)
                focal_node = all_nodes[focal_id]
                candidate_node = all_nodes[matched_id]
                new_node = _Node(
                    type_id=new_id, label=new_label,
                    profile=_merge_profiles([focal_node.profile, candidate_node.profile]),
                    is_leaf=False, definition=str(decision.get("new_parent_definition", "") or ""),
                )
                pool[new_id] = new_node
                all_nodes[new_id] = new_node
                synthesized.append(SynthesizedType(
                    type_id=new_id, canonical_label=new_label, definition=new_node.definition,
                    child_type_ids=[focal_id, matched_id],
                ))
                for cid in (focal_id, matched_id):
                    cnode = all_nodes[cid]
                    edges.append(_make_edge(cid, new_id, cnode.profile, new_node.profile, confidence))
                    pool.pop(cid, None)
                return

            break  # unrecognized relation value; treat like "none"

    # Both of the following mutate shared state keyed by current_anchor
    # (unresolved_children / pool / edges), same as every step inside the
    # loop above -- acquire that anchor's lock here too rather than leave
    # this tail end unprotected.
    with anchor_locks.acquire(current_anchor):
        if depth >= config.max_depth:
            # Deferred, not destroyed -- see module docstring and CLAUDE.md
            # §16.2.7. current_anchor may be None (deferred right at the top
            # level, e.g. a pathologically deep chain of "parent" descents
            # before ever settling) -- bucket key "" represents that case.
            bucket_key = current_anchor or ""
            unresolved_children.setdefault(bucket_key, []).append(focal_id)
            pool.pop(focal_id, None)
            return

        if current_anchor is not None and focal_id in pool:
            # Settled: either a "parent" chain bottomed out (no further
            # child fit), or a "child" absorption just happened and focal
            # itself now belongs at current_anchor's level.
            focal_node = pool[focal_id]
            anchor_node = all_nodes[current_anchor]
            anchor_node.profile = _merge_profiles([anchor_node.profile, focal_node.profile])
            edges.append(_make_edge(focal_id, current_anchor, focal_node.profile, anchor_node.profile, 0.5))
            pool.pop(focal_id, None)
        # else current_anchor is None: focal never matched anything at the
        # top level and simply stays in `pool` as a root -- no action
        # needed, it's already there.


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
    Induce a subClassOf hierarchy over the flat type vocabulary T* via a
    single priority-ordered pass -- see the module docstring and CLAUDE.md
    §16 for the full design.

    Requires relation_vocab's CanonicalRelation.subject_types / object_types
    to already hold canonical type_ids (relation_dedup.update_relation_type_map()
    must have been called first) so relation-signature profiles can be built
    directly at the type_id level.
    """
    config = config or HierarchyConfig()
    embedder = ContrieverEmbedder(model_name=contriever_model, device=device)

    pool = _initial_pool(type_vocab, relation_vocab, config)
    all_nodes: dict[str, _Node] = dict(pool)
    edges: list[HierarchyEdge] = []
    synthesized: list[SynthesizedType] = []
    synthetic_counter = [0]
    counter_lock = threading.Lock()   # protects synthetic_counter across concurrently-placed bands 2+
    unresolved_children: dict[str, list[str]] = defaultdict(list)

    pinned_roots: set[str] = set()
    if config.seed_roots:
        pinned_roots = _apply_seed_roots(config.seed_roots, pool, all_nodes, edges, synthetic_counter)
        num_seed_edges = sum(1 for e in edges if e.is_seed)
        logger.info(
            "Seeded with %d pinned root(s): %s (%d nested seed edge(s) below them)",
            len(pinned_roots), sorted(pool[r].label for r in pinned_roots), num_seed_edges,
        )

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
                    relation_vocab, llm_extractor, embedding_by_id, config, unresolved_children,
                    anchor_locks, counter_lock,
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
                        relation_vocab, llm_extractor, embedding_by_id, config, unresolved_children,
                        anchor_locks, counter_lock,
                    ): focal_id
                    for focal_id in band if focal_id in pool
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Priority band {band_idx}/{len(bands)}"):
                    future.result()  # re-raise any worker exception instead of silently swallowing it

    hierarchy = _build_type_hierarchy(edges, roots=list(pool.keys()))
    num_unresolved = sum(len(v) for v in unresolved_children.values())
    logger.info(
        "Hierarchy induction complete: %d root(s), %d edge(s), %d synthesized type(s), "
        "%d node(s) deferred past max_depth across %d bucket(s)",
        len(hierarchy.roots), len(edges), len(synthesized), num_unresolved, len(unresolved_children),
    )

    return HierarchyInductionResult(
        hierarchy=hierarchy, synthesized_types=synthesized,
        unresolved_children=dict(unresolved_children),
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
