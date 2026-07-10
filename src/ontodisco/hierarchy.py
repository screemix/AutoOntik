"""
Hierarchy Induction via Adaptive HAC-Recursive LLM Merging
==========================================================

Induces a subClassOf type hierarchy over the flat type vocabulary T*.

Instead of Chain-of-Layer flat prompting, this module uses the HAC dendrogram
structure to adaptively batch similar types and recursively ask the LLM to
organise them. The key insight: HAC already encodes a similarity-based nesting,
so we use it to feed the LLM semantically coherent batches of increasing scope.

Pipeline:
  Phase 1 — Non-LLM signals (instance subsumption + Hearst patterns)
  Phase 2 — Adaptive HAC-recursive LLM hierarchy induction
  Phase 3 — Ensemble all three signals
  Phase 4 — Post-processing (Hasse reduction, cycle breaking)
  Phase 5 — Top-down validation pass

"""

from __future__ import annotations

import re
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.cluster import hierarchy as scipy_hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances

from src.ontodisco.utils.dedup_base import (
    CanonicalItem,
    ContrieverEmbedder,
    DeduplicationResult,
    normalize_label,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HierarchyEdge:
    child_type_id: str
    parent_type_id: str
    instance_subsumption_score: Optional[float] = None
    hearst_score: Optional[float] = None
    llm_score: Optional[float] = None
    ensemble_score: float = 0.0
    is_direct: bool = True


@dataclass
class TypeHierarchy:
    edges: list[HierarchyEdge]
    children: dict[str, list[str]]   # parent_type_id → [child_type_ids]
    parents: dict[str, list[str]]    # child_type_id → [parent_type_ids]
    roots: list[str]                 # type_ids with no parents


@dataclass
class HierarchyConfig:
    # -- Adaptive batching (Phase 2) --
    # Similarity thresholds, high→low. At each level we cut the dendrogram,
    # form batches, and ask the LLM to organise each batch.
    # High threshold → tight clusters (easy, intra-domain).
    # Low threshold → loose clusters (cross-domain, structural).
    batch_thresholds: list[float] = field(
        default_factory=lambda: [0.85, 0.70, 0.55]
    )
    max_batch_size: int = 15   # cap per LLM call (accuracy degrades beyond ~20)
    min_batch_size: int = 2    # skip singletons

    # -- Instance subsumption (Phase 1, Signal 1) --
    min_instances_for_subsumption: int = 20
    subsumption_child_threshold: float = 0.80
    subsumption_synonym_threshold: float = 0.50

    # -- Hearst patterns (Phase 1, Signal 2) --
    lexical_containment_weight: float = 0.3

    # -- Ensemble (Phase 3) --
    w_subsumption: float = 0.40
    w_hearst: float = 0.30
    w_llm: float = 0.30
    edge_threshold: float = 0.5

    # -- Subsumption-informed batching (Phase 2) --
    subsumption_pull_threshold: float = 0.6

    # -- Validation (Phase 5) --
    validation_confidence_threshold: float = 0.5
    max_children_per_validation_call: int = 20


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Non-LLM Signals
# ═══════════════════════════════════════════════════════════════════════════════

def compute_instance_subsumption(
    type_vocab: DeduplicationResult,
    entity_type_map: dict[str, list[str]],
    config: HierarchyConfig,
) -> dict[tuple[str, str], float]:
    """
    Compute instance-subsumption scores for all ordered type pairs.

    entity_type_map: canonical_entity_id → list of type_ids that entity belongs to.
        Built from the triplets: for each entity, collect the type_ids of all
        its mentions.

    Returns:
        Dict of (child_type_id, parent_type_id) → subsumption score.
        Only includes pairs where score >= subsumption_child_threshold
        and the reverse score <= subsumption_synonym_threshold
        (filtering out synonyms and weak overlaps).
    """
    type_to_entities: dict[str, set[str]] = defaultdict(set)
    for entity_id, type_ids in entity_type_map.items():
        for tid in type_ids:
            type_to_entities[tid].add(entity_id)

    eligible_types = [
        tid for tid, entities in type_to_entities.items()
        if len(entities) >= config.min_instances_for_subsumption
    ]

    scores: dict[tuple[str, str], float] = {}

    for child_tid in eligible_types:
        child_ents = type_to_entities[child_tid]
        for parent_tid in eligible_types:
            if child_tid == parent_tid:
                continue

            parent_ents = type_to_entities[parent_tid]
            overlap = len(child_ents & parent_ents)

            forward = overlap / len(child_ents)    # child → parent
            reverse = overlap / len(parent_ents)   # parent → child

            if forward >= config.subsumption_child_threshold:
                if reverse >= config.subsumption_synonym_threshold:
                    logger.warning(
                        "Possible synonym: %s ↔ %s (forward=%.2f, reverse=%.2f). "
                        "Should have been merged in type canonicalization.",
                        child_tid, parent_tid, forward, reverse,
                    )
                    continue
                scores[(child_tid, parent_tid)] = forward

    logger.info(
        "Instance subsumption: %d eligible types, %d candidate edges",
        len(eligible_types), len(scores),
    )
    return scores


# ── Hearst patterns ───────────────────────────────────────────────────────────

HEARST_PATTERNS = [
    # "NP such as NP" → NP[1] IS-A NP[0]
    (re.compile(r"\b({T1})\s+such\s+as\s+({T2})\b", re.IGNORECASE), "t2_isa_t1"),
    # "NP and other NP" → NP[0] IS-A NP[1]
    (re.compile(r"\b({T1})\s+and\s+other\s+({T2})\b", re.IGNORECASE), "t1_isa_t2"),
    # "NP including NP" → NP[1] IS-A NP[0]
    (re.compile(r"\b({T1})\s+including\s+({T2})\b", re.IGNORECASE), "t2_isa_t1"),
    # "NP, especially NP" → NP[1] IS-A NP[0]
    (re.compile(r"\b({T1}),?\s+especially\s+({T2})\b", re.IGNORECASE), "t2_isa_t1"),
    # "NP or other NP" → NP[0] IS-A NP[1]
    (re.compile(r"\b({T1})\s+or\s+other\s+({T2})\b", re.IGNORECASE), "t1_isa_t2"),
]


def _build_hearst_regex(
    label_a: str, label_b: str, pattern_template: re.Pattern, direction: str
) -> tuple[re.Pattern, str, str]:
    """Compile a Hearst pattern for a specific pair of type labels."""
    escaped_a = re.escape(label_a)
    escaped_b = re.escape(label_b)
    pattern_str = pattern_template.pattern.replace("{T1}", escaped_a).replace("{T2}", escaped_b)
    compiled = re.compile(pattern_str, re.IGNORECASE)

    if direction == "t2_isa_t1":
        return compiled, label_b, label_a   # child, parent
    else:
        return compiled, label_a, label_b


def compute_hearst_scores(
    type_vocab: DeduplicationResult,
    corpus_texts: list[str],
    config: HierarchyConfig,
) -> dict[tuple[str, str], float]:
    """
    Scan corpus for Hearst patterns between canonical type labels.
    Also checks lexical containment (e.g. "documentary film" contains "film").

    Returns:
        Dict of (child_type_id, parent_type_id) → hearst score.
    """
    labels = [ct.canonical_label for ct in type_vocab.items.values()]
    label_to_id = {ct.canonical_label: ct.item_id for ct in type_vocab.items.values()}
    instance_counts = {ct.item_id: ct.mention_count for ct in type_vocab.items.values()}

    raw_counts: dict[tuple[str, str], int] = defaultdict(int)

    # -- Hearst pattern scanning --
    corpus_text = "\n".join(corpus_texts)

    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            for pattern_template, direction in HEARST_PATTERNS:
                regex, child_label, parent_label = _build_hearst_regex(
                    label_a, label_b, pattern_template, direction,
                )
                matches = regex.findall(corpus_text)
                if matches:
                    child_id = label_to_id[child_label]
                    parent_id = label_to_id[parent_label]
                    raw_counts[(child_id, parent_id)] += len(matches)

                # Also check reversed order
                regex_rev, child_label_rev, parent_label_rev = _build_hearst_regex(
                    label_b, label_a, pattern_template, direction,
                )
                matches_rev = regex_rev.findall(corpus_text)
                if matches_rev:
                    child_id = label_to_id[child_label_rev]
                    parent_id = label_to_id[parent_label_rev]
                    raw_counts[(child_id, parent_id)] += len(matches_rev)

    # -- Lexical containment --
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            tokens_a = set(label_a.lower().split())
            tokens_b = set(label_b.lower().split())

            # "documentary film" contains "film" → "documentary film" IS-A "film"
            if tokens_b < tokens_a:  # b is a strict subset of a's tokens
                child_id = label_to_id[label_a]
                parent_id = label_to_id[label_b]
                raw_counts[(child_id, parent_id)] += config.lexical_containment_weight

            elif tokens_a < tokens_b:
                child_id = label_to_id[label_b]
                parent_id = label_to_id[label_a]
                raw_counts[(child_id, parent_id)] += config.lexical_containment_weight

    # -- Normalise by log(1 + instance_count(child)) --
    scores: dict[tuple[str, str], float] = {}
    for (child_id, parent_id), count in raw_counts.items():
        child_instances = instance_counts.get(child_id, 0)
        scores[(child_id, parent_id)] = count / np.log1p(max(child_instances, 1))

    # Normalise to [0, 1] range
    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

    logger.info("Hearst patterns: %d candidate edges", len(scores))
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Adaptive HAC-Recursive LLM Hierarchy Induction
# ═══════════════════════════════════════════════════════════════════════════════

def _build_linkage_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute scipy linkage matrix from L2-normalised embeddings."""
    dist_matrix = cosine_distances(embeddings)
    condensed = squareform(dist_matrix, checks=False)
    return scipy_hierarchy.linkage(condensed, method="average")


def _cut_dendrogram(
    linkage_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Cut the dendrogram at a cosine-similarity threshold, return flat cluster labels."""
    distance_threshold = 1.0 - threshold
    return scipy_hierarchy.fcluster(
        linkage_matrix, t=distance_threshold, criterion="distance"
    )


def _split_oversized_batch(
    members: list[str],
    embeddings_map: dict[str, np.ndarray],
    max_size: int,
) -> list[list[str]]:
    """
    If a cluster exceeds max_batch_size, sub-divide it using HAC at a tighter
    threshold until all sub-batches fit.
    """
    if len(members) <= max_size:
        return [members]

    member_embeddings = np.array([embeddings_map[m] for m in members])
    linkage_matrix = _build_linkage_matrix(member_embeddings)

    # Binary search for a threshold that gives sub-clusters <= max_size
    for n_clusters in range(2, len(members)):
        sub_labels = scipy_hierarchy.fcluster(
            linkage_matrix, t=n_clusters, criterion="maxclust"
        )
        sub_groups: dict[int, list[str]] = defaultdict(list)
        for label, member in zip(sub_labels, members):
            sub_groups[int(label)].append(member)

        if all(len(g) <= max_size for g in sub_groups.values()):
            return list(sub_groups.values())

    return [[m] for m in members]


def _augment_batch_with_subsumption(
    batch_type_ids: set[str],
    subsumption_scores: dict[tuple[str, str], float],
    type_id_to_obj: dict[str, CanonicalItem],
    max_batch_size: int,
    pull_threshold: float = 0.6,
) -> set[str]:
    """
    Pull types with high subsumption scores into an existing batch.

    For each type NOT in the batch, if it has a subsumption score
    >= pull_threshold with any type already in the batch, it becomes
    a candidate. Candidates are added in descending order of their
    best subsumption score, up to max_batch_size.
    """
    if not subsumption_scores:
        return batch_type_ids

    candidates: dict[str, float] = {}

    for (child_id, parent_id), score in subsumption_scores.items():
        if score < pull_threshold:
            continue

        if child_id in batch_type_ids and parent_id not in batch_type_ids:
            if parent_id in type_id_to_obj:
                candidates[parent_id] = max(candidates.get(parent_id, 0.0), score)

        elif parent_id in batch_type_ids and child_id not in batch_type_ids:
            if child_id in type_id_to_obj:
                candidates[child_id] = max(candidates.get(child_id, 0.0), score)

    if not candidates:
        return batch_type_ids

    augmented = set(batch_type_ids)
    for tid, _ in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        if len(augmented) >= max_batch_size:
            break
        augmented.add(tid)

    if len(augmented) > len(batch_type_ids):
        logger.debug(
            "Subsumption augmentation: %d → %d types in batch",
            len(batch_type_ids), len(augmented),
        )

    return augmented


def _format_batch_for_llm(
    batch_types: list[CanonicalItem],
    all_labels: list[str],
    existing_edges: list[tuple[str, str]],
) -> tuple[str, str, str]:
    """Format the batch types, existing structure, and full vocabulary for the prompt."""
    batch_lines = []
    for ct in batch_types:
        examples = ", ".join(ct.surface_forms[:3])
        batch_lines.append(
            f"- {ct.canonical_label} ({ct.mention_count} instances, "
            f"surface forms: {examples})"
        )
    batch_str = "\n".join(batch_lines)

    if existing_edges:
        struct_lines = [
            f"- \"{child}\" IS-A \"{parent}\""
            for child, parent in existing_edges
        ]
        struct_block = (
            "ALREADY ESTABLISHED HIERARCHY (from previous analysis):\n"
            + "\n".join(struct_lines)
        )
    else:
        struct_block = ""

    vocab_str = ", ".join(f'"{l}"' for l in all_labels)

    return batch_str, struct_block, vocab_str


def _parse_batch_llm_response(
    response: dict | list | str,
    label_to_type_id: dict[str, str],
) -> list[tuple[str, str, float]]:
    """
    Parse LLM response from hierarchy_batch prompt.

    Returns list of (child_type_id, parent_type_id, confidence).
    """
    edges = []

    if isinstance(response, str):
        logger.warning("LLM returned unparseable string for hierarchy batch")
        return edges

    if isinstance(response, list):
        response = {"edges": response}

    for edge in response.get("edges", []):
        child_label = normalize_label(edge.get("child", ""))
        parent_label = normalize_label(edge.get("parent", ""))
        confidence = float(edge.get("confidence", 0.5))

        child_id = label_to_type_id.get(child_label)
        parent_id = label_to_type_id.get(parent_label)

        if child_id and parent_id and child_id != parent_id:
            edges.append((child_id, parent_id, confidence))
        elif not child_id:
            logger.warning("LLM referenced unknown child type: %r", child_label)
        elif not parent_id:
            logger.warning("LLM referenced unknown parent type: %r", parent_label)

    return edges


def induce_hierarchy_hac_recursive(
    type_vocab: DeduplicationResult,
    config: HierarchyConfig,
    llm_client,
    embedder: ContrieverEmbedder,
    prompt_template: str,
    subsumption_scores: dict[tuple[str, str], float] | None = None,
) -> dict[tuple[str, str], float]:
    """
    Phase 2: Adaptive HAC-recursive hierarchy induction.

    Algorithm:
      1. Embed all canonical type labels.
      2. Build the full HAC linkage matrix (dendrogram).
      3. For each threshold level (high → low):
         a. Cut dendrogram at this threshold → flat clusters.
         b. For each cluster with 2+ members:
            - Filter out types already resolved at tighter thresholds.
            - If remaining batch > max_batch_size, subdivide via HAC.
            - Send batch to LLM: "organise these types hierarchically".
            - Collect proposed edges with confidence scores.
      4. At each successive level, the LLM sees larger, more diverse batches
         and can connect subtrees discovered at tighter levels.

    Returns:
        Dict of (child_type_id, parent_type_id) → LLM confidence score.
    """
    types_list = list(type_vocab.items.values())
    all_labels = [ct.canonical_label for ct in types_list]
    label_to_type_id = {
        normalize_label(ct.canonical_label): ct.item_id
        for ct in types_list
    }
    # Also map surface forms
    for ct in types_list:
        for sf in ct.surface_forms:
            label_to_type_id.setdefault(normalize_label(sf), ct.item_id)

    type_id_to_obj = {ct.item_id: ct for ct in types_list}

    if len(types_list) < 2:
        return {}

    # ── Step 1: Embed ─────────────────────────────────────────────────────────
    embeddings = embedder.embed(all_labels)
    embeddings_map = {label: embeddings[i] for i, label in enumerate(all_labels)}

    # ── Step 2: Build dendrogram ──────────────────────────────────────────────
    linkage_matrix = _build_linkage_matrix(embeddings)

    # ── Step 3: Multi-resolution processing ───────────────────────────────────
    llm_edges: dict[tuple[str, str], float] = {}
    has_parent: set[str] = set()  # type_ids that already got a parent
    established_edges: list[tuple[str, str]] = []  # (child_label, parent_label)

    for threshold in sorted(config.batch_thresholds, reverse=True):
        logger.info("Processing threshold level %.2f", threshold)

        cluster_labels = _cut_dendrogram(linkage_matrix, threshold)

        # Group types by cluster
        clusters: dict[int, list[CanonicalItem]] = defaultdict(list)
        for ct, cl in zip(types_list, cluster_labels):
            clusters[int(cl)].append(ct)

        for cluster_id, members in clusters.items():
            if len(members) < config.min_batch_size:
                continue

            # Filter to types not yet fully resolved
            unresolved = [
                ct for ct in members
                if ct.item_id not in has_parent
            ]
            # But include resolved types as context (they might be parents)
            resolved_in_batch = [
                ct for ct in members
                if ct.item_id in has_parent
            ]

            if len(unresolved) < config.min_batch_size:
                continue

            # Augment batch with subsumption-related types
            if subsumption_scores:
                unresolved_ids = {ct.item_id for ct in unresolved}
                augmented_ids = _augment_batch_with_subsumption(
                    unresolved_ids,
                    subsumption_scores,
                    type_id_to_obj,
                    config.max_batch_size,
                    config.subsumption_pull_threshold,
                )
                for tid in augmented_ids - unresolved_ids:
                    unresolved.append(type_id_to_obj[tid])

            # Subdivide oversized batches
            member_labels = [ct.canonical_label for ct in unresolved]
            sub_batches = _split_oversized_batch(
                member_labels, embeddings_map, config.max_batch_size,
            )

            for sub_batch_labels in sub_batches:
                if len(sub_batch_labels) < config.min_batch_size:
                    continue

                batch_types = [
                    type_id_to_obj[label_to_type_id[normalize_label(l)]]
                    for l in sub_batch_labels
                ]

                # Include resolved types from this cluster as extra context
                for ct in resolved_in_batch:
                    if ct not in batch_types and len(batch_types) < config.max_batch_size:
                        batch_types.append(ct)

                # Collect existing edges relevant to this batch
                batch_type_ids = {ct.item_id for ct in batch_types}
                relevant_existing = [
                    (c, p) for c, p in established_edges
                    if label_to_type_id.get(normalize_label(c)) in batch_type_ids
                    or label_to_type_id.get(normalize_label(p)) in batch_type_ids
                ]

                batch_str, struct_block, vocab_str = _format_batch_for_llm(
                    batch_types, all_labels, relevant_existing,
                )

                prompt = prompt_template.format(
                    batch_types=batch_str,
                    existing_structure_block=struct_block,
                    all_type_labels=vocab_str,
                )

                try:
                    response = llm_client.get_completion(
                        system_prompt="You are an ontology engineer.",
                        user_prompt=prompt,
                        transform_to_json=True,
                    )
                except Exception:
                    logger.exception(
                        "LLM call failed for batch at threshold %.2f, cluster %d",
                        threshold, cluster_id,
                    )
                    continue

                new_edges = _parse_batch_llm_response(response, label_to_type_id)

                for child_id, parent_id, confidence in new_edges:
                    key = (child_id, parent_id)
                    if key not in llm_edges or llm_edges[key] < confidence:
                        llm_edges[key] = confidence

                    has_parent.add(child_id)

                    child_label = type_id_to_obj[child_id].canonical_label
                    parent_label = type_id_to_obj[parent_id].canonical_label
                    established_edges.append((child_label, parent_label))

        logger.info(
            "After threshold %.2f: %d LLM edges, %d types with parents",
            threshold, len(llm_edges), len(has_parent),
        )

    return llm_edges


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_signals(
    subsumption_scores: dict[tuple[str, str], float],
    hearst_scores: dict[tuple[str, str], float],
    llm_scores: dict[tuple[str, str], float],
    config: HierarchyConfig,
) -> list[HierarchyEdge]:
    """
    Combine three signals into final scored edges.

    For each candidate (child, parent) pair that appears in ANY signal,
    compute:
        ensemble = w_sub * sub_score + w_hearst * hearst_score + w_llm * llm_score

    Accept edge if ensemble_score >= edge_threshold.
    """
    all_pairs = set(subsumption_scores) | set(hearst_scores) | set(llm_scores)

    edges = []
    for child_id, parent_id in all_pairs:
        sub_score = subsumption_scores.get((child_id, parent_id))
        h_score = hearst_scores.get((child_id, parent_id))
        llm_score = llm_scores.get((child_id, parent_id))

        ensemble = (
            config.w_subsumption * (sub_score or 0.0)
            + config.w_hearst * (h_score or 0.0)
            + config.w_llm * (llm_score or 0.0)
        )

        if ensemble >= config.edge_threshold:
            edges.append(HierarchyEdge(
                child_type_id=child_id,
                parent_type_id=parent_id,
                instance_subsumption_score=sub_score,
                hearst_score=h_score,
                llm_score=llm_score,
                ensemble_score=ensemble,
                is_direct=True,
            ))

    edges.sort(key=lambda e: e.ensemble_score, reverse=True)
    logger.info(
        "Ensemble: %d candidate pairs → %d edges above threshold %.2f",
        len(all_pairs), len(edges), config.edge_threshold,
    )
    return edges


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def _build_adjacency(edges: list[HierarchyEdge]) -> dict[str, set[str]]:
    """Build child → set of parent_ids adjacency from edges."""
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.is_direct:
            adj[e.child_type_id].add(e.parent_type_id)
    return adj


def _ancestors(node: str, parent_map: dict[str, set[str]]) -> set[str]:
    """Compute all ancestors of a node via BFS."""
    visited = set()
    frontier = list(parent_map.get(node, set()))
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(parent_map.get(current, set()))
    return visited


def hasse_reduce(edges: list[HierarchyEdge]) -> list[HierarchyEdge]:
    """
    Remove transitive shortcuts (Hasse diagram reduction).

    If A→B and B→C both exist, and A→C also exists, mark A→C as is_direct=False.
    The redundant edge is kept for analysis but won't appear in OWL output.
    """
    parent_map = _build_adjacency(edges)
    edge_lookup = {(e.child_type_id, e.parent_type_id): e for e in edges}

    for edge in edges:
        if not edge.is_direct:
            continue

        child, parent = edge.child_type_id, edge.parent_type_id

        # Check if parent is reachable from child through other direct edges
        # (i.e., there's an indirect path child → ... → parent)
        other_parents = parent_map.get(child, set()) - {parent}
        for other_parent in other_parents:
            if parent in _ancestors(other_parent, parent_map):
                edge.is_direct = False
                logger.debug(
                    "Hasse reduction: removed transitive edge %s → %s "
                    "(path through %s)",
                    child, parent, other_parent,
                )
                break

    direct_count = sum(1 for e in edges if e.is_direct)
    logger.info(
        "Hasse reduction: %d edges → %d direct + %d transitive",
        len(edges), direct_count, len(edges) - direct_count,
    )
    return edges


def break_cycles(edges: list[HierarchyEdge]) -> list[HierarchyEdge]:
    """
    Detect and break cycles using Kahn's topological sort.

    When a cycle is found, remove the edge with the lowest ensemble_score
    in the cycle. Repeat until the graph is a DAG.
    """
    direct_edges = [e for e in edges if e.is_direct]

    while True:
        # Build adjacency for topological sort
        children_of: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)
        all_nodes: set[str] = set()

        for e in direct_edges:
            children_of[e.parent_type_id].append(e.child_type_id)
            in_degree[e.child_type_id] = in_degree.get(e.child_type_id, 0) + 1
            all_nodes.add(e.child_type_id)
            all_nodes.add(e.parent_type_id)

        # Kahn's algorithm
        queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
        visited_count = 0
        visited_set: set[str] = set()

        while queue:
            node = queue.pop()
            visited_set.add(node)
            visited_count += 1
            for child in children_of.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if visited_count == len(all_nodes):
            break  # no cycles

        # Find a cycle: nodes not visited are in cycles
        cycle_nodes = all_nodes - visited_set
        cycle_edges = [
            e for e in direct_edges
            if e.child_type_id in cycle_nodes and e.parent_type_id in cycle_nodes
        ]

        if not cycle_edges:
            break

        weakest = min(cycle_edges, key=lambda e: e.ensemble_score)
        logger.warning(
            "Breaking cycle: removing edge %s → %s (score=%.3f)",
            weakest.child_type_id, weakest.parent_type_id, weakest.ensemble_score,
        )
        weakest.is_direct = False
        direct_edges = [e for e in direct_edges if e.is_direct]

    non_direct = [e for e in edges if not e.is_direct]
    return direct_edges + non_direct


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Top-Down Validation
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_validation_response(
    response: dict | list | str,
    label_to_type_id: dict[str, str],
) -> list[dict]:
    """Parse LLM validation response into structured verdicts."""
    if isinstance(response, str):
        logger.warning("Validation LLM returned unparseable string")
        return []

    if isinstance(response, list):
        response = {"validations": response}

    results = []
    for v in response.get("validations", []):
        child_label = normalize_label(v.get("child", ""))
        child_id = label_to_type_id.get(child_label)
        if not child_id:
            continue

        verdict = v.get("verdict", "valid").lower()
        confidence = float(v.get("confidence", 0.5))

        result = {
            "child_id": child_id,
            "verdict": verdict,
            "confidence": confidence,
        }

        if verdict == "invalid":
            suggested = normalize_label(v.get("suggested_parent", "none"))
            result["suggested_parent_id"] = label_to_type_id.get(suggested)
        elif verdict == "too_deep":
            move_under = normalize_label(v.get("move_under", ""))
            result["move_under_id"] = label_to_type_id.get(move_under)

        results.append(result)

    return results


def validate_top_down(
    edges: list[HierarchyEdge],
    type_vocab: DeduplicationResult,
    config: HierarchyConfig,
    llm_client,
    prompt_template: str,
) -> list[HierarchyEdge]:
    """
    Phase 5: Walk the hierarchy top-down. For each parent, present its
    children to the LLM and ask it to confirm, reject, or rearrange.

    This catches errors from Phase 2 where a bad early merge propagated
    up the tree: seeing children in context of their parent + siblings
    gives the LLM a correction opportunity.
    """
    direct_edges = [e for e in edges if e.is_direct]
    type_id_to_obj = {ct.item_id: ct for ct in type_vocab.items.values()}

    all_labels = [ct.canonical_label for ct in type_vocab.items.values()]
    label_to_type_id = {
        normalize_label(ct.canonical_label): ct.item_id
        for ct in type_vocab.items.values()
    }
    for ct in type_vocab.items.values():
        for sf in ct.surface_forms:
            label_to_type_id.setdefault(normalize_label(sf), ct.item_id)

    # Build parent → children map from direct edges
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    for e in direct_edges:
        parent_to_children[e.parent_type_id].append(e.child_type_id)

    # Find roots (parents that are not children of anything)
    all_children = {e.child_type_id for e in direct_edges}
    all_parents = {e.parent_type_id for e in direct_edges}
    roots = all_parents - all_children

    # BFS top-down from roots
    edges_to_remove: set[tuple[str, str]] = set()
    edges_to_add: list[tuple[str, str, float]] = []

    queue = list(roots)
    visited: set[str] = set()

    while queue:
        parent_id = queue.pop(0)
        if parent_id in visited:
            continue
        visited.add(parent_id)

        children_ids = parent_to_children.get(parent_id, [])
        if not children_ids:
            continue

        # Queue children for next level
        queue.extend(children_ids)

        parent_obj = type_id_to_obj.get(parent_id)
        if not parent_obj:
            continue

        # Batch children for validation (respect max per call)
        for i in range(0, len(children_ids), config.max_children_per_validation_call):
            batch_children_ids = children_ids[i:i + config.max_children_per_validation_call]
            children_list_str = "\n".join(
                f"- {type_id_to_obj[cid].canonical_label} "
                f"({type_id_to_obj[cid].mention_count} instances)"
                for cid in batch_children_ids
                if cid in type_id_to_obj
            )

            vocab_str = ", ".join(f'"{l}"' for l in all_labels)

            prompt = prompt_template.format(
                parent_label=parent_obj.canonical_label,
                children_list=children_list_str,
                all_type_labels=vocab_str,
            )

            try:
                response = llm_client.get_completion(
                    system_prompt="You are an ontology engineer.",
                    user_prompt=prompt,
                    transform_to_json=True,
                )
            except Exception:
                logger.exception(
                    "Validation LLM call failed for parent %s",
                    parent_obj.canonical_label,
                )
                continue

            verdicts = _parse_validation_response(response, label_to_type_id)

            for v in verdicts:
                child_id = v["child_id"]
                verdict = v["verdict"]
                confidence = v["confidence"]

                if verdict == "invalid" and confidence >= config.validation_confidence_threshold:
                    edges_to_remove.add((child_id, parent_id))
                    suggested = v.get("suggested_parent_id")
                    if suggested and suggested != child_id:
                        edges_to_add.append((child_id, suggested, confidence))
                    logger.info(
                        "Validation: REMOVING %s → %s (invalid, conf=%.2f)",
                        type_id_to_obj.get(child_id, child_id),
                        parent_obj.canonical_label, confidence,
                    )

                elif verdict == "too_deep" and confidence >= config.validation_confidence_threshold:
                    move_under = v.get("move_under_id")
                    if move_under and move_under != child_id:
                        edges_to_remove.add((child_id, parent_id))
                        edges_to_add.append((child_id, move_under, confidence))
                        logger.info(
                            "Validation: MOVING %s from under %s to under %s",
                            type_id_to_obj.get(child_id, child_id),
                            parent_obj.canonical_label,
                            type_id_to_obj.get(move_under, move_under),
                        )

    # Apply removals
    result_edges = [
        e for e in edges
        if (e.child_type_id, e.parent_type_id) not in edges_to_remove
        or not e.is_direct
    ]

    # Apply additions
    existing = {(e.child_type_id, e.parent_type_id) for e in result_edges}
    for child_id, parent_id, confidence in edges_to_add:
        if (child_id, parent_id) not in existing:
            result_edges.append(HierarchyEdge(
                child_type_id=child_id,
                parent_type_id=parent_id,
                llm_score=confidence,
                ensemble_score=confidence,
                is_direct=True,
            ))
            existing.add((child_id, parent_id))

    logger.info(
        "Validation: removed %d edges, added %d edges",
        len(edges_to_remove), len(edges_to_add),
    )

    return result_edges


# ═══════════════════════════════════════════════════════════════════════════════
#  Build final TypeHierarchy
# ═══════════════════════════════════════════════════════════════════════════════

def _build_type_hierarchy(
    edges: list[HierarchyEdge],
    all_type_ids: list[str],
) -> TypeHierarchy:
    """Assemble the TypeHierarchy dataclass from a list of edges."""
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)

    for e in edges:
        if e.is_direct:
            children[e.parent_type_id].append(e.child_type_id)
            parents[e.child_type_id].append(e.parent_type_id)

    child_set = set(parents.keys())
    roots = [tid for tid in all_type_ids if tid not in child_set]

    return TypeHierarchy(
        edges=edges,
        children=dict(children),
        parents=dict(parents),
        roots=roots,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def induce_hierarchy(
    type_vocab: DeduplicationResult,
    corpus_texts: list[str],
    entity_type_map: dict[str, list[str]],
    config: HierarchyConfig,
    llm_client,
    embedder: ContrieverEmbedder,
    *,
    hierarchy_batch_prompt: str,
    hierarchy_validate_prompt: str,
) -> TypeHierarchy:
    """
    Full hierarchy induction pipeline.

    Args:
        type_vocab:         Flat type vocabulary from type canonicalization.
        corpus_texts:       Raw corpus text strings (for Hearst pattern scanning).
        entity_type_map:    canonical_entity_id → [type_ids] mapping
                            (for instance subsumption).
        config:             Hierarchy induction configuration.
        llm_client:         LLM client with get_completion() method.
        embedder:           Contriever embedder instance.
        hierarchy_batch_prompt:    Prompt template for Phase 2 (batch hierarchy).
        hierarchy_validate_prompt: Prompt template for Phase 5 (validation).

    Returns:
        TypeHierarchy — the directed acyclic forest of subClassOf edges.
    """
    all_type_ids = list(type_vocab.items.keys())
    logger.info("Starting hierarchy induction for %d types", len(all_type_ids))

    if len(all_type_ids) < 2:
        logger.info("Fewer than 2 types; returning empty hierarchy")
        return _build_type_hierarchy([], all_type_ids)

    # ── Phase 1: Non-LLM signals ─────────────────────────────────────────────
    logger.info("Phase 1: Computing non-LLM signals")

    subsumption_scores = compute_instance_subsumption(
        type_vocab, entity_type_map, config,
    )
    hearst_scores = compute_hearst_scores(
        type_vocab, corpus_texts, config,
    )

    # ── Phase 2: Adaptive HAC-recursive LLM hierarchy ────────────────────────
    logger.info("Phase 2: Adaptive HAC-recursive LLM hierarchy induction")

    llm_scores = induce_hierarchy_hac_recursive(
        type_vocab, config, llm_client, embedder,
        prompt_template=hierarchy_batch_prompt,
        subsumption_scores=subsumption_scores,
    )

    # ── Phase 3: Ensemble ─────────────────────────────────────────────────────
    logger.info("Phase 3: Ensembling signals")

    edges = ensemble_signals(subsumption_scores, hearst_scores, llm_scores, config)

    # ── Phase 4: Post-processing ──────────────────────────────────────────────
    logger.info("Phase 4: Hasse reduction + cycle breaking")

    edges = hasse_reduce(edges)
    edges = break_cycles(edges)

    # ── Phase 5: Top-down validation ──────────────────────────────────────────
    logger.info("Phase 5: Top-down validation")

    edges = validate_top_down(
        edges, type_vocab, config, llm_client,
        prompt_template=hierarchy_validate_prompt,
    )

    # Re-run post-processing after validation (may have introduced new edges)
    edges = hasse_reduce(edges)
    edges = break_cycles(edges)

    # ── Build final hierarchy ─────────────────────────────────────────────────
    hierarchy = _build_type_hierarchy(edges, all_type_ids)

    logger.info(
        "Hierarchy induction complete: %d direct edges, %d roots, %d types with parents",
        sum(1 for e in hierarchy.edges if e.is_direct),
        len(hierarchy.roots),
        len(hierarchy.parents),
    )

    return hierarchy
