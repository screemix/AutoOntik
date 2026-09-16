"""
Shared Deduplication Infrastructure
====================================

Base module for surface-form deduplication. Provides:
  - ContrieverEmbedder — dense embedding via Meta's Contriever
  - normalize_label    — Unicode NFKC + lowercase + whitespace collapse
  - cluster_hdbscan    — HDBSCAN-based candidate clustering
  - verify_clusters_with_llm — LLM merge/split verification loop
  - deduplicate()      — the shared 4-step pipeline (embed → HDBSCAN → LLM → build)

Both type_dedup.py and entity_dedup.py are thin wrappers around this module.
"""

from __future__ import annotations

import unicodedata
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import dotenv
dotenv.load_dotenv()
import os

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalItem:
    """One canonical item after deduplication (base for types and entities)."""
    item_id: str
    canonical_label: str
    count_per_normalized: int = 0
    surface_forms: list[str] = field(default_factory=list)
    surface_form_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class DeduplicationResult:
    """Full output of a deduplication step."""
    items: dict[str, CanonicalItem]       # item_id → CanonicalItem
    surface_to_id: dict[str, str]         # raw surface form → item_id
    num_raw: int = 0
    num_canonical: int = 0
    reduction_pct: float = 0.0
    normalized_to_raws: dict[str, set[str]] = field(default_factory=dict)
    surface_form_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


# ═══════════════════════════════════════════════════════════════════════════════
#  Contriever Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class ContrieverEmbedder:
    """
    Embeds text strings using Meta's Contriever model (facebook/contriever).

    Produces 768-dimensional L2-normalised embeddings via mean pooling
    over the last hidden states (masked for padding).
    """

    def __init__(self, model_name: str = "facebook/contriever", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Prefer safetensors, but don't make the whole pipeline depend on HF's
        # safetensors-convert Space being up: when a cached revision ships only
        # pytorch_model.bin, use_safetensors=True makes transformers POST to
        # that Space and parse its response, which fails with an opaque
        # JSONDecodeError whenever the service is down or rate-limiting --
        # aborting a multi-hour run before the first LLM call. Fall back to the
        # already-cached .bin weights instead.
        load_kwargs = dict(trust_remote_code=True, token=os.getenv("HF_KEY"))
        try:
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True, **load_kwargs)
        except Exception as exc:
            logger.warning(
                "Loading %s with use_safetensors=True failed (%s: %s); retrying without it",
                model_name, type(exc).__name__, exc,
            )
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.api_key = os.getenv("HF_KEY")

    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Text Normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_label(label: str) -> str:
    """
    Normalise a raw surface form: Unicode NFKC, lowercase, whitespace collapse.
    """
    text = unicodedata.normalize("NFKC", label)
    text = text.lower()
    text = " ".join(text.split())
    return text



# ═══════════════════════════════════════════════════════════════════════════════
#  HDBSCAN Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_hdbscan(
    embeddings: np.ndarray,
    threshold: float = 0.8,
    *,
    min_cluster_size: int = 2,
    min_samples: int = 1,
) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN, parameterised to approximate the
    "merge if cosine similarity >= threshold" semantics, while additionally letting
    variable-density substructure *within* that threshold band form separate
    clusters instead of one fixed global cut.

    Args:
        embeddings:       (N, D) array of L2-normalised embeddings.
        threshold:        Cosine *similarity* threshold. Converted internally to
                           `cluster_selection_epsilon = 1 - threshold` (a cosine
                           *distance*): pairs farther apart than this can never
                           land in the same cluster, but within it HDBSCAN is
                           free to find whatever locally-stable groupings exist.
        min_cluster_size: HDBSCAN's minimum cluster size (clamped to >= 2 --
                           HDBSCAN itself requires this).
        min_samples:      HDBSCAN's core-distance neighbour count. Left at 1
                           (the most permissive setting available) to stay
                           close to a flat pairwise-threshold merge rather than
                           imposing a density requirement no HAC/FAISS call
                           site here ever needed.

    `allow_single_cluster=True` is required: HDBSCAN's default rejects a
    pool that has no viable sub-split at the very top of its internal
    hierarchy and marks EVERY point as noise instead of calling it one
    cluster -- verified empirically (not a hypothetical edge case) to
    otherwise silently noise-out an entire well-formed group of 5+
    near-duplicate labels that obviously belong together.

    Points HDBSCAN leaves unclustered (label -1, "noise") become their own
    singleton cluster in the returned labels, matching every other cluster_fn
    in this pipeline: unclustered means "no confident merge partner", not
    "dropped".

    Returns:
        Array of integer cluster labels, shape (N,).
    """
    n = len(embeddings)
    if n <= 1:
        return np.zeros(n, dtype=int)

    distance_matrix = cosine_distances(embeddings).astype(np.float64)
    np.fill_diagonal(distance_matrix, 0.0)
    epsilon = max(0.0, 1.0 - threshold)

    clusterer = HDBSCAN(
        metric="precomputed",
        min_cluster_size=max(2, min_cluster_size),
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        allow_single_cluster=True,
    )
    raw_labels = clusterer.fit_predict(distance_matrix)

    labels = raw_labels.copy()
    next_id = int(labels.max()) + 1 if (labels >= 0).any() else 0
    for i in range(n):
        if labels[i] == -1:
            labels[i] = next_id
            next_id += 1
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM Verification
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_cluster_response(
    groups: Optional[list],
    exc: Optional[BaseException],
    members: list[str],
    cluster_id,
) -> list[tuple[str, list[str]]]:
    """Turn one raw LLM response (or exception) for one cluster into
    [(canonical_label, [matched_members]), ...], with every fallback
    verify_clusters_with_llm has always had: a failed call or unparseable
    response keeps the HDBSCAN grouping as one group; a label the LLM
    returned but that doesn't match any input member is kept as-is; a
    member the LLM silently dropped from every group becomes its own
    singleton. Shared by the normal per-cluster path and the oversized-
    cluster stitching call below, so both get identical robustness.

    Guarantee: never returns more than len(members) groups, AND never returns
    a given input member in more than one group -- so the total member count
    summed across every returned group is always <= len(members) too, not
    just the group count. This is what lets deduplicate_with_rounds()'s
    convergence check ("a round produced no further reduction") mean
    anything -- HDBSCAN partitions each round's pool labels across clusters,
    so if every cluster's own output is bounded by its own input size, the
    round's total output is bounded by the round's total input, and pool
    size can only shrink or hold steady, never grow. .)
    """
    # UNVERIFIED CLUSTERS MUST NOT MERGE. HDBSCAN is candidate GENERATION, not a
    # merge decision -- it groups by embedding proximity, which for e.g. country
    # names or person names is high across genuinely distinct entities. The LLM
    # call is the only thing that ever decides "these are the same". So when that
    # call fails or its response can't be parsed, the honest outcome is "unknown",
    # and this module's stated policy (see cluster_entity_names.txt: "when in
    # doubt, keep entities SEPARATE") makes the safe direction SPLIT, not merge.

    if exc is not None:
        logger.exception(
            "LLM verification failed for cluster %s (%d members), keeping them SEPARATE "
            "(unverified clusters are never merged)", cluster_id, len(members), exc_info=exc,
        )
        return [(m, [m]) for m in members]

    if not groups:
        if groups is not None:
            logger.warning(
                "Could not parse LLM response for cluster %s (%d members), keeping them SEPARATE",
                cluster_id, len(members),
            )
        return [(m, [m]) for m in members]

    # Validate every group up front -- a single malformed group (missing
    # canonical_label/members) makes the whole response's structure
    # suspect, and partially trusting it risks the fallback below
    # duplicating members a valid sibling group already claimed. Falling
    # back once for the whole response, rather than once per malformed
    # group, is what keeps the "at most len(members) groups" guarantee
    # unconditional instead of merely typical.
    for group in groups:
        if not group.get("canonical_label") or not group.get("members"):
            logger.warning(
                "Malformed group in LLM response for cluster %s (%d members), keeping them SEPARATE",
                cluster_id, len(members),
            )
            return [(m, [m]) for m in members]

    result: list[tuple[str, list[str]]] = []
    claimed: set[str] = set()
    members_lower = {m.lower(): m for m in members}

    for group in groups:
        canonical = group["canonical_label"]
        matched_members = []
        for gm in group["members"]:
            gm_lower = gm.lower().strip()
            if gm_lower in claimed:
                logger.warning(
                    "LLM assigned label %r to more than one group for cluster %s; "
                    "keeping its first assignment, dropping the duplicate",
                    gm, cluster_id,
                )
                continue
            claimed.add(gm_lower)
            if gm_lower in members_lower:
                matched_members.append(members_lower[gm_lower])
            else:
                logger.warning(
                    "LLM returned label %r not matching any cluster %s member %s; "
                    "keeping LLM form as-is",
                    gm, cluster_id, members,
                )
                matched_members.append(gm)
        if matched_members:
            result.append((canonical.lower().strip(), matched_members))

    for member in members:
        if member.lower().strip() not in claimed:
            logger.warning("LLM dropped label '%s' from cluster %s, adding as singleton", member, cluster_id)
            result.append((member, [member]))

    if len(result) > len(members):
        # Unconditional safety net: whatever pathological shape the LLM's
        # groups took (e.g. hallucinated extra groups with no real
        # members), never let this cluster's own output exceed its own
        # input size -- see the guarantee in the docstring above.
        # Same direction rule as the failure paths above, and if anything more
        # clear-cut: a response with MORE groups than members was splitting
        # aggressively, so collapsing it into one merged item inverts the only
        # signal the LLM actually gave. Keep them separate.
        logger.warning(
            "LLM response for cluster %s produced %d groups from %d input members "
            "(more groups than members); keeping them SEPARATE",
            cluster_id, len(result), len(members),
        )
        return [(m, [m]) for m in members]

    return result


def verify_clusters_with_llm(
    clusters: dict[int, list[str]],
    llm_verifier: object = None,
    *,
    verbose: bool = True,
    token_log_every: int = 10,
    surface_form_type: str = 'entity_type',
    member_context: dict[str, str] | None = None,
    max_workers: int = 8,
    max_cluster_size: int = 40,
) -> list[tuple[str, list[str]]]:
    """
    Send each multi-member HDBSCAN cluster to the LLM for merge/split verification.

    The llm_verifier must implement verify_cluster_with_llm(members: list[str])
    returning a list of dicts with "canonical_label" and "members" keys.

    member_context: optional normalised label -> evidence string, forwarded to
    llm_verifier.verify_cluster_with_llm(..., member_context=...) so the LLM sees
    e.g. relational-context evidence alongside each candidate. Only members
    present in this cluster's list are included in the sub-dict passed down;
    labels with no evidence are simply omitted (not padded with "").

    max_workers: multi-member clusters are independent of each other (no
    shared mutable state to serialize on, unlike hierarchy_induction.py's
    node placement), so their LLM calls run concurrently on a
    ThreadPoolExecutor -- same pattern as induce_hierarchy()'s band-2+
    placement. Singleton clusters never reach the pool (no LLM call needed).

    max_cluster_size: size-capping + stitching for oversized post-merge
    clusters cap how large a single cluster can grow -- a cluster well
    past this size produces an oversized prompt that's slow and whose JSON
    response the LLM is prone to truncating, silently dropping members into
    accidental singletons rather than making a real merge/split call on
    them (see the "LLM dropped label" warning above). A cluster over
    max_cluster_size is split into contiguous alphabetically-sorted
    sub-batches of at most that size, each resolved independently and
    concurrently like any other cluster, then reconciled with ONE extra
    "stitching" call per oversized cluster (skipped if the sub-batches
    already collapsed to a single group) that treats each sub-batch's own
    canonical_label as a candidate and asks whether any should be reunified
    -- they were only split apart by the size cap, not because anything
    found them unrelated. The split itself doesn't need to be
    embedding-smart for this to be correct: stitching is what recovers any
    merge the split's arbitrary boundary would otherwise have missed. If the
    stitch pool itself would exceed max_cluster_size (a cluster oversized
    enough that its sub-batches still survive as more than max_cluster_size
    labels -- meaning most of the original cluster wasn't actually
    duplicated to begin with, not that a boundary was unlucky), stitching is
    skipped entirely rather than forced through an equally oversized call:
    the sub-batch groups are kept as-is, separate. A cluster that size is a
    sign HDBSCAN over-clustered upstream (see CLAUDE.md L3), not something
    one more LLM call can meaningfully reconcile.

    Returns:
        List of (canonical_label, [member_labels]) tuples.
    """
    cluster_items = list(clusters.items())

    # ── Build work items: one per cluster, or several per oversized cluster ──
    # (origin_idx, work_members) -- origin_idx groups sub-batches back to the
    # cluster they were split from; phase 2 walks by origin_idx, not by work
    # item, to keep output order tied to the ORIGINAL cluster order.
    work_items: list[tuple[int, list[str]]] = []
    origin_work_indices: dict[int, list[int]] = defaultdict(list)
    # A trailing sub-batch can end up with exactly one member (e.g. 5 members
    # split at max_cluster_size=2 -> chunks of 2, 2, 1) -- that's a singleton,
    # not a candidate cluster, so it's kept out of work_items entirely (no LLM
    # call) same as any other singleton, and folded back in during phase 2.
    origin_singletons: dict[int, list[str]] = defaultdict(list)

    for origin_idx, (cluster_id, members) in enumerate(cluster_items):
        if len(members) <= 1:
            continue
        if len(members) <= max_cluster_size:
            origin_work_indices[origin_idx].append(len(work_items))
            work_items.append((origin_idx, members))
        else:
            sub_batches = sorted(members)
            for start in range(0, len(sub_batches), max_cluster_size):
                chunk = sub_batches[start:start + max_cluster_size]
                if len(chunk) == 1:
                    origin_singletons[origin_idx].append(chunk[0])
                else:
                    origin_work_indices[origin_idx].append(len(work_items))
                    work_items.append((origin_idx, chunk))

    # ── Phase 1: fire every work item's LLM call concurrently ────────────────
    # Results are collected keyed by work-item index; phase 2 re-groups them
    # by origin_idx so output order never depends on thread completion timing
    # (downstream code assigns item IDs via enumerate(verified_groups)).
    raw_results: dict[int, tuple[Optional[list], Optional[BaseException]]] = {}

    if work_items:
        pbar = tqdm(
            total=len(work_items),
            desc="LLM cluster verify",
            unit="cluster",
            disable=not verbose,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for work_idx, (origin_idx, members) in enumerate(work_items):
                call_kwargs = {"surface_form_type": surface_form_type}
                if member_context is not None:
                    call_kwargs["member_context"] = {
                        m: member_context[m] for m in members if m in member_context
                    }
                futures[executor.submit(llm_verifier.verify_cluster_with_llm, members, **call_kwargs)] = work_idx

            for completed, future in enumerate(as_completed(futures), start=1):
                work_idx = futures[future]
                try:
                    raw_results[work_idx] = (future.result(), None)
                except Exception as exc:
                    raw_results[work_idx] = (None, exc)
                pbar.update(1)

                if (
                    verbose
                    and token_log_every > 0
                    and completed % token_log_every == 0
                    and llm_verifier is not None
                    and hasattr(llm_verifier, "calculate_used_tokens")
                ):
                    prompt_tokens, completion_tokens = llm_verifier.calculate_used_tokens()
                    total_tokens = prompt_tokens + completion_tokens
                    pbar.write(
                        f"Tokens spent: {total_tokens} "
                        f"(prompt={prompt_tokens}, completion={completion_tokens})"
                    )
        pbar.close()

    # ── Phase 2: assemble verified_groups sequentially, in original order,
    #    stitching any oversized cluster's sub-batches back together ────────
    verified_groups: list[tuple[str, list[str]]] = []

    for origin_idx, (cluster_id, members) in enumerate(cluster_items):
        if len(members) == 1:
            verified_groups.append((members[0], members))
            continue

        work_indices = origin_work_indices[origin_idx]
        singleton_chunks = origin_singletons.get(origin_idx, [])
        sub_results: list[tuple[str, list[str]]] = []
        for work_idx in work_indices:
            _, work_members = work_items[work_idx]
            groups, exc = raw_results[work_idx]
            sub_results.extend(_parse_cluster_response(groups, exc, work_members, cluster_id))
        # Trailing singleton chunks (e.g. 5 members split at max_cluster_size=2
        # -> chunks of 2, 2, 1) never got their own LLM call, but they were
        # still split off from the rest of the cluster purely by the size cap
        # -- fold them in as their own sub-group so stitching below gets a
        # chance to reunite them with a real sub-batch group, same as any
        # other sub-group.
        for label in singleton_chunks:
            sub_results.append((label, [label]))

        if len(work_indices) + len(singleton_chunks) <= 1 or len(sub_results) <= 1:
            # Not oversized (one work item covered the whole cluster, with no
            # trailing singleton chunk), or everything collapsed to a single
            # group -- nothing to stitch either way.
            verified_groups.extend(sub_results)
            continue

        label_to_members: dict[str, list[str]] = {}
        for canonical, sub_members in sub_results:
            key = canonical.lower().strip()
            if key in label_to_members:
                label_to_members[key] = label_to_members[key] + sub_members
            else:
                label_to_members[key] = sub_members

        stitch_labels = list(label_to_members.keys())
        if len(stitch_labels) > max_cluster_size:
            # The stitch pool itself can exceed max_cluster_size -- a big
            # enough oversized cluster survives its sub-batches as more than
            # max_cluster_size labels (e.g. 2249 members -> 57 sub-batches ->
            # 1050 surviving labels). Forcing all of them through one stitch
            # call recreates exactly the oversized/slow/truncation-prone
            # call this whole mechanism exists to avoid. 
            logger.warning(
                "Cluster %s: size %d over max_cluster_size=%d, split into %d sub-batches -> "
                "%d surviving sub-groups, which itself exceeds max_cluster_size -- skipping "
                "stitching, keeping sub-groups separate",
                cluster_id, len(members), max_cluster_size, len(work_indices), len(sub_results),
            )
            verified_groups.extend(sub_results)
            continue

        logger.info(
            "Cluster %s: size %d over max_cluster_size=%d, split into %d sub-batches -> "
            "%d sub-groups, stitching",
            cluster_id, len(members), max_cluster_size, len(work_indices), len(sub_results),
        )
        try:
            stitch_groups = llm_verifier.verify_cluster_with_llm(
                stitch_labels, surface_form_type=surface_form_type,
            )
            stitch_exc = None
        except Exception as exc:
            stitch_groups, stitch_exc = None, exc

        stitched = _parse_cluster_response(stitch_groups, stitch_exc, stitch_labels, f"{cluster_id}.stitch")
        for canonical, matched_labels in stitched:
            reunited: list[str] = []
            for label in matched_labels:
                reunited.extend(label_to_members.get(label.lower().strip(), [label]))
            verified_groups.append((canonical, reunited))

    return verified_groups


# ═══════════════════════════════════════════════════════════════════════════════
#  Base Deduplication Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_and_count(
    all_labels: list[str],
    normalizer: Callable[[str], str],
) -> tuple[list[str], dict[str, set[str]], dict[str, int], dict[str, int]]:
    """Shared "Step 1" of deduplicate(): normalise raw labels and compute
    per-normalized-label frequency statistics. Used by both deduplicate()
    and deduplicate_with_rounds() (which needs the same bookkeeping outside
    of a single deduplicate() call, to survive across multiple rounds).

    Returns (unique_labels, normalized_to_raws, count_per_normalized, surface_form_counts).
    """
    normalized_to_raws: dict[str, set[str]] = defaultdict(set)
    surface_form_counts: dict[str, int] = defaultdict(int)
    count_per_normalized: dict[str, int] = defaultdict(int)

    for label in all_labels:
        norm = normalizer(label)
        normalized_to_raws[norm].add(label)
        surface_form_counts[label] += 1
        count_per_normalized[norm] += 1

    unique_labels = sorted(normalized_to_raws.keys())

    logger.info(
        "After normalisation: %d unique labels (from %d raw labels)",
        len(unique_labels), len(set(all_labels)),
    )

    return unique_labels, normalized_to_raws, count_per_normalized, surface_form_counts


def deduplicate(
    all_labels: list[str],
    embedder: ContrieverEmbedder,
    llm_verifier = None,
    surface_form_type: str = 'entity_type',
    *,
    normalizer: Callable[[str], str] = normalize_label,
    embedding_text_fn: Callable[[str], str] | None = None,
    cluster_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    cluster_postprocess_fn: Callable[[dict[int, list[str]]], dict[int, list[str]]] | None = None,
    member_context: dict[str, str] | None = None,
    hac_threshold: float = 0.8,
    embed_batch_size: int = 64,
    id_prefix: str = "item",
    id_width: int = 4,
    build_item: Callable[[str, str, list[str], int, dict[str, int]], CanonicalItem] | None = None,
    max_workers: int = 8,
    max_cluster_size: int = 40,
) -> DeduplicationResult:
    """
    Shared deduplication pipeline: normalise → embed → cluster → LLM verify → build result.

    Accepts all labels (with duplicates) and computes normalisation, frequency
    statistics, and per-surface-form counts internally.

    Args:
        all_labels:          All raw labels, including duplicates.
        llm_verifier:        Object with verify_cluster_with_llm(members) method.
        embedder:            ContrieverEmbedder instance.
        normalizer:          Function to normalise a raw label. Defaults to
                             normalize_label (NFKC + lowercase + whitespace collapse).
        embedding_text_fn:   Optional function mapping a normalised label to the
                             text to embed (e.g. type-augmented for entities).
                             If None, embeds normalised labels directly.
        cluster_fn:          Custom clustering function: (embeddings) → cluster_labels.
                             When provided, replaces HDBSCAN for candidate generation.
                             When None (default), uses cluster_hdbscan with hac_threshold.
        cluster_postprocess_fn: Optional function: (clusters: dict[cluster_id, [labels]])
                             → clusters. Runs after clustering but before LLM
                             verification, so it can additionally MERGE separate
                             clusters together using evidence outside the label
                             embedding (e.g. relation_context.merge_clusters_by_relation_signature,
                             which unions clusters whose relation-signature TF-IDF
                             cosine similarity exceeds a threshold). Cannot split
                             clusters, only coarsen them further.
        member_context:      Optional normalised label → human-readable evidence
                             string, shown to the LLM verifier alongside each
                             candidate (e.g. "film (context: often object of:
                             directed, starred in)"). Purely additive context;
                             does not affect clustering.
        hac_threshold:       Cosine similarity threshold for HDBSCAN (used only when
                             cluster_fn is None) -- see cluster_hdbscan() for how this
                             maps to cluster_selection_epsilon. Named hac_threshold for
                             config/call-site compatibility with pre-HDBSCAN callers.
        embed_batch_size:    Batch size for Contriever.
        id_prefix:           Prefix for generated IDs (e.g. "type", "ent").
        id_width:            Zero-pad width for IDs.
        build_item:          Optional factory to build custom CanonicalItem subclasses.
                             Signature: (item_id, canonical_label, surface_forms,
                             mention_count, surface_form_counts) → CanonicalItem.
                             If None, builds a default CanonicalItem.
        max_workers:         Thread pool size for concurrent LLM cluster
                             verification (see verify_clusters_with_llm).

    Returns:
        DeduplicationResult with canonical items, surface form mappings,
        and per-surface-form frequency statistics.
    """
    # ── Step 1: Normalise and compute frequency statistics ────────────────────
    unique_labels, normalized_to_raws, count_per_normalized, surface_form_counts = (
        _normalize_and_count(all_labels, normalizer)
    )

    # ── Degenerate case: 0 or 1 unique labels ────────────────────────────────
    if len(unique_labels) < 2:
        items: dict[str, CanonicalItem] = {}
        surface_to_id: dict[str, str] = {}
        for idx, norm_label in enumerate(unique_labels):
            item_id = f"{id_prefix}_{idx:0{id_width}d}"
            sfs = sorted(normalized_to_raws.get(norm_label, {norm_label}) | {norm_label})
            sf_counts = {sf: surface_form_counts.get(sf, 0) for sf in sfs}
            count = count_per_normalized[norm_label]

            if build_item:
                item = build_item(item_id, norm_label, sfs, count, sf_counts)
            else:
                item = CanonicalItem(item_id=item_id, canonical_label=norm_label,
                                     count_per_normalized=count,
                                     surface_forms=sfs,
                                     surface_form_counts=sf_counts)
            items[item_id] = item
            for sf in sfs:
                surface_to_id[sf] = item_id
            surface_to_id[norm_label] = item_id

        return DeduplicationResult(
            items=items, surface_to_id=surface_to_id,
            num_raw=len(unique_labels), num_canonical=len(items),
            reduction_pct=0.0,
            normalized_to_raws=dict(normalized_to_raws),
            surface_form_counts=dict(surface_form_counts),
        )

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    if embedding_text_fn is not None:
        texts = [embedding_text_fn(norm_label) for norm_label in unique_labels]
    else:
        texts = unique_labels
    embeddings = embedder.embed(texts, batch_size=embed_batch_size)

    # ── Step 3: Candidate clustering ────────────────────────────────────────
    if cluster_fn is not None:
        cluster_labels = cluster_fn(embeddings)
    else:
        cluster_labels = cluster_hdbscan(embeddings, threshold=hac_threshold)

    clusters: dict[int, list[str]] = defaultdict(list)
    for norm_label, cl in zip(unique_labels, cluster_labels):
        clusters[int(cl)].append(norm_label)
    clusters = dict(clusters)

    logger.info(
        "Clustering produced %d clusters (%d with 2+ members, requiring LLM verification)",
        len(clusters), sum(1 for v in clusters.values() if len(v) > 1),
    )

    # ── Step 3b: Optional cluster-merge postprocessing ───────────────────────
    # e.g. relation_context.merge_clusters_by_relation_signature: additionally
    # union clusters whose relation-signature similarity is high, even if their
    # label embeddings never put them in the same HAC cluster.
    if cluster_postprocess_fn is not None:
        before = len(clusters)
        clusters = cluster_postprocess_fn(clusters)
        logger.info(
            "Cluster postprocessing: %d clusters → %d clusters",
            before, len(clusters),
        )

    multi_member_sizes = [len(v) for v in clusters.values() if len(v) > 1]
    if multi_member_sizes:
        logger.info("Mean cluster size for clusters with 2+ members: %.1f",
                     sum(multi_member_sizes) / len(multi_member_sizes))

    # ── Step 4: LLM verification ─────────────────────────────────────────────
    if llm_verifier is not None:
        verified_groups = verify_clusters_with_llm(
            clusters, llm_verifier, surface_form_type=surface_form_type,
            member_context=member_context, max_workers=max_workers,
            max_cluster_size=max_cluster_size,
        )
    else:
        verified_groups: list[tuple[str, list[str]]] = []
        for _, members in clusters.items():
            verified_groups.append((members[0], members))

    # ── Step 5: Build result ─────────────────────────────────────────────────
    return _assemble_dedup_result(
        verified_groups, normalized_to_raws=normalized_to_raws,
        count_per_normalized=count_per_normalized, surface_form_counts=surface_form_counts,
        num_raw=len(unique_labels), id_prefix=id_prefix, id_width=id_width, build_item=build_item,
    )


def _assemble_dedup_result(
    verified_groups: list[tuple[str, list[str]]],
    *,
    normalized_to_raws: dict[str, set[str]],
    count_per_normalized: dict[str, int],
    surface_form_counts: dict[str, int],
    num_raw: int,
    id_prefix: str,
    id_width: int,
    build_item: Callable[[str, str, list[str], int, dict[str, int]], CanonicalItem] | None,
) -> DeduplicationResult:
    """Shared "Step 5" of deduplicate(): turn (canonical_label, [normalized
    members]) groups plus the Step-1 normalisation bookkeeping into a full
    DeduplicationResult. Used by both deduplicate() (single pass) and
    deduplicate_with_rounds() (after all rounds have converged), so the
    result-shape logic exists in exactly one place."""
    items: dict[str, CanonicalItem] = {}
    surface_to_id: dict[str, str] = {}

    for idx, (canonical_label, members) in enumerate(verified_groups):
        item_id = f"{id_prefix}_{idx:0{id_width}d}"

        all_surface_forms: set[str] = set()
        total_count = 0
        for member in members:
            all_surface_forms.update(normalized_to_raws.get(member, {member}))
            all_surface_forms.add(member)
            total_count += count_per_normalized.get(member, 0)

        sorted_forms = sorted(all_surface_forms)
        sf_counts = {sf: surface_form_counts.get(sf, 0) for sf in sorted_forms}

        if build_item:
            item = build_item(item_id, canonical_label, sorted_forms, total_count,
                              sf_counts)
        else:
            item = CanonicalItem(
                item_id=item_id, canonical_label=canonical_label,
                surface_forms=sorted_forms, count_per_normalized=total_count,
                surface_form_counts=sf_counts,
            )
        items[item_id] = item

        for sf in all_surface_forms:
            surface_to_id[sf] = item_id
        surface_to_id[canonical_label] = item_id

    num_canonical = len(items)
    reduction = (1 - num_canonical / max(num_raw, 1)) * 100

    logger.info(
        "Deduplication complete: %d → %d canonical items (%.1f%% reduction)",
        num_raw, num_canonical, reduction,
    )

    return DeduplicationResult(
        items=items,
        surface_to_id=surface_to_id,
        num_raw=num_raw,
        num_canonical=num_canonical,
        reduction_pct=reduction,
        normalized_to_raws=dict(normalized_to_raws),
        surface_form_counts=dict(surface_form_counts),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Round-Based Deduplication (convergence loop)
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate_with_rounds(
    all_labels: list[str],
    embedder: ContrieverEmbedder,
    llm_verifier = None,
    surface_form_type: str = 'entity_type',
    *,
    normalizer: Callable[[str], str] = normalize_label,
    embedding_text_fn: Callable[[str], str] | None = None,
    cluster_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    cluster_postprocess_fn: Callable[[dict[int, list[str]]], dict[int, list[str]]] | None = None,
    member_context: dict[str, str] | None = None,
    hac_threshold: float = 0.8,
    embed_batch_size: int = 64,
    id_prefix: str = "item",
    id_width: int = 4,
    build_item: Callable[[str, str, list[str], int, dict[str, int]], CanonicalItem] | None = None,
    max_workers: int = 8,
    max_cluster_size: int = 40,
    max_rounds: int = 1,
) -> DeduplicationResult:
    """
    Round-based generalization of deduplicate(): normalise once, then
    repeatedly embed -> cluster -> LLM-verify the current pool of surviving
    canonical labels, folding merges back in, until a round produces no
    further reduction in pool size (nothing left to merge) or max_rounds is
    reached. Same convergence rule entity_dedup._run_partition_merge_rounds
    already uses per parent-partition (a merge in round N can pull two
    labels' embeddings close enough to only become clusterable in round
    N+1) -- generalized here off entity_dedup's compound "name [type]"
    labels to plain labels, so type_dedup and relation_dedup can use it too.

    No new LLM-facing signal is needed for convergence: "a round produced no
    further reduction" is a pure pool-size check, exactly like entity_dedup's
    existing loop. max_rounds=1 behaves identically to deduplicate().

    All other args have the same meaning as deduplicate() -- see its
    docstring.
    """
    unique_labels, normalized_to_raws, count_per_normalized, surface_form_counts = (
        _normalize_and_count(all_labels, normalizer)
    )

    if len(unique_labels) < 2:
        # Degenerate case identical to deduplicate()'s early return -- just
        # delegate rather than duplicate it.
        return deduplicate(
            all_labels, embedder, llm_verifier, surface_form_type,
            normalizer=normalizer, embedding_text_fn=embedding_text_fn,
            cluster_fn=cluster_fn, cluster_postprocess_fn=cluster_postprocess_fn,
            member_context=member_context, hac_threshold=hac_threshold,
            embed_batch_size=embed_batch_size, id_prefix=id_prefix, id_width=id_width,
            build_item=build_item, max_workers=max_workers, max_cluster_size=max_cluster_size,
        )

    # pool: current canonical label -> set of ORIGINAL round-1 normalized
    # labels folded into it so far (keys into normalized_to_raws /
    # count_per_normalized / surface_form_counts, established once above and
    # never recomputed -- only the grouping into pool entries changes).
    pool: dict[str, set[str]] = {label: {label} for label in unique_labels}

    for round_num in range(1, max(max_rounds, 1) + 1):
        if len(pool) < 2:
            break

        pool_labels = sorted(pool.keys())
        if embedding_text_fn is not None:
            texts = [embedding_text_fn(label) for label in pool_labels]
        else:
            texts = pool_labels
        embeddings = embedder.embed(texts, batch_size=embed_batch_size)

        if cluster_fn is not None:
            cluster_labels = cluster_fn(embeddings)
        else:
            cluster_labels = cluster_hdbscan(embeddings, threshold=hac_threshold)

        clusters: dict[int, list[str]] = defaultdict(list)
        for label, cl in zip(pool_labels, cluster_labels):
            clusters[int(cl)].append(label)
        clusters = dict(clusters)

        if cluster_postprocess_fn is not None:
            clusters = cluster_postprocess_fn(clusters)

        if all(len(members) == 1 for members in clusters.values()):
            logger.info("deduplicate_with_rounds round %d: no candidate clusters, stopping", round_num)
            break

        if llm_verifier is not None:
            verified_groups = verify_clusters_with_llm(
                clusters, llm_verifier, surface_form_type=surface_form_type,
                member_context=member_context, max_workers=max_workers,
                max_cluster_size=max_cluster_size,
            )
        else:
            verified_groups = [(members[0], members) for members in clusters.values()]

        # Fold this round's merges into a fresh pool, keyed by the (possibly
        # re-chosen) canonical label. Two independently-resolved groups
        # landing on the identical canonical_label string are unioned rather
        # than treated as a key collision -- a reasonable outcome (they'll
        # almost certainly re-cluster next round anyway, since identical
        # text embeds identically), and avoids ever using a disambiguated
        # key as a real canonical label.
        next_pool: dict[str, set[str]] = defaultdict(set)
        for canonical_label, members in verified_groups:
            merged_raws: set[str] = set()
            for m in members:
                merged_raws |= pool.get(m, {m})
            next_pool[canonical_label] |= merged_raws
        next_pool = dict(next_pool)

        progressed = len(next_pool) < len(pool)
        logger.info(
            "deduplicate_with_rounds round %d: %d -> %d canonical labels",
            round_num, len(pool), len(next_pool),
        )
        pool = next_pool
        if not progressed:
            break
    else:
        logger.info("deduplicate_with_rounds reached max_rounds=%d, stopping", max_rounds)

    verified_groups = [(label, sorted(raws)) for label, raws in pool.items()]
    return _assemble_dedup_result(
        verified_groups, normalized_to_raws=normalized_to_raws,
        count_per_normalized=count_per_normalized, surface_form_counts=surface_form_counts,
        num_raw=len(unique_labels), id_prefix=id_prefix, id_width=id_width, build_item=build_item,
    )
