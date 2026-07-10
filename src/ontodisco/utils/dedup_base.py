"""
Shared Deduplication Infrastructure
====================================

Base module for surface-form deduplication. Provides:
  - ContrieverEmbedder — dense embedding via Meta's Contriever
  - normalize_label    — Unicode NFKC + lowercase + whitespace collapse
  - cluster_hac        — Hierarchical Agglomerative Clustering
  - verify_clusters_with_llm — LLM merge/split verification loop
  - deduplicate()      — the shared 4-step pipeline (embed → HAC → LLM → build)

Both type_dedup.py and entity_dedup.py are thin wrappers around this module.
"""

from __future__ import annotations

import unicodedata
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
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
    surface_form_counts: dict[str, int] = field(default_factory=defaultdict(int))


@dataclass
class DeduplicationResult:
    """Full output of a deduplication step."""
    items: dict[str, CanonicalItem]       # item_id → CanonicalItem
    surface_to_id: dict[str, str]         # raw surface form → item_id
    num_raw: int = 0
    num_canonical: int = 0
    reduction_pct: float = 0.0
    normalized_to_raws: dict[str, set[str]] = field(default_factory=dict)
    surface_form_counts: dict[str, int] = field(default_factory=defaultdict(int))


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
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True, trust_remote_code=True, token=os.getenv("HF_KEY")).to(self.device)
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
#  HAC Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_hac(
    embeddings: np.ndarray,
    threshold: float = 0.8,
    linkage: str = "average",
) -> np.ndarray:
    """
    Cluster embeddings using Hierarchical Agglomerative Clustering.

    Args:
        embeddings:  (N, D) array of L2-normalised embeddings.
        threshold:   Cosine *similarity* threshold (converted to distance internally).
        linkage:     "average", "complete", or "single".

    Returns:
        Array of integer cluster labels, shape (N,).
    """
    distance_threshold = 1.0 - threshold
    distance_matrix = cosine_distances(embeddings)

    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=int)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage,
    )

    return clustering.fit_predict(distance_matrix)


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_clusters_with_llm(
    clusters: dict[int, list[str]],
    llm_verifier: object = None,
    *,
    verbose: bool = True,
    token_log_every: int = 10,
) -> list[tuple[str, list[str]]]:
    """
    Send each multi-member HAC cluster to the LLM for merge/split verification.

    The llm_verifier must implement verify_cluster_with_llm(members: list[str])
    returning a list of dicts with "canonical_label" and "members" keys.

    Returns:
        List of (canonical_label, [member_labels]) tuples.
    """
    verified_groups: list[tuple[str, list[str]]] = []
    cluster_items = list(clusters.items())

    pbar = tqdm(
        cluster_items,
        desc="LLM cluster verify",
        unit="cluster",
        disable=not verbose,
    )

    for i, (cluster_id, members) in enumerate(pbar, start=1):
        if len(members) == 1:
            verified_groups.append((members[0], members))
        else:
            try:
                groups = llm_verifier.verify_entity_type_cluster_with_llm(members)
            except Exception:
                logger.exception("LLM verification failed for cluster %d, keeping HAC grouping", cluster_id)
                verified_groups.append((members[0], members))
                groups = None

            if groups is not None and not groups:
                logger.warning("Could not parse LLM response for cluster %d, keeping HAC grouping", cluster_id)
                verified_groups.append((members[0], members))
            elif groups:
                for group in groups:
                    canonical = group.get("canonical_label", "")
                    group_members = group.get("members", [])

                    if not canonical or not group_members:
                        logger.warning("Could not parse LLM response for cluster %d, keeping HAC grouping", cluster_id)
                        verified_groups.append((members[0], members))
                        continue

                    matched_members = []
                    members_lower = {m.lower(): m for m in members}

                    for gm in group_members:
                        gm_lower = gm.lower().strip()
                        if gm_lower in members_lower:
                            matched_members.append(members_lower[gm_lower])
                        else:
                            logger.warning(
                                "LLM returned label %r not matching any cluster %d member %s; "
                                "keeping LLM form as-is",
                                gm, cluster_id, members,
                            )
                            matched_members.append(gm)

                    verified_groups.append((canonical.lower().strip(), matched_members))

                all_assigned = set()
                for group in groups:
                    for gm in group.get("members", []):
                        all_assigned.add(gm.lower().strip())

                for member in members:
                    if member.lower().strip() not in all_assigned:
                        logger.warning("LLM dropped label '%s' from cluster %d, adding as singleton", member, cluster_id)
                        verified_groups.append((member, [member]))

        if (
            verbose
            and token_log_every > 0
            and i % token_log_every == 0
            and llm_verifier is not None
            and hasattr(llm_verifier, "calculate_used_tokens")
        ):
            prompt_tokens, completion_tokens = llm_verifier.calculate_used_tokens()
            total_tokens = prompt_tokens + completion_tokens
            pbar.write(
                f"Tokens spent: {total_tokens} "
                f"(prompt={prompt_tokens}, completion={completion_tokens})"
            )

    return verified_groups


# ═══════════════════════════════════════════════════════════════════════════════
#  Base Deduplication Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def deduplicate(
    all_labels: list[str],
    llm_verifier,
    embedder: ContrieverEmbedder,
    *,
    normalizer: Callable[[str], str] = normalize_label,
    embedding_text_fn: Callable[[str], str] | None = None,
    cluster_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    hac_threshold: float = 0.8,
    hac_linkage: str = "average",
    embed_batch_size: int = 64,
    id_prefix: str = "item",
    id_width: int = 4,
    build_item: Callable[[str, str, list[str], int, dict[str, int]], CanonicalItem] | None = None,
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
                             When provided, replaces HAC for candidate generation.
                             When None (default), uses cluster_hac with hac_threshold
                             and hac_linkage.
        hac_threshold:       Cosine similarity threshold for HAC (used only when
                             cluster_fn is None).
        hac_linkage:         HAC linkage method (used only when cluster_fn is None).
        embed_batch_size:    Batch size for Contriever.
        id_prefix:           Prefix for generated IDs (e.g. "type", "ent").
        id_width:            Zero-pad width for IDs.
        build_item:          Optional factory to build custom CanonicalItem subclasses.
                             Signature: (item_id, canonical_label, surface_forms,
                             mention_count, surface_form_counts) → CanonicalItem.
                             If None, builds a default CanonicalItem.

    Returns:
        DeduplicationResult with canonical items, surface form mappings,
        and per-surface-form frequency statistics.
    """
    # ── Step 1: Normalise and compute frequency statistics ────────────────────
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
        cluster_labels = cluster_hac(embeddings, threshold=hac_threshold, linkage=hac_linkage)

    clusters: dict[int, list[str]] = defaultdict(list)
    for norm_label, cl in zip(unique_labels, cluster_labels):
        clusters[int(cl)].append(norm_label)

    multi_member_sizes = [len(v) for v in clusters.values() if len(v) > 1]
    logger.info(
        "Clustering produced %d clusters (%d with 2+ members, requiring LLM verification)",
        len(clusters), len(multi_member_sizes),
    )
    if multi_member_sizes:
        logger.info("Mean cluster size for clusters with 2+ members: %.1f",
                     sum(multi_member_sizes) / len(multi_member_sizes))

    # ── Step 4: LLM verification ─────────────────────────────────────────────
    verified_groups = verify_clusters_with_llm(clusters, llm_verifier)

    # ── Step 5: Build result ─────────────────────────────────────────────────
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
    reduction = (1 - num_canonical / max(len(unique_labels), 1)) * 100

    logger.info(
        "Deduplication complete: %d → %d canonical items (%.1f%% reduction)",
        len(unique_labels), num_canonical, reduction,
    )

    return DeduplicationResult(
        items=items,
        surface_to_id=surface_to_id,
        num_raw=len(unique_labels),
        num_canonical=num_canonical,
        reduction_pct=reduction,
        normalized_to_raws=dict(normalized_to_raws),
        count_per_normalized=dict(count_per_normalized),
    )
