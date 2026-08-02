"""
Pipeline Orchestration
=======================

Runs the currently-implemented steps of the ontology discovery pipeline from
a single YAML config:

    Relation Canonicalization -> Type Canonicalization -> Hierarchy Induction
    -> Entity Deduplication + Class Assignment -> Constraint Induction

(Serialization is not implemented yet and is out of scope here -- add it as
its own checkpointed step once it exists.)

Step order is NOT "entity dedup, relation dedup, hierarchy induction" in that
naive reading -- relation canonicalization runs FIRST and entity dedup runs
LAST, because:
  - type_dedup.deduplicate_types() optionally takes the relation vocabulary
    as a `relation_result` signal (relation-signature-assisted clustering,
    see type_dedup.py) to help it avoid the exact hypernym/hyponym-as-synonym
    over-merges we found in onto_artifacts/verified_groups.json -- this
    pipeline always supplies it, rather than leaving it optional.
  - At the point deduplicate_relations() runs, CanonicalRelation.subject_types
    / object_types are still raw type labels (type dedup hasn't produced T*
    yet); relation_dedup.update_relation_type_map() is what later resolves
    them to canonical type_ids, and it must run exactly once, AFTER type
    dedup, BEFORE hierarchy induction (see the checkpointing note below for
    why this step is deliberately never checkpointed on its own).
  - entity_dedup.deduplicate_entities() needs canonical type_ids (from type
    dedup) to build its compound "name [type]" labels -- that's what makes
    class assignment fall out of clustering for free -- AND needs the
    induced TypeHierarchy (from hierarchy induction) to gate candidate
    merges by shared immediate parent (see entity_dedup.py's module
    docstring). So it can only run after BOTH of those steps.
  - constraints.induce_constraints() runs LAST of all: it needs the
    canonical type vocabulary, the canonical relation vocabulary WITH
    subject_types/object_types already resolved to type_ids (the same
    update_relation_type_map() bridge hierarchy induction depends on), and
    the induced TypeHierarchy for its hierarchy-generalization stage (LCA
    over observed domain/range types). It re-resolves the raw triplets
    itself (nothing upstream keeps the joint per-triple (subject_type,
    object_type) pairing for a relation), so it does not depend on
    entity_dedup's output at all -- it's ordered last only because it's the
    remaining orchestrated step, not because of a data dependency on entity
    dedup specifically.

Each of the five LLM-backed steps (relation dedup, type dedup, hierarchy
induction, entity dedup, constraint induction) is checkpointed to
`output_dir/checkpoints/run_<n>/<step>.pkl` via pickle (preserves the
dataclasses / sets exactly, no custom serialization needed). Every run gets
its own `run_<n>` folder (n = 1, 2, 3, ... auto-incremented) so successive
runs never clobber each other's checkpoints or metadata:
  - resume=False (a fresh run) always allocates a NEW run_<n> folder --
    n = 1 + the highest existing run number under output_dir/checkpoints
    (0 if none exist yet).
  - resume=True continues the MOST RECENT run_<n> folder (creating run_1 if
    none exists yet), so a crash mid-pipeline never forces re-paying for
    already-completed (expensive, LLM-backed) work, and never silently
    starts a fresh run number instead of picking up where it left off.

Each run folder also gets its own `run_metadata.json`, capturing everything
needed to reproduce that specific run: the full resolved PipelineConfig
(LLM model/base_url, embedding model/device/batch size, every step's
hyperparameters) plus corpus stats, step durations, and start/end
timestamps.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from src.ontodisco.constraints import ConstraintConfig, RelationConstraint, induce_constraints
from src.ontodisco.entity_dedup import EntityDeduplicationResult, deduplicate_entities
from src.ontodisco.hierarchy_induction import (
    HierarchyConfig,
    HierarchyInductionResult,
    induce_hierarchy,
)
from src.ontodisco.relation_dedup import (
    RelationDeduplicationResult,
    deduplicate_relations,
    update_relation_type_map,
)
from src.ontodisco.type_dedup import CanonicalType, TypeDeduplicationResult, deduplicate_types
from src.ontodisco.utils.dedup_base import normalize_label
from src.ontodisco.utils.openai_utils import LLMTripletExtractor

logger = logging.getLogger(__name__)

STEP_NAMES = (
    "relation_dedup", "type_dedup", "hierarchy_induction", "entity_dedup", "constraints",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"      # env var to read the key from -- never put keys in YAML
    base_url: str = "https://api.openai.com/v1"
    proxy_key_env: Optional[str] = None      # optional: env var holding an HTTP(S) proxy URL, routed
                                              # through httpx.Client(proxy=...) same as run_judge_eval.py's
                                              # judge client; None (default) means no proxy -- direct connection


@dataclass
class EmbeddingConfig:
    contriever_model: str = "facebook/contriever"
    device: Optional[str] = None
    embed_batch_size: int = 64


@dataclass
class TypeCanonicalizationConfig:
    hac_threshold: float = 0.75
    relation_context_top_k: int = 5
    max_merge_rounds: int = 5          # embed->cluster->LLM-verify passes; stops early once a
                                        # round produces no further reduction (dedup_base.deduplicate_with_rounds)
    max_parallel_workers: int = 8      # thread pool size for concurrent LLM cluster verification
    max_cluster_size: int = 40         # clusters over this size are split into sub-batches + stitched
                                        # back together, instead of one oversized LLM call (dedup_base.verify_clusters_with_llm)


@dataclass
class RelationCanonicalizationConfig:
    similarity_threshold: float = 0.75
    max_merge_rounds: int = 5
    max_parallel_workers: int = 8
    max_cluster_size: int = 40


@dataclass
class EntityCanonicalizationConfig:
    similarity_threshold: float = 0.85
    max_merge_rounds: int = 5
    max_parallel_workers: int = 8      # thread pool size for concurrent LLM cluster verification
    max_cluster_size: int = 40


@dataclass
class PipelineConfig:
    corpus_id: str = "my_corpus"
    input_path: str = ""                     # JSONL file of raw triplets
    output_dir: str = "./output"

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    type_canonicalization: TypeCanonicalizationConfig = field(default_factory=TypeCanonicalizationConfig)
    relation_canonicalization: RelationCanonicalizationConfig = field(default_factory=RelationCanonicalizationConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    entity_canonicalization: EntityCanonicalizationConfig = field(default_factory=EntityCanonicalizationConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)


def _from_dict(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys (with a warning)
    and leaving the dataclass's own defaults for any keys that are absent."""
    valid_fields = {f.name for f in fields(cls)}
    unknown = set(data) - valid_fields
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in data.items() if k in valid_fields})


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig(
        corpus_id=raw.get("corpus_id", PipelineConfig.corpus_id),
        input_path=raw.get("input_path", ""),
        output_dir=raw.get("output_dir", "./output"),
        llm=_from_dict(LLMConfig, raw.get("llm", {})),
        embedding=_from_dict(EmbeddingConfig, raw.get("embedding", {})),
        type_canonicalization=_from_dict(TypeCanonicalizationConfig, raw.get("type_canonicalization", {})),
        relation_canonicalization=_from_dict(RelationCanonicalizationConfig, raw.get("relation_canonicalization", {})),
        hierarchy=_from_dict(HierarchyConfig, raw.get("hierarchy", {})),
        entity_canonicalization=_from_dict(EntityCanonicalizationConfig, raw.get("entity_canonicalization", {})),
        constraints=_from_dict(ConstraintConfig, raw.get("constraints", {})),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Input loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_triplets(path: str | Path) -> list[dict]:
    """Load raw triplets from a JSONL file (one triplet dict per line, with
    at least subject/subject_type/relation/object/object_type keys)."""
    triplets: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    logger.info("Loaded %d triplets from %s", len(triplets), path)
    return triplets


def load_seed_roots(path: str | Path) -> list:
    """Load a seed hierarchy (hierarchy_induction.HierarchyConfig.seed_roots'
    shape -- a flat list of labels, or nested {"label", "children"} dicts)
    from a standalone YAML file, so a reusable seed ontology like DOLCE
    (configs/seeds/dolce.yaml) doesn't have to be duplicated inline in every
    pipeline config that wants it. Accepts either a file whose top-level
    content IS the list, or one that wraps it under a `seed_roots:` key
    (matching how the same structure is written inline under `hierarchy:`
    in a pipeline config) -- the latter is the documented/recommended form,
    since it self-labels the file's contents."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("seed_roots", [])
    raise ValueError(f"Seed roots file {path!r} must contain a list or a 'seed_roots:' mapping")


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def _existing_run_numbers(checkpoints_root: Path) -> list[int]:
    if not checkpoints_root.exists():
        return []
    numbers = []
    for child in checkpoints_root.iterdir():
        if child.is_dir():
            m = _RUN_DIR_RE.match(child.name)
            if m:
                numbers.append(int(m.group(1)))
    return sorted(numbers)


def _resolve_run_dir(output_dir: Path, *, resume: bool) -> Path:
    """
    Pick (and create) the `checkpoints/run_<n>` directory for this invocation.

    resume=True continues the highest-numbered existing run (or starts
    run_1 if none exist). resume=False always allocates a brand new
    run_<n> one past the highest existing number, so fresh runs never
    overwrite a previous run's checkpoints/metadata.
    """
    checkpoints_root = output_dir / "checkpoints"
    existing = _existing_run_numbers(checkpoints_root)

    if resume:
        run_number = existing[-1] if existing else 1
    else:
        run_number = (existing[-1] + 1) if existing else 1

    run_dir = checkpoints_root / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using run directory: %s (resume=%s)", run_dir, resume)
    return run_dir


def _checkpoint_path(run_dir: Path, step_name: str) -> Path:
    return run_dir / f"{step_name}.pkl"


def _save_checkpoint(run_dir: Path, step_name: str, obj) -> None:
    path = _checkpoint_path(run_dir, step_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Checkpoint saved: %s", path)


def _load_checkpoint(run_dir: Path, step_name: str):
    path = _checkpoint_path(run_dir, step_name)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Checkpoint loaded (resume): %s", path)
    return obj


def _has_checkpoint(run_dir: Path, step_name: str) -> bool:
    return _checkpoint_path(run_dir, step_name).exists()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OntoDiscoResult:
    relation_vocab: RelationDeduplicationResult   # subject_types/object_types already resolved to type_ids
    type_vocab: TypeDeduplicationResult
    hierarchy_result: HierarchyInductionResult
    entity_vocab: EntityDeduplicationResult
    constraints: list[RelationConstraint]


def run_pipeline(config: PipelineConfig, *, resume: bool = False) -> OntoDiscoResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _resolve_run_dir(output_dir, resume=resume)
    start_time = datetime.now(timezone.utc)

    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {config.llm.api_key_env!r} (config.llm.api_key_env) is not set"
        )
    proxy = os.environ.get(config.llm.proxy_key_env) if config.llm.proxy_key_env else None
    if config.llm.proxy_key_env and not proxy:
        logger.warning(
            "config.llm.proxy_key_env=%r is set but that environment variable is empty -- "
            "proceeding without a proxy", config.llm.proxy_key_env,
        )
    llm_extractor = LLMTripletExtractor(
        api_key=api_key, model=config.llm.model, base_url=config.llm.base_url, proxy=proxy,
    )

    triplets = load_triplets(config.input_path)
    step_durations: dict[str, float] = {}
    token_usage_by_step: dict[str, dict] = {}

    # Resolve an external seed-ontology file (if any) before hierarchy
    # induction runs, mirroring how config.input_path is only read here at
    # run time rather than at load_config() time. Doing this unconditionally
    # -- even under --resume, when hierarchy_induction's checkpoint may make
    # induce_hierarchy() a no-op this run -- keeps config.hierarchy.seed_roots
    # (and therefore run_metadata.json's config dump) an accurate record of
    # the seed hierarchy actually in effect, not just a dangling file path.
    if config.hierarchy.seed_roots_path:
        if config.hierarchy.seed_roots:
            logger.warning(
                "hierarchy.seed_roots_path is set; ignoring the %d inline seed_roots "
                "entry/entries already present in hierarchy.seed_roots",
                len(config.hierarchy.seed_roots),
            )
        config.hierarchy.seed_roots = load_seed_roots(config.hierarchy.seed_roots_path)
        logger.info(
            "Loaded %d top-level seed root(s) from %s",
            len(config.hierarchy.seed_roots), config.hierarchy.seed_roots_path,
        )

    # ── Step: Relation Canonicalization ───────────────────────────────────────
    # Checkpointed in its RAW (pre-type-map-resolution) form deliberately --
    # see update_relation_type_map() below for why.
    t0 = time.monotonic()
    usage0 = llm_extractor.get_usage_snapshot()
    if resume and _has_checkpoint(run_dir, "relation_dedup"):
        relation_vocab = _load_checkpoint(run_dir, "relation_dedup")
    else:
        logger.info("=== Step: Relation Canonicalization ===")
        relation_vocab = deduplicate_relations(
            triplets=triplets,
            llm_extractor=llm_extractor,
            contriever_model=config.embedding.contriever_model,
            similarity_threshold=config.relation_canonicalization.similarity_threshold,
            embed_batch_size=config.embedding.embed_batch_size,
            device=config.embedding.device,
            max_merge_rounds=config.relation_canonicalization.max_merge_rounds,
            max_parallel_workers=config.relation_canonicalization.max_parallel_workers,
            max_cluster_size=config.relation_canonicalization.max_cluster_size,
        )
        _save_checkpoint(run_dir, "relation_dedup", relation_vocab)
    step_durations["relation_dedup"] = time.monotonic() - t0
    token_usage_by_step["relation_dedup"] = _diff_usage(usage0, llm_extractor.get_usage_snapshot())

    # ── Step: Type Canonicalization ───────────────────────────────────────────
    # Always given the relation vocabulary as a disambiguating signal (see
    # module docstring) -- this pipeline never runs it "bare".
    t0 = time.monotonic()
    usage0 = llm_extractor.get_usage_snapshot()
    if resume and _has_checkpoint(run_dir, "type_dedup"):
        type_vocab = _load_checkpoint(run_dir, "type_dedup")
    else:
        logger.info("=== Step: Type Canonicalization ===")
        raw_type_labels = _collect_type_surface_forms(triplets)
        type_vocab = deduplicate_types(
            raw_type_labels=raw_type_labels,
            llm_extractor=llm_extractor,
            contriever_model=config.embedding.contriever_model,
            hac_threshold=config.type_canonicalization.hac_threshold,
            embed_batch_size=config.embedding.embed_batch_size,
            device=config.embedding.device,
            relation_result=relation_vocab,
            relation_context_top_k=config.type_canonicalization.relation_context_top_k,
            max_merge_rounds=config.type_canonicalization.max_merge_rounds,
            max_parallel_workers=config.type_canonicalization.max_parallel_workers,
            max_cluster_size=config.type_canonicalization.max_cluster_size,
        )
        _save_checkpoint(run_dir, "type_dedup", type_vocab)
    step_durations["type_dedup"] = time.monotonic() - t0
    token_usage_by_step["type_dedup"] = _diff_usage(usage0, llm_extractor.get_usage_snapshot())

    # ── Bridge: resolve relation_vocab's subject_types/object_types to type_ids ──
    # Deliberately NOT its own checkpoint: it mutates relation_vocab in place,
    # and update_relation_type_map() is not idempotent -- calling it twice on
    # an already-resolved relation_vocab would look up type_ids as if they
    # were raw surface forms and silently wipe subject_types/object_types to
    # empty sets. Always re-derive it fresh, in-process, from the RAW
    # relation_vocab checkpoint + the type_vocab checkpoint; it's a pure
    # lookup with no LLM calls, so re-running it costs nothing.
    relation_vocab = update_relation_type_map(relation_vocab, type_vocab)

    # ── Step: Hierarchy Induction ─────────────────────────────────────────────
    t0 = time.monotonic()
    usage0 = llm_extractor.get_usage_snapshot()
    if resume and _has_checkpoint(run_dir, "hierarchy_induction"):
        hierarchy_result = _load_checkpoint(run_dir, "hierarchy_induction")
    else:
        logger.info("=== Step: Hierarchy Induction ===")
        hierarchy_result = induce_hierarchy(
            type_vocab, relation_vocab, llm_extractor,
            contriever_model=config.embedding.contriever_model,
            device=config.embedding.device,
            config=config.hierarchy,
        )
        _save_checkpoint(run_dir, "hierarchy_induction", hierarchy_result)
    step_durations["hierarchy_induction"] = time.monotonic() - t0
    token_usage_by_step["hierarchy_induction"] = _diff_usage(usage0, llm_extractor.get_usage_snapshot())

    # ── Bridge: fold hierarchy induction's synthesized types into type_vocab ──
    # Same rationale/pattern as update_relation_type_map above: cheap, pure,
    # idempotent, so always re-derive it rather than checkpoint it on its
    # own. Must run before entity_dedup/constraints so both see verbalized
    # labels for synthesized parent types, not just their raw type_ids.
    type_vocab = _merge_synthesized_types_into_vocab(type_vocab, hierarchy_result)

    # ── Step: Entity Deduplication + Class Assignment ─────────────────────────
    # Runs LAST, after the type hierarchy exists: entity_dedup.py gates
    # candidate merges by shared immediate parent in the induced TypeHierarchy
    # (see its module docstring), and class assignment falls out of using
    # canonical type_ids in the compound labels -- neither is available until
    # both type_dedup and hierarchy_induction have run.
    t0 = time.monotonic()
    usage0 = llm_extractor.get_usage_snapshot()
    if resume and _has_checkpoint(run_dir, "entity_dedup"):
        entity_vocab = _load_checkpoint(run_dir, "entity_dedup")
    else:
        logger.info("=== Step: Entity Deduplication ===")
        entity_vocab = deduplicate_entities(
            triplets=triplets,
            type_vocab=type_vocab,
            hierarchy=hierarchy_result.hierarchy,
            llm_extractor=llm_extractor,
            contriever_model=config.embedding.contriever_model,
            similarity_threshold=config.entity_canonicalization.similarity_threshold,
            embed_batch_size=config.embedding.embed_batch_size,
            device=config.embedding.device,
            max_merge_rounds=config.entity_canonicalization.max_merge_rounds,
            max_parallel_workers=config.entity_canonicalization.max_parallel_workers,
            max_cluster_size=config.entity_canonicalization.max_cluster_size,
        )
        _save_checkpoint(run_dir, "entity_dedup", entity_vocab)
    step_durations["entity_dedup"] = time.monotonic() - t0
    token_usage_by_step["entity_dedup"] = _diff_usage(usage0, llm_extractor.get_usage_snapshot())

    # ── Step: Domain/Range Constraint Induction ───────────────────────────────
    # Doesn't depend on entity_dedup's output at all (see module docstring) --
    # ordered last only because it's the remaining orchestrated step. Needs
    # relation_vocab's subject_types/object_types already resolved to type_ids
    # (the update_relation_type_map() bridge above, same precondition
    # hierarchy induction has) and the induced TypeHierarchy for its
    # hierarchy-generalization (LCA) stage.
    t0 = time.monotonic()
    usage0 = llm_extractor.get_usage_snapshot()
    if resume and _has_checkpoint(run_dir, "constraints"):
        constraints_result = _load_checkpoint(run_dir, "constraints")
    else:
        logger.info("=== Step: Constraint Induction ===")
        constraints_result = induce_constraints(
            triplets=triplets,
            type_vocab=type_vocab,
            relation_vocab=relation_vocab,
            hierarchy=hierarchy_result.hierarchy,
            llm_extractor=llm_extractor,
            config=config.constraints,
        )
        _save_checkpoint(run_dir, "constraints", constraints_result)
    step_durations["constraints"] = time.monotonic() - t0
    token_usage_by_step["constraints"] = _diff_usage(usage0, llm_extractor.get_usage_snapshot())

    _write_run_metadata(
        run_dir, config, triplets, relation_vocab, type_vocab, hierarchy_result, entity_vocab,
        constraints_result, step_durations, token_usage_by_step, start_time,
    )

    return OntoDiscoResult(
        relation_vocab=relation_vocab, type_vocab=type_vocab, hierarchy_result=hierarchy_result,
        entity_vocab=entity_vocab, constraints=constraints_result,
    )


def _merge_synthesized_types_into_vocab(
    type_vocab: TypeDeduplicationResult,
    hierarchy_result: HierarchyInductionResult,
) -> TypeDeduplicationResult:
    """Fold each SynthesizedType minted during hierarchy induction into the
    type vocabulary as a real CanonicalType (SynthesizedType's own docstring
    says callers must do this; nothing previously did). Without this, any
    consumer that resolves a type_id to a display label -- constraint
    domain/range signatures land on synthesized parents constantly, since
    that's exactly what LCA over the hierarchy tends to produce -- falls
    back to the raw "type_h0007"-style id instead of its verbalized name.
    Mutates type_vocab in place (same "cheap, idempotent, recompute after
    loading checkpoints" pattern as update_relation_type_map) and returns it."""
    for st in hierarchy_result.synthesized_types:
        if st.type_id in type_vocab.items:
            continue
        type_vocab.items[st.type_id] = CanonicalType(
            item_id=st.type_id,
            canonical_label=st.canonical_label,
            surface_forms=[st.canonical_label],
        )
        type_vocab.surface_to_id[normalize_label(st.canonical_label)] = st.type_id
    return type_vocab


def _collect_type_surface_forms(triplets: list[dict]) -> list[str]:
    labels: list[str] = []
    for triplet in triplets:
        for type_key in ("subject_type", "object_type"):
            raw_type = triplet.get(type_key, "").strip()
            if raw_type:
                labels.append(raw_type)
    return labels


def _diff_usage(before: dict, after: dict) -> dict:
    """Per-step token/cost usage: after - before, for the two
    LLMTripletExtractor.get_usage_snapshot() calls bracketing one step. A
    resumed (checkpoint-loaded) step makes no LLM calls, so its diff is
    correctly all zeros rather than omitted."""
    return {key: after[key] - before[key] for key in before}


def _write_run_metadata(
    run_dir: Path,
    config: PipelineConfig,
    triplets: list[dict],
    relation_vocab: RelationDeduplicationResult,
    type_vocab: TypeDeduplicationResult,
    hierarchy_result: HierarchyInductionResult,
    entity_vocab: EntityDeduplicationResult,
    constraints_result: list[RelationConstraint],
    step_durations: dict[str, float],
    token_usage_by_step: dict[str, dict],
    start_time: datetime,
) -> None:
    end_time = datetime.now(timezone.utc)
    metadata = {
        "run_id": run_dir.name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "corpus_id": config.corpus_id,
        "num_triplets": len(triplets),
        "num_canonical_relations": relation_vocab.num_canonical_relations,
        "num_canonical_types": type_vocab.num_canonical_types,
        "num_synthesized_types": len(hierarchy_result.synthesized_types),
        "num_hierarchy_edges": len(hierarchy_result.hierarchy.edges),
        "num_hierarchy_roots": len(hierarchy_result.hierarchy.roots),
        "num_canonical_entities": entity_vocab.num_canonical_entities,
        "num_relation_constraints": len(constraints_result),
        "num_constraints_by_strength": {
            strength: sum(1 for c in constraints_result if c.strength.value == strength)
            for strength in ("hard", "soft", "hint")
        },
        "step_durations_seconds": step_durations,
        # Per-step {prompt_tokens, completion_tokens, total_tokens, cost_usd},
        # computed as a before/after snapshot diff around each step (see
        # _diff_usage) -- a resumed (checkpoint-loaded) step correctly shows
        # all zeros, since it made no LLM calls this run.
        "token_usage_by_step": token_usage_by_step,
        # Full resolved config (LLM model/base_url, embedding model/device/
        # batch size, every step's hyperparameters) so this specific run is
        # reproducible from the metadata file alone. api_key_env only records
        # which env var was read, never the key value itself.
        "config": asdict(config),
    }
    path = run_dir / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Run metadata written: %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the ontology discovery pipeline")
    parser.add_argument("--config", required=True, help="Path to a pipeline YAML config")
    parser.add_argument("--resume", action="store_true", help="Skip steps with an existing checkpoint")
    args = parser.parse_args()

    pipeline_config = load_config(args.config)
    run_pipeline(pipeline_config, resume=args.resume)
