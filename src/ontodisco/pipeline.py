"""
Pipeline Orchestration
=======================

Runs the currently-implemented steps of the ontology discovery pipeline from
a single YAML config:

    Relation Canonicalization -> Type Canonicalization -> Hierarchy Induction
    -> Entity Deduplication + Class Assignment

(Domain/Range Constraint Induction and Serialization are not implemented yet
and are out of scope here -- add them as their own checkpointed steps once
they exist.)

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

Each of the four LLM-backed steps (relation dedup, type dedup, hierarchy
induction, entity dedup) is checkpointed to
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
from src.ontodisco.type_dedup import TypeDeduplicationResult, deduplicate_types
from src.ontodisco.utils.openai_utils import LLMTripletExtractor

logger = logging.getLogger(__name__)

STEP_NAMES = ("relation_dedup", "type_dedup", "hierarchy_induction", "entity_dedup")


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"      # env var to read the key from -- never put keys in YAML
    base_url: str = "https://api.openai.com/v1"


@dataclass
class EmbeddingConfig:
    contriever_model: str = "facebook/contriever"
    device: Optional[str] = None
    embed_batch_size: int = 64


@dataclass
class TypeCanonicalizationConfig:
    hac_threshold: float = 0.75
    relation_signature_merge_threshold: float = 0.8
    relation_context_top_k: int = 5


@dataclass
class RelationCanonicalizationConfig:
    similarity_threshold: float = 0.75


@dataclass
class EntityCanonicalizationConfig:
    similarity_threshold: float = 0.85


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
    llm_extractor = LLMTripletExtractor(
        api_key=api_key, model=config.llm.model, base_url=config.llm.base_url,
    )

    triplets = load_triplets(config.input_path)
    step_durations: dict[str, float] = {}

    # ── Step: Relation Canonicalization ───────────────────────────────────────
    # Checkpointed in its RAW (pre-type-map-resolution) form deliberately --
    # see update_relation_type_map() below for why.
    t0 = time.monotonic()
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
        )
        _save_checkpoint(run_dir, "relation_dedup", relation_vocab)
    step_durations["relation_dedup"] = time.monotonic() - t0

    # ── Step: Type Canonicalization ───────────────────────────────────────────
    # Always given the relation vocabulary as a disambiguating signal (see
    # module docstring) -- this pipeline never runs it "bare".
    t0 = time.monotonic()
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
            relation_signature_merge_threshold=config.type_canonicalization.relation_signature_merge_threshold,
            relation_context_top_k=config.type_canonicalization.relation_context_top_k,
        )
        _save_checkpoint(run_dir, "type_dedup", type_vocab)
    step_durations["type_dedup"] = time.monotonic() - t0

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

    # ── Step: Entity Deduplication + Class Assignment ─────────────────────────
    # Runs LAST, after the type hierarchy exists: entity_dedup.py gates
    # candidate merges by shared immediate parent in the induced TypeHierarchy
    # (see its module docstring), and class assignment falls out of using
    # canonical type_ids in the compound labels -- neither is available until
    # both type_dedup and hierarchy_induction have run.
    t0 = time.monotonic()
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
        )
        _save_checkpoint(run_dir, "entity_dedup", entity_vocab)
    step_durations["entity_dedup"] = time.monotonic() - t0

    _write_run_metadata(
        run_dir, config, triplets, relation_vocab, type_vocab, hierarchy_result, entity_vocab,
        step_durations, start_time,
    )

    return OntoDiscoResult(
        relation_vocab=relation_vocab, type_vocab=type_vocab, hierarchy_result=hierarchy_result,
        entity_vocab=entity_vocab,
    )


def _collect_type_surface_forms(triplets: list[dict]) -> list[str]:
    labels: list[str] = []
    for triplet in triplets:
        for type_key in ("subject_type", "object_type"):
            raw_type = triplet.get(type_key, "").strip()
            if raw_type:
                labels.append(raw_type)
    return labels


def _write_run_metadata(
    run_dir: Path,
    config: PipelineConfig,
    triplets: list[dict],
    relation_vocab: RelationDeduplicationResult,
    type_vocab: TypeDeduplicationResult,
    hierarchy_result: HierarchyInductionResult,
    entity_vocab: EntityDeduplicationResult,
    step_durations: dict[str, float],
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
        "step_durations_seconds": step_durations,
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
