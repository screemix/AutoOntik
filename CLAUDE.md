# CLAUDE.md — Ontology Discovery Pipeline Specification

> **Purpose**: This file is the canonical specification for the **Automatic Ontology Discovery Pipeline**
> for constructing high-quality, typed knowledge graphs from raw text corpora. It is intended to be
> read by both human developers and AI coding assistants (hence the name). **This revision is written
> against the actual implementation under `src/ontodisco/` as of the `dev` branch** — every module,
> data structure, and algorithm description below was checked against the code that produces it. Where
> the implementation diverges from an earlier design intent, or where a described step is not yet
> wired into the orchestrated pipeline, that is called out explicitly rather than silently glossed
> over.
>
> **Research context**: This pipeline operationalizes the approach described in the paper
> *"From Surface Forms to Ontologies: Discovering Type Hierarchies and Property Constraints
> for Graph-RAG"*. It extends KGGen (arXiv 2502.09956) with typed extraction, type/relation
> canonicalization, hierarchy induction, and domain/range constraint mining. The hierarchy-induction
> algorithm in particular has evolved substantially past what the paper originally proposed — see
> Step 3 below.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Shared Data Structures](#2-shared-data-structures)
3. [Step 0 — Typed Triplet Extraction](#3-step-0--typed-triplet-extraction)
4. [Step 1 — Relation Canonicalization](#4-step-1--relation-canonicalization)
5. [Step 2 — Type Canonicalization](#5-step-2--type-canonicalization)
6. [Step 3 — Hierarchy Induction](#6-step-3--hierarchy-induction)
7. [Step 4 — Entity Deduplication & Class Assignment](#7-step-4--entity-deduplication--class-assignment)
8. [Step 5 — Domain/Range Constraint Induction](#8-step-5--domainrange-constraint-induction)
9. [Step 6 — Ontology Serialization](#9-step-6--ontology-serialization)
10. [Pipeline Orchestration](#10-pipeline-orchestration)
11. [Configuration Reference](#11-configuration-reference)
12. [Testing Requirements](#12-testing-requirements)
13. [Logging and Reproducibility](#13-logging-and-reproducibility)
14. [Known Limitations and Design Rationale](#14-known-limitations-and-design-rationale)

---

## 1. Architecture Overview

`src/ontodisco/pipeline.py::run_pipeline()` orchestrates four steps, in this order — **not** the
"extraction → entities → types → relations → hierarchy → constraints → serialization" order a naive
reading of a typed-KG pipeline would suggest. Relation canonicalization runs **first** and entity
deduplication runs **last**, because of real data dependencies between the steps (see the docstring
of `pipeline.py` for the full argument). Extraction, constraint induction, and serialization exist
as standalone, working code but are **not** called by `run_pipeline()` today.

```
┌───────────────────────────────────────────────────────────────────────┐
│  INPUT: pre-extracted raw triplets, one JSON dict per line             │
│  {"subject", "subject_type", "relation", "object", "object_type", ...} │
│  Produced upstream by Step 0 (LLMTripletExtractor) — NOT orchestrated  │
│  as part of run_pipeline(); loaded via pipeline.load_triplets()        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 1: Relation Canonicalization  (relation_dedup.py)   [RUNS FIRST] │
│  • HDBSCAN candidate generation over the full cosine-distance matrix   │
│  • LLM merge/split verification (prompts/cluster_relations.txt)        │
│  • Records subject_types / object_types (still RAW labels) per relation│
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  RelationDeduplicationResult
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 2: Type Canonicalization  (type_dedup.py)                        │
│  • HDBSCAN on Contriever embeddings + LLM merge/split verification     │
│  • ALWAYS given Step 1's relation vocabulary as a disambiguating signal│
│    (relation-signature cluster-merge pre-LLM + PPMI context per label) │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  TypeDeduplicationResult (T*)
                               │
                               ▼  bridge: update_relation_type_map() resolves
                                  relation_vocab's subject_types/object_types
                                  from raw labels into canonical type_ids
                               │  (not checkpointed on its own — see §10)
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 3: Hierarchy Induction  (hierarchy_induction.py)                 │
│  • Recursive rounds of HDBSCAN on a FUSED similarity (label embedding +│
│    relation-argument-profile cosine) over T* + parent surrogates       │
│  • Every multi-member cluster → LLM "merge"/"no_parent" decision       │
│    (prompts/hierarchy_cluster_action.txt); may invent new parent labels│
│  • Duplicate-label reconciliation every round + final root-stitching   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  TypeHierarchy (single-parent forest)
                               │  + SynthesizedType[] (new abstract types)
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  STEP 4: Entity Deduplication + Class Assignment (entity_dedup.py)     │
│                                                             [RUNS LAST]│
│  • Compound "name [canonical type]" labels (identity = name + type)    │
│  • Parent-partitioned HDBSCAN candidates, gated by shared immediate    │
│    parent in TypeHierarchy (identical type, or siblings — nothing else)│
│  • LLM merge/split verification (prompts/cluster_entity_names.txt)     │
│  • Class assignment falls out of clustering (CanonicalEntity.type_ids) │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  OUTPUT: OntoDiscoResult                                                │
│  { relation_vocab, type_vocab, hierarchy_result, entity_vocab }         │
│  Each step checkpointed to output_dir/checkpoints/run_<n>/<step>.pkl     │
└───────────────────────────────────────────────────────────────────────┘

╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍  implemented, but NOT called by run_pipeline()  ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍

┌───────────────────────────────────────────────────────────────────────┐
│  Domain/Range Constraint Induction (constraints.py)                    │
│  induce_constraints() is fully implemented and independently callable, │
│  given (triplets, type_vocab, relation_vocab, hierarchy, llm_extractor) │
└───────────────────────────────────────────────────────────────────────┘

╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍  NOT IMPLEMENTED AT ALL  ╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍╍

┌───────────────────────────────────────────────────────────────────────┐
│  Ontology Serialization — no src/ontodisco/serialization.py exists.     │
│  §9 below is a design target only.                                     │
└───────────────────────────────────────────────────────────────────────┘
```

### Design principles (as actually followed by the code)

1. **Schema-free extraction**: Step 0's prompt does not constrain `subject_type`/`object_type` to any
   pre-defined vocabulary — T* still *emerges* from whatever the LLM writes down, even though the
   extraction step itself isn't corpus-orchestrated (see §3).
2. **Separation of concerns, without a shared schema module**: there is no central `types.py` — each
   step's dataclasses live in the module that produces them (`type_dedup.py`, `relation_dedup.py`,
   `entity_dedup.py`, `hierarchy_induction.py`, `constraints.py`), subclassing a small shared base
   (`CanonicalItem` / `DeduplicationResult`) in `utils/dedup_base.py`. Raw triplets are never wrapped
   in a dataclass at all — they flow as plain `dict`s end to end.
3. **LLM-agnosticism via one concrete client, not an abstract interface**: every LLM call goes through
   a single `LLMTripletExtractor` (`utils/openai_utils.py`), which wraps the `openai` Python client
   pointed at whatever OpenAI-compatible `base_url` the config supplies — real runs in this repo have
   used OpenAI models and an internal gateway serving GigaChat/Llama/Qwen/gpt-oss (see
   `configs/*.yaml`). There is no separate Anthropic/Ollama/vLLM adapter or `LLMClient` protocol class.
4. **Reproducibility, partial**: every call uses `temperature=0` (`get_completion()`). There is
   **no** per-call JSONL log of prompt+response (see §13) — only aggregate token/cost counters and a
   `run_metadata.json` per run. Majority voting exists as a mechanism in hierarchy induction
   (`col_num_votes`/`col_vote_agreement`) but defaults to 1 vote (off) and isn't used anywhere else;
   every other LLM-verification call in the pipeline is a single, unvoted call per cluster.
5. **Incremental processing at step granularity, not chunk granularity**: each of the four
   orchestrated steps (relation dedup, type dedup, hierarchy induction, entity dedup) is checkpointed
   as a whole to `output_dir/checkpoints/run_<n>/<step>.pkl` via `pickle`. There is no finer-grained
   per-chunk/per-item checkpointing within a step — a crash mid-step re-runs that entire step from
   scratch on resume.

---

## 2. Shared Data Structures

There is no `src/ontodisco/types.py`. Each step's output dataclasses live in the module that produces
them. A small shared base pair lives in `src/ontodisco/utils/dedup_base.py` and is subclassed by the
type/relation/entity vocabularies:

```python
# ─────────────────────────────────────────────────────────────────────────────
# src/ontodisco/utils/dedup_base.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CanonicalItem:
    """Base for CanonicalType / CanonicalRelation / CanonicalEntity."""
    item_id: str
    canonical_label: str
    count_per_normalized: int = 0
    surface_forms: list[str] = field(default_factory=list)
    surface_form_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class DeduplicationResult:
    """Base for TypeDeduplicationResult / RelationDeduplicationResult / EntityDeduplicationResult."""
    items: dict[str, CanonicalItem]         # item_id -> CanonicalItem
    surface_to_id: dict[str, str]           # raw/normalized surface form -> item_id
    num_raw: int = 0
    num_canonical: int = 0
    reduction_pct: float = 0.0
    normalized_to_raws: dict[str, set[str]] = field(default_factory=dict)
    surface_form_counts: dict[str, int] = field(default_factory=dict)
```

Raw triplets are **never** wrapped in a dataclass — there is no `RawTriplet` or `CanonicalTriplet`.
They flow as plain `dict`s (`{"subject", "subject_type", "relation", "object", "object_type",
"qualifiers"}`), loaded straight from a JSONL file by `pipeline.load_triplets()`. Every downstream
step (relation dedup, type dedup, entity dedup, constraint induction) independently re-resolves
whatever raw fields it needs into canonical ids directly from these dicts, rather than consuming one
shared canonicalized-triplet representation.

**Type Vocabulary** (`type_dedup.py`) — no `definition`, `example_entities`, or `instance_count`
fields exist, and there is no per-type list of member entities on the vocabulary itself (that
direction lives on `CanonicalEntity.type_ids` instead — see Step 4):

```python
@dataclass
class CanonicalType(CanonicalItem):
    """One entry in the flat type vocabulary T*. Adds nothing beyond
    CanonicalItem; type_id / type_count are properties aliasing
    item_id / count_per_normalized."""
    @property
    def type_id(self) -> str: return self.item_id
    @property
    def type_count(self) -> int: return self.count_per_normalized


@dataclass
class TypeDeduplicationResult(DeduplicationResult):
    """Exposes .types / .surface_to_type_id / .num_raw_types /
    .num_canonical_types as renamed views onto the base fields."""
```

**Relation Vocabulary** (`relation_dedup.py`):

```python
@dataclass
class CanonicalRelation(CanonicalItem):
    subject_types: set[str] = field(default_factory=set)   # raw type labels until
    object_types: set[str] = field(default_factory=set)    # update_relation_type_map() runs
    @property
    def relation_id(self) -> str: return self.item_id
```

This is a deliberate deviation from an earlier design that kept a single unordered `arg_types: set`
(union of subject and object types, to stay symmetric under grammatical inversion). The implemented
`CanonicalRelation` keeps `subject_types`/`object_types` as two **separate** sets, because Step 5
(constraint induction) needs domain and range separately. The "union, role-blind" idea is still real,
but it happens one layer up: `relation_context.build_relation_counts()` sums subject- and object-slot
occurrences into the *same* per-type distributional dimension when building relation-argument
profiles for type dedup (Step 2) and hierarchy induction (Step 3).

**Entity Vocabulary** (`entity_dedup.py`) — absent from the original design entirely:

```python
@dataclass
class CanonicalEntity(CanonicalItem):
    """Class assignment falls out of clustering: type_ids is the union of
    canonical type_ids across every mention merged into this entity — there
    is no separate 'class assignment' pass."""
    type_ids: set[str] = field(default_factory=set)
    @property
    def entity_id(self) -> str: return self.item_id
```

**Type Hierarchy** (`hierarchy_induction.py`) — redesigned; see Step 3 for why:

```python
@dataclass
class HierarchyEdge:
    child_type_id: str
    parent_type_id: str
    relation_signature_score: Optional[float]  # Weeds precision(child, parent) — informational only,
                                                # never used to gate or score the edge
    llm_score: Optional[float]                 # the LLM's own merge confidence (0.0–1.0)
    ensemble_score: float                      # == llm_score; kept only for downstream-code
                                                # compatibility — there is no actual ensemble
    is_direct: bool = True                      # ALWAYS True: transitive shortcuts cannot occur by
                                                 # construction (see Step 3) — no Hasse reduction exists

@dataclass
class TypeHierarchy:
    edges: list[HierarchyEdge]
    children: dict[str, list[str]]   # parent_type_id -> [child_type_ids]
    parents: dict[str, list[str]]    # child_type_id -> [parent_type_id] — length is ALWAYS <= 1
                                      # (a single-parent forest, not a multi-parent DAG)
    roots: list[str]

@dataclass
class SynthesizedType:
    """A new abstract type invented by the LLM during hierarchy induction
    (e.g. 'vehicle' for {car, bicycle}), rather than promoting an existing
    T* member as the parent. Callers must fold these into the type
    vocabulary alongside Step 2's output."""
    type_id: str
    canonical_label: str
    definition: str
    child_type_ids: list[str]

@dataclass
class HierarchyInductionResult:
    hierarchy: TypeHierarchy
    synthesized_types: list[SynthesizedType] = field(default_factory=list)
```

**Relation Constraints** (`constraints.py`) — implemented, simpler than the original design:

```python
class ConstraintStrength(Enum):
    HARD = "hard"
    SOFT = "soft"
    HINT = "hint"
    # Thresholds are NOT hardcoded on the enum — they come from ConstraintConfig
    # (hard_threshold=0.90, soft_threshold=0.50, hint_threshold=0.20 by default).

@dataclass
class RelationConstraint:
    relation_id: str
    domain_type_id: str
    range_type_id: str
    support: int              # joint triple count backing this (domain, range) pair
    total: int                # total resolved+direction-corrected triples for this relation
    pca_confidence: float     # support / total
    strength: ConstraintStrength
    # No constraint_id, coverage, source, or llm_evidence fields — those
    # existed in the original design but are not part of the implemented
    # dataclass.
```

**Pipeline result** (`pipeline.py`) — there is no `Ontology` dataclass (that belonged to
Serialization, which is unimplemented — see §9):

```python
@dataclass
class OntoDiscoResult:
    relation_vocab: RelationDeduplicationResult   # subject_types/object_types already resolved to type_ids
    type_vocab: TypeDeduplicationResult
    hierarchy_result: HierarchyInductionResult
    entity_vocab: EntityDeduplicationResult
    # No `constraints` field — induce_constraints() isn't called by run_pipeline() yet.
```

---

## 3. Step 0 — Typed Triplet Extraction

**Status**: Implemented as a single callable method, **not orchestrated** as a corpus-level pipeline
step anywhere in `src/`. There is no chunking module, no `Document`/`Chunk` dataclass, and no
batch-runner that walks a corpus calling this once per chunk with checkpointing. `run_pipeline()`'s
only input is an already-produced JSONL file of raw triplet dicts
(`pipeline.load_triplets(config.input_path)`); whatever produced that JSONL is external to the
orchestrated pipeline today (this repo's `test.ipynb` / ad-hoc scripts call the extractor directly
against a MongoDB-backed corpus, outside of `pipeline.py`).

**Module**: `src/ontodisco/utils/openai_utils.py` — `LLMTripletExtractor.extract_triplets_from_text()`
**Prompt**: `src/ontodisco/utils/prompts/prompt_1_with_types_and_qualifiers.txt`

What the implemented method actually does:
- One `get_completion()` call per input text at `temperature=0` (no explicit `max_tokens` override —
  uses the client/model default), wrapped in a `tenacity` retry (`stop_after_attempt(5)`, random
  exponential backoff on the outer decorator; an inner manual retry counter tracks repair attempts).
- The prompt asks for Wikidata-style extraction: a JSON object with a `"triplets"` list, each item
  `{"subject", "relation", "object", "qualifiers": [...], "subject_type", "object_type"}`. Qualifiers
  are themselves relation/object pairs attached to a triplet (e.g. `{"relation": "point in time",
  "object": "1903"}` on an award-received triplet) — they must always be attached to a triplet, never
  standalone.
- JSON parsing (`extract_json()`) tries a raw `json.loads` first, then a fenced-code-block regex, then
  an inline-brace regex. There is **no separate "repair prompt" call** on parse failure as originally
  designed — a failure that survives all parsing attempts propagates as an exception, appended to the
  system prompt as context on the next retry attempt, and re-raised once attempts are exhausted.
- No `extraction_error` bookkeeping, no per-chunk error-rate threshold, and no `triplet_id`/`doc_id`/
  `evidence_span` fields exist anywhere. Once a triplet dict exists, every downstream step only ever
  reads `subject`, `subject_type`, `relation`, `object`, `object_type` — `qualifiers` is extracted but
  not consumed downstream.

---

## 4. Step 1 — Relation Canonicalization

**Module**: `src/ontodisco/relation_dedup.py` (thin wrapper around the shared
`utils/dedup_base.py::deduplicate()` pipeline)
**Runs FIRST** in `run_pipeline()` — not after type canonicalization, as a naive reading of a typed-KG
pipeline would assume. Reason: Step 2 (type canonicalization) is *always* given the relation
vocabulary as a disambiguating signal in this codebase, so the relation vocabulary must already exist
by the time type dedup runs.

**Input**: `list[dict]` raw triplets (only `relation`, `subject_type`, `object_type` are read)
**Output**: `RelationDeduplicationResult`

1. `collect_relation_surface_forms()` collects the raw `relation` string from every triplet as a plain
   label — **not** a compound "predicate [type]" label. Unlike entity dedup (Step 4), relation surface
   forms are bare strings in the implemented code.
2. Separately, `relation_2_subject_types` / `relation_2_object_types` dicts group each triplet's *raw*
   `subject_type` / `object_type` strings under their (unnormalized) `relation` string. These become
   `CanonicalRelation.subject_types` / `.object_types` once merged, and remain raw type labels until
   the bridge step below runs.
3. **Candidate generation is HDBSCAN** (`dedup_base.cluster_hdbscan`) over the full pairwise
   cosine-distance matrix of Contriever embeddings — `similarity_threshold` (default 0.85 in the
   function signature; `configs/default.yaml` sets 0.75) is converted internally to
   `cluster_selection_epsilon = 1 - similarity_threshold`. This replaced an earlier FAISS
   nearest-neighbor + union-find design (chosen originally for its O(N × top_k) memory footprint at
   scale); the current implementation reintroduces the O(N²) memory cost this vocabulary's ~10k raw
   surface forms make tractable (~400MB), at the benefit of HDBSCAN's density-adaptive clustering
   instead of a flat top-k-limited neighbor search — see Step 2 for the same trade-off and the
   `allow_single_cluster=True` requirement `cluster_hdbscan()` depends on.
4. Every multi-member cluster goes to
   `LLMTripletExtractor.verify_cluster_with_llm(members, surface_form_type='relation')`
   (`prompts/cluster_relations.txt`): merge-all-or-split-into-subgroups, one call per cluster. There is
   **no argument-signature-based forced split** for relations — the `arg_sig_split_threshold` concept
   from the original design is not implemented here; `subject_types`/`object_types` are recorded as a
   byproduct for later use (relation-context profiling in Step 2/3, constraint induction in Step 5),
   not consulted during the relation-merge decision itself.
5. **Bridge — `update_relation_type_map(relation_result, type_result)`**: called from `pipeline.py`
   between Steps 2 and 3, this resolves every relation's `subject_types`/`object_types` from raw
   labels to canonical `type_id`s via `type_vocab.surface_to_id`, mutating `relation_result` in place.
   It is deliberately **not idempotent** and **not checkpointed on its own** — calling it twice on an
   already-resolved `relation_result` would look up type_ids as if they were still raw surface forms
   and silently empty out `subject_types`/`object_types`. `pipeline.py` always re-derives it fresh from
   the raw Step 1 checkpoint plus the Step 2 checkpoint on every invocation (including `--resume`),
   since it's a pure dict lookup with no LLM calls — cheap to redo, dangerous to double-apply.

---

## 5. Step 2 — Type Canonicalization

**Module**: `src/ontodisco/type_dedup.py` (thin wrapper around `dedup_base.deduplicate()`)
**Input**: raw `subject_type`/`object_type` surface forms from every triplet
(`pipeline._collect_type_surface_forms()`), plus — always, in `run_pipeline()` — Step 1's
`RelationDeduplicationResult`
**Output**: `TypeDeduplicationResult`

1. Candidate generation is **HDBSCAN** (`dedup_base.cluster_hdbscan`, an
   `sklearn.cluster.HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1,
   cluster_selection_epsilon=1 - hac_threshold, allow_single_cluster=True)` over Contriever cosine
   distance), cut at `hac_threshold` (default 0.85 in the function signature; `configs/default.yaml`
   sets `type_canonicalization.hac_threshold: 0.75` — name kept for config compatibility with the
   HAC-based implementation this replaced). Type labels are embedded as their bare normalized string —
   there is no definition/example-augmented embedding text as an earlier design proposed.
   `allow_single_cluster=True` is required: without it, HDBSCAN's default behavior rejects a pool with
   no viable sub-split at the very top of its internal hierarchy and marks the *entire* pool as noise
   instead of one cluster — verified empirically against this pipeline's own label-embedding
   distributions, not a hypothetical edge case. Points HDBSCAN leaves unclustered (noise) become their
   own singleton cluster, same as an unmatched label under the earlier HAC cut.
2. **Relation-signature-assisted disambiguation** (`relation_result` is always supplied by
   `run_pipeline()`, never `None`):
   - `cluster_postprocess_fn = relation_context.merge_clusters_by_relation_signature(...)`: **before**
     LLM verification, additionally unions any two HDBSCAN clusters whose *summed* TF-IDF-weighted
     relation-argument profile cosine similarity is `>= relation_signature_merge_threshold` (default
     0.8; `default.yaml`: 0.75). This can only coarsen clusters further, never split them, and
     clusters with no relation evidence at all are left untouched.
   - `member_context`: each candidate label shown to the LLM verifier is annotated with a
     PPMI-weighted `"often appears with: <relation labels>"` string
     (`relation_context.describe_relation_context`, top `relation_context_top_k` = 5 dimensions) —
     concrete distributional evidence beyond the bare label.
3. Every multi-member (post-merge) cluster goes to
   `verify_cluster_with_llm(members, surface_form_type='entity_type')`
   (`prompts/cluster_entity_types.txt`). This prompt enforces an explicit **hyponymy ("kind of") test
   that takes priority over context/relation overlap**: e.g. "documentary film" and "film" must be
   SPLIT even though a documentary film participates in nearly every relation "film" does; only true
   synonyms with *no* hypernym/hyponym relationship (e.g. "movie"/"film") are merged. This guard exists
   because an earlier prompt version silently merged general/specific pairs as synonyms — see
   `tests/unit/test_type_dedup.py::test_hypernym_types_are_not_merged`, driven by real observed pipeline
   output (`onto_artifacts/verified_groups.json`, e.g. "journal"+"academic journal",
   "aircraft"+"military aircraft"). That mistake is unrecoverable once hierarchy induction runs on top
   of an already-flattened vocabulary, so `tests/unit/test_type_dedup.py` also pins the prompt text
   itself against regression.
4. No `min_type_freq` noise-filtering config exists in the implemented code — every observed type
   label becomes (or is merged into) a `CanonicalType`, regardless of how rarely it occurs.

### Planned extension: size-capping + stitching for oversized post-merge clusters

**Status: design only.** `merge_clusters_by_relation_signature` (point 2 above) can only grow a
cluster, never shrink it, and nothing downstream caps the resulting size before it goes to a single
`verify_cluster_with_llm()` call — unlike Step 3 (hierarchy induction), which splits any
`max_batch_size`-exceeding cluster via `_split_oversized` before LLM resolution. The designed fix
generalizes that same pattern to the shared `dedup_base.deduplicate()` pipeline (so it would apply to
relation, type, and entity dedup alike, since all three call it):
- Skip the relation-signature merge step entirely for any HDBSCAN cluster already at or above the size
  cap — it doesn't need rescuing, it's already well-supported by the primary embedding signal, and
  running the merge step on it risks pulling in more members via a noisier signal for no clear benefit.
- Split any cluster that ends up oversized *from merging* (not from HDBSCAN alone) via HAC into
  LLM-batch-sized sub-clusters, exactly as Step 3 already does, and resolve each sub-cluster
  independently.
- Run one additional "stitching" LLM call per split cluster, asking whether any of the
  independently-resolved sub-groups should actually be reunified — since they were only split apart
  because of the size cap, not because HDBSCAN or the relation-signature merge step ever considered
  them unrelated.

---

## 6. Step 3 — Hierarchy Induction

**Module**: `src/ontodisco/hierarchy_induction.py` (relation-argument profiles built by
`src/ontodisco/relation_context.py`)
**Input**: `TypeDeduplicationResult` (T*), and `RelationDeduplicationResult` **after**
`update_relation_type_map()` has resolved its `subject_types`/`object_types` to canonical type_ids —
a hard precondition stated in the module docstring.
**Output**: `HierarchyInductionResult` = `TypeHierarchy` (single-parent forest) + `list[SynthesizedType]`

> **This is a complete redesign relative to an earlier two-signal-ensemble specification**
> (relation-signature-containment Weeds Precision + Chain-of-Layer LLM majority voting, combined via
> fixed weights, thresholded, then Hasse-reduced). None of that ensemble/threshold/reduction machinery
> exists in the implemented code. The actual mechanism is a single **recursive fused-similarity
> clustering** loop, described below.

### The recursive loop

1. **Initial pool**: every canonical type in T* becomes a leaf `_Node`, carrying its PPMI-weighted
   relation-argument profile (`relation_context.build_relation_counts` + `build_type_relation_profiles`,
   `config.relation_signature_weighting` default `"ppmi"`).
2. **Fused similarity**: over the current pool, combine label-embedding cosine similarity (Contriever)
   with relation-profile cosine similarity (`relation_context.cosine_sim_sparse`) as
   `w_embedding * emb_sim + (1 - w_embedding) * rel_sim` (`config.fusion_weight_embedding`, default
   0.5). Pairs where either side has an empty relation profile fall back to pure embedding similarity
   rather than being penalized.
3. **HDBSCAN round cut** (`_hdbscan_round_cut`): `sklearn.cluster.HDBSCAN(metric="precomputed",
   min_cluster_size=config.min_batch_size, min_samples=1, cluster_selection_epsilon=1 -
   hac_threshold_floor, allow_single_cluster=True)` over the fused distance matrix. This replaced an
   earlier HAC-linkage cut at a *percentile of that round's own merge distances*
   (`round_cut_percentile`, now removed from `HierarchyConfig`): instead of one flat cut distance for
   the whole pool, HDBSCAN's stability-based selection can accept a tighter cluster in one part of the
   pool and a looser one in another, within the same round — `config.hac_threshold_floor` (default
   0.35) now only sets a hard similarity *floor* (nodes farther apart than this can never land in the
   same cluster), not the cut itself. Nodes HDBSCAN leaves as noise (label `-1`) simply stay in the
   pool, retried next round — the same outcome a below-floor node had under the old percentile cut.
   `allow_single_cluster=True` is required here for the same reason as Step 2: without it, a pool with
   no viable internal sub-split gets marked entirely as noise instead of one cluster.
4. **Oversized-cluster splitting**: any multi-member cluster larger than `config.max_batch_size`
   (default 15) is still subdivided via HAC (`_split_oversized`, unchanged) on the same fused-similarity
   submatrix, `maxclust` criterion, into the fewest sub-clusters that each fit. This is deliberately
   *not* replaced with HDBSCAN's own `max_cluster_size` parameter: empirically, `max_cluster_size`
   doesn't compose reliably with `cluster_selection_epsilon` + `allow_single_cluster=True` (the epsilon
   merge can override the size cap), and `_split_oversized`'s job — deterministically chop an
   *already-identified* oversized group into LLM-batch-sized pieces — is a narrower, different concern
   from "which nodes belong together" in the first place, so keeping it HAC-based here doesn't
   reintroduce the problems HDBSCAN was adopted to fix elsewhere.
5. **LLM resolution per batch** (`prompts/hierarchy_cluster_action.txt`,
   `LLMTripletExtractor.resolve_hierarchy_cluster()`): for each batch, the LLM partitions it into
   `"merge"` groups (naming a parent — preferring an existing member's label over inventing one — with
   a `confidence` in [0,1]), `"no_parent"` groups (members stay siblings, retried in a later round), or
   `"same_concept"` groups. `same_concept` was added to handle independently-synthesized abstractions
   that turn out to be the identical concept under different wording (e.g. two unrelated batches each
   inventing "entity"): forcing a fake parent/child edge between them (`"merge"`) is wrong, and declining
   via `"no_parent"` previously had no downstream effect at all — a correctly-spotted duplicate just sat
   in the pool forever as two separate nodes. A confirmed `"same_concept"` group is instead collapsed
   immediately via `_collapse_duplicate_nodes` — the same survivor-selection/edge-rewriting logic step 7's
   exact-label pass already used, just invoked directly from batch resolution instead of only from that
   separate whole-pool scan. `same_concept` is restricted to synthesized nodes only (`_Node.is_leaf ==
   False`) — a group naming an original T* leaf type is rejected and logged, since leaf identity was Step
   2's call to make, not this step's to relitigate. `_resolve_cluster_with_llm` also rejects (not
   partially applies) any group whose type_id was already claimed elsewhere in the same LLM response —
   e.g. one group naming a type as a promoted parent while another names it as a `same_concept` member —
   leaving the untouched type_ids in the pool for a later round rather than crashing or silently
   corrupting the hierarchy.
   `config.col_num_votes` (default 1 = no voting) allows repeated independent calls reconciled by
   tally-based majority (`config.col_vote_agreement`, which now tallies `same_concept` groups the same
   way as `merge`), but the shipped config never turns this on.

   **Known gap: multi-level chains landing in one HDBSCAN batch.** If HDBSCAN's fused-similarity cut
   puts a whole chain in one batch together — e.g. `horror_film`, `film`, and `audiovisual_work`, where
   `horror_film IS-A film IS-A audiovisual_work` — the prompt's own rule explicitly forbids the LLM from
   expressing both levels in one response ("if 'film' is promoted to parent one group, it must NOT also
   appear as a 'member' of any other group in this same response"), and `_resolve_cluster_with_llm`'s
   `claimed`-set check enforces that: only one of the two edges can be accepted per response, by design.
   This is *usually* safe, just slower — the rejected level's types stay in the pool, `film` and
   `audiovisual_work`'s unchanged embeddings mean they very likely land in the same batch again next
   round, and the second edge goes through cleanly once there's no competing claim in that later
   response. But the current conflict-resolution rule doesn't actually guarantee this: if the LLM
   disregards the "pick one role" instruction and returns both groups anyway, `claimed` rejects whichever
   group is processed *second*, based purely on the order the groups appear in the LLM's JSON list — not
   on which resolution is safe. If the group naming `audiovisual_work` as `film`'s parent happens to be
   listed (and thus applied) before the group naming `film` as `horror_film`'s parent, `film` gets popped
   from the pool as `audiovisual_work`'s child *before* `horror_film`'s attachment to it is ever applied
   — permanently losing the more specific level, not just delaying it, since `_materialize_parent`'s
   "parent no longer in pool" guard then silently drops the `horror_film` group instead of retrying it
   against the (now-gone) `film` node. **The needed fix** (not yet implemented): when `claimed` detects
   this specific pattern — a type_id used as a promoted *parent* in one group and as a *member* in
   another, within the same response — resolve the conflict by always keeping the group where it's the
   *parent* (letting it absorb its child first) and deferring the group where it's a *member*, rather
   than resolving by list order. This is the same "receiving before promotion" principle the planned
   Weeds-precision pairwise track uses (see below), applied here to a single response's internal
   conflict-resolution instead of across pairwise calls.
6. **Parent materialization**: a promoted-existing-member parent has its profile updated in place
   (`_merge_profiles`); an invented parent becomes a new synthetic node (`type_hNNNN` id), recorded as a
   `SynthesizedType`, and re-enters the pool for later rounds. Either way its children leave the pool
   and one `HierarchyEdge` per child is appended (`relation_signature_score` = Weeds
   precision(child, parent), purely informational; `llm_score`/`ensemble_score` = the LLM's own
   confidence).
7. **Duplicate-label reconciliation** (every round, regardless of whether steps 3–6 found anything):
   independent LLM batches routinely reinvent the same abstract parent label (e.g. five different
   batches each proposing "entity"). This pass finds SYNTHESIZED nodes (never original T* leaf types —
   that split was Step 2's call to make) sharing a normalized label, auto-merges pairs whose relation
   profiles agree (`cosine_sim_sparse >= config.duplicate_label_profile_merge_floor`, default 0.3 —
   deliberately **one-directional**: high similarity confirms a merge, but low/zero similarity does
   *not* confirm a split, since large rolled-up abstract profiles are often too sparse for cosine to be
   a reliable negative signal), and batches the rest into one LLM disambiguation call
   (`prompts/hierarchy_label_disambiguation.txt`, `config.duplicate_label_llm_batch_size` groups per
   call).
8. **Stopping rule**: the loop runs until `config.max_depth` (default 8; `default.yaml`: 15) rounds, or
   the pool drops below 2, or `config.max_consecutive_stall_rounds` (default 2) consecutive rounds make
   no progress from *either* the HDBSCAN-cut mechanism or duplicate-label reconciliation.
9. **Root-stitching pass**: after the main loop, a few more rounds
   (`config.root_stitch_max_rounds`, default 3) run scoped to just the leftover pool (now small), with
   a relaxed similarity floor (`config.root_stitch_threshold_floor`, default 0.15) — this catches
   near-duplicate roots differing only in wording (e.g. "geographic entity" vs. "geographical entity").

Because a parent is only ever formed from nodes already in the *previous* round's pool, the result is
**acyclic and non-transitive by construction**: `HierarchyEdge.is_direct` is always `True`, and there
is no Hasse-diagram reduction step (nothing to reduce). `TypeHierarchy.parents[child]` always has
length ≤ 1 — this is a **single-parent forest**, not the general multi-parent DAG the original design
allowed for.

### Planned extension: independent Weeds-precision pairwise track, replacing fused clustering

**Status: design only** — nothing below this point exists in `hierarchy_induction.py` yet, and it
changes step 2 above once built: **candidate clustering reverts to pure label-embedding cosine
similarity, dropping the profile-cosine fusion.** Once relation-signature evidence has its own
dedicated discovery mechanism (below), folding it into the clustering metric too is redundant — and it
reintroduces the exact problem already rejected earlier in this design's history: cosine and PPMI
profile similarity are different-natured signals (different dimensionality, different concentration-
of-measure behavior, different sparsity characteristics), and blending them into one metric via a fixed
weight (`fusion_weight_embedding`) either hides that incompatibility inside an unprincipled scale
hyperparameter or lets one signal dilute the other exactly where it's strongest. Splitting them into two
independent tracks — pure embedding clustering, and a separate asymmetric Weeds-precision mechanism —
avoids that trade-off entirely instead of just relocating it.

Why a dedicated relation-signature mechanism is needed at all: `HierarchyEdge.relation_signature_score`
(Weeds precision) is currently computed only as post-hoc metadata on an edge the clustering step already
produced (step 6) — never as an independent way to propose one. This misses genuine IS-A pairs that are
lexically distant but relationally contained: once a broad candidate parent has absorbed several other
children, its embedding similarity to a lexically-distant true child stays low no matter how the
clustering metric is weighted, even though the child's relations remain a clean subset of the parent's —
exactly the asymmetric-vs-symmetric distinction Weeds precision exists to capture (e.g. "film" vs. a
broader "audiovisual work" that has already absorbed several unrelated siblings like "song" or
"podcast").

The designed mechanism:
- **Both-direction Weeds precision as a 3-way router.** For a candidate pair (A, B), compute
  `weeds_precision(A,B)` and `weeds_precision(B,A)`. (high, low) or (low, high) → asymmetric
  containment → propose as a `"merge"` candidate, direction given by which side is high. (high, high) →
  propose as a `"same_concept"` candidate — the same signature a true synonym pair already produces,
  generalized to pairs that don't happen to share a label at all. (low, low) → not enough evidence,
  propose nothing.
- **Cheap prefiltering.** An inverted index (`relation_id -> [type_ids with nonzero weight]`) avoids a
  full O(n²) scan — only pairs sharing at least one relation dimension are ever compared, since
  disjoint profiles can't have containment.
- **Bottom-up scheduling by candidate in-degree.** Order proposed-parent candidates by how many other
  pool nodes currently propose them as parent (ascending). Few proposers suggests a specific candidate
  (e.g. "film" — a handful of genuine subtypes); many proposers suggests a generic one (e.g. "entity" —
  broad relation coverage subsumes almost everything). Resolving low-in-degree candidates first, in
  tiers (all current lowest-in-degree pairs processed together each round, not one at a time),
  mirrors "attach to the most specific plausible parent" and avoids routing something through an overly
  generic candidate before a more specific one has had a chance to claim it.
- **Genuinely pairwise LLM calls, not multi-candidate batches.** Each proposed pair gets its own
  independent LLM call (a 4-option question: synonymous / A-is-parent-of-B / B-is-parent-of-A / none)
  rather than bundling several candidates for the same proposed parent into one multi-group response.
  This was chosen specifically to avoid a same-response internal-consistency failure mode a batched
  design has: if one candidate for a proposed parent P turns out to be a synonym of P itself while
  others are genuinely P's children, a single JSON response naming P as both a reused parent and a
  `same_concept` member is self-contradictory (exactly the case step 5's `claimed`-set guard has to
  reject). Independent pairwise calls decompose the same multi-way relationship into separately-decided
  edges/collapses that get composed against `pool` afterward — "P parents A and B" and "P is the same
  concept as C" are both simultaneously valid outcomes once decomposed this way, with no single-response
  consistency constraint blocking that valid combination. Conflicts that remain (e.g. a stale type_id
  already popped by an earlier decision applied this same round) are handled by the same defensive
  `pool.get()`/already-resolved checks the batch path already uses, not a new mechanism.
- **Stall-before-promotion ordering.** `_merge_profiles` only ever adds non-negative PPMI-weighted
  values, so a parent's coverage of any relation dimension is monotonically non-decreasing as it absorbs
  more children — which means `weeds_precision(fixed_child, growing_parent)` can only increase, never
  decrease, round over round. Consequently, a node must complete one full round *gaining no new
  children* before it becomes eligible to be promoted as someone else's child. Without this rule, a
  borderline sibling (e.g. "horror film," whose score against "film" only clears threshold once "film"
  has absorbed a couple of other children first) can permanently lose its chance to attach to the
  correct, more specific parent if that parent is promoted upward and popped from the pool one round
  too early. Because of the monotonicity above, "wait for one stall round, then promote" is not an
  arbitrary safety margin — no future round can make a currently-non-qualifying candidate newly qualify
  against a parent that just went a full round gaining nothing, since further qualification can only
  come from further absorption.
- **Within a round, apply "receives a child" decisions before checking "becomes a child" eligibility —
  never the reverse, and never by arbitrary decision order.** This is the concrete mechanism that makes
  stall-before-promotion actually hold, not just a restated intent. Worked example: suppose in-degree
  ties put `(horror_film, film)` and `(film, audiovisual_work)` in the *same* round's tier — a real
  possibility the in-degree heuristic reduces but doesn't rule out. Both pairwise calls can go out
  independently and both can come back confirmed. Applying them requires a fixed two-pass order: first,
  apply every decision where a node *receives* a new child this round (`film` absorbs `horror_film`,
  its profile updated, `film` itself still in the pool). Only then check which nodes are even eligible
  to be applied as someone else's child this round — and `film`, having just received a child in the
  pass that only completed a moment ago, fails the stall-before-promotion check for *this* round by
  construction. So `(film, audiovisual_work)` is not applied now; it's deferred to the round after
  `film` goes one full round absorbing nothing. The pairwise call's "yes" answer isn't wrong or
  discarded — it's simply not actionable yet. Getting the pass order backwards (checking "becomes a
  child" eligibility before all "receives a child" decisions for this round are known) is exactly how
  the bad outcome happens: `film` gets popped as `audiovisual_work`'s child while `horror_film`'s
  decision is still pending, and when that pending decision is finally applied, `film` is no longer in
  `pool` — `horror_film` permanently loses the correct, more specific attachment rather than merely
  waiting for it.
- **Two sequential phases, not an interleaved round loop.** The Weeds-precision pairwise track is not
  one step folded into the existing round loop — it *is* its own complete phase, run first: repeat
  its own in-degree-tiered rounds (each respecting stall-before-promotion) until no pair anywhere in
  the pool clears the containment/synonym thresholds any more, i.e. until it reaches its own fixed
  point. Only once that phase is fully exhausted does the pool hand off to phase two — the existing
  embedding-only clustering round loop (steps 2–9 above, minus the profile fusion) — which then runs
  on whatever the pairwise phase left behind. Embedding-based clustering never runs concurrently with,
  or interleaved round-by-round against, the pairwise phase; it only ever sees the pairwise phase's
  final, fully-settled output pool.

The two tracks remain complementary even though they run as separate phases rather than fused into one
metric: the pairwise Weeds-precision phase catches relationally-linked pairs regardless of how
lexically distant they are (its whole purpose); the embedding-clustering phase that follows catches
whatever's left — pairs that are lexically close but whose relation profiles were too thin, noisy, or
disjoint for Weeds precision to ever propose. Running Weeds-precision to exhaustion *first* means the
pool handed to embedding clustering is already maximally reduced and rolled-up, so embedding clustering
only ever has to resolve what relational evidence genuinely couldn't.

---

## 7. Step 4 — Entity Deduplication & Class Assignment

*(Absent from the original design entirely — added because class assignment has to happen somewhere,
and doing it as a byproduct of entity-name clustering avoids a separate assignment pass.)*

**Module**: `src/ontodisco/entity_dedup.py`
**Runs LAST** in `run_pipeline()` — needs both the canonical type_ids from Step 2 and the induced
`TypeHierarchy` from Step 3.
**Input**: raw triplets, `TypeDeduplicationResult`, `TypeHierarchy`
**Output**: `EntityDeduplicationResult`

- Every entity mention becomes a compound label `"name [canonical type label]"`
  (`collect_entity_surface_forms`) — entity identity is `(name, type)`, not just name, so homonymous
  entities with different types (`"Paris [city]"` vs. `"Paris [person]"`) are never merged, by
  construction. Mentions whose type didn't resolve to a known `type_id` are skipped (the entity may
  still canonicalize via its other, resolvable mentions).
- Candidate generation partitions compound-label embeddings by `_parent_key` FIRST, then runs HDBSCAN
  (`dedup_base.cluster_hdbscan`) independently *within* each partition (`_cluster_hdbscan_with_parent_gate`)
  — the **hard, symbolic gate** is the partition itself: two candidates are only ever compared if
  `_parent_key` agrees for both — i.e. their types are literally identical, or they are siblings under
  the exact same immediate parent in `TypeHierarchy`. No other relatedness counts (not
  grandparent/grandchild, not "any shared ancestor"). This replaced an earlier FAISS-NN + post-hoc gate
  design (retrieve top-k neighbors from the full index, then filter pairs by parent match); partitioning
  first is equivalent — candidates in different partitions could never have merged either way — but
  cheaper (HDBSCAN only ever compares candidates that could possibly merge) and avoids materializing one
  global distance matrix across the whole entity vocabulary. Both designs avoid the recall loss of an
  even earlier KMeans hard-blocking approach, which could silently drop true synonym pairs that happened
  to land in different embedding-space blocks — partitioning by exact hierarchy parentage is a symbolic,
  exact gate, not an approximate embedding-space one.
- Multi-member candidate clusters are LLM-verified (`prompts/cluster_entity_names.txt`,
  `surface_form_type='entity'`), with an explicit precision-over-recall instruction ("when in doubt,
  keep entities SEPARATE").
- **Class assignment falls out of clustering for free**: `CanonicalEntity.type_ids` is simply the
  union of canonical type_ids across all mentions merged into that entity — there is no separate
  assignment pass, and no `entity_to_type_ids` map on the type vocabulary (that direction lives on the
  entity object, not the type object).
- `tests/unit/test_entity_dedup.py` pins the parent-gate behavior directly: same-name mentions typed as
  parent/child (e.g. "film" vs. "documentary film") are *not* merged even though their name embeddings
  are near-identical; same-name mentions typed as siblings under one parent *are* merged, and the
  resulting entity's `type_ids` is the union of both; same-name mentions under two disjoint hierarchy
  roots (e.g. "Amazon" typed "ocean current" vs. "film") are *not* merged.

---

## 8. Step 5 — Domain/Range Constraint Induction

**Status**: Fully implemented and self-consistent, but **not yet called by `run_pipeline()`** — it is
not part of `OntoDiscoResult`. `induce_constraints()` can be invoked standalone once the four
orchestrated steps' outputs exist.

**Module**: `src/ontodisco/constraints.py`
**Input**: `List[dict]` (the raw triplet dicts loaded by `pipeline.load_triplets` — there is no
`CanonicalTriplet`/entity-level canonicalization stage in this codebase; `subject`/`object` stay raw
surface strings throughout), `TypeDeduplicationResult`, `RelationDeduplicationResult` (with
`subject_types`/`object_types` already resolved to type_ids via
`relation_dedup.update_relation_type_map()`), `TypeHierarchy`, `LLMTripletExtractor`
**Output**: `List[RelationConstraint]`

### Specification

Nothing upstream of this step retains the *joint* per-triple `(subject_type, object_type)` pairing
for a relation — `CanonicalRelation.subject_types`/`object_types` are marginal sets, not paired
counts. So this step starts by re-resolving the raw triplets directly.

#### Stage 0: Re-resolve raw triplets

For each raw triplet dict, look up `subject_type_id`/`object_type_id` via
`type_vocab.surface_to_id[normalize_label(subject_type)]` and `relation_id` via
`relation_vocab.surface_to_id[relation]` (same fallback pattern as
`relation_dedup._resolve_type_ids`). Triplets that fail to resolve are dropped and logged; this
also gives, per triplet, the *raw predicate string* actually used (needed for Stage 1).

#### Stage 1: Relation-direction resolution

Relation canonicalization (Step 1) merges surface forms role-blind, so a canonical relation can
silently pool a predicate with its grammatical inverse — e.g. `"directed"` and `"was directed by"`
both resolve to the same `relation_id`, but with subject/object swapped between them. Left
uncorrected, this fragments the domain/range support for a perfectly well-typed relation into two
competing, individually-weak orientations (pooling 3× forward + 2× reverse "directed"/"was directed
by" triples gives only 60%/40% support for either domain choice, when the real signal is 100% support
once direction is normalized).

Statistical detection of this (comparing observed subject/object type pairs) does **not** work in
general: subject-side and object-side types always share *some* common ancestor eventually (the
hierarchy forest has few roots), and plenty of genuinely directional relations (`influenced`,
`reports to`) have identical domain/range types to begin with — so type overlap can't distinguish
"merged inverse" from "genuinely same-type-both-sides." Direction is resolved from the one signal
that actually carries it: the literal surface form.

For every canonical relation with more than one surface form (single-surface-form relations are
trivially "forward", no LLM call needed), one LLM call classifies each surface form in
`CanonicalRelation.surface_forms` against the canonical label as:
- `"forward"` — same subject/object roles as the canonical label.
- `"inverted"` — subject/object swapped relative to the canonical label.

See `prompts/relation_direction.txt` / `LLMTripletExtractor.classify_relation_direction()`.

Triplets whose original predicate was classified `"inverted"` have their `subject_type_id` and
`object_type_id` swapped before Stage 2.

**Known simplification**: this catches merged grammatical inverses (e.g. "directed" / "was
directed by"), but not genuinely order-free relations (e.g. "collaborated with") where a single
surface form gets extracted with subject/object swapped across sentences purely by chance — domain
and range are always walked independently (see Stage 2), so such a relation may end up with a
narrower or oddly-split domain/range instead of one unified type. Deliberately not handled: an
earlier design considered an explicit "symmetric" classification that pooled subject-role and
object-role types together, but domain-LCA ≠ range-LCA is *also* the expected, correct signature
of a genuinely asymmetric relation (e.g. "directed": domain=person, range=film don't converge
either) — so this can't be inferred from the LCA computation itself, only from a priori knowledge
that the relation is order-free, and the added classification complexity wasn't judged worth it.

#### Stage 2: Hierarchy generalization (exact LCA, no threshold)

For a relation's domain (subject-role types) and range (object-role types), walked independently,
build the set of observed type_ids and compute their **lowest common ancestor** in the
single-parent `TypeHierarchy` forest (`hierarchy.parents[child_type_id] -> [parent_type_id]`):

1. For each observed type, its ancestor path is the leaf-to-root chain obtained by repeatedly
   following `hierarchy.parents[t][0]` until a node in `hierarchy.roots` is reached.
2. The LCA is the node common to every observed type's ancestor path that is closest to the
   leaves (deepest / most specific).
3. If no such node exists (the observed types span disconnected trees of the hierarchy forest —
   the relation genuinely connects unrelated type families), partition the observed types by which
   root each descends from, and compute the LCA independently *within* each partition. This yields
   more than one signature type for that role — each becomes a separate `RelationConstraint` row
   (the dataclass already allows several constraints per relation).

No pruning or support-based filtering happens at this stage — every observed type, however rare,
is folded into whichever signature (partition) it belongs to. There is no new config value here;
this is a deterministic graph computation, not a statistical decision.

#### Stage 3: Confidence scoring (reuses existing thresholds only)

Regroup the (direction-corrected) triplets by their resolved `(domain_signature, range_signature)`
pair (looked up per-triplet via the Stage-2 mapping, not a cross-product of the marginal
signatures — this preserves the true joint support) and count:

```
support(r, T_domain, T_range) = |{ triplets of r whose (domain_signature, range_signature) == (T_domain, T_range) }|
pca_conf(r, T_domain, T_range) = support(r, T_domain, T_range) / total_triples(r)
```

This is the same PCA-confidence formula as before — only the numerator/denominator now come from
direction-corrected, hierarchy-generalized signatures instead of raw leaf types. Strength
assignment (`constraints._assign_strength`):

```python
def _assign_strength(pca_confidence: float, config: ConstraintConfig) -> Optional[ConstraintStrength]:
    if pca_confidence >= config.hard_threshold: return ConstraintStrength.HARD
    if pca_confidence >= config.soft_threshold: return ConstraintStrength.SOFT
    if pca_confidence >= config.hint_threshold: return ConstraintStrength.HINT
    return None  # discard; too weak to be useful
```

`config.hard_threshold` / `soft_threshold` / `hint_threshold` (`ConstraintConfig`, defaults 0.90 /
0.50 / 0.20) are the *only* tunables this whole step uses — applied once, at the very end, purely to
label an already-fully-determined signature type. There is no `min_support` or LLM fill-in pass:
sparsity is handled structurally by Stage 2's generalization (a rare leaf type gets folded into a
well-attested ancestor automatically).

```python
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

    Returns a list of RelationConstraints. Multiple constraints per relation
    are allowed (e.g. one per disconnected hierarchy partition, or a
    genuinely disjunctive domain/range).
    """
```

---

## 9. Step 6 — Ontology Serialization

> **STATUS: NOT IMPLEMENTED.** No `src/ontodisco/serialization.py` file exists in this codebase.
> Everything below is a **design target**, kept so the eventual implementation has a spec to build
> against — do not assume any of it exists, and do not point other code at it.

**Planned module**: `src/ontodisco/serialization.py`
**Planned input**: an assembled ontology object (Steps 1–5's outputs; there is no `Ontology`
dataclass yet — see §2)
**Planned output**: three files written to `config.output_dir`:
  - `ontology.ttl` — OWL 2 ontology in Turtle format
  - `shapes.ttl` — SHACL shapes (Core for HARD/SOFT constraints)
  - `schema.jsonld` — Lightweight JSON-LD schema for non-Semantic-Web consumers

### Design sketch

**`ontology.ttl`** format:
```turtle
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix od:   <https://ontodisco.example.org/ontology/> .

# For each CanonicalType:
od:type_0017 a owl:Class ;
    rdfs:label "biographical film"@en .
    # NOTE: CanonicalType has no `definition` field today — an rdfs:comment
    # would need that field added upstream first, or to be generated fresh
    # at serialization time.

# For each HierarchyEdge (all edges are is_direct=True today — see §6):
od:type_0017 rdfs:subClassOf od:type_0003 .   # biographical film ⊆ film

# For each CanonicalRelation, using constraints.py's RelationConstraint if available:
od:rel_0008 a owl:ObjectProperty ;
    rdfs:label "directed"@en ;
    rdfs:domain od:type_0022 ;    # from the HARD/SOFT RelationConstraint, if one exists
    rdfs:range  od:type_0003 .
```

**`shapes.ttl`** format (SHACL Core):
```turtle
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix od:   <https://ontodisco.example.org/ontology/> .

# One NodeShape per CanonicalRelation with HARD or SOFT constraints:
od:shape_rel_0008 a sh:NodeShape ;
    sh:targetSubjectsOf od:rel_0008 ;
    sh:property [
        sh:path od:rel_0008 ;
        sh:class od:type_0003 ;   # range constraint
        sh:severity sh:Violation  # HARD → Violation; SOFT → Warning
    ] .
```

**`schema.jsonld`** (compact, for application developers):
```json
{
  "@context": { "od": "https://ontodisco.example.org/ontology/" },
  "types": [
    { "id": "od:type_0017", "label": "biographical film", "subClassOf": "od:type_0003" }
  ],
  "relations": [
    {
      "id": "od:rel_0008", "label": "directed",
      "domain": "od:type_0022", "range": "od:type_0003",
      "pcaConfidence": 0.94, "strength": "hard"
    }
  ]
}
```

Whenever this step is built, it should be added to `run_pipeline()` as a fifth checkpointed step
(after entity dedup), consuming `OntoDiscoResult` plus a freshly-run `induce_constraints()` call.

---

## 10. Pipeline Orchestration

**Module**: `src/ontodisco/pipeline.py`

`run_pipeline(config: PipelineConfig, *, resume: bool = False) -> OntoDiscoResult` runs the four
orchestrated steps in order — Relation Canonicalization → Type Canonicalization → Hierarchy Induction
→ Entity Deduplication — with a bridge call (`update_relation_type_map()`) between Steps 2 and 3 (see
§4). It reads a JSONL file of pre-extracted triplets (`config.input_path`, loaded via
`load_triplets()`); it does **not** take `documents`/chunks, and does not itself run extraction,
constraint induction, or serialization.

```python
@dataclass
class OntoDiscoResult:
    relation_vocab: RelationDeduplicationResult
    type_vocab: TypeDeduplicationResult
    hierarchy_result: HierarchyInductionResult
    entity_vocab: EntityDeduplicationResult
```

### Checkpointing

Each of the four steps is checkpointed via `pickle` to
`output_dir/checkpoints/run_<n>/<step_name>.pkl`, where `step_name` is one of `relation_dedup`,
`type_dedup`, `hierarchy_induction`, `entity_dedup` (`pipeline.STEP_NAMES`). The bridge
(`update_relation_type_map`) is **deliberately never checkpointed on its own** — see §4 for why
(it's not idempotent, and re-deriving it from the raw checkpoints is free since it makes no LLM
calls).

Every invocation gets its own `run_<n>` folder, auto-numbered:
- `resume=False` (a fresh run) always allocates a **new** `run_<n>` folder — one past the highest
  existing run number under `output_dir/checkpoints` (`run_1` if none exist yet). Fresh runs never
  clobber a previous run's checkpoints.
- `resume=True` continues the **most recent** existing `run_<n>` folder (or creates `run_1` if none
  exists), so a crash mid-pipeline never forces re-paying for already-completed, LLM-backed work, and
  never silently starts a fresh run number instead of picking up where it left off.

Each run folder also gets its own `run_metadata.json` — see §13 for its schema.

### CLI usage

There is no `python -m ontodisco run` entry point (no `src/ontodisco/__main__.py` exists). The actual
CLI is `pipeline.py`'s own `if __name__ == "__main__":` block, invoked as a module from the repo root:

```bash
# Full run
python -m src.ontodisco.pipeline --config configs/default.yaml

# Resume after crash (continues the most recent run_<n> folder)
python -m src.ontodisco.pipeline --config configs/default.yaml --resume
```

There is no `--steps` flag for ablation-style partial runs — all four steps always run (subject to
per-step checkpoint skipping under `--resume`).

---

## 11. Configuration Reference

All configuration is a single YAML file loaded into `PipelineConfig` via `pipeline.load_config()`.
Unknown keys in the YAML are ignored with a logged warning (`_from_dict()`); missing keys fall back to
the dataclass's own default. There is no `pipeline_version` or top-level version field read by the
loader (it's accepted by neither `PipelineConfig` nor `load_config()` — any such key in the YAML would
just be a silently-ignored unknown key today).

```yaml
# configs/default.yaml — mirrors the actually-loaded PipelineConfig fields

corpus_id: "my_corpus"
input_path: "./data/musique_initial_triplets.jsonl"   # JSONL, one raw triplet dict per line
output_dir: "./output"

# ── LLM client (LLMConfig) ────────────────────────────────────────────────────
llm:
  model: "gpt-4o-mini"                          # dataclass default; default.yaml overrides per-run
  api_key_env: "OPENAI_API_KEY"                 # name of the env var holding the key — never the key itself
  base_url: "https://api.openai.com/v1"

# ── Embedding client (EmbeddingConfig) ────────────────────────────────────────
embedding:
  contriever_model: "facebook/contriever"
  device: null                                  # "cuda" | "cpu" | null (auto)
  embed_batch_size: 64

# ── Step 1: Relation Canonicalization (RelationCanonicalizationConfig) ────────
relation_canonicalization:
  similarity_threshold: 0.75    # HDBSCAN cluster_selection_epsilon = 1 - this (dataclass default: 0.85)

# ── Step 2: Type Canonicalization (TypeCanonicalizationConfig) ───────────────
type_canonicalization:
  hac_threshold: 0.75                            # HDBSCAN cluster_selection_epsilon = 1 - this
                                                  # (dataclass default: 0.85; name kept for compatibility
                                                  # with the HAC-based implementation this replaced)
  relation_signature_merge_threshold: 0.75       # dataclass default: 0.8
  relation_context_top_k: 5

# ── Step 3: Hierarchy Induction (HierarchyConfig, defined in hierarchy_induction.py) ──
hierarchy:
  fusion_weight_embedding: 0.5      # weight on label-embedding cosine vs. relation-profile cosine
  relation_signature_weighting: "ppmi"   # "ppmi" | "tfidf" | "raw"
  hac_threshold_floor: 0.35         # HDBSCAN cluster_selection_epsilon = 1 - this, per round -- a
                                     # similarity FLOOR, not a fixed per-round cut (HDBSCAN picks the
                                     # cut itself within that floor; round_cut_percentile, used by the
                                     # earlier HAC-based cut, no longer exists)
  max_depth: 15                     # safety cap on number of rounds (dataclass default: 8)
  max_batch_size: 15                # max cluster size sent to the LLM at once (re-split via HAC if
                                     # exceeded -- _split_oversized, kept as-is, see Step 3)
  min_batch_size: 2                 # minimum cluster size worth sending to the LLM (== HDBSCAN's
                                     # min_cluster_size)
  embed_batch_size: 64
  context_top_k: 5                  # top relation dimensions shown per candidate
  col_num_votes: 1                  # independent LLM calls per batch; 1 = no voting
  col_vote_agreement: 1             # min votes required to accept a merge (only if col_num_votes > 1)
  # Not set in default.yaml (dataclass defaults apply):
  #   duplicate_label_profile_merge_floor: 0.3
  #   duplicate_label_llm_batch_size: 30
  #   max_consecutive_stall_rounds: 2
  #   root_stitch_threshold_floor: 0.15
  #   root_stitch_max_rounds: 3

# ── Step 4: Entity Deduplication + Class Assignment (EntityCanonicalizationConfig) ──
entity_canonicalization:
  similarity_threshold: 0.85   # HDBSCAN cluster_selection_epsilon = 1 - this, applied within each
                                # parent-key partition (see Step 4)
```

**Not part of `PipelineConfig` / `default.yaml`** (exist as standalone dataclasses in their own
modules, since the steps that use them aren't wired into `run_pipeline()` yet):
- `constraints.ConstraintConfig` (`hard_threshold`, `soft_threshold`, `hint_threshold`) — see §8.
- A `SerializationConfig` does not exist at all — see §9.

There is also no `chunking:` or `extraction:` config section — Step 0 isn't corpus-orchestrated (§3),
so there is no chunk-size/tokenizer/backend configuration to speak of; `LLMTripletExtractor` is
constructed directly with `api_key`/`model`/`base_url` wherever it's used.

**All four canonicalization/hierarchy steps now cluster via HDBSCAN** (`dedup_base.cluster_hdbscan`,
shared by Steps 1, 2, and 4; a bespoke variant in `hierarchy_induction._hdbscan_round_cut` for Step 3,
since it clusters a precomputed *fused* similarity matrix rather than raw embeddings) — this replaced
the original design's HAC (Step 2) and FAISS-NN + union-find (Steps 1, 4) candidate generation.
`allow_single_cluster=True` is load-bearing everywhere it's used: verified empirically that without it,
HDBSCAN's default behavior can silently mark an entire well-formed, obviously-single cluster as noise
whenever there's no viable internal sub-split, rather than returning it as one cluster. The trade-off
accepted for this rewrite: Steps 1 and 4 (relation and entity dedup) used FAISS specifically for its
O(N × top_k) memory footprint at scale; clustering the full pairwise cosine-distance matrix via HDBSCAN
reintroduces O(N²) memory (workable at this corpus's actual scale — ~10k raw relation surface forms —
but a real regression for much larger vocabularies). Step 4 mitigates this somewhat by partitioning on
the parent-key gate *before* running HDBSCAN, so each HDBSCAN call only ever sees one hierarchy
sibling-group's worth of candidates, not the whole entity vocabulary at once.

---

## 12. Testing Requirements

**Current state**: only `tests/unit/` exists, with three files —
`test_type_dedup.py`, `test_entity_dedup.py`, `test_logging_config.py`. There is no
`tests/integration/`, `tests/e2e/`, `tests/regression/`, or `tests/fixtures/` directory yet. All
existing tests mock the LLM verifier and the embedder (a small `_FakeEmbedder`/`_FakeLLMVerifier`
pair per test file, keyed on deterministic label→vector lookups) — no test makes a real API call or
downloads a real Contriever checkpoint.

What's actually covered today:
- `test_type_dedup.py` — pins the hypernym/hyponym merge guard in `cluster_entity_types.txt` (kind-of
  pairs must split even when HDBSCAN clusters them together; true synonyms must still merge), plus a
  regression test on the prompt text itself.
- `test_entity_dedup.py` — pins the parent-gate in `entity_dedup.py`: same-name mentions typed as
  parent/child are not merged; siblings under one parent are merged (and their `type_ids` union
  correctly); mentions under disjoint hierarchy roots are not merged.
- `test_logging_config.py` — pins `ONTODISCO_LOG_LEVEL`/`ONTODISCO_LIB_LOG_LEVEL` behavior in
  `utils/logging_config.py`.

There is currently no test coverage for `relation_dedup.py`, `hierarchy_induction.py`,
`constraints.py`, or `pipeline.py` itself. Target categories for future work (not yet built):

| Category | Location | Notes |
|---|---|---|
| Unit tests (per module) | `tests/unit/` | Extend to relation_dedup, hierarchy_induction, constraints, pipeline |
| Integration tests (step → step) | `tests/integration/` (does not exist yet) | E.g. relation_dedup → update_relation_type_map → hierarchy_induction |
| End-to-end (`run_pipeline()` on a tiny corpus) | `tests/e2e/` (does not exist yet) | Real checkpointing/resume behavior |
| Regression (output snapshots) | `tests/regression/` (does not exist yet) | Guard against silent hierarchy/constraint drift |

```bash
# What actually runs today:
pytest tests/unit -v
```

---

## 13. Logging and Reproducibility

There is **no** `output_dir/llm_calls.jsonl` or any other per-call log of full prompt+response text.
Logging is standard Python `logging`, configured once by
`src/ontodisco/utils/logging_config.py::configure_logging()`:
- `ONTODISCO_LOG_LEVEL` (default `INFO`) controls the level for everything under the `src.ontodisco`
  logger namespace.
- `ONTODISCO_LIB_LOG_LEVEL` (default `WARNING`) caps a fixed list of noisy third-party loggers
  (`pymongo`, `urllib3`, `httpcore`, `httpx`, `huggingface_hub`, `transformers`, `filelock`,
  `matplotlib`, `fsspec`, `asyncio`, `openai`, `charset_normalizer`) so `DEBUG` on project code doesn't
  flood the console with library noise.

**Token/cost accounting**: `LLMTripletExtractor` accumulates `prompt_tokens_num` /
`completion_tokens_num` / `current_cost` across every `get_completion()` call, readable via
`calculate_used_tokens()` / `calculate_cost()`. `dedup_base.verify_clusters_with_llm()` periodically
logs a running token total every `token_log_every` clusters (default 10) during a dedup step's LLM
verification loop. None of this is written to a file automatically — it's log-line-only unless a
caller explicitly persists it.

**`run_metadata.json`** — written by `pipeline._write_run_metadata()` to
`output_dir/checkpoints/run_<n>/run_metadata.json` after every `run_pipeline()` call:

```json
{
  "run_id": "run_3",
  "start_time": "...",
  "end_time": "...",
  "corpus_id": "my_corpus",
  "num_triplets": 38492,
  "num_canonical_relations": 89,
  "num_canonical_types": 312,
  "num_synthesized_types": 41,
  "num_hierarchy_edges": 271,
  "num_hierarchy_roots": 6,
  "num_canonical_entities": 8124,
  "step_durations_seconds": {
    "relation_dedup": 54.1,
    "type_dedup": 87.2,
    "hierarchy_induction": 341.0,
    "entity_dedup": 213.4
  },
  "config": { "...": "asdict(config) -- full resolved PipelineConfig, api_key_env only records which env var was read, never the key value" }
}
```

There is no `config_hash`, no `total_llm_tokens` field, and no `pipeline_version` field in this
metadata — all three existed in the original design but are not written by the implemented code.

---

## 14. Known Limitations and Design Rationale

### L1: Relation canonicalization has no argument-signature-based forced split
Unlike type/entity dedup, `relation_dedup.py` doesn't consult `subject_types`/`object_types` when
deciding whether to merge two predicate surface forms — that evidence is recorded but only consumed
downstream (relation-context profiling, constraint induction). A genuinely polysemous predicate label
(two unrelated relations that happen to share a surface form) can only be split by the LLM
verification call itself, with no symbolic argument-signature gate backing it up the way Step 2/4 have
one for types/entities.

### L2: Single-parent forest, not a multi-parent DAG
`hierarchy_induction.py`'s recursive design guarantees `TypeHierarchy.parents[child]` has length ≤ 1.
A type that genuinely belongs under two unrelated parents (e.g. "amphibious vehicle" under both
"vehicle" and "watercraft") cannot be represented — it gets assigned to whichever parent its first
accepted merge happens to produce. This is a structural consequence of the "parent only ever formed
from the previous round's pool" construction that also gives acyclicity and non-transitivity for free
(see §6) — the two properties are the same trade-off, not independent decisions.

### L3: HDBSCAN candidate clustering trades FAISS's scalability for adaptive quality
Steps 1, 2, and 4 all cluster via `dedup_base.cluster_hdbscan()`, which runs HDBSCAN over the full
pairwise cosine-distance matrix rather than a FAISS top-k neighbor search. This is workable at this
corpus's actual scale (Step 1's relation vocabulary is ~10k raw surface forms, a ~400MB float32
distance matrix) but reintroduces O(N²) memory that FAISS's O(N × top_k) footprint was specifically
adopted to avoid — a real regression for a much larger relation or entity vocabulary. Step 4 (entity
dedup) partially offsets this by partitioning candidates on the parent-key gate *before* clustering, so
each HDBSCAN call only ever sees one hierarchy sibling-group at a time rather than the whole entity
vocabulary; Steps 1 and 2 have no equivalent partition and would need one (or a return to FAISS-based
candidate generation feeding into a per-neighborhood HDBSCAN pass) if their vocabularies grow much
larger. Separately, `dedup_base.cluster_hdbscan()` depends on `allow_single_cluster=True` — verified
empirically (against this codebase's actual label-embedding distributions, not a synthetic corner case)
that HDBSCAN's default rejects a pool with no viable internal sub-split and marks it *entirely* as
noise instead of one cluster, which would otherwise have silently discarded whole well-formed groups of
mutually-similar surface forms as "no merge" with no error or warning.

### L4: Duplicate-label reconciliation's asymmetric confirm-only-not-deny logic
As documented in `hierarchy_induction.py`'s module docstring: two SYNTHESIZED nodes sharing a
normalized label are auto-merged only on *high* relation-profile cosine similarity; low or zero
similarity is treated as "ambiguous, ask the LLM" rather than "confirmed different," because rolled-up
abstract-type profiles are frequently sparse enough that even genuine duplicates land at cosine ≈ 0
from sparsity alone, not real semantic difference. This was verified empirically on this corpus's
"entity"/"concept"/"status" duplicates. The cost is more LLM disambiguation calls than a symmetric
threshold would need; the alternative (treating low similarity as a confirmed split) was found to
leave true duplicates split forever.

### L5: SHACL-Core cannot express all Wikidata property constraints
Relevant once §9 is actually built: as shown by Ferranti et al. (Semantic Web 15(6), 2024), 9 of
Wikidata's 32 constraint types require SHACL-SPARQL. The design sketch in §9 only plans for
SHACL-Core; complex constraints (e.g., "inverse of", "symmetric") would need to be stored in JSON-LD
with a note rather than as SHACL shapes.

### L6: Stochastic LLM outputs, majority voting mostly unused
Despite `temperature=0` on every call, some LLM APIs are not fully deterministic (especially under
server-side request batching). Majority voting exists as a mechanism (`HierarchyConfig.col_num_votes`/
`col_vote_agreement`) but ships disabled (`col_num_votes=1`) — every LLM-verification call in the
pipeline (relation/type/entity cluster verification, relation-direction classification, duplicate-label
disambiguation) is currently a single, unvoted call. Absolute reproducibility across API/model
versions is not guaranteed; pin the model version string in `configs/*.yaml`.

### L7: Polysemous entity types
Type canonicalization (Step 2) assumes the same surface form always refers to the same type unless the
LLM verification call decides otherwise (via the hyponymy/homonym distinction in
`cluster_entity_types.txt`). This fails for highly polysemous terms not caught by that single call
(e.g. "bank" as financial institution vs. river bank in a small cluster where nothing forces a split).
Entity dedup's compound `"name [type]"` labels prevent *cross-type* name collisions (`"Paris [city]"`
vs. `"Paris [person]"`) but do nothing for a type label itself being polysemous — a dedicated
word-sense-disambiguation step would improve precision here.

### L8: Candidate generation for hierarchy edges currently depends entirely on fused-similarity clustering
Every candidate parent/child/same-concept pair the LLM ever sees in `hierarchy_induction.py` first has
to clear the fused embedding+profile *cosine* similarity floor in some round (§6, steps 2–3) —
`HierarchyEdge.relation_signature_score` (Weeds precision) is computed only as post-hoc metadata on an
edge the clustering step already produced, never as an independent way to propose one. This misses
genuine IS-A pairs that are lexically distant but relationally contained (a broad parent that has
absorbed several other children can have mediocre similarity to a lexically-distant true child even
though the child's relations remain a clean subset of the parent's — cosine dilutes on the parent's
extra unrelated mass, Weeds precision doesn't). See §6's planned extension for the two-phase
asymmetric-pairwise-then-embedding-clustering design intended to close this gap.

### Design choice: why flat canonicalization before hierarchy?
An alternative design would induce the hierarchy *jointly* with type canonicalization. The sequential
design (Steps 1–2 before Step 3) was kept for three reasons:
1. It is more modular and easier to ablate.
2. Flat canonicalization is a solved(-ish) problem (CESI, KGGen); adding hierarchy jointly would
   make the LLM prompts significantly more complex and error-prone.
3. Error propagation is easier to analyse: if the hierarchy is wrong, the canonical types that fed
   into it can be inspected directly, whereas a joint model's errors would be entangled.

---

*End of CLAUDE.md.*
