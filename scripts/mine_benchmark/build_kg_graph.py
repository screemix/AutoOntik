"""
Build a single kg-gen-format KG graph from the ontodisco pipeline output
============================================================================

Independent, standalone step: resolves the raw MINE triplets into canonical
(entity, relation, entity) triples using the canonical vocabularies produced
by run_pipeline_mine.py (type_dedup / relation_dedup / entity_dedup
checkpoints), and writes them out as a single kg-gen-compatible Graph JSON
file (the same {"entities": [...], "edges": [...], "relations": [[s,r,o],
...]} schema KGGen.export_graph() writes -- see
https://github.com/stair-lab/kg-gen/blob/main/src/kg_gen/models.py).

This script only needs this repo's existing venv (torch/transformers/sklearn
for unpickling the checkpoints) -- it does NOT need kg-gen or
sentence-transformers. run_judge_eval.py (a separate script, run in a
separate Python 3.10+ conda env with kg-gen installed) consumes the JSON file
this script produces and knows nothing about ontodisco's internal
dataclasses -- so graph construction and judge evaluation can be re-run
independently of one another, exactly like the two-stage kg-gen MINE
workflow (generate once, evaluate against it any number of times, e.g. with
a different judge model later).

Triplet -> canonical triple resolution mirrors the pattern
constraints.py::_resolve_triplets already uses for type_id/relation_id
lookup, and entity_dedup.py::collect_entity_surface_forms's compound-label
construction for entity_id lookup:
    1. subject_type_id  = type_vocab.surface_to_id[normalize(subject_type)]
    2. object_type_id   = type_vocab.surface_to_id[normalize(object_type)]
    3. relation_id       = relation_vocab.surface_to_id[relation]
    4. compound_subject = "{subject} [{canonical subject_type label}]"
       subject_entity_id = entity_vocab.surface_to_id[compound_subject]
       (object symmetric)
Triplets that fail any of these four lookups are dropped (logged, not
silently ignored) -- e.g. a type/relation left as noise by its dedup step,
or an entity mention skipped by collect_entity_surface_forms because its
type never resolved.

Usage:
    python -m scripts.mine_benchmark.build_kg_graph \\
        --triplets data/mine/triplets.jsonl \\
        --output-dir output/mine \\
        --graph-output output/mine/kg_graph.json
    # --run picks a specific run_<n>; default is the most recent run.
    # --use-qualifiers additionally folds each triplet's qualifiers (e.g.
    #   "point in time: 1903") into the graph as extra raw-string edges off
    #   a synthesized per-triplet statement node ("{subject} {relation}
    #   {object}"), linked back in via invented has_subject/has_object
    #   edges so retrieval can actually reach them. Qualifiers are
    #   otherwise extracted (Step 0) but never used anywhere downstream.
    #   Without --graph-output, the default filename switches to
    #   kg_graph_qualifiers.json so both variants can be built side by
    #   side and A/B'd in evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.ontodisco.entity_dedup import _make_compound, _normalize_compound
from src.ontodisco.pipeline import _existing_run_numbers, _load_checkpoint, load_triplets
from src.ontodisco.utils.dedup_base import normalize_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_run_dir(output_dir: Path, run: int | None) -> Path:
    checkpoints_root = output_dir / "checkpoints"
    if run is not None:
        run_dir = checkpoints_root / f"run_{run}"
        if not run_dir.exists():
            raise FileNotFoundError(f"No such run directory: {run_dir}")
        return run_dir
    existing = _existing_run_numbers(checkpoints_root)
    if not existing:
        raise FileNotFoundError(f"No runs found under {checkpoints_root}")
    return checkpoints_root / f"run_{existing[-1]}"


def _resolve_type_id(raw_type: str, type_vocab) -> str | None:
    return type_vocab.surface_to_id.get(normalize_label(raw_type)) or type_vocab.surface_to_id.get(raw_type)


def _resolve_relation_id(raw_relation: str, relation_vocab) -> str | None:
    return relation_vocab.surface_to_id.get(raw_relation) or relation_vocab.surface_to_id.get(normalize_label(raw_relation))


def _resolve_entity_id(raw_name: str, canonical_type_label: str, entity_vocab) -> str | None:
    compound = _make_compound(raw_name, canonical_type_label)
    return entity_vocab.surface_to_id.get(compound) or entity_vocab.surface_to_id.get(_normalize_compound(compound))


HAS_SUBJECT_EDGE = "has_subject"
HAS_OBJECT_EDGE = "has_object"


def _add_qualifiers(triplet: dict, subject_label: str, relation_label: str, object_label: str,
                     entities: set[str], edges: set[str], relations: set[tuple[str, str, str]]) -> int:
    """Qualifiers describe the STATEMENT (subject, relation, object) as a
    whole, not the object entity alone -- attaching them directly to
    object_label would collide different statements' qualifiers onto one
    shared canonical entity (e.g. two different people "receiving" the same
    canonical award in different years would both dump their "point in
    time" qualifier onto the one shared "award" node, with no way to tell
    them apart again). Instead we synthesize one statement node per
    triplet (raw string, uncanonicalized like everything else qualifiers
    touch) and hang the qualifiers off THAT, linked back to subject/object
    via two invented (not LLM-sourced) edges so retrieval's embedding-match
    + graph-walk can actually reach it -- a qualifier node with no path
    back to a real entity would never surface in retrieved context.
    """
    qualifiers = triplet.get("qualifiers") or []
    added = 0
    statement_label = None
    for qualifier in qualifiers:
        if not isinstance(qualifier, dict):
            continue
        q_relation = str(qualifier.get("relation", "")).strip()
        q_object = str(qualifier.get("object", "")).strip()
        if not q_relation or not q_object:
            continue

        if statement_label is None:
            statement_label = f"{subject_label} {relation_label} {object_label}"
            entities.add(statement_label)
            edges.add(HAS_SUBJECT_EDGE)
            edges.add(HAS_OBJECT_EDGE)
            relations.add((statement_label, HAS_SUBJECT_EDGE, subject_label))
            relations.add((statement_label, HAS_OBJECT_EDGE, object_label))

        entities.add(q_object)
        edges.add(q_relation)
        relations.add((statement_label, q_relation, q_object))
        added += 1
    return added


def build_graph(triplets: list[dict], type_vocab, relation_vocab, entity_vocab,
                 use_qualifiers: bool = False) -> tuple[dict, dict]:
    entities: set[str] = set()
    edges: set[str] = set()
    relations: set[tuple[str, str, str]] = set()

    skip_counts = {"subject_type": 0, "object_type": 0, "relation": 0,
                    "subject_entity": 0, "object_entity": 0}
    num_resolved = 0
    num_qualifiers_added = 0
    num_triplets_with_qualifiers = 0

    for triplet in triplets:
        raw_subject = triplet.get("subject", "").strip()
        raw_object = triplet.get("object", "").strip()
        raw_relation = triplet.get("relation", "").strip()
        raw_subject_type = triplet.get("subject_type", "").strip()
        raw_object_type = triplet.get("object_type", "").strip()
        if not (raw_subject and raw_object and raw_relation and raw_subject_type and raw_object_type):
            continue

        subject_type_id = _resolve_type_id(raw_subject_type, type_vocab)
        if subject_type_id is None:
            skip_counts["subject_type"] += 1
            continue
        object_type_id = _resolve_type_id(raw_object_type, type_vocab)
        if object_type_id is None:
            skip_counts["object_type"] += 1
            continue
        relation_id = _resolve_relation_id(raw_relation, relation_vocab)
        if relation_id is None:
            skip_counts["relation"] += 1
            continue

        subject_type_label = type_vocab.types[subject_type_id].canonical_label
        object_type_label = type_vocab.types[object_type_id].canonical_label

        subject_entity_id = _resolve_entity_id(raw_subject, subject_type_label, entity_vocab)
        if subject_entity_id is None:
            skip_counts["subject_entity"] += 1
            continue
        object_entity_id = _resolve_entity_id(raw_object, object_type_label, entity_vocab)
        if object_entity_id is None:
            skip_counts["object_entity"] += 1
            continue

        subject_label = entity_vocab.entities[subject_entity_id].canonical_label
        object_label = entity_vocab.entities[object_entity_id].canonical_label
        relation_label = relation_vocab.relations[relation_id].canonical_label

        entities.add(subject_label)
        entities.add(object_label)
        edges.add(relation_label)
        relations.add((subject_label, relation_label, object_label))
        num_resolved += 1

        if use_qualifiers:
            added = _add_qualifiers(triplet, subject_label, relation_label, object_label,
                                     entities, edges, relations)
            if added:
                num_qualifiers_added += added
                num_triplets_with_qualifiers += 1

    logger.info(
        "Resolved %d/%d triplets into %d unique canonical relations over %d entities / %d edge labels",
        num_resolved, len(triplets), len(relations), len(entities), len(edges),
    )
    logger.info("Skip breakdown: %s", skip_counts)
    if use_qualifiers:
        logger.info(
            "Qualifiers: %d added from %d triplets (statement nodes + has_subject/has_object linking edges)",
            num_qualifiers_added, num_triplets_with_qualifiers,
        )

    graph_dict = {
        "entities": sorted(entities),
        "edges": sorted(edges),
        "relations": sorted(relations),
        "entity_clusters": None,
        "edge_clusters": None,
        "entity_metadata": None,
    }
    stats = {
        "num_input_triplets": len(triplets),
        "num_resolved_triplets": num_resolved,
        "num_entities": len(entities),
        "num_edge_labels": len(edges),
        "num_relation_triples": len(relations),
        "skip_counts": skip_counts,
        "use_qualifiers": use_qualifiers,
        "num_qualifiers_added": num_qualifiers_added,
        "num_triplets_with_qualifiers": num_triplets_with_qualifiers,
    }
    return graph_dict, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", default="data/mine/triplets.jsonl")
    parser.add_argument("--output-dir", default="output/mine",
                         help="Same output_dir passed to run_pipeline_mine.py")
    parser.add_argument("--run", type=int, default=None,
                         help="Specific run_<n> to load; default is the most recent.")
    parser.add_argument("--graph-output", default=None,
                         help="Defaults to output-dir/kg_graph.json, or "
                              "output-dir/kg_graph_qualifiers.json if --use-qualifiers is set "
                              "-- so both variants can be built side by side and compared "
                              "in evaluation without overwriting each other.")
    parser.add_argument("--use-qualifiers", action="store_true",
                         help="Fold each triplet's qualifiers (e.g. 'point in time: 1903') into "
                              "the graph as extra edges off a synthesized per-triplet statement "
                              "node, instead of discarding them (the current default pipeline "
                              "behavior -- qualifiers are extracted but never used downstream).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_dir = _resolve_run_dir(output_dir, args.run)
    logger.info("Loading canonical vocabularies from %s", run_dir)

    type_vocab = _load_checkpoint(run_dir, "type_dedup")
    relation_vocab = _load_checkpoint(run_dir, "relation_dedup")
    entity_vocab = _load_checkpoint(run_dir, "entity_dedup")

    triplets = load_triplets(args.triplets)
    graph_dict, stats = build_graph(triplets, type_vocab, relation_vocab, entity_vocab,
                                     use_qualifiers=args.use_qualifiers)

    if args.graph_output is not None:
        graph_output = Path(args.graph_output)
    else:
        filename = "kg_graph_qualifiers.json" if args.use_qualifiers else "kg_graph.json"
        graph_output = output_dir / filename
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_output, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2, ensure_ascii=False)
    logger.info("Wrote KG graph to %s", graph_output)

    stats_output = graph_output.with_name(graph_output.stem + "_stats.json")
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote stats to %s", stats_output)


if __name__ == "__main__":
    main()
