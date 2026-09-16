"""Export raw triplets in fully-canonicalized form, one JSON object per line.

Each raw triplet dict from the extraction JSONL is re-resolved against a
pipeline run's checkpoints so that downstream consumers get canonical ids and
labels instead of surface forms:

    subject/object  -> canonical ENTITY   (entity_dedup.pkl)
    subject_type /
    object_type     -> canonical TYPE     (type_dedup.pkl)
    relation        -> canonical RELATION (relation_dedup.pkl)

plus every ``source_*`` / ``*_id`` field the extractor attached to the triplet,
so each exported row still points back at the document it came from.

This deliberately reuses build_kg_graph.py's resolvers rather than
reimplementing the lookup, so the export and the MINE graph can never disagree
about what a triplet resolves to. That module documents the lookup chain and
the compound-label reconstruction in detail; the short version is that entity
identity is ``name + canonical type label``, not name alone.

Rows whose relation or types fail to resolve are, by default, still emitted
with nulls in those fields and ``"resolved": false`` -- dropping them silently
is how the six zero-triplet MINE essays went unnoticed. Pass --only-resolved
for the strict subset.

Usage:
    python -m scripts.export_refined_triplets \
        --triplets data/mine/triplets_gptoss_v2.jsonl \
        --output-dir output/mine --run 13 \
        --output output/mine/refined_triplets_run13.jsonl

    # most recent run under ./output, include the hierarchy's primary type
    python -m scripts.export_refined_triplets \
        --triplets data/musique_initial_triplets.jsonl --with-hierarchy
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from src.ontodisco.pipeline import _load_checkpoint, load_triplets
from scripts.mine_benchmark.build_kg_graph import (
    _entity_label,
    _resolve_entity_id,
    _resolve_relation_id,
    _resolve_run_dir,
    _resolve_type_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Fields the extractor attaches to say where a triplet came from. Anything
# matching is copied through verbatim -- MINE uses source_essay_id, MuSiQue
# uses sample_id, and a future corpus can add its own without touching this.
PROVENANCE_HINTS = ("source_", "sample_id", "doc_id", "document_id", "chunk_id", "essay_id")


def _provenance(triplet: dict) -> dict:
    return {k: v for k, v in triplet.items() if any(h in k for h in PROVENANCE_HINTS)}


def _entity_block(raw_name, raw_type, type_vocab, entity_vocab, hierarchy=None) -> dict:
    """Resolve one entity slot. Type resolution can succeed while entity
    resolution fails (the mention was skipped by
    collect_entity_surface_forms), so the two are reported independently
    rather than collapsed into one all-or-nothing flag."""
    block = {"name": raw_name, "raw_type": raw_type,
             "entity_id": None, "canonical_name": None,
             "type_id": None, "type_label": None, "type_ids": None, "primary_type_id": None}
    type_id = _resolve_type_id(raw_type, type_vocab) if raw_type else None
    if type_id is None:
        return block
    type_label = type_vocab.types[type_id].canonical_label
    block["type_id"], block["type_label"] = type_id, type_label

    entity_id = _resolve_entity_id(raw_name, type_label, entity_vocab)
    if entity_id is None:
        return block
    entity = entity_vocab.entities[entity_id]
    block["entity_id"] = entity_id
    block["canonical_name"] = _entity_label(entity_id, entity_vocab)
    # The authoritative class assignment is the SET of types across every
    # mention merged into this entity (CLAUDE.md sec.7) -- primary_type_id is a
    # convenience LCA over that set, and is None when the set spans
    # disconnected hierarchy trees, so both are exported.
    block["type_ids"] = sorted(entity.type_ids)
    block["primary_type_id"] = entity.primary_type_id
    return block


def _resolve_qualifiers(triplet, type_vocab, relation_vocab, entity_vocab) -> list:
    out = []
    for q in triplet.get("qualifiers") or []:
        if not isinstance(q, dict):
            continue
        q_rel = str(q.get("relation", "")).strip()
        rel_id = _resolve_relation_id(q_rel, relation_vocab) if q_rel else None
        row = {
            "raw_relation": q_rel,
            "relation_id": rel_id,
            "relation_label": relation_vocab.items[rel_id].canonical_label if rel_id else None,
        }
        row["object"] = _entity_block(str(q.get("object", "")).strip(),
                                      str(q.get("object_type", "")).strip() or None,
                                      type_vocab, entity_vocab)
        out.append(row)
    return out


def export(triplets, type_vocab, relation_vocab, entity_vocab, out_path: Path,
           *, only_resolved: bool = False, include_qualifiers: bool = True) -> dict:
    stats = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for triplet in triplets:
            stats["total"] += 1
            raw_relation = str(triplet.get("relation", "")).strip()
            relation_id = _resolve_relation_id(raw_relation, relation_vocab) if raw_relation else None

            subject = _entity_block(str(triplet.get("subject", "")).strip(),
                                    str(triplet.get("subject_type", "")).strip() or None,
                                    type_vocab, entity_vocab)
            obj = _entity_block(str(triplet.get("object", "")).strip(),
                                str(triplet.get("object_type", "")).strip() or None,
                                type_vocab, entity_vocab)

            fully = bool(relation_id and subject["entity_id"] and obj["entity_id"])
            stats["resolved" if fully else "partial"] += 1
            if not relation_id:
                stats["unresolved_relation"] += 1
            for tag, blk in (("subject", subject), ("object", obj)):
                if blk["type_id"] is None:
                    stats[f"unresolved_{tag}_type"] += 1
                elif blk["entity_id"] is None:
                    stats[f"unresolved_{tag}_entity"] += 1

            if only_resolved and not fully:
                continue

            row = {
                "subject": subject,
                "relation": {"raw": raw_relation, "relation_id": relation_id,
                             "label": relation_vocab.items[relation_id].canonical_label if relation_id else None},
                "object": obj,
                "resolved": fully,
                "source": _provenance(triplet),
            }
            if include_qualifiers:
                row["qualifiers"] = _resolve_qualifiers(triplet, type_vocab, relation_vocab, entity_vocab)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return dict(stats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--triplets", required=True, help="raw extraction JSONL the run was built from")
    p.add_argument("--output-dir", default="output",
                   help="pipeline output_dir holding checkpoints/ (NOTE: output/ and output/mine/ "
                        "number runs INDEPENDENTLY -- run_13 exists under both and they are "
                        "different runs)")
    p.add_argument("--run", type=int, default=None, help="run number (default: most recent)")
    p.add_argument("--output", default=None, help="destination JSONL (default: <run_dir>/refined_triplets.jsonl)")
    p.add_argument("--only-resolved", action="store_true",
                   help="emit only fully-resolved rows instead of flagging partials")
    p.add_argument("--no-qualifiers", action="store_true", help="omit resolved qualifiers")
    args = p.parse_args()

    run_dir = _resolve_run_dir(Path(args.output_dir), args.run)
    logger.info("Loading canonical vocabularies from %s", run_dir)
    type_vocab = _load_checkpoint(run_dir, "type_dedup")
    relation_vocab = _load_checkpoint(run_dir, "relation_dedup")
    entity_vocab = _load_checkpoint(run_dir, "entity_dedup")

    triplets = load_triplets(args.triplets)
    out_path = Path(args.output) if args.output else run_dir / "refined_triplets.jsonl"
    stats = export(triplets, type_vocab, relation_vocab, entity_vocab, out_path,
                   only_resolved=args.only_resolved, include_qualifiers=not args.no_qualifiers)

    logger.info("Wrote %d row(s) to %s", stats.get("written", 0), out_path)
    logger.info("Fully resolved: %d/%d (%.1f%%)", stats.get("resolved", 0), stats["total"],
                100.0 * stats.get("resolved", 0) / max(1, stats["total"]))
    unresolved = {k: v for k, v in stats.items() if k.startswith("unresolved_")}
    if unresolved:
        logger.info("Unresolved breakdown: %s", unresolved)


if __name__ == "__main__":
    main()
