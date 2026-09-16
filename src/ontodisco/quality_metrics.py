"""
Zero-LLM quality metrics over a finished pipeline run
======================================================

Every metric here is computed from checkpoints alone -- no API calls, no
judge, seconds rather than hours. That is the entire point: evaluating a
config change previously meant a full pipeline run plus a MINE judge pass,
which is too slow to ablate against, so knobs accumulated untuned (see
CLAUDE.md sec. 16.4). These four numbers make a change measurable
immediately, and three of them have a demonstrated relationship to
downstream quality:

  giant_component_share   fraction of graph nodes in the largest connected
                          component. Predicted MINE accuracy across three
                          graphs built from the same corpus: KGGen 86.5% ->
                          81.87% accuracy, AutoOntic 51.1% -> 77.80%,
                          AutoOntic/gpt-4o 32.4% -> 62.93%. MINE retrieval is
                          seed-match + fixed-depth walk, so the giant
                          component IS the reachability ceiling.

  nodes_per_triple        distinct entities divided by triples (LOWER is
                          better -- it means entities recur across statements
                          instead of each triple minting its own endpoints).
                          AutoOntic 0.942 vs KGGen 0.769 on MINE. CAUTION:
                          this metric is gameable -- emitting extra degenerate
                          triples over existing entities improves it while
                          making the graph worse (observed with a two-stage
                          extraction pilot that scored 1.25 by inventing
                          statements). Never optimize it without also watching
                          triple precision.

  orphan_rate             hierarchy nodes that are roots with no children --
                          neither placed under anything nor parenting anything.
                          0% on MINE (672 types) vs 30.7% on clinrecs (4140
                          types), which is what exposed the fixed candidate
                          budget starving large vocabularies.

  residual_synonym_pairs  canonical labels still unmerged despite high
                          embedding similarity -- an upper bound on missed
                          merges, since a guarantee would need all-pairs LLM
                          verification (8.5M type pairs at 4140 types). Not a
                          guarantee, a measurable bound.

Usage:
    python -m src.ontodisco.quality_metrics --run-dir output/mine/checkpoints/run_7 \\
        [--graph output/mine/kg_graph_qualifiers_gptoss_v3.json]

run_pipeline() also calls compute_metrics() and folds the result into
run_metadata.json under "quality_metrics".
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph structure
# ═══════════════════════════════════════════════════════════════════════════════

def graph_metrics(triples: list[tuple[str, str, str]]) -> dict:
    """Connectivity of the (subject, relation, object) triple set.

    Nodes are the entities actually appearing in triples -- an entity declared
    in the vocabulary but never used in any triple is not reachable by a graph
    walk and is not counted here.
    """
    if not triples:
        return {"num_triples": 0, "num_nodes": 0, "nodes_per_triple": None,
                "num_components": 0, "giant_component_size": 0,
                "giant_component_share": None, "degree_one_share": None}

    adjacency: dict[str, set[str]] = defaultdict(set)
    degree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for subject, _relation, obj in triples:
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
        degree[subject] += 1
        degree[obj] += 1
        nodes.add(subject)
        nodes.add(obj)

    seen: set[str] = set()
    largest = 0
    num_components = 0
    for node in nodes:
        if node in seen:
            continue
        num_components += 1
        size = 0
        stack = [node]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            size += 1
            stack.extend(n for n in adjacency[current] if n not in seen)
        largest = max(largest, size)

    return {
        "num_triples": len(triples),
        "num_nodes": len(nodes),
        "nodes_per_triple": round(len(nodes) / len(triples), 4),
        "num_components": num_components,
        "giant_component_size": largest,
        "giant_component_share": round(largest / len(nodes), 4),
        "degree_one_share": round(sum(1 for n in nodes if degree[n] == 1) / len(nodes), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Hierarchy structure
# ═══════════════════════════════════════════════════════════════════════════════

def hierarchy_metrics(hierarchy_result) -> dict:
    """Shape of the induced hierarchy. An "orphan" is a root with no children:
    the algorithm neither found it a parent nor found it any child, which on a
    large vocabulary usually means its true parent was never among the
    candidates it was shown, not that it is genuinely top-level."""
    hierarchy = hierarchy_result.hierarchy
    nodes: set[str] = set(hierarchy.parents) | set(hierarchy.children) | set(hierarchy.roots)
    for edge in hierarchy.edges:
        nodes.add(edge.child_type_id)
        nodes.add(edge.parent_type_id)
    if not nodes:
        return {"num_nodes": 0, "num_edges": 0, "num_roots": 0, "num_orphans": 0,
                "orphan_rate": None, "max_depth": 0, "mean_depth": None}

    orphans = [r for r in hierarchy.roots if not hierarchy.children.get(r)]

    depths = []
    for node in nodes:
        depth = 0
        current = node
        walked: set[str] = set()
        while hierarchy.parents.get(current):
            if current in walked:      # defensive: a cycle would hang this walk
                break
            walked.add(current)
            current = hierarchy.parents[current][0]
            depth += 1
        depths.append(depth)

    unresolved = getattr(hierarchy_result, "unresolved_children", None) or {}
    return {
        "num_nodes": len(nodes),
        "num_edges": len(hierarchy.edges),
        "num_roots": len(hierarchy.roots),
        "num_orphans": len(orphans),
        "orphan_rate": round(len(orphans) / len(nodes), 4),
        "num_synthesized": len(hierarchy_result.synthesized_types),
        "num_deferred_unresolved": sum(len(v) for v in unresolved.values()),
        "max_depth": max(depths),
        "mean_depth": round(sum(depths) / len(depths), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Residual synonymy
# ═══════════════════════════════════════════════════════════════════════════════

def residual_synonym_metrics(
    labels: list[str], embedder, *, thresholds=(0.95, 0.92, 0.90), embed_batch_size: int = 128,
) -> dict:
    """Count canonical labels that survived deduplication despite scoring
    above `thresholds` against each other -- an upper bound on missed merges.

    Upper bound, not a count of errors: a high-similarity pair may be a
    genuine hypernym/hyponym split that the type prompt deliberately keeps
    apart ("economic outcome" vs "outcome"). What makes it useful is the
    comparison across runs, and the fact that a pathological embedding space
    shows up immediately -- on Russian with an English-trained encoder,
    unrelated pairs scored 0.93 while true synonyms sat at 0.54, giving 681
    surviving pairs at 0.92 against 0 for the English vocabulary.
    """
    if len(labels) < 2:
        return {"num_labels": len(labels), "pairs_above": {}}
    embeddings = np.asarray(embedder.embed(labels, batch_size=embed_batch_size), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = embeddings @ embeddings.T
    upper = np.triu_indices(len(labels), 1)
    flat = similarity[upper]
    return {
        "num_labels": len(labels),
        "pairs_above": {str(t): int((flat >= t).sum()) for t in thresholds},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    *,
    type_vocab=None,
    relation_vocab=None,
    entity_vocab=None,
    hierarchy_result=None,
    constraints=None,
    triples: Optional[list[tuple[str, str, str]]] = None,
    embedder=None,
) -> dict:
    """Assemble every metric available from whatever was passed. Each section
    is skipped (not errored) when its input is absent, so this is safe to call
    on a partially-complete run."""
    metrics: dict = {}

    if hierarchy_result is not None:
        metrics["hierarchy"] = hierarchy_metrics(hierarchy_result)

    if triples:
        metrics["graph"] = graph_metrics(triples)

    for name, vocab in (("types", type_vocab), ("relations", relation_vocab), ("entities", entity_vocab)):
        if vocab is None:
            continue
        items = vocab.items
        singletons = sum(1 for i in items.values() if len(i.surface_forms) <= 1)
        section = {
            "num_raw": vocab.num_raw,
            "num_canonical": len(items),
            "reduction_pct": round(vocab.reduction_pct, 2),
            "never_merged_share": round(singletons / len(items), 4) if items else None,
        }
        if embedder is not None and name in ("types", "relations"):
            section["residual_synonyms"] = residual_synonym_metrics(
                [i.canonical_label for i in items.values()], embedder,
            )
        metrics[name] = section

    if constraints is not None:
        by_strength: dict = defaultdict(int)
        hard_low_support = 0
        for c in constraints:
            by_strength[c.strength.value] += 1
            if c.strength.value == "hard" and c.support < 3:
                hard_low_support += 1
        metrics["constraints"] = {
            "num_constraints": len(constraints),
            "by_strength": dict(by_strength),
            "hard_with_support_under_3": hard_low_support,
        }

    return metrics


def _load(run_dir: Path, name: str):
    path = run_dir / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="output_dir/checkpoints/run_<n>")
    parser.add_argument("--graph", default=None, help="optional kg-gen-format graph JSON for graph metrics")
    parser.add_argument("--output", default=None, help="write JSON here (default: stdout only)")
    parser.add_argument("--no-embeddings", action="store_true",
                         help="skip residual-synonym metrics (avoids loading Contriever)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    triples = None
    if args.graph:
        with open(args.graph, "r", encoding="utf-8") as f:
            triples = [tuple(t) for t in json.load(f)["relations"]]

    embedder = None
    if not args.no_embeddings:
        from src.ontodisco.utils.dedup_base import ContrieverEmbedder
        embedder = ContrieverEmbedder()

    metrics = compute_metrics(
        type_vocab=_load(run_dir, "type_dedup"),
        relation_vocab=_load(run_dir, "relation_dedup"),
        entity_vocab=_load(run_dir, "entity_dedup"),
        hierarchy_result=_load(run_dir, "hierarchy_induction"),
        constraints=_load(run_dir, "constraints"),
        triples=triples,
        embedder=embedder,
    )
    rendered = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
