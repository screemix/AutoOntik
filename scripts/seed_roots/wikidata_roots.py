"""
Wikidata Seed-Root Exploration
=================================

Standalone research/exploration script for the "Wikidata seed roots" branch
of the seeded-hierarchy-induction experiment (see CLAUDE.md's "Seeded
Hierarchy Induction" section). NOT wired into hierarchy_induction.py --
this just surveys what Wikidata's class system (the `subclass of` / P279
predicate) actually offers near the top, so a small usable seed set can be
picked by hand afterward.

Three complementary queries against the public Wikidata Query Service
(https://query.wikidata.org/sparql, no auth required):

1. `find_root_candidates()`: DIRECT P279 children of a given root (Q35120
   "entity" by default) that themselves have at least one subclass of their
   own -- i.e. genuine intermediate categories (e.g. `location`,
   `individual entity`) rather than childless leaf-like concepts. This
   replaced an earlier `find_true_roots()` that scanned the WHOLE P279
   graph for parentless items (things with no `subclass of` statement of
   their own): that surfaced mostly orphan/pseudo-roots like chemical
   elements (`lutetium`, `holmium`, ...) which only appear "rootlike"
   because of a missing/inconsistent P279 link upstream, not because
   they're meaningful top-level categories -- and they aren't even children
   of `entity` in the first place. Restricting to actual children of
   Q35120, and requiring subclasses of their own, grounds the candidate set
   in the one hierarchy this script otherwise builds from (see
   `collect_seed_tree()`).

2. `collect_seed_tree()`: a breadth-first survey starting from Q35120
   "entity" (default) -- documented by Wikidata's own
   Wikidata:WikiProject_Ontology/Top-level_ontology_list as the root of the
   class hierarchy ("the root at entity (Q35120)"), and confirmed
   empirically by Doğan & Patel-Schneider (2025), "A Multi-Axial Mindset for
   Ontology Design: Lessons from Wikidata's Polyhierarchical Structure"
   (arXiv:2512.12260): 4,210,960 classes are reachable from it via P279
   chains, vs. 28,648 that aren't. Expands the branches with the most
   direct subclasses (a breadth/hub-ness proxy)
   until at least `--target-n` candidate classes are collected. This is
   deliberately NOT "the true roots" (per the task: "not strictly roots") --
   it's a practical shortlist of high-level classes worth considering as
   seeds, each with its own subclass count (structural breadth) and
   instance count (real-world usage) so they can be compared.

3. `find_classlike_instances()`: (2) only ever follows P279 edges, so it is
   structurally blind to Wikidata items that function as categories via P31
   ("instance of") instead -- a well-documented inconsistency in how the
   community applies the P31/P279 distinction. For each node (2) collects,
   this checks whether any of ITS P31 instances themselves have P279
   subclasses of their own (i.e. they're "instances" that are also acting
   as classes). This is EXPLORATORY ONLY: the decision (see CLAUDE.md
   §15.3.1) is to keep the actual candidate seed pool P279-only at this
   abstract/high level, for comparability with the NER and DOLCE branches'
   purely taxonomic type inventories -- (3)'s output documents a real
   Wikidata data-quality finding but is not folded into (2)'s result.

Usage:
    python -m scripts.seed_roots.wikidata_roots
    python -m scripts.seed_roots.wikidata_roots --root Q35120 --target-n 15
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "AutoOnto-seed-roots-research/0.1 (https://github.com/; research use, contact via repo)"


def run_sparql(query: str, max_retries: int = 4) -> list[dict]:
    """POST a SPARQL query to the Wikidata Query Service, return the raw
    `results.bindings` list. Retries with backoff on 429/5xx -- WDQS is a
    shared public endpoint with real rate limits."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
                timeout=60,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited (429), waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]
        except requests.RequestException as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning("SPARQL query failed (attempt %d/%d): %s -- retrying in %ds",
                           attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"SPARQL query failed after {max_retries} attempts: {last_err}")


def _qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def get_labels_and_descriptions(qids: list[str]) -> dict[str, dict]:
    """Batch label/description lookup for a small, already-known set of
    QIDs via VALUES. Kept as a SEPARATE query from the aggregation queries
    below on purpose: SERVICE wikibase:label applied over a large
    intermediate result set (thousands of candidate roots/children, before
    GROUP BY/LIMIT narrows it down) is what caused this script's queries to
    time out against WDQS's ~60s budget -- confirmed empirically. Labeling
    only the final, already-small (<=~30) result set is fast."""
    if not qids:
        return {}
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
    SELECT ?item ?itemLabel ?itemDescription WHERE {{
      VALUES ?item {{ {values} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    bindings = run_sparql(query)
    return {
        _qid_from_uri(b["item"]["value"]): {
            "label": b.get("itemLabel", {}).get("value", ""),
            "description": b.get("itemDescription", {}).get("value", ""),
        }
        for b in bindings
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Query 1: root candidates -- direct children of `entity` that have subclasses
# ═══════════════════════════════════════════════════════════════════════════

def find_root_candidates(
    root_qid: str = "Q35120", list_limit: int = 200, count_cap: int = 5000,
) -> list[dict]:
    """Direct P279 children of root_qid (Q35120 "entity" by default) that
    themselves have at least one subclass of their own -- e.g. `location`,
    `individual entity` -- excluding children that are leaf-like (no
    subclasses), the way e.g. a specific named instance-ish concept with
    nothing under it would be.

    list_limit bounds the initial child scan; Wikidata's own
    WikiProject_Ontology page documents ~22 direct children of Q35120 as of
    writing (see CLAUDE.md's "Seeded Hierarchy Induction" notes), so 200 is
    comfortably an upper bound, not a truncation risk. Subclass/instance
    counts reuse `_capped_counts_per_child` (see its docstring for why
    counts are capped rather than exact -- large subtrees can otherwise
    blow the endpoint's ~60s budget)."""
    list_query = f"SELECT ?child WHERE {{ ?child wdt:P279 wd:{root_qid} . }} LIMIT {list_limit}"
    child_qids = [_qid_from_uri(b["child"]["value"]) for b in run_sparql(list_query)]
    if not child_qids:
        return []

    subclass_counts = _capped_counts_per_child(child_qids, "P279", count_cap=count_cap)
    instance_counts = _capped_counts_per_child(child_qids, "P31", count_cap=count_cap)

    candidates = [
        {
            "qid": q,
            "num_subclasses": subclass_counts.get(q, 0),
            "num_instances": instance_counts.get(q, 0),
        }
        for q in child_qids
        if subclass_counts.get(q, 0) > 0
    ]
    candidates.sort(key=lambda c: -c["num_subclasses"])

    labels = get_labels_and_descriptions([c["qid"] for c in candidates])
    for c in candidates:
        c["label"] = labels.get(c["qid"], {}).get("label", "")
        c["description"] = labels.get(c["qid"], {}).get("description", "")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
#  Query 2: breadth-first seed-tree survey from a known top node
# ═══════════════════════════════════════════════════════════════════════════

def _capped_counts_per_child(child_qids: list[str], predicate: str, count_cap: int = 5000) -> dict[str, int]:
    """For each of a bounded set of child QIDs, count how many things point
    at it via `predicate` (P279 for subclasses, P31 for instances) -- CAPPED
    at count_cap per child.

    A naive single query with `?child wdt:P279 wd:ROOT . OPTIONAL { ?gc
    wdt:P279 ?child }` GROUP BY ?child times out in practice: some Wikidata
    classes have subtrees so large (e.g. Q13196193 "part", encountered
    empirically while building this script -- millions of things eventually
    subclass it) that COUNT(DISTINCT ?grandchild) over even ONE such child
    blows the shared endpoint's ~60s budget, taking every other child in the
    same query down with it.

    Fix: one independently-LIMITed subquery PER child, unioned together.
    Each subquery's LIMIT bounds the actual triple scan (not just the
    output row count) before COUNT(*) runs, so no single child's subtree
    size can blow the whole query's budget regardless of how big it
    actually is. A capped count (e.g. "5000+") is exactly as useful as an
    exact count for this script's only use of it: ranking children by
    breadth to decide which to expand next.

    Chunked into small batches rather than one big UNION of all child_qids:
    a single request unioning ~60 children (encountered empirically for
    Q13196193 "part", which alone has 60 direct subclasses) came back as a
    503 from the endpoint's front proxy, independent of the per-block LIMIT
    fix above -- the request itself was too large/complex, not merely slow.
    Small fixed-size batches avoid that regardless of how wide any given
    parent's child list is."""
    if not child_qids:
        return {}
    batch_size = 10
    counts: dict[str, int] = {}
    for i in range(0, len(child_qids), batch_size):
        batch = child_qids[i : i + batch_size]
        blocks = "\nUNION\n".join(
            f"{{ SELECT ?child (COUNT(*) AS ?n) WHERE {{ "
            f"BIND(wd:{q} AS ?child) "
            f"{{ SELECT ?x WHERE {{ ?x wdt:{predicate} wd:{q} . }} LIMIT {count_cap} }} "
            f"}} GROUP BY ?child }}"
            for q in batch
        )
        query = f"SELECT ?child ?n WHERE {{\n{blocks}\n}}"
        bindings = run_sparql(query)
        counts.update({_qid_from_uri(b["child"]["value"]): int(b["n"]["value"]) for b in bindings})
        time.sleep(1.5)
    return counts


def get_children(qid: str, limit: int = 30, list_limit: int = 60) -> list[dict]:
    """Direct P279 (subclass of) children of qid, each with its own direct
    subclass count (breadth/hub-ness proxy, CAPPED -- see
    _capped_counts_per_child) and instance count (P31 reverse -- real-world
    usage signal, informational only, not used for ranking)."""
    # Step 1: just the list of direct children -- a single indexed hop, no
    # aggregation, so this is fast regardless of how large any child's OWN
    # subtree later turns out to be. list_limit bounds how many children we
    # bother scoring in step 2 (a class with hundreds of direct subclasses
    # is itself informative -- "hits list_limit" is a valid finding), which
    # in turn bounds the size of the per-child UNION query below.
    list_query = f"SELECT ?child WHERE {{ ?child wdt:P279 wd:{qid} . }} LIMIT {list_limit}"
    child_qids = [_qid_from_uri(b["child"]["value"]) for b in run_sparql(list_query)]
    if not child_qids:
        return []

    subclass_counts = _capped_counts_per_child(child_qids, "P279")
    instance_counts = _capped_counts_per_child(child_qids, "P31")

    children = [
        {
            "qid": q,
            "num_subclasses": subclass_counts.get(q, 0),
            "num_instances": instance_counts.get(q, 0),
        }
        for q in child_qids
    ]
    children.sort(key=lambda c: -c["num_subclasses"])
    children = children[:limit]

    labels = get_labels_and_descriptions([c["qid"] for c in children])
    for c in children:
        c["label"] = labels.get(c["qid"], {}).get("label", "")
        c["description"] = labels.get(c["qid"], {}).get("description", "")
    return children


def collect_seed_tree(
    root_qid: str, root_label: str, target_n: int = 10,
    top_k_per_level: int = 6, max_depth: int = 3,
) -> dict:
    """BFS from root_qid, always expanding the widest (most-subclasses)
    unexpanded node next, until at least target_n classes are collected or
    max_depth is reached. Returns {"root": {...}, "nodes": {qid: {...,
    "parent": qid, "depth": int}}} -- a flat map is easier to serialize and
    re-derive a tree from than nested dicts, and avoids representing a node
    twice if BFS reaches it via two parents (P279 is a DAG, not a tree)."""
    nodes: dict[str, dict] = {
        root_qid: {"qid": root_qid, "label": root_label, "description": "",
                    "num_subclasses": None, "num_instances": None,
                    "parent": None, "depth": 0}
    }
    frontier = [(root_qid, 0)]
    seen_children_of: set[str] = set()

    while frontier and len(nodes) < target_n:
        # Expand the widest not-yet-expanded node at the shallowest depth first.
        frontier.sort(key=lambda x: x[1])
        parent_qid, depth = frontier.pop(0)
        if parent_qid in seen_children_of or depth >= max_depth:
            continue
        seen_children_of.add(parent_qid)

        logger.info("Expanding %s (%s) at depth %d -- have %d/%d nodes",
                    parent_qid, nodes[parent_qid]["label"], depth, len(nodes), target_n)
        children = get_children(parent_qid, limit=top_k_per_level)
        time.sleep(1)  # be polite to the shared public endpoint

        for child in children:
            if child["qid"] in nodes:
                continue  # already reached via another parent (P279 is a DAG)
            child["parent"] = parent_qid
            child["depth"] = depth + 1
            nodes[child["qid"]] = child
            frontier.append((child["qid"], depth + 1))
            if len(nodes) >= target_n:
                break

    return {"root_qid": root_qid, "root_label": root_label, "nodes": nodes}


# ═══════════════════════════════════════════════════════════════════════════
#  Query 3: P31 (instance of) as a hierarchy signal, not just a side metric
# ═══════════════════════════════════════════════════════════════════════════

def find_classlike_instances(qid: str, list_limit: int = 30, top_k: int = 10) -> list[dict]:
    """collect_seed_tree() only ever follows P279 (subclass of) edges, but
    Wikidata's own modeling guidance (Wikidata:WikiProject_Ontology) and
    community discussion repeatedly flag that the P31/P279 boundary is not
    applied consistently: some items that function as categories in
    practice -- e.g. "metaclass"-style items, or classes the contributor
    who added them tagged as P31 "instance of" a parent rather than P279
    "subclass of" it -- are structurally invisible to a pure-P279 traversal
    no matter how it's parameterized.

    This surfaces that missed signal directly: for a given qid, fetch its
    P31 instances, then check (reusing the already-bounded
    _capped_counts_per_child machinery -- an "instance" being pointed at by
    P279 from other items is exactly the same shape of question as "does
    this child have subclasses") which of those instances themselves have
    P279 subclasses of their own. An "instance of qid" that is itself a
    superclass of other things is a class-like node collect_seed_tree()
    structurally cannot discover, since it never appears as a P279 child of
    qid at all."""
    list_query = f"SELECT ?instance WHERE {{ ?instance wdt:P31 wd:{qid} . }} LIMIT {list_limit}"
    instance_qids = [_qid_from_uri(b["instance"]["value"]) for b in run_sparql(list_query)]
    if not instance_qids:
        return []

    subclass_counts = _capped_counts_per_child(instance_qids, "P279")
    results = [
        {"qid": q, "num_subclasses": subclass_counts.get(q, 0)}
        for q in instance_qids if subclass_counts.get(q, 0) > 0
    ]
    results.sort(key=lambda c: -c["num_subclasses"])
    results = results[:top_k]

    labels = get_labels_and_descriptions([c["qid"] for c in results])
    for c in results:
        c["label"] = labels.get(c["qid"], {}).get("label", "")
        c["description"] = labels.get(c["qid"], {}).get("description", "")
    return results


def print_tree(tree: dict) -> None:
    nodes = tree["nodes"]
    children_of: dict[str, list[str]] = {}
    for qid, node in nodes.items():
        children_of.setdefault(node["parent"], []).append(qid)

    def _print(qid: str, indent: int = 0):
        node = nodes[qid]
        stats = ""
        if node["num_subclasses"] is not None:
            stats = f"  [subclasses={node['num_subclasses']}, instances={node['num_instances']}]"
        print(f"{'  ' * indent}- {node['label']} ({qid}){stats}")
        for child_qid in sorted(children_of.get(qid, []), key=lambda q: -(nodes[q]["num_subclasses"] or 0)):
            _print(child_qid, indent + 1)

    _print(tree["root_qid"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Q35120", help="QID to start the seed-tree BFS from (default: 'entity').")
    parser.add_argument("--root-label", default="entity")
    parser.add_argument("--target-n", type=int, default=10,
                         help="Minimum number of high-level classes to collect (default: 10).")
    parser.add_argument("--top-k-per-level", type=int, default=6,
                         help="Max children expanded per node per BFS step.")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--root-candidates-list-limit", type=int, default=200,
                         help="Max direct children of --root to scan for the root-candidates query.")
    parser.add_argument("--output", default="output/seed_roots/wikidata_tree.json")
    args = parser.parse_args()

    logger.info("=== Query 1: root candidates -- children of %s (%s) with subclasses of their own ===",
                args.root, args.root_label)
    root_candidates = find_root_candidates(args.root, list_limit=args.root_candidates_list_limit)
    logger.info("Found %d root candidates (of the direct children scanned):", len(root_candidates))
    for r in root_candidates:
        print(f"  - {r['label']} ({r['qid']}): {r['num_subclasses']} subclasses, {r['num_instances']} instances")

    logger.info("=== Query 2: seed-tree BFS from %s (%s) ===", args.root, args.root_label)
    tree = collect_seed_tree(
        args.root, args.root_label, target_n=args.target_n,
        top_k_per_level=args.top_k_per_level, max_depth=args.max_depth,
    )
    print()
    print_tree(tree)

    logger.info("=== Query 3: P31 (instance of) class-like instances, per collected node ===")
    classlike_instances: dict[str, list[dict]] = {}
    for qid, node in tree["nodes"].items():
        found = find_classlike_instances(qid)
        time.sleep(2)  # extra spacing between nodes -- query 3 issues several
                        # batched requests per node on top of queries 1-2, and
                        # sustained 429s were observed in practice without this
        if found:
            classlike_instances[qid] = found
            print(f"\n{node['label']} ({qid}) -- class-like P31 instances (invisible to pure P279 BFS):")
            for c in found:
                print(f"  - {c['label']} ({c['qid']}): {c['num_subclasses']} subclasses of its own")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"root_candidates": root_candidates, "seed_tree": tree, "classlike_instances": classlike_instances},
            f, indent=2, ensure_ascii=False,
        )
    logger.info("Wrote results to %s", output_path)


if __name__ == "__main__":
    main()
