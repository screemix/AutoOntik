"""
Expand Accepted Root Candidates
==================================

Follow-up to wikidata_roots.py: collect_seed_tree() shares ONE node budget
(--target-n) across the whole BFS from `entity`, always expanding the
single widest not-yet-expanded node next -- so with a small target_n it
exhausts its budget on one branch (e.g. "part") before most of the
depth-1 candidates in `root_candidates` ever get their own turn, and
before candidates outside get_children()'s top-k-per-level cut (e.g.
"location", "collective entity") ever enter the frontier at all. See
CLAUDE.md's "Seeded Hierarchy Induction" notes / the accompanying research
conversation for the full explanation.

This script instead expands EVERY `root_candidates` entry marked
`"accept": true` in an existing output/seed_roots/wikidata_tree.json,
independently -- one get_children() call per accepted candidate, not one
shared BFS budget -- and merges the results into that same file's
`seed_tree.nodes` map (adding the candidate itself at depth 1, if not
already present, and its fetched children at depth 2).

Usage:
    python -m scripts.seed_roots.expand_accepted_roots
    python -m scripts.seed_roots.expand_accepted_roots --children-per-node 15
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from scripts.seed_roots.wikidata_roots import get_children, print_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def expand_accepted_candidates(
    data: dict, children_per_node: int = 10, list_limit: int = 30,
) -> dict:
    """Mutates and returns data["seed_tree"]["nodes"] in place: for every
    root_candidates entry with accept=true, ensures it's present as a
    depth-1 node under root_qid, then fetches (and adds, at depth 2) its
    own direct P279 children via a dedicated get_children() call -- no
    shared BFS budget, so every accepted candidate gets expanded regardless
    of how many subclasses its siblings happen to have."""
    root_qid = data["seed_tree"]["root_qid"]
    nodes = data["seed_tree"]["nodes"]
    accepted = [rc for rc in data["root_candidates"] if rc.get("accept") is True]
    logger.info("Expanding %d accepted root candidate(s)", len(accepted))

    for rc in accepted:
        qid = rc["qid"]
        if qid not in nodes:
            nodes[qid] = {
                "qid": qid, "label": rc["label"], "description": rc.get("description", ""),
                "num_subclasses": rc.get("num_subclasses"), "num_instances": rc.get("num_instances"),
                "parent": root_qid, "depth": 1,
            }

        logger.info("Fetching children of %s (%s)", qid, rc["label"])
        children = get_children(qid, limit=children_per_node, list_limit=list_limit)
        time.sleep(1.5)  # polite spacing between candidates, same as collect_seed_tree()

        added = 0
        for child in children:
            if child["qid"] in nodes:
                continue  # already present (possibly reached via another parent -- P279 is a DAG)
            child["parent"] = qid
            child["depth"] = nodes[qid]["depth"] + 1
            nodes[child["qid"]] = child
            added += 1
        logger.info("  -> %d child(ren) added (%d returned by the query)", added, len(children))

    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/seed_roots/wikidata_tree.json")
    parser.add_argument("--children-per-node", type=int, default=10,
                         help="Max children kept per accepted candidate, ranked by subclass count.")
    parser.add_argument("--list-limit", type=int, default=30,
                         help="Max raw children scanned per accepted candidate before ranking/trimming.")
    args = parser.parse_args()

    path = Path(args.input)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = expand_accepted_candidates(data, children_per_node=args.children_per_node, list_limit=args.list_limit)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Updated %s (seed_tree.nodes now has %d node(s))", path, len(data["seed_tree"]["nodes"]))

    print()
    print_tree(data["seed_tree"])


if __name__ == "__main__":
    main()
