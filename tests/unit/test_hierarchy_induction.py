"""Unit tests for hierarchy_induction's decomposed placement algorithm.

The placement decision is two single-purpose questions per level (parent
check, then -- only if that finds nothing -- child check), never one
five-way choice, and synthesis only ever happens as a capacity-management
response to an overflowing level (_regroup_level), never as a per-comparison
judgment. See the module docstring and CLAUDE.md §6 for the measured
reasoning (framing-experiment accuracy, the historical-period bulk-claim
replay, the cross-branch duplicate-minting cases).

These tests pin:
  1. Single-parent forest holds through both parent-descent chains and
     child-check absorption (no double-parenting).
  2. same_concept (via parent_check) collapses duplicates correctly.
  3. Regrouping fires exactly when a level exceeds candidate_batch_size, and
     the newly-grouped node is attached (never left floating).
  4. The label-collision safeguard reuses an existing label instead of
     minting a duplicate node for it.
  5. The child-check pathology backstop rejects a bulk claim but still
     applies a small, genuine one.
  6. Top-level candidates are the live root set, never unrouted pool members.

The LLM and the Contriever embedder are both faked -- no test here makes an
API call or downloads a checkpoint.
"""
import hashlib

import numpy as np

from src.ontodisco import hierarchy_induction as hi
from src.ontodisco.hierarchy_induction import HierarchyConfig, induce_hierarchy
from src.ontodisco.relation_dedup import RelationDeduplicationResult
from src.ontodisco.type_dedup import CanonicalType, TypeDeduplicationResult


class _FakeEmbedder:
    """Deterministic label -> vector, stable across processes (hash-derived,
    not Python's salted hash())."""

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, texts, batch_size=None):
        out = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            vec = np.random.default_rng(seed).normal(size=8)
            out.append(vec / np.linalg.norm(vec))
        return np.stack(out)


class _FakeLLM:
    """Scripted answers for the three decomposed calls, keyed by
    (focal_label, frozenset(candidate_labels)) for the two checks and by
    frozenset(candidate_labels) for regroup. Unscripted -> the safe default
    (no match / no children / no groups). Every call is recorded so a test
    can assert on exactly what was asked."""

    def __init__(self, parent_script=None, child_script=None, regroup_script=None):
        self.parent_script = parent_script or {}
        self.child_script = child_script or {}
        self.regroup_script = regroup_script or {}
        self.parent_asked = []
        self.child_asked = []
        self.regroup_asked = []

    def check_hierarchy_parent(self, focal_label, candidate_labels, **kwargs):
        key = (focal_label, frozenset(candidate_labels))
        self.parent_asked.append((focal_label, tuple(sorted(candidate_labels))))
        return dict(self.parent_script.get(key, {"parent": None}))

    def check_hierarchy_children(self, focal_label, candidate_labels, **kwargs):
        key = (focal_label, frozenset(candidate_labels))
        self.child_asked.append((focal_label, tuple(sorted(candidate_labels))))
        return dict(self.child_script.get(key, {"children": []}))

    def regroup_hierarchy_siblings(self, candidate_labels):
        key = frozenset(candidate_labels)
        self.regroup_asked.append(tuple(sorted(candidate_labels)))
        return [dict(g) for g in self.regroup_script.get(key, [])]


def _vocabs(labels):
    """A type vocabulary of `labels`, in order, with ids type_0000.. -- and
    an EMPTY relation vocabulary, so every in-degree is 0, every node lands
    in one band, and placement order is exactly `labels` order."""
    items, surface = {}, {}
    for i, label in enumerate(labels):
        type_id = f"type_{i:04d}"
        items[type_id] = CanonicalType(item_id=type_id, canonical_label=label, surface_forms=[label])
        surface[label] = type_id
    type_vocab = TypeDeduplicationResult(
        items=items, surface_to_id=surface, num_raw=len(labels), num_canonical=len(labels),
    )
    return type_vocab, RelationDeduplicationResult(items={}, surface_to_id={})


def _run(labels, parent_script=None, child_script=None, regroup_script=None, monkeypatch=None, **cfg):
    monkeypatch.setattr(hi, "ContrieverEmbedder", _FakeEmbedder)
    type_vocab, relation_vocab = _vocabs(labels)
    llm = _FakeLLM(parent_script, child_script, regroup_script)
    result = induce_hierarchy(
        type_vocab, relation_vocab, llm, triplets=[],
        config=HierarchyConfig(max_parallel_workers=1, **cfg),
    )
    return result, llm


def _by_id(labels, result):
    d = {f"type_{i:04d}": lab for i, lab in enumerate(labels)}
    d.update({s.type_id: s.canonical_label for s in result.synthesized_types})
    return d


def test_parent_descent_chain_and_child_absorption_keep_single_parent_forest(monkeypatch):
    """entity <- vehicle <- car via parent-descent; then truck arrives,
    fails to descend past vehicle, and absorbs car as its own child (a
    retroactive correction) -- car's parent must change cleanly from
    vehicle to truck, never leaving it with two parents."""
    labels = ["entity", "vehicle", "car", "truck"]
    parent_script = {
        ("vehicle", frozenset(["entity"])): {"parent": "entity"},
        ("car", frozenset(["entity"])): {"parent": "entity"},
        ("car", frozenset(["vehicle"])): {"parent": "vehicle"},
        ("truck", frozenset(["entity"])): {"parent": "entity"},
        ("truck", frozenset(["vehicle"])): {"parent": "vehicle"},
        ("truck", frozenset(["car"])): {"parent": None},
    }
    child_script = {
        ("truck", frozenset(["car"])): {"children": ["car"]},
    }
    result, llm = _run(labels, parent_script, child_script, monkeypatch=monkeypatch)
    h = result.hierarchy

    multi = {c: p for c, p in h.parents.items() if len(p) > 1}
    assert not multi, f"single-parent forest violated: {multi}"

    by_id = _by_id(labels, result)
    parent_of = {by_id[c]: by_id[p[0]] for c, p in h.parents.items()}
    assert parent_of["vehicle"] == "entity"
    assert parent_of["car"] == "truck", "truck's child-check must retroactively reparent car"
    assert parent_of["truck"] == "vehicle", "truck settles where it stopped descending"
    assert [by_id[r] for r in h.roots] == ["entity"]


def test_same_concept_via_parent_check_collapses_duplicates(monkeypatch):
    labels = ["thing", "vehicle", "automobile"]
    parent_script = {
        ("vehicle", frozenset(["thing"])): {"parent": "thing"},
        ("automobile", frozenset(["thing"])): {"parent": "thing"},
        ("automobile", frozenset(["vehicle"])): {"parent": "vehicle", "same_concept": True},
    }
    result, llm = _run(labels, parent_script, monkeypatch=monkeypatch)
    h = result.hierarchy
    by_id = _by_id(labels, result)

    assert len(h.roots) == 1 and by_id[h.roots[0]] == "thing"
    remaining_labels = sorted(by_id[t] for t in (set(h.parents) | set(h.children) | set(h.roots)))
    assert "automobile" not in remaining_labels, "collapsed node must not survive as its own entry"
    assert "vehicle" in remaining_labels


def test_regroup_fires_only_on_overflow_and_result_is_attached(monkeypatch):
    """With candidate_batch_size=3, a 5th root-level arrival must trigger
    regroup on the 4 existing roots BEFORE its own placement check runs --
    proactive capping, not reactive cleanup. The minted group must be
    attached (a new root here, never left floating), and its members must
    no longer be roots themselves."""
    labels = ["p", "q", "r", "s", "t"]
    regroup_script = {
        frozenset(["p", "q", "r", "s"]): [
            {"label": "pq_group", "definition": "a shared abstraction", "members": ["p", "q"]},
        ],
    }
    result, llm = _run(
        labels, regroup_script=regroup_script, monkeypatch=monkeypatch,
        candidate_batch_size=3, max_regroup_input=90, max_regroup_rounds=3,
    )
    h = result.hierarchy
    by_id = _by_id(labels, result)

    assert llm.regroup_asked, "regroup should have fired at all"
    assert sorted(llm.regroup_asked[0]) == ["p", "q", "r", "s"], llm.regroup_asked

    # t's own parent/child checks must have seen the CONSOLIDATED level
    # (post-regroup), not the raw 4-item overflow.
    t_parent_candidates = next(cands for (focal, cands) in llm.parent_asked if focal == "t")
    assert len(t_parent_candidates) <= 3, t_parent_candidates

    root_labels = sorted(by_id[r] for r in h.roots)
    assert "pq_group" in root_labels
    assert "p" not in root_labels and "q" not in root_labels

    parent_of = {by_id[c]: by_id[p[0]] for c, p in h.parents.items()}
    assert parent_of["p"] == "pq_group"
    assert parent_of["q"] == "pq_group"
    assert not any(len(p) > 1 for p in h.parents.values())


def test_regroup_reuses_an_existing_label_instead_of_minting_a_duplicate(monkeypatch):
    """If regroup proposes a label that's ALREADY a real T* type (or, in a
    full run, something already synthesized elsewhere), it must attach the
    group's members under that EXISTING node -- never mint a second node
    under the same name. Measured on MINE run_13: two independent regroup
    calls minted 'biological entity' (colliding with a real type) and
    'biological concept' (a near-miss of the same underlying grouping) for
    overlapping content."""
    labels = ["existing_parent", "p", "q", "r", "s"]
    regroup_script = {
        frozenset(["existing_parent", "p", "q", "r"]): [
            {"label": "existing_parent", "definition": "should be reused, not re-minted", "members": ["p", "q"]},
        ],
    }
    result, llm = _run(
        labels, regroup_script=regroup_script, monkeypatch=monkeypatch,
        candidate_batch_size=3, max_regroup_input=90, max_regroup_rounds=3,
    )
    h = result.hierarchy
    by_id = _by_id(labels, result)

    assert not result.synthesized_types, "reusing an existing label must not synthesize a new node"

    existing_id = "type_0000"  # "existing_parent"'s original id
    assert h.children.get(existing_id, []), "existing_parent should have gained children"
    parent_of = {c: p[0] for c, p in h.parents.items()}
    p_id, q_id = "type_0001", "type_0002"
    assert parent_of[p_id] == existing_id
    assert parent_of[q_id] == existing_id
    # existing_parent itself must still resolve to exactly one entry across
    # roots/children -- no duplicate id sneaking in from regroup's own
    # bookkeeping.
    assert sum(1 for r in h.roots if r == existing_id) <= 1


def test_bulk_child_claim_is_rejected_but_small_claim_is_applied(monkeypatch):
    """A 'child' decision claiming most of the batch is answering a looser
    question than the one asked -- reject outright, don't truncate. A small,
    genuine multi-child claim must still go through."""
    labels = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "grabby"]
    others = [l for l in labels if l != "grabby"]
    child_script = {
        ("grabby", frozenset(others)): {"children": others},  # bulk claim: all 6
    }
    result, _ = _run(
        labels, child_script=child_script, monkeypatch=monkeypatch,
        max_child_claim_share=0.34,
    )
    h = result.hierarchy
    assert not h.edges, f"bulk claim should have been rejected, got {len(h.edges)} edge(s)"
    assert len(h.roots) == len(labels)

    labels2 = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "grabby"]
    child_script2 = {
        ("grabby", frozenset(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])):
            {"children": ["alpha", "beta"]},
    }
    result2, _ = _run(
        labels2, child_script=child_script2, monkeypatch=monkeypatch,
        max_child_claim_share=0.34,
    )
    by_id2 = _by_id(labels2, result2)
    parent_of2 = {by_id2[c]: by_id2[p[0]] for c, p in result2.hierarchy.parents.items()}
    assert parent_of2 == {"alpha": "grabby", "beta": "grabby"}, parent_of2


def test_top_level_candidates_are_roots_not_unplaced_pool_members(monkeypatch):
    """An unplaced node is not a position in the tree, so it must never be
    offered as a top-level candidate. With four types and no matches at all,
    the Nth node placed may only ever see the N-1 already-settled roots."""
    labels = ["alpha", "beta", "gamma", "delta"]
    result, llm = _run(labels, monkeypatch=monkeypatch)  # every check answers empty/None

    seen = [cands for (focal, cands) in llm.parent_asked]
    assert seen == [("alpha",), ("alpha", "beta"), ("alpha", "beta", "gamma")], llm.parent_asked
    assert len(result.hierarchy.roots) == 4
    assert not result.hierarchy.edges
