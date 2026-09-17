"""
Tests for scripts/webapp/lib/sandbox.py -- the pure-Python diff/reparent
logic behind the Reparent Lab demo page. No Streamlit, no real LLM/embedder
calls; mirrors tests/unit/test_entity_dedup.py's synthetic-hierarchy fixture
style.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.webapp.lib.sandbox import (
    demo_reparent, diff_affected_constraints, diff_merge_candidates,
)
from src.ontodisco.constraints import ConstraintConfig, ConstraintStrength, RelationConstraint
from src.ontodisco.entity_dedup import CanonicalEntity, EntityDeduplicationResult
from src.ontodisco.hierarchy_induction import HierarchyEdge, TypeHierarchy
from src.ontodisco.relation_dedup import CanonicalRelation, RelationDeduplicationResult
from src.ontodisco.type_dedup import CanonicalType, TypeDeduplicationResult


def _edge(child, parent):
    return HierarchyEdge(child_type_id=child, parent_type_id=parent,
                          relation_signature_score=None, llm_score=0.9, ensemble_score=0.9)


def _hierarchy():
    # film -> {documentary film}; ocean current is an unrelated root;
    # biographical film starts as its OWN root (not yet under film).
    edges = [_edge("type_doc", "type_film")]
    return TypeHierarchy(
        edges=edges,
        children={"type_film": ["type_doc"]},
        parents={"type_doc": ["type_film"]},
        roots=["type_film", "type_ocean", "type_bio"],
    )


def _type_vocab():
    types = {
        "type_film": CanonicalType(item_id="type_film", canonical_label="film"),
        "type_doc": CanonicalType(item_id="type_doc", canonical_label="documentary film"),
        "type_bio": CanonicalType(item_id="type_bio", canonical_label="biographical film"),
        "type_ocean": CanonicalType(item_id="type_ocean", canonical_label="ocean current"),
    }
    surface_to_id = {v.canonical_label: k for k, v in types.items()}
    return TypeDeduplicationResult(items=types, surface_to_id=surface_to_id, num_raw=4, num_canonical=4)


# ═══════════════════════════════════════════════════════════════════════════
#  demo_reparent
# ═══════════════════════════════════════════════════════════════════════════

def test_demo_reparent_rejects_cycle():
    h = _hierarchy()
    result = demo_reparent(h, "type_film", "type_doc")  # film under its own child
    assert not result.ok
    assert "cycle" in result.reason.lower()
    assert result.hierarchy is h  # unchanged, same object


def test_demo_reparent_rejects_noop():
    h = _hierarchy()
    result = demo_reparent(h, "type_doc", "type_film")  # already the case
    assert not result.ok


def test_demo_reparent_moves_and_rebuilds_structure():
    h = _hierarchy()
    result = demo_reparent(h, "type_bio", "type_film")  # bio: root -> child of film
    assert result.ok
    new_h = result.hierarchy
    assert new_h is not h  # never mutates the input
    assert new_h.parents["type_bio"] == ["type_film"]
    assert set(new_h.children["type_film"]) == {"type_doc", "type_bio"}
    assert "type_bio" not in new_h.roots
    # original untouched
    assert "type_bio" in h.roots
    assert "type_bio" not in h.parents


# ═══════════════════════════════════════════════════════════════════════════
#  diff_merge_candidates
# ═══════════════════════════════════════════════════════════════════════════

def _entity(item_id, label, type_ids):
    return CanonicalEntity(item_id=item_id, canonical_label=label, type_ids=set(type_ids))


def test_diff_merge_candidates_finds_new_pair_after_move():
    old_h = _hierarchy()
    new_h = demo_reparent(old_h, "type_bio", "type_film").hierarchy

    entity_vocab = EntityDeduplicationResult(
        items={
            "e1": _entity("e1", "Human Planet [documentary film]", {"type_doc"}),
            "e2": _entity("e2", "Human Planet [biographical film]", {"type_bio"}),
            "e3": _entity("e3", "Amazon [ocean current]", {"type_ocean"}),
        },
        surface_to_id={}, num_raw=3, num_canonical=3,
    )
    affected = {"type_bio", "type_film", "type_doc"}

    candidates = diff_merge_candidates(old_h, new_h, entity_vocab, affected)
    pairs = {(c["entity_a_id"], c["entity_b_id"]) for c in candidates}
    assert ("e1", "e2") in pairs or ("e2", "e1") in pairs
    assert not any("e3" in p for p in pairs)  # unaffected type, never touched


def test_diff_merge_candidates_empty_when_nothing_newly_shared():
    h = _hierarchy()
    same_h = demo_reparent(h, "type_bio", "type_ocean").hierarchy  # bio moves, but not near type_doc
    entity_vocab = EntityDeduplicationResult(
        items={
            "e1": _entity("e1", "X [documentary film]", {"type_doc"}),
            "e2": _entity("e2", "Y [biographical film]", {"type_bio"}),
        },
        surface_to_id={}, num_raw=2, num_canonical=2,
    )
    candidates = diff_merge_candidates(h, same_h, entity_vocab, {"type_bio", "type_ocean"})
    assert candidates == []


# ═══════════════════════════════════════════════════════════════════════════
#  diff_affected_constraints
# ═══════════════════════════════════════════════════════════════════════════

def _relation_vocab():
    rels = {
        "rel_directed": CanonicalRelation(item_id="rel_directed", canonical_label="directed"),
        "rel_unrelated": CanonicalRelation(item_id="rel_unrelated", canonical_label="located in"),
    }
    return RelationDeduplicationResult(
        items=rels,
        surface_to_id={"directed": "rel_directed", "located in": "rel_unrelated"},
        num_raw=2, num_canonical=2,
    )


def test_diff_affected_constraints_detects_signature_change():
    old_h = _hierarchy()
    new_h = demo_reparent(old_h, "type_bio", "type_film").hierarchy
    tv = _type_vocab()
    rv = _relation_vocab()

    # Old constraint: "directed" ranges over documentary film alone.
    old_constraints = [
        RelationConstraint(relation_id="rel_directed", domain_type_id="type_doc",
                            range_type_id="type_doc", support=2, total=2,
                            pca_confidence=1.0, strength=ConstraintStrength.HARD),
        RelationConstraint(relation_id="rel_unrelated", domain_type_id="type_ocean",
                            range_type_id="type_ocean", support=5, total=5,
                            pca_confidence=1.0, strength=ConstraintStrength.HARD),
    ]
    triplets = [
        {"subject": "A", "subject_type": "documentary film", "relation": "directed",
         "object": "B", "object_type": "documentary film"},
        {"subject": "C", "subject_type": "biographical film", "relation": "directed",
         "object": "D", "object_type": "biographical film"},
    ]
    affected = {"type_bio", "type_film", "type_doc"}

    rows = diff_affected_constraints(old_h, new_h, old_constraints, triplets, tv, rv,
                                      ConstraintConfig(), affected)
    relations_touched = {r["relation"] for r in rows}
    assert "directed" in relations_touched
    assert "located in" not in relations_touched  # unaffected relation must not be recomputed/shown
