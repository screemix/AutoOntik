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

import pickle

from scripts.webapp.lib.data import RunBundle
from scripts.webapp.lib.sandbox import (
    affected_type_ids, demo_delete_subtree, demo_reparent, diff_affected_constraints,
    diff_delete_impact, diff_merge_candidates, save_sandbox_as_run,
)
from src.ontodisco.constraints import ConstraintConfig, ConstraintStrength, RelationConstraint
from src.ontodisco.entity_dedup import CanonicalEntity, EntityDeduplicationResult
from src.ontodisco.hierarchy_induction import HierarchyEdge, HierarchyInductionResult, TypeHierarchy
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
#  demo_delete_subtree / diff_delete_impact
# ═══════════════════════════════════════════════════════════════════════════

def test_demo_delete_subtree_removes_node_and_descendants_only():
    h = _hierarchy()  # type_film -> type_doc; type_ocean and type_bio are separate roots
    result = demo_delete_subtree(h, "type_film")
    assert result.ok
    assert result.deleted_ids == {"type_film", "type_doc"}
    new_h = result.hierarchy
    assert new_h is not h  # never mutates the input
    assert "type_film" not in new_h.roots and "type_film" not in new_h.parents
    assert "type_doc" not in new_h.children.get("type_film", [])
    assert "type_doc" not in new_h.parents  # its only parent was deleted too
    # untouched siblings survive
    assert "type_ocean" in new_h.roots
    assert "type_bio" in new_h.roots
    # original untouched
    assert "type_film" in h.roots
    assert h.children["type_film"] == ["type_doc"]


def test_demo_delete_subtree_rejects_unknown_node():
    h = _hierarchy()
    result = demo_delete_subtree(h, "type_nonexistent")
    assert not result.ok
    assert result.hierarchy is h


def test_diff_delete_impact_reassigns_fully_deleted_vs_narrows_partial():
    # e1's ONLY type (type_doc) is being deleted -> reassigned to type_doc's
    # own former parent (type_film), NOT dropped. e2 has one deleted type
    # and one surviving type -> narrowed, keeps type_ocean. e3 has no
    # deleted type at all -> untouched.
    old_h = _hierarchy()  # type_film -> type_doc; type_ocean, type_bio are separate roots
    new_h = demo_delete_subtree(old_h, "type_doc").hierarchy

    entity_vocab = EntityDeduplicationResult(
        items={
            "e1": _entity("e1", "Oppenheimer [documentary film]", {"type_doc"}),
            "e2": _entity("e2", "Human Planet [mixed]", {"type_doc", "type_ocean"}),
            "e3": _entity("e3", "Amazon [ocean current]", {"type_ocean"}),
        },
        surface_to_id={}, num_raw=3, num_canonical=3,
    )
    constraints = [
        RelationConstraint(relation_id="rel_directed", domain_type_id="type_doc",
                            range_type_id="type_doc", support=2, total=2,
                            pca_confidence=1.0, strength=ConstraintStrength.HARD),
        RelationConstraint(relation_id="rel_unrelated", domain_type_id="type_ocean",
                            range_type_id="type_ocean", support=5, total=5,
                            pca_confidence=1.0, strength=ConstraintStrength.HARD),
    ]

    impact = diff_delete_impact(old_h, new_h, entity_vocab, constraints, {"type_doc"})

    reassigned_by_id = {r["entity_id"]: r for r in impact["reassigned_entities"]}
    narrowed_ids = {r["entity_id"] for r in impact["narrowed_entities"]}
    assert set(reassigned_by_id) == {"e1"}
    assert reassigned_by_id["e1"]["reassigned_type_id"] == "type_film"  # its nearest surviving ancestor
    assert narrowed_ids == {"e2"}
    assert "e3" not in reassigned_by_id and "e3" not in narrowed_ids

    stale_relation_ids = {c["relation_id"] for c in impact["stale_constraints"]}
    assert stale_relation_ids == {"rel_directed"}  # only the one referencing type_doc


def test_diff_delete_impact_reassigns_to_none_when_no_ancestor_survives():
    # type_bio is itself a ROOT with no parent -- an entity typed ONLY
    # type_bio has nowhere to go once it's deleted.
    old_h = _hierarchy()
    new_h = demo_delete_subtree(old_h, "type_bio").hierarchy
    entity_vocab = EntityDeduplicationResult(
        items={"e1": _entity("e1", "Some Biopic [biographical film]", {"type_bio"})},
        surface_to_id={}, num_raw=1, num_canonical=1,
    )

    impact = diff_delete_impact(old_h, new_h, entity_vocab, [], {"type_bio"})

    assert len(impact["reassigned_entities"]) == 1
    assert impact["reassigned_entities"][0]["reassigned_type_id"] is None


def test_save_sandbox_as_run_reassigns_rather_than_drops_deleted_entities(tmp_path):
    """The actually-persisted behavior must match what diff_delete_impact
    previews: an entity whose only type was deleted survives in
    entity_dedup.pkl, reassigned to its nearest surviving ancestor, not
    dropped."""
    old_h = _hierarchy()
    new_h = demo_delete_subtree(old_h, "type_doc").hierarchy

    run_dir = tmp_path / "checkpoints" / "run_1"
    run_dir.mkdir(parents=True)
    entity_vocab = EntityDeduplicationResult(
        items={
            "e1": _entity("e1", "Oppenheimer [documentary film]", {"type_doc"}),
            "e2": _entity("e2", "Human Planet [mixed]", {"type_doc", "type_ocean"}),
        },
        surface_to_id={}, num_raw=2, num_canonical=2,
    )
    bundle = RunBundle(
        run_dir=run_dir, type_vocab=_type_vocab(), relation_vocab=None,
        hierarchy_result=HierarchyInductionResult(hierarchy=old_h),
        entity_vocab=entity_vocab, constraints=[], triplets_path=None,
    )

    new_dir = save_sandbox_as_run(bundle, new_h, deleted_type_ids={"type_doc"})

    with open(new_dir / "entity_dedup.pkl", "rb") as f:
        saved = pickle.load(f)
    assert set(saved.items) == {"e1", "e2"}  # e1 survives -- never dropped
    assert saved.items["e1"].type_ids == {"type_film"}  # reassigned to its former parent
    assert saved.items["e1"].primary_type_id == "type_film"
    assert saved.items["e2"].type_ids == {"type_ocean"}  # narrowed, unrelated to reassignment


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


def test_affected_type_ids_includes_existing_children_at_destination():
    """Regression test: moving a node under a new parent must flag that
    parent's PRE-EXISTING children as affected too, since they're exactly
    who the moved node might now match -- not just the moved node's own
    subtree/ancestor path. Mirrors a real bug found against run_17 (moving
    'physical quantity' under 'quality', which already had 'physical
    property' as a child, returned 0 merge candidates until this was
    fixed -- 'physical quantity' and 'physical property' both had a
    'humidity' entity that should have become a candidate pair)."""
    h = TypeHierarchy(
        edges=[_edge("type_pp", "type_quality")],
        children={"type_quality": ["type_pp"]},
        parents={"type_pp": ["type_quality"]},
        roots=["type_pq", "type_quality"],
    )
    new_h = demo_reparent(h, "type_pq", "type_quality").hierarchy

    affected = affected_type_ids(h, new_h, "type_pq")
    assert "type_pp" in affected, (
        "the destination's pre-existing child must be in the affected set, "
        "or diff_merge_candidates can never find it as a match"
    )

    entity_vocab = EntityDeduplicationResult(
        items={
            "e1": _entity("e1", "humidity [physical quantity]", {"type_pq"}),
            "e2": _entity("e2", "humidity [physical property]", {"type_pp"}),
        },
        surface_to_id={}, num_raw=2, num_canonical=2,
    )
    candidates = diff_merge_candidates(h, new_h, entity_vocab, affected)
    pairs = {(c["entity_a_id"], c["entity_b_id"]) for c in candidates}
    assert ("e1", "e2") in pairs or ("e2", "e1") in pairs


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


def test_diff_affected_constraints_resolves_synthesized_type_labels():
    """Regression test: a constraint's new domain/range signature is
    frequently the LCA of several observed types, which is often a
    SYNTHESIZED type (hierarchy induction's own invented abstraction) --
    those never appear in type_vocab.items, so the label must come from
    synthesized_labels instead of falling back to the raw id."""
    # type_doc and type_bio are both children of a SYNTHESIZED parent
    # (never in type_vocab.items) that isn't present in _hierarchy()'s
    # normal fixture -- build a small dedicated one here.
    h = TypeHierarchy(
        edges=[_edge("type_doc", "type_synth"), _edge("type_bio", "type_synth")],
        children={"type_synth": ["type_doc", "type_bio"]},
        parents={"type_doc": ["type_synth"], "type_bio": ["type_synth"]},
        roots=["type_synth"],
    )
    tv = _type_vocab()  # has type_doc/type_bio but NOT type_synth
    rv = _relation_vocab()
    constraints = []  # nothing pre-existing -- every row here is "(new)"
    triplets = [
        {"subject": "A", "subject_type": "documentary film", "relation": "directed",
         "object": "B", "object_type": "documentary film"},
        {"subject": "C", "subject_type": "biographical film", "relation": "directed",
         "object": "D", "object_type": "biographical film"},
    ]
    # affected_ids must include type_synth itself for the relation to even
    # be picked up, but here we're diffing against an EMPTY old constraint
    # list, so pass the relation's own types directly.
    affected = {"type_doc", "type_bio", "type_synth"}
    # Force the relation to be "affected" by seeding one throwaway old
    # constraint referencing type_doc (diff_affected_constraints only scans
    # relations with an EXISTING constraint touching affected_ids).
    constraints = [
        RelationConstraint(relation_id="rel_directed", domain_type_id="type_doc",
                            range_type_id="type_doc", support=1, total=1,
                            pca_confidence=1.0, strength=ConstraintStrength.HARD),
    ]

    rows = diff_affected_constraints(
        h, h, constraints, triplets, tv, rv, ConstraintConfig(), affected,
        synthesized_labels={"type_synth": "audiovisual work"},
    )
    assert rows, "expected at least one row for the affected relation"
    assert any(r["domain"] == "audiovisual work" for r in rows), rows
    assert not any(r["domain"] == "type_synth" for r in rows)  # never the raw id
