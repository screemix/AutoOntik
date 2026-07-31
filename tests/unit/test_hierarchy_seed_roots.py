"""
Tests for hierarchy_induction.py's nested seed-root support.

config.seed_roots can now express a multi-level seed hierarchy (e.g. DOLCE's
Endurant -> Physical Endurant / Non-Physical Endurant), not just a flat list
of root labels. `_apply_seed_roots` is pure data manipulation (no embedder/LLM
calls), so it's tested directly rather than through the full induce_hierarchy()
pipeline.
"""

from __future__ import annotations

from src.ontodisco.hierarchy_induction import HierarchyEdge, _Node, _apply_seed_roots, _node_context
from src.ontodisco.relation_dedup import CanonicalRelation, RelationDeduplicationResult


def _make_pool(*labels: str) -> dict[str, _Node]:
    pool: dict[str, _Node] = {}
    for i, label in enumerate(labels):
        type_id = f"type_{i:04d}"
        pool[type_id] = _Node(type_id=type_id, label=label, profile={}, is_leaf=True)
    return pool


def test_flat_seed_list_is_backward_compatible():
    pool = _make_pool("Endurant", "Perdurant")
    all_nodes = dict(pool)
    edges: list[HierarchyEdge] = []

    pinned_roots = _apply_seed_roots(["Endurant", "Perdurant"], pool, all_nodes, edges, [0])

    assert pinned_roots == set(pool.keys())
    assert edges == []  # no nesting -> no seed edges


def test_nested_seed_creates_seed_edge_and_hides_child_from_pool():
    pool = _make_pool("Physical Object")  # will be matched as a nested child
    all_nodes = dict(pool)
    edges: list[HierarchyEdge] = []
    counter = [0]

    seed_items = [
        {
            "label": "Endurant",
            "children": [
                {"label": "Physical Endurant", "children": ["Physical Object"]},
                "Non-Physical Endurant",
            ],
        },
    ]
    pinned_roots = _apply_seed_roots(seed_items, pool, all_nodes, edges, counter)

    # Only "Endurant" is a root -- it's the only one with no seed parent.
    assert len(pinned_roots) == 1
    (root_id,) = pinned_roots
    assert all_nodes[root_id].label == "Endurant"
    assert root_id in pool

    # Nested labels were synthesized/matched but never added as pool roots.
    physical_endurant_id = next(
        tid for tid, n in all_nodes.items() if n.label == "Physical Endurant"
    )
    non_physical_id = next(
        tid for tid, n in all_nodes.items() if n.label == "Non-Physical Endurant"
    )
    physical_object_id = next(
        tid for tid, n in all_nodes.items() if n.label == "Physical Object"
    )
    assert physical_endurant_id not in pool
    assert non_physical_id not in pool
    # "Physical Object" pre-existed as a real pool member; nesting it under a
    # seed parent must pop it out of pool so it isn't ALSO independently
    # routed through a priority band as an unplaced leaf.
    assert physical_object_id not in pool

    # Seed edges: Physical Endurant -> Endurant, Non-Physical Endurant ->
    # Endurant, Physical Object -> Physical Endurant. All marked is_seed=True
    # with no LLM confidence.
    assert len(edges) == 3
    by_child = {e.child_type_id: e for e in edges}
    assert by_child[physical_endurant_id].parent_type_id == root_id
    assert by_child[non_physical_id].parent_type_id == root_id
    assert by_child[physical_object_id].parent_type_id == physical_endurant_id
    for e in edges:
        assert e.is_seed is True
        assert e.llm_score is None
        assert e.is_direct is True


def test_seed_label_matching_existing_root_type_stays_pinned_and_in_pool():
    pool = _make_pool("Endurant")  # already present in T*
    all_nodes = dict(pool)
    edges: list[HierarchyEdge] = []

    pinned_roots = _apply_seed_roots(["Endurant"], pool, all_nodes, edges, [0])

    assert len(pinned_roots) == 1
    (root_id,) = pinned_roots
    assert root_id in pool
    assert pool[root_id].label == "Endurant"
    # No synthesized duplicate was created for the matched label.
    assert len(all_nodes) == 1


def test_description_becomes_definition_for_synthesized_node_only():
    pool = _make_pool("Endurant")  # already present in T*, no definition field on CanonicalType
    all_nodes = dict(pool)
    edges: list[HierarchyEdge] = []

    seed_items = [
        {"label": "Endurant", "description": "should be ignored -- Endurant already exists in T*"},
        {"label": "Perdurant", "description": "only partially present at any time it exists"},
    ]
    _apply_seed_roots(seed_items, pool, all_nodes, edges, [0])

    endurant = next(n for n in all_nodes.values() if n.label == "Endurant")
    perdurant = next(n for n in all_nodes.values() if n.label == "Perdurant")
    # Matched existing type: definition untouched (stays "" as it always was).
    assert endurant.definition == ""
    # Freshly synthesized: description flows into _Node.definition.
    assert perdurant.definition == "only partially present at any time it exists"


def test_node_context_reaches_llm_prompt_for_synthesized_seed_node():
    # A freshly-synthesized seed node has an empty relation profile, so
    # describe_relation_context alone would return "" -- the LLM would see
    # a bare label with zero context. definition must fill that gap.
    node = _Node(type_id="type_seed0000", label="conceptual entity", profile={}, is_leaf=False,
                 definition="a type of entity")
    rel_vocab = RelationDeduplicationResult(items={}, surface_to_id={})

    ctx = _node_context(node, {node.type_id: node.profile}, rel_vocab, top_k=5)

    assert ctx == "a type of entity"
    # Mirrors resolve_hierarchy_relation's own parenthetical rendering
    # (openai_utils.py) -- label + description together, in parens.
    assert f"{node.label} (context: {ctx})" == "conceptual entity (context: a type of entity)"


def test_node_context_combines_definition_with_relation_signature():
    directed = CanonicalRelation(item_id="rel_0001", canonical_label="directed")
    rel_vocab = RelationDeduplicationResult(items={"rel_0001": directed}, surface_to_id={})
    node = _Node(type_id="type_0001", label="film", profile={"rel_0001": 5.0}, is_leaf=True,
                 definition="a work of visual art")

    ctx = _node_context(node, {node.type_id: node.profile}, rel_vocab, top_k=5)

    assert ctx == "a work of visual art; often appears with: directed"


def test_duplicate_nested_label_under_two_parents_keeps_first_and_warns(caplog):
    pool: dict[str, _Node] = {}
    all_nodes: dict[str, _Node] = {}
    edges: list[HierarchyEdge] = []

    seed_items = [
        {"label": "Endurant", "children": ["Concept"]},
        {"label": "Perdurant", "children": ["Concept"]},
    ]
    _apply_seed_roots(seed_items, pool, all_nodes, edges, [0])

    concept_edges = [e for e in edges if all_nodes[e.child_type_id].label == "Concept"]
    assert len(concept_edges) == 1  # second occurrence ignored, not a second edge
    assert "already has a seed parent" in caplog.text
