"""
Tests for scripts/webapp/lib/graph.py's Graph-page helpers: color_group_for
(nearest-selected-ancestor coloring) and type_selection_subgraph (core +
directly-connected-neighbor node selection). Mirrors
tests/unit/test_sandbox.py's synthetic-hierarchy fixture style (same
film/documentary-film/ocean-current shape).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from scripts.webapp.lib.graph import GraphIndex, color_group_for, type_selection_subgraph
from src.ontodisco.hierarchy_induction import HierarchyEdge, TypeHierarchy


def _edge(child, parent):
    return HierarchyEdge(child_type_id=child, parent_type_id=parent,
                          relation_signature_score=None, llm_score=0.9, ensemble_score=0.9)


def _hierarchy():
    # film -> {documentary film -> biographical documentary}; ocean current
    # is an unrelated root.
    edges = [
        _edge("type_doc", "type_film"),
        _edge("type_bio_doc", "type_doc"),
    ]
    return TypeHierarchy(
        edges=edges,
        children={"type_film": ["type_doc"], "type_doc": ["type_bio_doc"]},
        parents={"type_doc": ["type_film"], "type_bio_doc": ["type_doc"]},
        roots=["type_film", "type_ocean"],
    )


def test_selected_type_colors_as_itself():
    h = _hierarchy()
    assert color_group_for("type_doc", h, {"type_doc"}) == "type_doc"


def test_descendant_of_selected_parent_colors_as_the_parent():
    h = _hierarchy()
    # type_bio_doc wasn't itself clicked, but its ancestor type_film was --
    # this is the whole point: selecting a broad parent shouldn't explode
    # into one legend color per leaf type.
    assert color_group_for("type_bio_doc", h, {"type_film"}) == "type_film"


def test_nearer_selected_ancestor_wins_over_a_farther_one():
    h = _hierarchy()
    # Both type_doc and type_film are selected; type_bio_doc's nearest
    # selected ancestor is the more specific type_doc, not type_film.
    assert color_group_for("type_bio_doc", h, {"type_doc", "type_film"}) == "type_doc"


def test_unrelated_root_does_not_match_a_disjoint_selection():
    h = _hierarchy()
    assert color_group_for("type_ocean", h, {"type_film"}) is None


def test_no_ancestor_selected_returns_none():
    h = _hierarchy()
    assert color_group_for("type_bio_doc", h, {"type_ocean"}) is None


def test_falsy_type_id_returns_none():
    h = _hierarchy()
    assert color_group_for(None, h, {"type_film"}) is None
    assert color_group_for("", h, {"type_film"}) is None


def _entity_graph_index():
    # A director (person) connects to a documentary film via "directed";
    # the film has no OTHER film-typed neighbor. If only "documentary film"
    # is selected, a pure induced subgraph (core-to-core edges only) would
    # show 1 node and 0 edges -- the exact bug this helper exists to avoid.
    g = nx.MultiDiGraph()
    g.add_node("e_person", label="Jane Director", type_id="type_person")
    g.add_node("e_doc", label="Some Documentary", type_id="type_doc")
    g.add_node("e_ocean", label="Amazon", type_id="type_ocean")
    g.add_edge("e_person", "e_doc", label="directed")
    return GraphIndex(graph=g)


def test_pure_core_selection_pulls_in_directly_connected_neighbor():
    index = _entity_graph_index()
    sub, truncated = type_selection_subgraph(index, {"type_doc"}, max_nodes=50)
    assert not truncated
    assert set(sub.nodes()) == {"e_doc", "e_person"}
    assert sub.number_of_edges() == 1
    assert "e_ocean" not in sub.nodes()  # unrelated entity, not pulled in


def test_core_selection_alone_when_over_cap_drops_neighbors_not_core():
    index = _entity_graph_index()
    sub, truncated = type_selection_subgraph(index, {"type_doc", "type_person"}, max_nodes=1)
    assert truncated
    assert sub.number_of_nodes() == 1
    # Whichever core node survived the cap, it must be a CORE node (person or
    # doc), never the unrelated ocean-current entity.
    assert set(sub.nodes()).issubset({"e_doc", "e_person"})


def test_neighbor_truncation_keeps_all_core_nodes_first():
    index = _entity_graph_index()
    # Both core nodes fit; only room for a partial/no neighbor set.
    sub, truncated = type_selection_subgraph(index, {"type_doc"}, max_nodes=1)
    assert truncated
    assert set(sub.nodes()) == {"e_doc"}  # the one core node, no room for its neighbor
