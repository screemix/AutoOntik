"""Entity graph viewer: click one or more types in the induced hierarchy
tree, and render the induced subgraph over every entity under any of them
(no separate hop-radius search -- type selection is the scoping mechanism).
The full KG is too large to render at once, so nothing is drawn until at
least one type is selected."""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.webapp.lib.data import require_run_bundle, load_triplets_cached, type_label_lookup  # noqa: E402
from scripts.webapp.lib.graph import build_graph, color_group_for, type_selection_subgraph  # noqa: E402
from scripts.webapp.components.type_tree_selector import type_tree_selector  # noqa: E402
from scripts.webapp.lib.sandbox import subtree_ids  # noqa: E402

st.set_page_config(page_title="Graph - AutoOntic", layout="wide")
st.title("Entity Graph")

bundle = require_run_bundle()

if bundle.entity_vocab is None or bundle.type_vocab is None or bundle.relation_vocab is None:
    st.error("This run is missing entity_dedup.pkl / type_dedup.pkl / relation_dedup.pkl -- "
             "all three are needed to resolve the graph.")
    st.stop()
if bundle.hierarchy_result is None:
    st.error("This run is missing hierarchy_induction.pkl -- needed to render the type tree.")
    st.stop()
triplets_path = bundle.triplets_path
if triplets_path is None:
    st.warning(
        "Couldn't auto-locate this run's raw triplets file (no run_metadata.json, or its "
        "input_path doesn't exist). Enter the path below -- it's the same JSONL file the "
        "pipeline was run on."
    )
    manual = st.text_input("Triplets JSONL path (relative to repo root, or absolute)", "")
    if not manual.strip():
        st.stop()
    candidate = Path(manual.strip())
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[3] / candidate
    if not candidate.exists():
        st.error(f"No such file: {candidate}")
        st.stop()
    triplets_path = candidate

triplets = load_triplets_cached(str(triplets_path))
index = build_graph(bundle.type_vocab, bundle.relation_vocab, bundle.entity_vocab,
                     str(triplets_path), triplets)

col1, col2, col3 = st.columns(3)
col1.metric("Entities", index.graph.number_of_nodes())
col2.metric("Edges", index.graph.number_of_edges())
col3.metric("Resolved triplets", f"{index.num_resolved}/{index.num_triplets_seen}")

st.divider()
st.subheader("Select types")
st.caption("Click a type node to include every entity under it. Selecting a parent "
           "colors its whole subtree as one group -- pick a more specific node for a "
           "finer-grained legend.")

hierarchy = bundle.hierarchy_result.hierarchy
label_lookup = type_label_lookup(bundle)
tree_nodes = [{"id": tid, "label": label} for tid, label in label_lookup.items()]
tree_edges = [{"child": e.child_type_id, "parent": e.parent_type_id} for e in hierarchy.edges]

selected_type_ids = type_tree_selector(tree_nodes, tree_edges, height=560, key="graph_type_tree")

if not selected_type_ids:
    st.info("Select one or more types above to render their entities.")
    st.stop()

st.divider()
st.subheader("Graph")
max_nodes = st.slider("Max nodes to render", 20, 400, 150, 10,
                       help="Selected-type entities plus everything directly connected to them "
                            "can be too large to usefully view -- this caps it. If the true set "
                            "is bigger, it's truncated and flagged below.")

included_type_ids = set()
for tid in selected_type_ids:
    included_type_ids |= subtree_ids(hierarchy, tid)

selected_ids_set = set(selected_type_ids)
# Directly-connected neighbors of ANY type are pulled in too, not just
# entities whose own type was selected -- a pure induced subgraph over only
# the selected types showed almost no edges, since most relations connect
# DIFFERENT types (e.g. "directed" links a person to a film) -- see
# type_selection_subgraph's own docstring for the measurement that drove this.
sub, truncated = type_selection_subgraph(index, included_type_ids, max_nodes)
if truncated:
    st.warning(
        f"The selected types (plus their directly-connected entities) exceed {max_nodes} "
        "nodes -- showing a truncated subset. Raise the node cap or narrow the type "
        "selection for a complete view."
    )
st.caption(f"Showing {sub.number_of_nodes()} node(s), {sub.number_of_edges()} edge(s)")

if sub.number_of_nodes() == 0:
    st.info("No entities resolve to the selected type(s) in this run's triplets.")
    st.stop()

from pyvis.network import Network  # noqa: E402

NEUTRAL_COLOR = "#B0B0B0"   # reserved for "connected, but not itself a selected type" -- never
                            # handed out to a real selected group, so it can't collide with one
TYPE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#CCB974", "#64B5CD", "#8C6D31",
]
color_groups = {}
has_neighbor_nodes = False
for _, d in sub.nodes(data=True):
    group = color_group_for(d.get("type_id"), hierarchy, selected_ids_set)
    if group is not None and group not in color_groups:
        color_groups[group] = TYPE_PALETTE[len(color_groups) % len(TYPE_PALETTE)]
    elif group is None:
        has_neighbor_nodes = True

net = Network(height="750px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
net.barnes_hut(gravity=-8000, spring_length=120)
for node_id, data in sub.nodes(data=True):
    group = color_group_for(data.get("type_id"), hierarchy, selected_ids_set)
    is_core = data.get("type_id") in included_type_ids
    color = color_groups.get(group, NEUTRAL_COLOR)
    t = data.get("type_label", "?")
    net.add_node(
        node_id,
        label=data.get("label", node_id),
        title=f"{data.get('label')}  [{t}]",
        color=color,
        size=24 if is_core else 14,
        borderWidth=3 if is_core else 1,
    )
for u, v, data in sub.edges(data=True):
    net.add_edge(u, v, label=data.get("label", ""), title=data.get("label", ""))

net.set_options("""
{
  "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}, "font": {"size": 10, "align": "middle"}},
  "physics": {"stabilization": {"iterations": 150}}
}
""")

html = net.generate_html(notebook=False)
components.html(html, height=780, scrolling=True)

st.markdown("**Legend**")
st.caption("Larger, bordered nodes are entities of a selected type; small plain nodes are "
           "directly-connected entities of other types, shown for context.")
legend_html = " &nbsp;&nbsp; ".join(
    f"<span style='color:{color}'>&#9679;</span> {label_lookup.get(group, group)}"
    for group, color in color_groups.items()
)
if has_neighbor_nodes:
    legend_html += f" &nbsp;&nbsp; <span style='color:{NEUTRAL_COLOR}'>&#9679;</span> (connected, not a selected type)"
st.markdown(legend_html, unsafe_allow_html=True)
