"""Entity graph viewer: search by name/type, then render a bounded k-hop
neighborhood. The full KG is too large to render at once, so nothing is
drawn until the user picks a seed and a hop radius."""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.webapp.lib.data import require_run_bundle, load_triplets_cached  # noqa: E402
from scripts.webapp.lib.graph import build_graph, search_entities, neighborhood  # noqa: E402

st.set_page_config(page_title="Graph - AutoOntic", layout="wide")
st.title("Entity Graph")

bundle = require_run_bundle()

if bundle.entity_vocab is None or bundle.type_vocab is None or bundle.relation_vocab is None:
    st.error("This run is missing entity_dedup.pkl / type_dedup.pkl / relation_dedup.pkl -- "
             "all three are needed to resolve the graph.")
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
st.subheader("Find a starting entity")

c1, c2 = st.columns([2, 1])
name_query = c1.text_input("Name contains", "")
type_options = ["(any type)"] + sorted(index.types_by_label.keys())
type_choice = c2.selectbox("Type", options=type_options)
type_filter = None if type_choice == "(any type)" else type_choice

if not name_query and type_filter is None:
    st.info("Type a name substring and/or pick a type to search.")
    st.stop()

matches = search_entities(index, name_query, type_filter, limit=300)
if not matches:
    st.warning("No entities match.")
    st.stop()

match_labels = [f"{m['label']}  [{m['type']}]" for m in matches]
picked = st.multiselect(
    f"{len(matches)} match(es) -- pick one or more as the neighborhood's seed(s)",
    options=range(len(matches)), format_func=lambda i: match_labels[i],
)
if not picked:
    st.stop()
seed_ids = [matches[i]["id"] for i in picked]

st.divider()
st.subheader("Neighborhood")
c1, c2 = st.columns(2)
k = c1.slider("Hops (k)", 0, 4, 1)
max_nodes = c2.slider("Max nodes to render", 20, 400, 150, 10,
                       help="The graph can be too large to usefully view -- this caps it. "
                            "If the true neighborhood is bigger, it's truncated and flagged below.")

sub, truncated = neighborhood(index, seed_ids, k, max_nodes)
if truncated:
    st.warning(
        f"The full {k}-hop neighborhood exceeds {max_nodes} nodes -- showing a truncated subset. "
        "Reduce k, lower the node cap's target, or narrow the seed selection for a complete view."
    )
st.caption(f"Showing {sub.number_of_nodes()} node(s), {sub.number_of_edges()} edge(s)")

if sub.number_of_nodes() == 0:
    st.info("Empty neighborhood.")
    st.stop()

from pyvis.network import Network  # noqa: E402

TYPE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
type_color = {}
for t in sorted({d.get("type_label", "?") for _, d in sub.nodes(data=True)}):
    type_color[t] = TYPE_PALETTE[len(type_color) % len(TYPE_PALETTE)]

net = Network(height="750px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
net.barnes_hut(gravity=-8000, spring_length=120)
for node_id, data in sub.nodes(data=True):
    t = data.get("type_label", "?")
    is_seed = node_id in seed_ids
    net.add_node(
        node_id,
        label=data.get("label", node_id),
        title=f"{data.get('label')}  [{t}]",
        color=type_color[t],
        size=26 if is_seed else 16,
        borderWidth=3 if is_seed else 1,
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

with st.expander("Legend"):
    for t, color in type_color.items():
        st.markdown(f"<span style='color:{color}'>&#9679;</span> {t}", unsafe_allow_html=True)
