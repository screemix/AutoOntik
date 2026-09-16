"""Hierarchy viewer -- the same tree scripts/visualize_hierarchy.py produces,
embedded directly rather than reimplemented, so this page can never drift
from that tool's own rendering."""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.webapp.lib.data import require_run_bundle  # noqa: E402
from scripts.visualize_hierarchy import build_tree, render_html  # noqa: E402

st.set_page_config(page_title="Hierarchy - AutoOntic", layout="wide")
st.title("Type Hierarchy")

bundle = require_run_bundle()

if bundle.hierarchy_result is None or bundle.type_vocab is None:
    st.error("This run has no hierarchy_induction.pkl / type_dedup.pkl checkpoint to show.")
    st.stop()

h = bundle.hierarchy_result.hierarchy
col1, col2, col3, col4 = st.columns(4)
col1.metric("Roots", len(h.roots))
col2.metric("Edges", len(h.edges))
col3.metric("Types (T*)", len(bundle.type_vocab.items))
col4.metric("Synthesized types", len(bundle.hierarchy_result.synthesized_types))

deferred = getattr(bundle.hierarchy_result, "deferred_low_support", None)
unresolved = getattr(bundle.hierarchy_result, "unresolved_children", None)
if deferred:
    st.info(f"{len(deferred)} type(s) deferred by min_type_support (queued, not placed).")
if unresolved:
    n = sum(len(v) for v in unresolved.values())
    if n:
        st.info(f"{n} type(s) deferred past max_depth across {len(unresolved)} bucket(s).")

tree = build_tree(bundle.hierarchy_result, bundle.type_vocab)
html = render_html(tree)
components.html(html, height=900, scrolling=True)
