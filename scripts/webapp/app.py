"""AutoOntic run viewer -- home page.

Run with:
    venv/bin/streamlit run scripts/webapp/app.py --server.port 8501 --server.address 0.0.0.0

Pick a run here; the choice is remembered (st.session_state) across the
Hierarchy / Constraints / Graph pages in the sidebar.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `from src...` / `from scripts...` imports

from scripts.webapp.lib.data import discover_runs, run_label, load_run  # noqa: E402

st.set_page_config(page_title="AutoOntic Run Viewer", layout="wide")

st.title("AutoOntic Run Viewer")
st.caption(
    "Browse an ontology-discovery pipeline run: the induced type hierarchy, "
    "domain/range constraints, and the entity graph."
)

runs = discover_runs()
if not runs:
    st.error(
        "No pipeline runs found. Expected at least one `<output_dir>/checkpoints/run_N/` "
        "folder with checkpoint .pkl files somewhere under the repo."
    )
    st.stop()

labels = [run_label(r) for r in runs]
default_idx = 0
if "run_dir" in st.session_state:
    try:
        default_idx = runs.index(st.session_state["run_dir"])
    except ValueError:
        pass

choice = st.selectbox(
    "Select a run",
    options=range(len(runs)),
    format_func=lambda i: labels[i],
    index=default_idx,
    help="output/ and output/mine/ (etc.) number runs INDEPENDENTLY -- the same "
         "'run_N' name can refer to two different runs, so the output_dir is always shown.",
)
st.session_state["run_dir"] = runs[choice]

bundle = load_run(runs[choice])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Relations", len(bundle.relation_vocab.items) if bundle.relation_vocab else "—")
col2.metric("Types", len(bundle.type_vocab.items) if bundle.type_vocab else "—")
h = bundle.hierarchy_result
col3.metric("Hierarchy roots", len(h.hierarchy.roots) if h else "—")
col4.metric("Entities", len(bundle.entity_vocab.items) if bundle.entity_vocab else "—")
col5.metric("Constraints", len(bundle.constraints) if bundle.constraints is not None else "—")

st.markdown(f"**Run path:** `{bundle.run_dir}`")
if bundle.triplets_path:
    st.markdown(f"**Raw triplets (for the Graph page):** `{bundle.triplets_path}`")
else:
    st.warning(
        "Could not locate this run's raw triplets file (no run_metadata.json, or its "
        "input_path doesn't exist on disk). The Graph page needs it -- you can still "
        "browse Hierarchy and Constraints."
    )

st.divider()
st.markdown(
    "Use the sidebar to open **Hierarchy**, **Constraints**, or **Graph**. "
    "The run selected here carries over to all three."
)
