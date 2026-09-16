"""Domain/range constraint viewer."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.webapp.lib.data import require_run_bundle  # noqa: E402

st.set_page_config(page_title="Constraints - AutoOntic", layout="wide")
st.title("Domain/Range Constraints")

bundle = require_run_bundle()

if not bundle.constraints:
    st.error("This run has no constraints.pkl checkpoint (or it's empty).")
    st.stop()

rv, tv = bundle.relation_vocab, bundle.type_vocab
relation_label = lambda rid: rv.items[rid].canonical_label if rid in rv.items else rid
type_label = lambda tid: tv.items[tid].canonical_label if tid in tv.items else tid

rows = [
    {
        "relation": relation_label(c.relation_id),
        "domain": type_label(c.domain_type_id),
        "range": type_label(c.range_type_id),
        "strength": c.strength.value,
        "confidence": round(c.pca_confidence, 3),
        "support": c.support,
        "total": c.total,
    }
    for c in bundle.constraints
]
df = pd.DataFrame(rows)

strength_order = {"hard": 0, "soft": 1, "hint": 2}
col1, col2, col3 = st.columns(3)
col1.metric("Constraints", len(df))
col2.metric("Hard", int((df["strength"] == "hard").sum()))
col3.metric("Distinct relations", df["relation"].nunique())

st.divider()

c1, c2, c3 = st.columns([2, 2, 3])
strength_filter = c1.multiselect("Strength", options=["hard", "soft", "hint"], default=["hard", "soft", "hint"])
relation_query = c2.text_input("Relation contains", "")
min_conf = c3.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

filtered = df[df["strength"].isin(strength_filter) & (df["confidence"] >= min_conf)]
if relation_query.strip():
    filtered = filtered[filtered["relation"].str.contains(relation_query.strip(), case=False, na=False)]
filtered = filtered.assign(_order=filtered["strength"].map(strength_order)).sort_values(
    ["_order", "confidence"], ascending=[True, False]
).drop(columns="_order")

st.caption(f"{len(filtered)} of {len(df)} constraints shown")
st.dataframe(filtered, use_container_width=True, hide_index=True)
