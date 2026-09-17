"""Reparent Lab: drag a branch of the induced type hierarchy onto a new
parent, see which entities newly become merge candidates and which
domain/range constraints are affected (both instant, no LLM), then
optionally submit a real, bounded, LLM-backed re-merge of just the affected
entities.

Everything here is an in-memory sandbox: `bundle` (loaded read-only by
require_run_bundle) is never mutated. The working hierarchy lives in
st.session_state and is deep-copied once per run selection. Saving to disk
is a separate, explicit action (save_sandbox_as_run) that always writes a
brand-new run directory -- the source run's checkpoints are never touched.
"""
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.webapp.lib.data import require_run_bundle, load_triplets_cached  # noqa: E402
from scripts.webapp.lib.sandbox import (  # noqa: E402
    affected_type_ids, demo_reparent, diff_affected_constraints,
    diff_merge_candidates, run_sandbox_remerge, save_sandbox_as_run,
)
from scripts.webapp.components.reparent_tree import reparent_tree  # noqa: E402
from src.ontodisco.constraints import ConstraintConfig  # noqa: E402
from src.ontodisco.utils.dedup_base import ContrieverEmbedder  # noqa: E402
from src.ontodisco.utils.openai_utils import LLMTripletExtractor  # noqa: E402


def _load_run_config(run_dir: Path) -> dict:
    """The exact PipelineConfig this run was actually built with, straight
    from run_metadata.json -- NOT a hardcoded config file, so Submit re-runs
    merges with the same model/embedder the rest of this ontology used
    rather than silently substituting a different one."""
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text()).get("config", {})

st.set_page_config(page_title="Reparent Lab - AutoOntic", layout="wide")
st.title("Reparent Lab")
st.caption(
    "Drag a node onto a new parent to preview a hierarchy edit. Sandbox only -- "
    "the original run's checkpoints are never modified."
)

bundle = require_run_bundle()
if bundle.hierarchy_result is None or bundle.type_vocab is None or bundle.entity_vocab is None:
    st.error("This run is missing hierarchy_induction.pkl / type_dedup.pkl / entity_dedup.pkl.")
    st.stop()

run_key = str(bundle.run_dir)
if st.session_state.get("sandbox_run_key") != run_key:
    st.session_state["sandbox_run_key"] = run_key
    st.session_state["sandbox_hierarchy"] = copy.deepcopy(bundle.hierarchy_result.hierarchy)
    st.session_state["sandbox_history"] = []
    st.session_state["sandbox_last_moved"] = None
    st.session_state["sandbox_merge_result"] = None
    st.session_state["sandbox_component_key"] = 0

hierarchy = st.session_state["sandbox_hierarchy"]
type_vocab = bundle.type_vocab

c1, c2, c3 = st.columns([3, 1, 1])
c1.caption(f"{len(st.session_state['sandbox_history'])} move(s) applied this session.")
if c2.button("Undo last move", disabled=not st.session_state["sandbox_history"]):
    st.session_state["sandbox_history"].pop()
    st.session_state["sandbox_hierarchy"] = copy.deepcopy(bundle.hierarchy_result.hierarchy)
    for child_id, new_parent_id in st.session_state["sandbox_history"]:
        st.session_state["sandbox_hierarchy"] = demo_reparent(
            st.session_state["sandbox_hierarchy"], child_id, new_parent_id).hierarchy
    st.session_state["sandbox_last_moved"] = None
    st.session_state["sandbox_merge_result"] = None
    st.session_state["sandbox_component_key"] += 1
    st.rerun()
if c3.button("Reset to original", disabled=not st.session_state["sandbox_history"]):
    st.session_state["sandbox_hierarchy"] = copy.deepcopy(bundle.hierarchy_result.hierarchy)
    st.session_state["sandbox_history"] = []
    st.session_state["sandbox_last_moved"] = None
    st.session_state["sandbox_merge_result"] = None
    st.session_state["sandbox_component_key"] += 1
    st.rerun()

# ── Tree ──────────────────────────────────────────────────────────────────
# Node ids come from TWO sources: type_vocab.items (Step 2's canonical
# types) AND hierarchy_result.synthesized_types (new abstract types minted
# DURING hierarchy induction, e.g. "type_h0003" -- never folded back into
# type_dedup.pkl, so they're invisible if you only iterate type_vocab.items).
# hierarchy.edges references both kinds of id interchangeably; omitting the
# synthesized half silently orphans every edge that points at one (several
# of run_17's own hierarchy roots are synthesized types) and breaks the
# client-side d3.stratify() call with a "missing parent" error.
synthesized_labels = {s.type_id: s.canonical_label for s in bundle.hierarchy_result.synthesized_types}
label_of = lambda tid: (
    type_vocab.items[tid].canonical_label if tid in type_vocab.items
    else synthesized_labels.get(tid, tid)
)
all_node_ids = set(type_vocab.items) | set(synthesized_labels)
nodes = [{"id": tid, "label": label_of(tid)} for tid in all_node_ids]
edges = [{"child": e.child_type_id, "parent": e.parent_type_id} for e in hierarchy.edges]
highlight_ids = list(st.session_state["sandbox_last_moved"] or [])

result = reparent_tree(
    nodes, edges, highlight_ids=highlight_ids, height=650,
    key=f"reparent_tree_{st.session_state['sandbox_component_key']}",
)

if result and result != st.session_state.get("sandbox_last_applied"):
    st.session_state["sandbox_last_applied"] = result
    outcome = demo_reparent(hierarchy, result["moved"], result["new_parent"])
    if outcome.ok:
        st.session_state["sandbox_hierarchy"] = outcome.hierarchy
        st.session_state["sandbox_history"].append((result["moved"], result["new_parent"]))
        st.session_state["sandbox_last_moved"] = {result["moved"], result["new_parent"]}
        st.session_state["sandbox_merge_result"] = None
        st.session_state["sandbox_component_key"] += 1
        st.rerun()
    else:
        st.error(f"Reparent rejected: {outcome.reason}")

if not st.session_state["sandbox_history"]:
    st.info("Drag a node onto another node to preview a reparent.")
    st.stop()

last_child_id, _ = st.session_state["sandbox_history"][-1]
old_hierarchy = copy.deepcopy(bundle.hierarchy_result.hierarchy)
for child_id, new_parent_id in st.session_state["sandbox_history"][:-1]:
    old_hierarchy = demo_reparent(old_hierarchy, child_id, new_parent_id).hierarchy
affected_ids = affected_type_ids(old_hierarchy, hierarchy, last_child_id)

st.divider()

# ── Merge candidates (instant) ───────────────────────────────────────────
st.subheader("Merge candidates")
candidates = diff_merge_candidates(old_hierarchy, hierarchy, bundle.entity_vocab, affected_ids)
if candidates:
    st.dataframe(pd.DataFrame(candidates)[["entity_a", "entity_b", "new_partition_key"]],
                 use_container_width=True, hide_index=True)
else:
    st.caption("No new merge candidates from this move.")

# ── Affected constraints (instant) ───────────────────────────────────────
st.subheader("Affected constraints")
if bundle.triplets_path and bundle.constraints and bundle.relation_vocab:
    triplets = load_triplets_cached(str(bundle.triplets_path))
    constraint_rows = diff_affected_constraints(
        old_hierarchy, hierarchy, bundle.constraints, triplets,
        type_vocab, bundle.relation_vocab, ConstraintConfig(), affected_ids,
    )
    if constraint_rows:
        st.dataframe(pd.DataFrame(constraint_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No constraints affected by this move.")
else:
    st.caption("No constraints/triplets available for this run to diff against.")

st.divider()

# ── Submit: real, bounded, LLM-backed re-merge ───────────────────────────
st.subheader("Submit")
st.caption(
    f"Re-opens every surface form under the {len(affected_ids)} affected type(s) and re-runs "
    "the real merge-round verification (actual LLM calls, bounded to this slice)."
)
run_config = _load_run_config(bundle.run_dir)
llm_cfg = run_config.get("llm", {})
embed_cfg = run_config.get("embedding", {})
entity_cfg = run_config.get("entity_canonicalization", {})
if not llm_cfg:
    st.caption(
        "⚠️ No run_metadata.json found for this run -- can't recover the exact model/embedder "
        "it was built with. Submit will be disabled."
    )

if st.button("Re-merge affected entities", type="primary", disabled=not llm_cfg):
    api_key_env = llm_cfg.get("api_key_env", "")
    api_key = os.environ.get(api_key_env)
    proxy_key_env = llm_cfg.get("proxy_key_env")
    proxy = os.environ.get(proxy_key_env) if proxy_key_env else None
    if not api_key:
        st.error(f"{api_key_env} is not set in the environment.")
    else:
        with st.spinner(f"Re-merging with {llm_cfg.get('model')} (the model this run was built with)..."):
            llm_extractor = LLMTripletExtractor(
                api_key=api_key, model=llm_cfg.get("model"), base_url=llm_cfg.get("base_url"), proxy=proxy)
            embedder = ContrieverEmbedder(
                embed_cfg.get("contriever_model", "facebook/contriever"), device=embed_cfg.get("device"))
            merged = run_sandbox_remerge(
                bundle.entity_vocab, type_vocab, hierarchy, affected_ids,
                llm_extractor, embedder,
                similarity_threshold=entity_cfg.get("similarity_threshold", 0.85),
                embed_batch_size=embed_cfg.get("embed_batch_size", 64),
                max_merge_rounds=entity_cfg.get("max_merge_rounds", 5),
                max_parallel_workers=entity_cfg.get("max_parallel_workers", 8),
                max_cluster_size=entity_cfg.get("max_cluster_size", 40),
            )
        st.session_state["sandbox_merge_result"] = merged
        pt, ct = llm_extractor.calculate_used_tokens()
        st.success(f"Re-merge complete: {len(merged)} canonical entities from this slice. "
                   f"Tokens: {pt + ct} (${llm_extractor.calculate_cost():.4f})")

if st.session_state.get("sandbox_merge_result"):
    merged = st.session_state["sandbox_merge_result"]
    st.dataframe(
        pd.DataFrame([{"canonical_label": e.canonical_label, "surface_forms": len(e.surface_forms)}
                      for e in merged]),
        use_container_width=True, hide_index=True,
    )
    if st.button("Save this sandbox as a new run"):
        new_dir = save_sandbox_as_run(bundle, hierarchy, merged)
        st.success(f"Saved to {new_dir}")
