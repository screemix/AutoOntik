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
from typing import Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.webapp.lib.data import require_run_bundle, load_triplets_cached, type_label_lookup  # noqa: E402
from scripts.webapp.lib.sandbox import (  # noqa: E402
    affected_type_ids, demo_delete_subtree, demo_reparent, diff_affected_constraints,
    diff_delete_impact, diff_merge_candidates, run_sandbox_remerge, save_sandbox_as_run,
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


@st.cache_resource(show_spinner="Loading embedder...")
def _load_embedder(model_name: str, device: Optional[str]) -> ContrieverEmbedder:
    return ContrieverEmbedder(model_name, device=device)

st.set_page_config(page_title="Ontology Lab - AutoOntic", layout="wide")
# No st.title/st.caption here on purpose -- the tree component's own small
# hint line already carries the instructions ("Drag a node onto a new
# parent, or click..."), and a full title block above it was redundant,
# stacked header space eating into the one thing this page is actually for.

bundle = require_run_bundle()
if bundle.hierarchy_result is None or bundle.type_vocab is None or bundle.entity_vocab is None:
    st.error("This run is missing hierarchy_induction.pkl / type_dedup.pkl / entity_dedup.pkl.")
    st.stop()

run_config = _load_run_config(bundle.run_dir)
llm_cfg = run_config.get("llm", {})
embed_cfg = run_config.get("embedding", {})
entity_cfg = run_config.get("entity_canonicalization", {})

run_key = str(bundle.run_dir)
if st.session_state.get("sandbox_run_key") != run_key:
    st.session_state["sandbox_run_key"] = run_key
    st.session_state["sandbox_hierarchy"] = copy.deepcopy(bundle.hierarchy_result.hierarchy)
    # A unified edit history so Undo/Reset can replay either kind of edit:
    # {"op": "reparent", "child": id, "new_parent": id} or
    # {"op": "delete", "node": id, "deleted_ids": frozenset(...)} -- deleted_ids
    # is captured at delete time (not recomputed later) since replaying deletes
    # in order against a freshly-rebuilt hierarchy is simplest done by just
    # re-applying demo_delete_subtree(node) again, which recomputes it fresh
    # anyway; kept here too for the live impact panel without recomputing.
    st.session_state["sandbox_history"] = []
    st.session_state["sandbox_last_moved"] = None
    st.session_state["sandbox_merge_result"] = None

hierarchy = st.session_state["sandbox_hierarchy"]
type_vocab = bundle.type_vocab


def _replay(entries, base_hierarchy):
    """Rebuild a hierarchy by replaying a prefix of sandbox_history from
    scratch. Shared by Undo, Reset, and the "hierarchy before the last edit"
    diff baseline -- one implementation for both edit kinds."""
    h = base_hierarchy
    for entry in entries:
        if entry["op"] == "reparent":
            h = demo_reparent(h, entry["child"], entry["new_parent"]).hierarchy
        else:
            h = demo_delete_subtree(h, entry["node"]).hierarchy
    return h


def _undo_last_move():
    st.session_state["sandbox_history"].pop()
    st.session_state["sandbox_hierarchy"] = _replay(
        st.session_state["sandbox_history"], copy.deepcopy(bundle.hierarchy_result.hierarchy))
    st.session_state["sandbox_last_moved"] = None
    st.session_state["sandbox_merge_result"] = None


def _reset_to_original():
    st.session_state["sandbox_hierarchy"] = copy.deepcopy(bundle.hierarchy_result.hierarchy)
    st.session_state["sandbox_history"] = []
    st.session_state["sandbox_last_moved"] = None
    st.session_state["sandbox_merge_result"] = None

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
label_lookup = type_label_lookup(bundle)
label_of = lambda tid: label_lookup.get(tid, tid)
nodes = [{"id": tid, "label": label} for tid, label in label_lookup.items()]
edges = [{"child": e.child_type_id, "parent": e.parent_type_id} for e in hierarchy.edges]
highlight_ids = list(st.session_state["sandbox_last_moved"] or [])

result = reparent_tree(
    nodes, edges, highlight_ids=highlight_ids, height=820,
    can_undo=bool(st.session_state["sandbox_history"]),
    # Deliberately STABLE across edits (keyed only by which run is loaded,
    # not by an incrementing counter) -- a changing key forces Streamlit to
    # tear down and recreate the iframe, wiping the component's own
    # client-side state (collapse-to-expand state, current zoom/pan) on
    # every single reparent/delete/undo. New args (updated nodes/edges)
    # already trigger a fresh render on the SAME iframe; sandbox_last_applied
    # below is what prevents re-processing a stale returned value, so
    # nothing here actually needed the key to change.
    key=f"reparent_tree_{run_key}",
)

if result and result != st.session_state.get("sandbox_last_applied"):
    st.session_state["sandbox_last_applied"] = result
    action = result.get("action")
    if action == "undo" and st.session_state["sandbox_history"]:
        _undo_last_move()
        st.rerun()
    elif action == "reset" and st.session_state["sandbox_history"]:
        _reset_to_original()
        st.rerun()
    elif action == "delete":
        outcome = demo_delete_subtree(hierarchy, result["node"])
        if outcome.ok:
            st.session_state["sandbox_hierarchy"] = outcome.hierarchy
            st.session_state["sandbox_history"].append(
                {"op": "delete", "node": result["node"], "deleted_ids": outcome.deleted_ids})
            st.session_state["sandbox_last_moved"] = None  # nothing left to highlight -- it's gone
            st.session_state["sandbox_merge_result"] = None
            st.rerun()
        else:
            st.error(f"Delete rejected: {outcome.reason}")
    elif "moved" in result:
        outcome = demo_reparent(hierarchy, result["moved"], result["new_parent"])
        if outcome.ok:
            st.session_state["sandbox_hierarchy"] = outcome.hierarchy
            st.session_state["sandbox_history"].append(
                {"op": "reparent", "child": result["moved"], "new_parent": result["new_parent"]})
            st.session_state["sandbox_last_moved"] = {result["moved"], result["new_parent"]}
            st.session_state["sandbox_merge_result"] = None
            st.rerun()
        else:
            st.error(f"Reparent rejected: {outcome.reason}")

if not st.session_state["sandbox_history"]:
    st.info("Drag a node onto another node to preview a reparent, or click a node then "
            "\"Delete subtree\" to remove it and its descendants.")
    st.stop()

history = st.session_state["sandbox_history"]
last_entry = history[-1]
old_hierarchy = _replay(history[:-1], copy.deepcopy(bundle.hierarchy_result.hierarchy))

# Deletions accumulate across the whole session (every deleted id stays
# gone even if a later edit is a reparent elsewhere) -- Save needs the full
# set, not just the last edit's.
all_deleted_ids = set()
for entry in history:
    if entry["op"] == "delete":
        all_deleted_ids |= set(entry["deleted_ids"])

st.divider()

embedder = _load_embedder(embed_cfg.get("contriever_model", "facebook/contriever"), embed_cfg.get("device"))

if last_entry["op"] == "delete":
    # ── Delete impact (instant) ──────────────────────────────────────────
    st.subheader("Delete impact")
    deleted_ids = set(last_entry["deleted_ids"])
    st.caption(f"Deleted {label_of(last_entry['node'])!r} and {len(deleted_ids) - 1} descendant type(s).")
    impact = diff_delete_impact(old_hierarchy, hierarchy, bundle.entity_vocab, bundle.constraints or [], deleted_ids)
    if impact["reassigned_entities"]:
        st.markdown(f"**{len(impact['reassigned_entities'])} entit(y/ies) would be reassigned** "
                    "(every one of their types was deleted -- moved to their nearest surviving "
                    "ancestor, or None if there isn't one):")
        reassigned_rows = [
            {"entity": r["entity"], "lost_types": r["lost_types"],
             "reassigned_type": label_of(r["reassigned_type_id"]) if r["reassigned_type_id"] else None}
            for r in impact["reassigned_entities"]
        ]
        st.dataframe(pd.DataFrame(reassigned_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No entities would need reassigning.")
    if impact["narrowed_entities"]:
        st.markdown(f"**{len(impact['narrowed_entities'])} entit(y/ies) would be narrowed** "
                    "(lose some types, keep others):")
        st.dataframe(pd.DataFrame(impact["narrowed_entities"])[["entity", "lost_types", "remaining_types"]],
                     use_container_width=True, hide_index=True)
    if impact["stale_constraints"]:
        st.markdown(f"**{len(impact['stale_constraints'])} constraint(s) would go stale** "
                    "(reference a deleted type as domain or range -- dropped if you save):")
        st.dataframe(pd.DataFrame(impact["stale_constraints"]), use_container_width=True, hide_index=True)
    affected_ids = None  # nothing to re-merge -- deletion needs no LLM step

else:
    affected_ids = affected_type_ids(old_hierarchy, hierarchy, last_entry["child"])

    # ── Merge candidates (instant) ───────────────────────────────────────
    st.subheader("Merge candidates")
    st.caption(
        "Sharing a hierarchy partition is necessary but not sufficient for a real duplicate -- "
        "filtered by embedding-label similarity, the same cheap pre-filter HDBSCAN applies "
        "before anything reaches the real LLM verifier."
    )
    similarity_threshold = st.slider("Min label similarity", 0.0, 1.0, 0.75, 0.05)
    candidates = diff_merge_candidates(
        old_hierarchy, hierarchy, bundle.entity_vocab, affected_ids,
        embedder=embedder, similarity_threshold=similarity_threshold,
    )
    if candidates:
        st.dataframe(pd.DataFrame(candidates)[["entity_a", "entity_b", "similarity", "new_partition_key"]],
                     use_container_width=True, hide_index=True)
    else:
        st.caption(f"No merge candidates above {similarity_threshold:.2f} similarity from this move.")

    # ── Affected constraints (instant) ────────────────────────────────────
    st.subheader("Affected constraints")
    if bundle.triplets_path and bundle.constraints and bundle.relation_vocab:
        triplets = load_triplets_cached(str(bundle.triplets_path))
        constraint_rows = diff_affected_constraints(
            old_hierarchy, hierarchy, bundle.constraints, triplets,
            type_vocab, bundle.relation_vocab, ConstraintConfig(), affected_ids,
            synthesized_labels=synthesized_labels,
        )
        if constraint_rows:
            st.dataframe(pd.DataFrame(constraint_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No constraints affected by this move.")
    else:
        st.caption("No constraints/triplets available for this run to diff against.")

    st.divider()

    # ── Submit: real, bounded, LLM-backed re-merge ────────────────────────
    st.subheader("Submit")
    st.caption(
        f"Re-opens every surface form under the {len(affected_ids)} affected type(s) and re-runs "
        "the real merge-round verification (actual LLM calls, bounded to this slice)."
    )
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
                merged = run_sandbox_remerge(
                    bundle.entity_vocab, type_vocab, hierarchy, affected_ids,
                    llm_extractor, embedder,
                    similarity_threshold=entity_cfg.get("similarity_threshold", 0.85),
                    embed_batch_size=embed_cfg.get("embed_batch_size", 64),
                    max_merge_rounds=entity_cfg.get("max_merge_rounds", 5),
                    max_parallel_workers=entity_cfg.get("max_parallel_workers", 8),
                    max_cluster_size=entity_cfg.get("max_cluster_size", 40),
                    synthesized_labels=synthesized_labels,
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

# ── Save: reachable whenever there's something to persist -- a completed
# re-merge, accumulated deletions, or both. Deletion needs no Submit step
# of its own (nothing to verify with an LLM), so it must be saveable on
# its own rather than gated behind sandbox_merge_result.
if st.session_state.get("sandbox_merge_result") or all_deleted_ids:
    st.divider()
    st.subheader("Save")
    if all_deleted_ids:
        st.caption(f"{len(all_deleted_ids)} deleted type(s) this session will also be applied to "
                   "entities and constraints when saved.")
    if st.button("Save this sandbox as a new run"):
        new_dir = save_sandbox_as_run(
            bundle, hierarchy, st.session_state.get("sandbox_merge_result"),
            deleted_type_ids=all_deleted_ids or None,
        )
        st.success(f"Saved to {new_dir}")
