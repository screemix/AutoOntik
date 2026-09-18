"""Custom Streamlit component: a read-only, collapsible D3 tree for
click-to-toggle multi-select of type nodes (used by the Graph page to pick
"everything under this category" instead of searching individual entity
names). Sibling of ../reparent_tree/ -- shares the same vendored d3.min.js
content, same stratify/collapse-filter/zoom-pan rendering approach, and the
same hand-written Streamlit<->component wire protocol (see that package's
own docstring for why it's hand-written rather than pulling in
streamlit-component-lib), but is a SEPARATE component: its click semantics
(toggle membership in a growing selection) are a different interaction than
reparent_tree's select-source-then-select-target flow, and reusing that flow
here would fire spurious reparent-shaped values. No drag, no Undo/Reset/
Delete controls -- this tree is never edited, only browsed and selected from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent
_component = components.declare_component("type_tree_selector", path=str(_COMPONENT_DIR))


def type_tree_selector(
    nodes: list[dict],
    edges: list[dict],
    selected_ids: Optional[list[str]] = None,
    height: int = 600,
    key: Optional[str] = None,
) -> list[str]:
    """Renders a collapsible tree (same +/- and zoom/pan as reparent_tree).
    `nodes`: [{"id", "label"}, ...]. `edges`: [{"child", "parent"}, ...].
    Clicking a node's circle toggles it in the selection; a "Clear
    selection" control lives in the component's own topbar (an external
    Streamlit button would need to bump `key` to signal it, which tears
    down the iframe and wipes zoom/collapse state -- see reparent_tree's
    own fix for the same issue). Returns the CURRENT full selection as a
    list of type_ids (never None -- defaults to [] both here and as the
    component's initial value, so callers don't need a None-check)."""
    result = _component(
        nodes=nodes, edges=edges, selected_ids=selected_ids or [],
        height=height, key=key, default=[],
    )
    return result if result is not None else []
