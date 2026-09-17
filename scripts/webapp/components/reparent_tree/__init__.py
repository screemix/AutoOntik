"""Custom Streamlit component: a draggable D3 tree for the Reparent Lab.

No JS build step and no CDN dependency at render time -- `vendor/d3.min.js`
is fetched once and committed locally (see vendor/README), and the
Streamlit<->component wire protocol (componentReady / render / setComponentValue
/ setFrameHeight postMessage events) is hand-written in index.html rather than
pulling in the `streamlit-component-lib` npm package, since that protocol is
small, public, and this avoids vendoring a bundle whose exact contents can't
be verified without a working npm toolchain in this environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent
_component = components.declare_component("reparent_tree", path=str(_COMPONENT_DIR))


def reparent_tree(
    nodes: list[dict],
    edges: list[dict],
    highlight_ids: Optional[list[str]] = None,
    height: int = 600,
    key: Optional[str] = None,
) -> Optional[dict]:
    """Renders a draggable tree. `nodes`: [{"id", "label", "depth"}, ...].
    `edges`: [{"child", "parent"}, ...]. Returns {"moved": id, "new_parent": id}
    on a completed drop (until the next distinct drop, per Streamlit's normal
    component-value semantics), else None."""
    return _component(
        nodes=nodes, edges=edges, highlight_ids=highlight_ids or [],
        height=height, key=key, default=None,
    )
