"""
Render an induced TypeHierarchy as a single self-contained, searchable HTML file.

Reads the pickled checkpoints written by src/ontodisco/pipeline.py
(output_dir/checkpoints/run_<n>/type_dedup.pkl and .../hierarchy_induction.pkl)
and emits a static HTML page with a collapsible forest view: each of the
(often 1000+) hierarchy roots is a top-level collapsed node; opening one
reveals its subtree. A search box filters the whole forest by label
substring and auto-expands the path to every match.

No server, no external JS/CSS -- open the output file directly in a browser.

Usage:
    # Uses the most recent run_<n> under output/checkpoints/
    python scripts/visualize_hierarchy.py --output-dir output --out output/hierarchy_viz.html

    # Pin a specific run
    python scripts/visualize_hierarchy.py --output-dir output --run 3
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

# The checkpoints are pickled dataclass instances from src.ontodisco.* --
# unpickling needs that package importable regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def _resolve_run_dir(output_dir: Path, run: int | None) -> Path:
    checkpoints_root = output_dir / "checkpoints"
    if run is not None:
        run_dir = checkpoints_root / f"run_{run}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"No such run directory: {run_dir}")
        return run_dir

    numbers = []
    if checkpoints_root.exists():
        for child in checkpoints_root.iterdir():
            m = _RUN_DIR_RE.match(child.name)
            if child.is_dir() and m:
                numbers.append(int(m.group(1)))
    if not numbers:
        raise FileNotFoundError(f"No run_<n> directories found under {checkpoints_root}")
    return checkpoints_root / f"run_{max(numbers)}"


def _load_checkpoint(run_dir: Path, name: str):
    path = run_dir / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_label_lookup(type_vocab, hierarchy_result):
    labels, counts, surface_forms, definitions, is_synth = {}, {}, {}, {}, {}
    for type_id, ct in type_vocab.items.items():
        labels[type_id] = ct.canonical_label
        counts[type_id] = ct.count_per_normalized
        surface_forms[type_id] = ct.surface_forms
        definitions[type_id] = ""
        is_synth[type_id] = False
    for st in hierarchy_result.synthesized_types:
        labels[st.type_id] = st.canonical_label
        counts[st.type_id] = 0
        surface_forms[st.type_id] = []
        definitions[st.type_id] = st.definition
        is_synth[st.type_id] = True
    return labels, counts, surface_forms, definitions, is_synth


def build_tree(hierarchy_result, type_vocab) -> list[dict]:
    hierarchy = hierarchy_result.hierarchy
    labels, counts, surface_forms, definitions, is_synth = _build_label_lookup(type_vocab, hierarchy_result)
    edge_by_child = {e.child_type_id: e for e in hierarchy.edges}

    def build_node(type_id: str, visited: frozenset[str]) -> dict:
        if type_id in visited:
            # Defensive only -- the forest is acyclic by construction; guards
            # against silently hanging if a checkpoint is ever malformed.
            return {"id": type_id, "label": f"[cycle: {labels.get(type_id, type_id)}]",
                    "synth": False, "definition": "", "surface": [], "count": 0,
                    "leafCount": 1, "score": None, "children": []}
        child_ids = hierarchy.children.get(type_id, [])
        children = [build_node(c, visited | {type_id}) for c in child_ids]
        children.sort(key=lambda c: -c["leafCount"])
        leaf_count = sum(c["leafCount"] for c in children) if children else 1
        own_count = counts.get(type_id, 0)
        total_count = own_count + sum(c["count"] for c in children)
        edge = edge_by_child.get(type_id)
        return {
            "id": type_id,
            "label": labels.get(type_id, type_id),
            "synth": is_synth.get(type_id, False),
            "definition": definitions.get(type_id, ""),
            "surface": [s for s in surface_forms.get(type_id, []) if s != labels.get(type_id)][:8],
            "count": total_count,
            "leafCount": leaf_count,
            "score": round(edge.ensemble_score, 2) if edge is not None else None,
            "children": children,
        }

    roots = [build_node(r, frozenset()) for r in hierarchy.roots]
    roots.sort(key=lambda r: -r["leafCount"])
    return roots


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Type Hierarchy Viewer</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1b1f24; --muted: #6b7280; --border: #e5e7eb;
    --hl: #fff3b0; --synth: #7c3aed; --synth-bg: #f3ecff;
    --leaf-bg: #f6f7f9; --accent: #2563eb;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #14171a; --fg: #e7e9ea; --muted: #9aa4af; --border: #2a2f35;
             --hl: #5a4a00; --synth: #b795f7; --synth-bg: #241a38; --leaf-bg: #1b1f24; --accent: #6ea8fe; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--fg); }
  header { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--border);
           padding: 12px 20px; display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
  h1 { font-size: 15px; margin: 0; font-weight: 600; white-space: nowrap; }
  #search { flex: 1 1 260px; min-width: 200px; padding: 7px 10px; border-radius: 6px;
            border: 1px solid var(--border); background: var(--bg); color: var(--fg); font-size: 14px; }
  .stats { font-size: 12px; color: var(--muted); white-space: nowrap; }
  .controls { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }
  button { font-size: 12px; padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border);
           background: var(--bg); color: var(--fg); cursor: pointer; }
  button:hover { border-color: var(--accent); color: var(--accent); }
  label.chk { display: flex; align-items: center; gap: 4px; cursor: pointer; }
  main { padding: 10px 20px 60px; max-width: 1100px; margin: 0 auto; overflow-x: auto; }
  ul.tree, ul.tree ul { list-style: none; margin: 0; padding-left: 20px; }
  ul.tree { padding-left: 0; }
  li.node { position: relative; margin: 1px 0; }
  .row { display: flex; align-items: baseline; gap: 6px; padding: 2px 4px; border-radius: 4px; cursor: pointer; }
  .row:hover { background: var(--leaf-bg); }
  .row.match .lbl { background: var(--hl); border-radius: 3px; padding: 0 2px; }
  .toggle { display: inline-block; width: 14px; text-align: center; color: var(--muted); font-size: 11px;
            user-select: none; flex: none; }
  .toggle.leafless { visibility: hidden; }
  .lbl { font-size: 13.5px; }
  .badge { font-size: 10.5px; color: var(--muted); border: 1px solid var(--border); border-radius: 8px;
           padding: 0px 6px; white-space: nowrap; }
  .badge.synth { color: var(--synth); border-color: var(--synth); background: var(--synth-bg); }
  .badge.score { color: var(--accent); }
  .def { font-size: 12px; color: var(--muted); font-style: italic; padding-left: 20px; }
  .surf { font-size: 11.5px; color: var(--muted); padding-left: 20px; }
  ul.children { display: none; }
  li.node.open > ul.children { display: block; }
  li.node.hidden { display: none; }
  #empty { display: none; color: var(--muted); padding: 30px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>Type Hierarchy</h1>
  <input id="search" type="text" placeholder="Search types… (label or surface form)" autocomplete="off">
  <div class="controls">
    <button id="expandAll">Expand all</button>
    <button id="collapseAll">Collapse all</button>
    <label class="chk"><input type="checkbox" id="hideSingletons"> Hide singleton roots</label>
  </div>
  <div class="stats" id="stats"></div>
</header>
<main>
  <ul class="tree" id="tree"></ul>
  <div id="empty">No matching types.</div>
</main>
<script>
const DATA = __TREE_JSON__;

const treeEl = document.getElementById('tree');
const searchEl = document.getElementById('search');
const statsEl = document.getElementById('stats');
const emptyEl = document.getElementById('empty');
const hideSingletonsEl = document.getElementById('hideSingletons');

function countAll(nodes) {
  let n = 0;
  for (const node of nodes) { n += 1; n += countAll(node.children); }
  return n;
}
const totalNodes = countAll(DATA);
const singletonRoots = DATA.filter(r => r.children.length === 0).length;
statsEl.textContent = `${DATA.length.toLocaleString()} roots · ${totalNodes.toLocaleString()} types total · ${singletonRoots.toLocaleString()} unmerged singleton roots`;

function nodeMatches(node, q) {
  if (node.label.toLowerCase().includes(q)) return true;
  if (node.surface.some(s => s.toLowerCase().includes(q))) return true;
  if (node.definition && node.definition.toLowerCase().includes(q)) return true;
  return false;
}

function renderNode(node, depth) {
  const li = document.createElement('li');
  li.className = 'node';
  li.dataset.id = node.id;

  const row = document.createElement('div');
  row.className = 'row';

  const toggle = document.createElement('span');
  toggle.className = 'toggle' + (node.children.length ? '' : ' leafless');
  toggle.textContent = node.children.length ? '▸' : '·';
  row.appendChild(toggle);

  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = node.label;
  row.appendChild(lbl);

  if (node.synth) {
    const b = document.createElement('span');
    b.className = 'badge synth';
    b.textContent = 'synthesized';
    row.appendChild(b);
  }
  if (node.leafCount > 1) {
    const b = document.createElement('span');
    b.className = 'badge';
    b.textContent = `${node.leafCount} types`;
    row.appendChild(b);
  }
  if (node.count > 0) {
    const b = document.createElement('span');
    b.className = 'badge';
    b.textContent = `${node.count.toLocaleString()} instances`;
    row.appendChild(b);
  }
  if (node.score !== null && node.score !== undefined) {
    const b = document.createElement('span');
    b.className = 'badge score';
    b.title = "hierarchy edge confidence (ensemble_score) to this node's parent";
    b.textContent = `conf ${node.score}`;
    row.appendChild(b);
  }

  row.addEventListener('click', () => {
    if (node.children.length) li.classList.toggle('open');
  });
  li.appendChild(row);

  if (node.definition) {
    const d = document.createElement('div');
    d.className = 'def';
    d.textContent = node.definition;
    li.appendChild(d);
  }
  if (node.surface.length) {
    const s = document.createElement('div');
    s.className = 'surf';
    s.textContent = 'aka: ' + node.surface.join(', ');
    li.appendChild(s);
  }

  if (node.children.length) {
    const ul = document.createElement('ul');
    ul.className = 'children';
    for (const child of node.children) ul.appendChild(renderNode(child, depth + 1));
    li.appendChild(ul);
  }
  return li;
}

function renderAll() {
  treeEl.innerHTML = '';
  for (const root of DATA) treeEl.appendChild(renderNode(root, 0));
}
renderAll();

function setOpenRecursive(li, open) {
  if (li.querySelector(':scope > ul.children')) li.classList.toggle('open', open);
  for (const child of li.querySelectorAll(':scope > ul.children > li.node')) setOpenRecursive(child, open);
}

document.getElementById('expandAll').addEventListener('click', () => {
  for (const li of treeEl.querySelectorAll('li.node')) setOpenRecursive(li, true);
});
document.getElementById('collapseAll').addEventListener('click', () => {
  for (const li of treeEl.querySelectorAll('li.node')) li.classList.remove('open');
});

hideSingletonsEl.addEventListener('change', applyFilter);
searchEl.addEventListener('input', applyFilter);

// Returns true if this <li> should be visible; toggles expansion/highlighting
// accordingly. hideSingletons only ever hides top-level roots with no
// children -- it must NOT hide ordinary leaves nested inside a real subtree
// (every leaf node also has zero children, so this has to be root-scoped).
function applyFilter() {
  const q = searchEl.value.trim().toLowerCase();
  const hideSingles = hideSingletonsEl.checked;

  function walk(li, node, isRoot) {
    const childLis = li.querySelectorAll(':scope > ul.children > li.node');
    let childMatch = false;
    let i = 0;
    for (const child of node.children) {
      if (walk(childLis[i], child, false)) childMatch = true;
      i++;
    }
    const row = li.querySelector(':scope > .row');
    let visible;
    if (q) {
      const self = nodeMatches(node, q);
      visible = self || childMatch;
      row.classList.toggle('match', self);
      if (visible && node.children.length) li.classList.add('open');
    } else {
      visible = !(isRoot && hideSingles && node.children.length === 0);
      row.classList.remove('match');
    }
    li.classList.toggle('hidden', !visible);
    return visible;
  }

  let visibleCount = 0;
  const rootLis = treeEl.querySelectorAll(':scope > li.node');
  for (let i = 0; i < DATA.length; i++) {
    if (walk(rootLis[i], DATA[i], true)) visibleCount++;
  }
  emptyEl.style.display = visibleCount === 0 ? 'block' : 'none';
}
</script>
</body>
</html>
"""


def render_html(tree: list[dict]) -> str:
    return HTML_TEMPLATE.replace("__TREE_JSON__", json.dumps(tree, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Pipeline output_dir containing checkpoints/")
    parser.add_argument("--run", type=int, default=None, help="Run number under checkpoints/run_<n> (default: most recent)")
    parser.add_argument("--out", default=None, help="Path to write the HTML file (default: <output-dir>/hierarchy_viz.html)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    out_path = Path(args.out) if args.out else output_dir / "hierarchy_viz.html"
    run_dir = _resolve_run_dir(output_dir, args.run)

    type_vocab = _load_checkpoint(run_dir, "type_dedup")
    hierarchy_result = _load_checkpoint(run_dir, "hierarchy_induction")

    tree = build_tree(hierarchy_result, type_vocab)
    out_path.write_text(render_html(tree), encoding="utf-8")
    print(f"Wrote {out_path} ({len(tree)} roots, from {run_dir}, open it directly in a browser)")


if __name__ == "__main__":
    main()
