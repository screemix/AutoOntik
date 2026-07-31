"""
Render induced RelationConstraints as a single self-contained, searchable HTML file.

Reads the pickled checkpoints written by src/ontodisco/pipeline.py
(output_dir/checkpoints/run_<n>/relation_dedup.pkl, type_dedup.pkl, and
constraints.pkl) and emits a static HTML page: one collapsible group per
canonical relation, each listing its domain/range constraint rows (support,
confidence, strength). A search box filters by relation/domain/range label,
and strength checkboxes let you show/hide hard/soft/hint rows.

No server, no external JS/CSS -- open the output file directly in a browser.

Usage:
    # Uses the most recent run_<n> under output/checkpoints/
    python scripts/visualize_constraints.py --output-dir output --out output/constraints_viz.html

    # Pin a specific run
    python scripts/visualize_constraints.py --output-dir output --run 6
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


def build_groups(constraints: list, relation_vocab, type_vocab) -> list[dict]:
    type_labels = {tid: ct.canonical_label for tid, ct in type_vocab.items.items()}
    relations = relation_vocab.relations

    by_relation: dict[str, list] = {}
    for c in constraints:
        by_relation.setdefault(c.relation_id, []).append(c)

    groups = []
    for relation_id, rows in by_relation.items():
        relation = relations.get(relation_id)
        relation_label = relation.canonical_label if relation is not None else relation_id
        surface_forms = [s for s in (relation.surface_forms if relation is not None else []) if s != relation_label][:8]

        rows_json = []
        for c in sorted(rows, key=lambda c: -c.pca_confidence):
            rows_json.append({
                "domainId": c.domain_type_id,
                "domain": type_labels.get(c.domain_type_id, c.domain_type_id),
                "rangeId": c.range_type_id,
                "range": type_labels.get(c.range_type_id, c.range_type_id),
                "support": c.support,
                "total": c.total,
                "confidence": round(c.pca_confidence, 3),
                "strength": c.strength.value,
            })

        groups.append({
            "id": relation_id,
            "relation": relation_label,
            "surface": surface_forms,
            "maxConfidence": round(max(c.pca_confidence for c in rows), 3),
            "topStrength": min((r["strength"] for r in rows_json), key=lambda s: {"hard": 0, "soft": 1, "hint": 2}[s]),
            "rows": rows_json,
        })

    groups.sort(key=lambda g: (-len(g["rows"]), g["relation"]))
    return groups


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Relation Constraints Viewer</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1b1f24; --muted: #6b7280; --border: #e5e7eb;
    --hl: #fff3b0; --leaf-bg: #f6f7f9; --accent: #2563eb;
    --hard: #16a34a; --hard-bg: #eafaf0; --soft: #d97706; --soft-bg: #fff6e8;
    --hint: #6b7280; --hint-bg: #f1f2f4;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #14171a; --fg: #e7e9ea; --muted: #9aa4af; --border: #2a2f35;
             --hl: #5a4a00; --leaf-bg: #1b1f24; --accent: #6ea8fe;
             --hard: #4ade80; --hard-bg: #123321; --soft: #fbbf24; --soft-bg: #3a2c0e;
             --hint: #9aa4af; --hint-bg: #23272c; }
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
  .controls { display: flex; gap: 10px; align-items: center; font-size: 12px; color: var(--muted); }
  button { font-size: 12px; padding: 5px 10px; border-radius: 6px; border: 1px solid var(--border);
           background: var(--bg); color: var(--fg); cursor: pointer; }
  button:hover { border-color: var(--accent); color: var(--accent); }
  label.chk { display: flex; align-items: center; gap: 4px; cursor: pointer; }
  main { padding: 10px 20px 60px; max-width: 1100px; margin: 0 auto; overflow-x: auto; }
  ul.groups { list-style: none; margin: 0; padding: 0; }
  li.group { margin: 3px 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .grow { display: flex; align-items: baseline; gap: 8px; padding: 8px 10px; cursor: pointer; }
  .grow:hover { background: var(--leaf-bg); }
  .toggle { display: inline-block; width: 14px; text-align: center; color: var(--muted); font-size: 11px;
            user-select: none; flex: none; }
  .rel-lbl { font-size: 14px; font-weight: 600; }
  .surf { font-size: 11.5px; color: var(--muted); }
  .badge { font-size: 10.5px; color: var(--muted); border: 1px solid var(--border); border-radius: 8px;
           padding: 0px 6px; white-space: nowrap; }
  .badge.hard { color: var(--hard); border-color: var(--hard); background: var(--hard-bg); }
  .badge.soft { color: var(--soft); border-color: var(--soft); background: var(--soft-bg); }
  .badge.hint { color: var(--hint); border-color: var(--hint); background: var(--hint-bg); }
  table.rows { display: none; width: 100%; border-collapse: collapse; }
  li.group.open table.rows { display: table; }
  table.rows th, table.rows td { text-align: left; padding: 5px 10px; font-size: 13px;
                                   border-top: 1px solid var(--border); }
  table.rows th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted);
                  font-weight: 600; }
  table.rows tr.row.match td.dr { background: var(--hl); }
  .arrow { color: var(--muted); padding: 0 4px; }
  .bar-wrap { display: flex; align-items: center; gap: 6px; min-width: 140px; }
  .bar-bg { flex: 1; height: 6px; border-radius: 3px; background: var(--border); overflow: hidden; }
  .bar-fill { height: 100%; }
  .bar-fill.hard { background: var(--hard); }
  .bar-fill.soft { background: var(--soft); }
  .bar-fill.hint { background: var(--hint); }
  .conf-text { font-size: 11.5px; color: var(--muted); white-space: nowrap; }
  li.group.hidden { display: none; }
  #empty { display: none; color: var(--muted); padding: 30px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>Relation Constraints</h1>
  <input id="search" type="text" placeholder="Search relations, domain or range types…" autocomplete="off">
  <div class="controls">
    <label class="chk"><input type="checkbox" id="fHard" checked> Hard</label>
    <label class="chk"><input type="checkbox" id="fSoft" checked> Soft</label>
    <label class="chk"><input type="checkbox" id="fHint" checked> Hint</label>
    <button id="expandAll">Expand all</button>
    <button id="collapseAll">Collapse all</button>
  </div>
  <div class="stats" id="stats"></div>
</header>
<main>
  <ul class="groups" id="groups"></ul>
  <div id="empty">No matching constraints.</div>
</main>
<script>
const DATA = __GROUPS_JSON__;

const groupsEl = document.getElementById('groups');
const searchEl = document.getElementById('search');
const statsEl = document.getElementById('stats');
const emptyEl = document.getElementById('empty');
const fHard = document.getElementById('fHard');
const fSoft = document.getElementById('fSoft');
const fHint = document.getElementById('fHint');

const totalRelations = DATA.length;
const totalRows = DATA.reduce((n, g) => n + g.rows.length, 0);
statsEl.textContent = `${totalRelations.toLocaleString()} relations · ${totalRows.toLocaleString()} constraints`;

function renderRow(row) {
  const tr = document.createElement('tr');
  tr.className = 'row';
  tr.dataset.strength = row.strength;

  const dr = document.createElement('td');
  dr.className = 'dr';
  dr.innerHTML = '';
  const domSpan = document.createElement('span');
  domSpan.textContent = row.domain;
  const arrow = document.createElement('span');
  arrow.className = 'arrow';
  arrow.textContent = '→';
  const rngSpan = document.createElement('span');
  rngSpan.textContent = row.range;
  dr.appendChild(domSpan);
  dr.appendChild(arrow);
  dr.appendChild(rngSpan);
  tr.appendChild(dr);

  const strengthTd = document.createElement('td');
  const b = document.createElement('span');
  b.className = 'badge ' + row.strength;
  b.textContent = row.strength;
  strengthTd.appendChild(b);
  tr.appendChild(strengthTd);

  const confTd = document.createElement('td');
  const wrap = document.createElement('div');
  wrap.className = 'bar-wrap';
  const barBg = document.createElement('div');
  barBg.className = 'bar-bg';
  const barFill = document.createElement('div');
  barFill.className = 'bar-fill ' + row.strength;
  barFill.style.width = Math.round(row.confidence * 100) + '%';
  barBg.appendChild(barFill);
  const confText = document.createElement('span');
  confText.className = 'conf-text';
  confText.textContent = `${(row.confidence * 100).toFixed(1)}%`;
  wrap.appendChild(barBg);
  wrap.appendChild(confText);
  confTd.appendChild(wrap);
  tr.appendChild(confTd);

  const supportTd = document.createElement('td');
  supportTd.className = 'conf-text';
  supportTd.textContent = `${row.support.toLocaleString()} / ${row.total.toLocaleString()}`;
  tr.appendChild(supportTd);

  return tr;
}

function renderGroup(group) {
  const li = document.createElement('li');
  li.className = 'group';
  li.dataset.id = group.id;

  const grow = document.createElement('div');
  grow.className = 'grow';

  const toggle = document.createElement('span');
  toggle.className = 'toggle';
  toggle.textContent = '▸';
  grow.appendChild(toggle);

  const lbl = document.createElement('span');
  lbl.className = 'rel-lbl';
  lbl.textContent = group.relation;
  grow.appendChild(lbl);

  const countBadge = document.createElement('span');
  countBadge.className = 'badge';
  countBadge.textContent = `${group.rows.length} constraint${group.rows.length === 1 ? '' : 's'}`;
  grow.appendChild(countBadge);

  const topBadge = document.createElement('span');
  topBadge.className = 'badge ' + group.topStrength;
  topBadge.textContent = 'best: ' + group.topStrength;
  grow.appendChild(topBadge);

  if (group.surface.length) {
    const s = document.createElement('span');
    s.className = 'surf';
    s.textContent = 'aka: ' + group.surface.join(', ');
    grow.appendChild(s);
  }

  grow.addEventListener('click', () => li.classList.toggle('open'));
  li.appendChild(grow);

  const table = document.createElement('table');
  table.className = 'rows';
  table.innerHTML = '<thead><tr><th>Domain → Range</th><th>Strength</th><th>Confidence</th><th>Support</th></tr></thead>';
  const tbody = document.createElement('tbody');
  for (const row of group.rows) tbody.appendChild(renderRow(row));
  table.appendChild(tbody);
  li.appendChild(table);

  return li;
}

function renderAll() {
  groupsEl.innerHTML = '';
  for (const group of DATA) groupsEl.appendChild(renderGroup(group));
}
renderAll();

document.getElementById('expandAll').addEventListener('click', () => {
  for (const li of groupsEl.querySelectorAll('li.group')) li.classList.add('open');
});
document.getElementById('collapseAll').addEventListener('click', () => {
  for (const li of groupsEl.querySelectorAll('li.group')) li.classList.remove('open');
});

function rowMatches(row, q) {
  return row.domain.toLowerCase().includes(q) || row.range.toLowerCase().includes(q);
}

function applyFilter() {
  const q = searchEl.value.trim().toLowerCase();
  const allowedStrengths = new Set();
  if (fHard.checked) allowedStrengths.add('hard');
  if (fSoft.checked) allowedStrengths.add('soft');
  if (fHint.checked) allowedStrengths.add('hint');

  const groupLis = groupsEl.querySelectorAll(':scope > li.group');
  let visibleCount = 0;

  for (let i = 0; i < DATA.length; i++) {
    const group = DATA[i];
    const li = groupLis[i];
    const rowTrs = li.querySelectorAll('tbody > tr.row');
    const relationSelfMatch = !q || group.relation.toLowerCase().includes(q) ||
      group.surface.some(s => s.toLowerCase().includes(q));

    let anyRowVisible = false;
    for (let j = 0; j < group.rows.length; j++) {
      const row = group.rows[j];
      const tr = rowTrs[j];
      const strengthOk = allowedStrengths.has(row.strength);
      const textMatch = !q || relationSelfMatch || rowMatches(row, q);
      const visible = strengthOk && textMatch;
      tr.classList.toggle('hidden', !visible);
      tr.style.display = visible ? '' : 'none';
      tr.classList.toggle('match', q && !relationSelfMatch && rowMatches(row, q));
      if (visible) anyRowVisible = true;
    }

    const groupVisible = anyRowVisible;
    li.classList.toggle('hidden', !groupVisible);
    if (groupVisible && q) li.classList.add('open');
    if (groupVisible) visibleCount++;
  }

  emptyEl.style.display = visibleCount === 0 ? 'block' : 'none';
}

searchEl.addEventListener('input', applyFilter);
fHard.addEventListener('change', applyFilter);
fSoft.addEventListener('change', applyFilter);
fHint.addEventListener('change', applyFilter);
</script>
</body>
</html>
"""


def render_html(groups: list[dict]) -> str:
    return HTML_TEMPLATE.replace("__GROUPS_JSON__", json.dumps(groups, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Pipeline output_dir containing checkpoints/")
    parser.add_argument("--run", type=int, default=None, help="Run number under checkpoints/run_<n> (default: most recent)")
    parser.add_argument("--out", default=None, help="Path to write the HTML file (default: <output-dir>/constraints_viz.html)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    out_path = Path(args.out) if args.out else output_dir / "constraints_viz.html"
    run_dir = _resolve_run_dir(output_dir, args.run)

    type_vocab = _load_checkpoint(run_dir, "type_dedup")
    relation_vocab = _load_checkpoint(run_dir, "relation_dedup")
    constraints = _load_checkpoint(run_dir, "constraints")

    groups = build_groups(constraints, relation_vocab, type_vocab)
    out_path.write_text(render_html(groups), encoding="utf-8")
    print(f"Wrote {out_path} ({len(groups)} relations, {len(constraints)} constraints, from {run_dir}, open it directly in a browser)")


if __name__ == "__main__":
    main()
