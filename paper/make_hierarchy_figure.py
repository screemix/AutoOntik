#!/usr/bin/env python3
"""Generates the AutoOntic hierarchy-induction worked-example figure as SVG:
all 8 placement decisions from a toy film-domain run, 2 rows x 4 columns,
matching src/ontodisco/hierarchy_induction.py's two-call (PARENT CHECK /
CHILD CHECK) + capacity-triggered regroup algorithm. Kept deliberately
minimal: a bordered box per step, a plain monospace tree, one caption line
-- no color-coding, no legend, no per-decision badges."""

import html

FONT_SANS = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"

INK = "#1A1A1A"
SUBINK = "#6B6B6B"
FAINT = "#A3A3A3"
PAGE_BG = "#FFFFFF"
BORDER = "#C9CCD3"

MARGIN = 32
GUTTER = 48           # horizontal space between boxes in the same row
ROW_GAP = 30           # vertical space between the two rows
PAD = 16               # inner padding between a box's border and its content
PANEL_W = 300          # content width; wide enough for "biographical film" at depth 3
BOX_W = PANEL_W + 2 * PAD
COLS = 4

TREE_SIZE = 19
TREE_LINE_H = 24.5
INDENT_PX = 23
STEP_LABEL_SIZE = 15.5
CAPTION_SIZE = 13.2
CAPTION_LINE_H = 18
CAPTION_MAX_CHARS = 34

parts = []


def esc(s):
    return html.escape(s, quote=False)


def add(s):
    parts.append(s)


def text(x, y, s, size=14, family=FONT_SANS, weight="normal", color=INK, anchor="start"):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')


def tspan_line(x, y, segs, size=TREE_SIZE, family=FONT_MONO):
    buf = [f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}">']
    for seg_text, color, weight, strike in segs:
        deco = ' text-decoration="line-through"' if strike else ""
        op = '0.4' if strike else '1'
        buf.append(f'<tspan fill="{color}" font-weight="{weight or "normal"}" opacity="{op}"{deco}>{esc(seg_text)}</tspan>')
    buf.append('</text>')
    add("".join(buf))


def draw_tree(x, y, lines):
    """lines: each entry is `segs` (depth 0) or `(depth, segs)`."""
    yy = y
    for entry in lines:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], int):
            depth, segs = entry
        else:
            depth, segs = 0, entry
        tspan_line(x + depth * INDENT_PX, yy, segs)
        yy += TREE_LINE_H
    return yy


def wrap_lines(s, max_chars):
    words = s.split()
    lines, cur = [], ""
    for wd in words:
        cand = (cur + " " + wd).strip()
        if len(cand) > max_chars and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def draw_caption(x, y, lines, color=SUBINK):
    yy = y
    for ln in lines:
        text(x, yy, ln, size=CAPTION_SIZE, color=color)
        yy += CAPTION_LINE_H
    return yy


cases = [
    dict(step="1 · audiovisual work", tree=[
        [("audiovisual work", INK, "bold", False)],
    ], caption="pool of roots is empty — settles as the first root."),

    dict(step="2 · documentary film", tree=[
        [("audiovisual work", SUBINK, None, False)],
        (1, [("└ documentary film", INK, "bold", False)]),
    ], caption="settles under audiovisual work — too shallow, fixed next."),

    dict(step="3 · film", tree=[
        [("audiovisual work", SUBINK, None, False)],
        (1, [("└ documentary film", SUBINK, None, True)]),
        [("└ film", INK, "bold", False)],
        (1, [("└ documentary film", INK, None, False)]),
    ], caption="reparented above documentary film — corrects a too-shallow attachment."),

    dict(step="4 · movie", tree=[
        [("audiovisual work", SUBINK, None, False)],
        (1, [("└ film ", INK, None, False), ("⊕movie", INK, "bold", False)]),
        (2, [("└ documentary film", INK, None, False)]),
    ], caption="same concept as film — merges, no new node."),

    dict(step="5 · war film", tree=[
        [("film", SUBINK, None, False)],
        (1, [("├ documentary film", SUBINK, None, False)]),
        (1, [("└ war film", INK, "bold", False)]),
    ], caption="settles as a plain sibling under film."),

    dict(step="6 · action film", tree=[
        [("film", SUBINK, None, False)],
        (1, [("├ documentary film", SUBINK, None, False)]),
        (1, [("├ war film", SUBINK, None, False)]),
        (1, [("└ action film", INK, "bold", False)]),
    ], caption="settles as a 3rd plain sibling — no shared parent yet."),

    dict(step="7 · regroup", tree=[
        [("film", SUBINK, None, False)],
        (1, [("├ documentary film", SUBINK, None, False)]),
        (1, [("└ genre film", INK, "bold", False)]),
        (2, [("├ war film", INK, None, False)]),
        (2, [("└ action film", INK, None, False)]),
    ], caption="biographical film's descent finds 3 siblings > B — regroup fires first."),

    dict(step="8 · final tree", tree=[
        [("audiovisual work", SUBINK, None, False)],
        (1, [("└ film ⊕movie", INK, None, False)]),
        (2, [("├ documentary film", INK, None, False)]),
        (3, [("└ biographical film", INK, "bold", False)]),
        (2, [("└ genre film", INK, None, False)]),
        (3, [("├ war film", INK, None, False)]),
        (3, [("└ action film", INK, None, False)]),
    ], caption="biographical film settles under documentary film."),
]

# ── measure: every box shares one uniform height, set by the tallest case ──
for c in cases:
    c["caption_lines"] = wrap_lines(c["caption"], CAPTION_MAX_CHARS)
    c["tree_h"] = len(c["tree"]) * TREE_LINE_H
    c["content_h"] = STEP_LABEL_SIZE + 14 + c["tree_h"] + 14 + len(c["caption_lines"]) * CAPTION_LINE_H

BOX_H = max(c["content_h"] for c in cases) + 2 * PAD

# ═══════════════════════════════ build ═══════════════════════════════
add(f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{FAINT}"/></marker></defs>')

for i, c in enumerate(cases):
    row, col = divmod(i, COLS)
    bx = MARGIN + col * (BOX_W + GUTTER)
    by = MARGIN + row * (BOX_H + ROW_GAP)
    add(f'<rect x="{bx}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="8" '
        f'fill="{PAGE_BG}" stroke="{BORDER}" stroke-width="1.4"/>')

    cx, cy = bx + PAD, by + PAD
    text(cx, cy + STEP_LABEL_SIZE * 0.78, c["step"], size=STEP_LABEL_SIZE, weight="bold", color=INK)
    tree_top = cy + STEP_LABEL_SIZE + 14
    tree_bottom = draw_tree(cx, tree_top, c["tree"])
    draw_caption(cx, tree_bottom + 14, c["caption_lines"])

    if col < COLS - 1:
        cx_mid = bx + BOX_W + GUTTER / 2
        cy_mid = by + BOX_H / 2
        add(f'<line x1="{cx_mid-12}" y1="{cy_mid}" x2="{cx_mid+10}" y2="{cy_mid}" stroke="{FAINT}" '
            f'stroke-width="1.5" marker-end="url(#arrow)"/>')

n_rows = (len(cases) + COLS - 1) // COLS
W = MARGIN + COLS * BOX_W + (COLS - 1) * GUTTER + MARGIN
H = MARGIN + n_rows * BOX_H + (n_rows - 1) * ROW_GAP + MARGIN

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAGE_BG}"/>' + "".join(parts) + '</svg>')

with open("hierarchy_induction.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print(f"wrote hierarchy_induction.svg ({W}x{H})")
