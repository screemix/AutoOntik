#!/usr/bin/env python3
"""Generates the AutoOntic hierarchy-induction worked-example figure as SVG,
matching the current two-call (PARENT CHECK / CHILD CHECK) + capacity-
triggered consolidation-call algorithm in src/ontodisco/hierarchy_induction.py."""

import html

W, H = 1720, 1150
MARGIN = 24
GAP = 18
COLS, ROWS = 3, 3
HEADER_H = 112
FOOTER_H = 30
GRID_TOP = MARGIN + HEADER_H + 18
GRID_W = W - 2 * MARGIN
GRID_H = H - GRID_TOP - MARGIN - FOOTER_H
CARD_W = (GRID_W - (COLS - 1) * GAP) / COLS
CARD_H = (GRID_H - (ROWS - 1) * GAP) / ROWS

FONT_SANS = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"

INK = "#1B1F2B"
SUBINK = "#5B6474"
FAINT = "#9AA3B2"
CARD_BG = "#FBFBFD"
CARD_BORDER = "#E4E7EE"
PAGE_BG = "#FFFFFF"

VIOLET = "#6D4CE0"
GRAY = "#9099AA"
ROSE = "#D6336C"
BLUE = "#2E6FE0"
TEAL = "#0E8C7F"
INKPANEL = "#33366B"

PARENT_COL = "#4C5FD5"
CHILD_COL = "#D6336C"
REGROUP_COL = "#0E8C7F"
MERGE_COL = "#2E6FE0"

parts = []


def esc(s):
    return html.escape(s, quote=False)


def add(s):
    parts.append(s)


def text(x, y, s, size=17, family=FONT_SANS, weight="normal", color=INK,
          style="normal", anchor="start", letter_spacing=None, opacity=1.0):
    ls = f' letter-spacing="{letter_spacing}"' if letter_spacing else ""
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" font-style="{style}" fill="{color}" '
        f'text-anchor="{anchor}" opacity="{opacity}"{ls}>{esc(s)}</text>')


def tspan_line(x, y, segs, size=19, family=FONT_MONO, weight="normal"):
    """segs: list of (text, color, weight_override|None, strike:bool)"""
    buf = [f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}">']
    for seg_text, color, w, strike in segs:
        wt = w or weight
        deco = ' text-decoration="line-through"' if strike else ""
        op = '0.45' if strike else '1'
        buf.append(f'<tspan fill="{color}" font-weight="{wt}" opacity="{op}"{deco}>{esc(seg_text)}</tspan>')
    buf.append('</text>')
    add("".join(buf))


def card(col, row, accent, title, highlighted=False, accent_label=None, double=False, dh=0):
    x = MARGIN + col * (CARD_W + GAP)
    y = GRID_TOP + row * (CARD_H + GAP)
    w = CARD_W
    h = CARD_H + dh
    rx = 14
    border_w = 2.6 if highlighted else 1.3
    border_c = accent if highlighted else CARD_BORDER
    add(f'<g>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{CARD_BG}" '
        f'stroke="{border_c}" stroke-width="{border_w}"/>')
    # accent top bar
    add(f'<path d="M {x} {y+rx} A {rx} {rx} 0 0 1 {x+rx} {y} L {x+w-rx} {y} '
        f'A {rx} {rx} 0 0 1 {x+w} {y+rx} L {x+w} {y+7} L {x} {y+7} Z" fill="{accent}"/>')
    tx = x + 24
    ty = y + 40
    text(tx, ty, title, size=18.5, weight="bold", color=INK, family=FONT_SANS)
    if accent_label:
        # own pill on its own row below the title, so a long title can
        # never collide with it regardless of either string's length.
        text(x + w - 22, y + 63, accent_label, size=11.5, weight="bold", color=accent,
             family=FONT_SANS, anchor="end", letter_spacing="0.6")
    add('</g>')
    return x, y, w, h, tx, ty


INDENT_PX = 30  # per tree-depth level; SVG collapses leading whitespace in
                # text content by default, so indentation MUST come from an
                # explicit x offset, never from leading space characters.


def body_lines(tx, start_y, lines, line_h=25, size=19, family=FONT_MONO):
    """lines: each entry is either `segs` (depth 0) or `(depth, segs)`,
    where segs is a list of (text, color, weight, strike) tuples. depth
    controls indentation via an explicit x offset (INDENT_PX per level)."""
    y = start_y
    for entry in lines:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], int):
            depth, segs = entry
        else:
            depth, segs = 0, entry
        if isinstance(segs, str):
            segs = [(segs, INK, "normal", False)]
        tspan_line(tx + depth * INDENT_PX, y, segs, size=size, family=family)
        y += line_h
    return y


def ann_lines(tx, start_y, lines, max_w, line_h=20, size=13.6):
    y = start_y
    for segs in lines:
        if isinstance(segs, str):
            segs = [(segs, SUBINK, "normal", False)]
        buf = [f'<text x="{tx}" y="{y}" font-family="{FONT_SANS}" font-size="{size}">']
        for seg_text, color, w, style in segs:
            style_attr = f' font-style="{style}"' if style else ''
            buf.append(f'<tspan fill="{color}" font-weight="{w or "normal"}"{style_attr}>{esc(seg_text)}</tspan>')
        buf.append('</text>')
        add("".join(buf))
        y += line_h
    return y


def note(tx, y, s, w, color=FAINT, size=12.8):
    text(tx, y, s, size=size, color=color, style="italic", family=FONT_SANS)


# ───────────────────────────── background ─────────────────────────────
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAGE_BG}"/>')

# ───────────────────────────── header / legend ─────────────────────────────
hx = MARGIN
hy = MARGIN + 24
text(hx, hy, "Hierarchy induction, one placement decision per step", size=24, weight="bold", color=INK)
text(hx, hy + 26, "Toy film-domain vocabulary, routed through the current two-call algorithm (parent check, then child check) with a capacity-triggered consolidation call between types.",
     size=14.2, color=SUBINK)

# legend chips -- fixed equal-width columns, two lines each, so nothing
# can ever collide regardless of label/description length. Terminology
# matches the paper's own prose (\S\ref{sec:hierarchy-induction}): a
# "consolidation call" fires BETWEEN types, not as a judgment ON one.
legend = [
    ("PARENT CHECK", PARENT_COL, "does focal belong under a candidate?"),
    ("CHILD CHECK", CHILD_COL, "does focal subsume a candidate?"),
    ("CONSOLIDATE", REGROUP_COL, "fires between types once a level exceeds B"),
    ("MERGE", MERGE_COL, "same_concept flag on a parent-check match"),
]
legend_w = (W - 2 * MARGIN) / len(legend)
ly = hy + 50
for i, (label, color, desc) in enumerate(legend):
    lx = hx + i * legend_w
    r = 5.5
    add(f'<circle cx="{lx+r}" cy="{ly-5}" r="{r}" fill="{color}"/>')
    text(lx + 2*r + 10, ly, label, size=13.4, weight="bold", color=color, family=FONT_SANS)
    text(lx + 2*r + 10, ly + 18, desc, size=11.8, color=SUBINK, family=FONT_SANS)

add(f'<line x1="{MARGIN}" y1="{MARGIN+HEADER_H+4}" x2="{W-MARGIN}" y2="{MARGIN+HEADER_H+4}" stroke="{CARD_BORDER}" stroke-width="1.2"/>')

# ═══════════════════════════════ Panel (a) ═══════════════════════════════
x, y, w, h, tx, ty = card(0, 0, VIOLET, "(a) Priority bands", accent_label="SETUP")
yy = ty + 30
items = ["1. audiovisual work", "2. documentary film", "3. film", "4. movie",
         "5. war film, action film, biographical film"]
for it in items:
    text(tx, yy, it, size=16.5, family=FONT_MONO, color=INK)
    yy += 23
yy += 10
yy = ann_lines(tx, yy, [
    [("ranked by Weeds-precision in-degree ", SUBINK, None, None),
     ("(τ, band tolerance δ)", SUBINK, None, "italic")],
], w - 48, line_h=19)
yy += 12
# caveat box
bx, bw = tx - 10, w - 2*(tx - x) + 20
bh = 58
add(f'<rect x="{bx}" y="{yy-18}" width="{bw}" height="{bh}" rx="8" fill="#FBEFF2" stroke="{ROSE}" stroke-width="1" opacity="0.9"/>')
ann_lines(tx, yy, [
    [("⚠ ", ROSE, "bold", None), ("documentary film outranks film here", ROSE, "bold", None)],
    [("in-degree is a frequency proxy, not true generality", SUBINK, None, "italic")],
    [("— a realistic mis-tiering risk the LLM must correct later.", SUBINK, None, "italic")],
], bw - 20, line_h=17.5, size=12.6)

# ═══════════════════════════════ Panel (b) ═══════════════════════════════
x, y, w, h, tx, ty = card(1, 0, GRAY, "(b) Step 1 — audiovisual work")
yy = ty + 34
yy = body_lines(tx, yy, [
    [("audiovisual work", INK, None, False), ("  ← new root", FAINT, "normal", False)],
], line_h=27)
yy += 26
yy = ann_lines(tx, yy, [
    [("roots = {} ", SUBINK, None, None), ("→ no candidates to compare against", SUBINK, None, None)],
], w - 48)
yy += 6
ann_lines(tx, yy, [[("settles as a new root", INK, "bold", None)]], w - 48)
note(tx, y + h - 24, "band 1 always runs first and sequentially", w)

# ═══════════════════════════════ Panel (c) ═══════════════════════════════
x, y, w, h, tx, ty = card(2, 0, GRAY, "(c) Step 2 — documentary film")
yy = ty + 34
yy = body_lines(tx, yy, [
    [("audiovisual work", INK, None, False)],
    (1, [("↳ documentary film", INK, None, False)]),
], line_h=27)
yy += 20
yy = ann_lines(tx, yy, [
    [("PARENT CHECK", PARENT_COL, "bold", None), (" vs {audiovisual work}", SUBINK, None, None)],
    [("→ match, not same concept → ", SUBINK, None, None), ("descend", INK, "bold", None)],
], w - 48, line_h=19)
yy += 10
yy = ann_lines(tx, yy, [
    [("no further candidates here → ", SUBINK, None, None), ("settle", INK, "bold", None)],
], w - 48)
note(tx, y + h - 24, "too shallow — film doesn't exist yet (fixed next)", w)

# ═══════════════════════════════ Panel (d) — retroactive correction ═══════
x, y, w, h, tx, ty = card(0, 1, ROSE, "(d) Step 3 — film", highlighted=True, accent_label="REPARENT")
yy = ty + 46
yy = body_lines(tx, yy, [
    [("audiovisual work", INK, None, False)],
    (1, [("↳ documentary film", SUBINK, None, True)]),
], line_h=25)
yy += 4
ann_lines(tx, yy, [[("parent of → reparent", ROSE, "bold", "italic")]], w - 48, line_h=18, size=12.6)
yy += 22
yy = body_lines(tx, yy, [
    [("↳ film", ROSE, "bold", False)],
    (1, [("↳ documentary film", INK, None, False)]),
], line_h=25)
yy += 14
yy = ann_lines(tx, yy, [
    [("PARENT CHECK", PARENT_COL, "bold", None), (" vs {audiovisual work} → match → descend", SUBINK, None, None)],
], w - 48, line_h=17.5, size=12.8)
yy = ann_lines(tx, yy, [
    [("PARENT CHECK", PARENT_COL, "bold", None), (" vs {documentary film} → no match", SUBINK, None, None)],
], w - 48, line_h=17.5, size=12.8)
yy = ann_lines(tx, yy, [
    [("CHILD CHECK", CHILD_COL, "bold", None), (" vs {documentary film} → match → ", SUBINK, None, None), ("REPARENT", CHILD_COL, "bold", None)],
], w - 48, line_h=17.5, size=12.8)
note(tx, y + h - 22, "a later, more general node corrects an earlier too-shallow attachment", w, color=ROSE)

# ═══════════════════════════════ Panel (e) — merge ═══════════════════════
x, y, w, h, tx, ty = card(1, 1, BLUE, "(e) Step 4 — movie", highlighted=True, accent_label="MERGE")
yy = ty + 46
yy = body_lines(tx, yy, [
    [("audiovisual work", INK, None, False)],
    (1, [("↳ film ", INK, None, False), ("⊕movie", BLUE, "bold", False)]),
    (2, [("↳ documentary film", INK, None, False)]),
], line_h=25)
yy += 16
yy = ann_lines(tx, yy, [
    [("PARENT CHECK", PARENT_COL, "bold", None), (" vs {audiovisual work} → descend", SUBINK, None, None)],
], w - 48, line_h=17.5, size=12.8)
yy = ann_lines(tx, yy, [
    [("PARENT CHECK", PARENT_COL, "bold", None), (" vs {film} → same_concept=true → ", SUBINK, None, None), ("MERGE", BLUE, "bold", None)],
], w - 48, line_h=17.5, size=12.8)
note(tx, y + h - 22, "“movie” folds in as a surface form — no new node created", w, color=BLUE)

# ═══════════════════════════════ Panel (f) ═══════════════════════════════
x, y, w, h, tx, ty = card(2, 1, GRAY, "(f) Step 5 — war film")
yy = ty + 30
yy = body_lines(tx, yy, [
    [("audiovisual work", INK, None, False)],
    (1, [("↳ film ⊕movie", INK, None, False)]),
    (2, [("├ documentary film", INK, None, False)]),
    (2, [("└ war film", INK, "bold", False)]),
], line_h=23)
yy += 14
yy = ann_lines(tx, yy, [
    [("PARENT", PARENT_COL, "bold", None), (" / ", SUBINK, None, None), ("CHILD CHECK", CHILD_COL, "bold", None),
     (" vs {documentary film}", SUBINK, None, None)],
], w - 48, line_h=17.2, size=12.6)
yy = ann_lines(tx, yy, [
    [("→ none / none → ", SUBINK, None, None), ("settles as a plain sibling", INK, "bold", None)],
], w - 48, line_h=17.2, size=12.6)
note(tx, y + h - 22, "film now has 2 children at this level", w)

# ═══════════════════════════════ Panel (g) ═══════════════════════════════
x, y, w, h, tx, ty = card(0, 2, GRAY, "(g) Step 6 — action film")
yy = ty + 30
yy = body_lines(tx, yy, [
    [("audiovisual work", INK, None, False)],
    (1, [("↳ film ⊕movie", INK, None, False)]),
    (2, [("├ documentary film", INK, None, False)]),
    (2, [("├ war film", INK, None, False)]),
    (2, [("└ action film", INK, "bold", False)]),
], line_h=21.5)
yy += 10
yy = ann_lines(tx, yy, [
    [("PARENT", PARENT_COL, "bold", None), (" / ", SUBINK, None, None), ("CHILD CHECK", CHILD_COL, "bold", None),
     (" vs {doc. film, war film}", SUBINK, None, None)],
], w - 48, line_h=16.8, size=12.3)
yy = ann_lines(tx, yy, [
    [("→ none / none → settles as a 3rd sibling", INK, "bold", None)],
], w - 48, line_h=16.8, size=12.3)
note(tx, y + h - 36, "no shared parent is invented here —", w)
note(tx, y + h - 20, "same_class is not a per-pair action anymore", w)

# ═══════════════════════════════ Panel (h) — CONSOLIDATE ═══════════════════
x, y, w, h, tx, ty = card(1, 2, TEAL, "(h) Step 7 — consolidation", highlighted=True, accent_label="CONSOLIDATE")
yy = ty + 42
yy = ann_lines(tx, yy, [
    [("biographical film descends to film’s level:", SUBINK, None, None)],
], w - 48, line_h=16)
yy = ann_lines(tx, yy, [
    [("3 candidates ", INK, "bold", None), ("> B (batch cap) → ", SUBINK, None, None), ("CONSOLIDATE", TEAL, "bold", None),
     (" fires first", SUBINK, None, None)],
], w - 48, line_h=17)
yy += 8
yy = body_lines(tx, yy, [
    [("↳ film", INK, None, False)],
    (1, [("├ documentary film", INK, None, False)]),
    (1, [("└ ", INK, None, False), ("genre film", TEAL, "bold", False), ("  (new)", TEAL, "normal", False)]),
    (2, [("├ war film", INK, None, False)]),
    (2, [("└ action film", INK, None, False)]),
], line_h=19.5)
yy += 12
note(tx, yy, "LLM groups TRUE siblings only (same kind of thing);", w, color=TEAL)
yy += 14
note(tx, yy, "documentary film stays out of the group.", w, color=TEAL)
yy += 19
note(tx, yy, "a capacity response, not a per-pair judgment —", w)
yy += 14
note(tx, yy, "never fires below B candidates", w)

# ═══════════════════════════════ Panel (i) — final ═══════════════════════
x, y, w, h, tx, ty = card(2, 2, INKPANEL, "(i) Step 8 — biographical film", highlighted=True, accent_label="FINAL TREE")
yy = ty + 46
yy = ann_lines(tx, yy, [
    [("PARENT CHECK", PARENT_COL, "bold", None), (" vs {doc. film, genre film}", SUBINK, None, None)],
], w - 48, line_h=16.5, size=12.2)
yy = ann_lines(tx, yy, [
    [("→ match (documentary film) → descend → settle", INK, "bold", None)],
], w - 48, line_h=16.5, size=12.2)
yy += 8
yy = body_lines(tx, yy, [
    [("audiovisual work", INK, None, False)],
    (1, [("↳ film ⊕movie", INK, None, False)]),
    (2, [("├ documentary film", INK, None, False)]),
], line_h=20.5)
# "biographical film" nests one level under documentary film, but the
# tree's vertical bar continuing down to "genre film" must stay at
# documentary film's own depth -- two different x offsets on one row, so
# this one line is placed directly rather than through body_lines.
tspan_line(tx + 2 * INDENT_PX, yy, [("│", INK, None, False)], size=19, family=FONT_MONO)
tspan_line(tx + 3 * INDENT_PX, yy, [("└ biographical film", INKPANEL, "bold", False)], size=19, family=FONT_MONO)
yy += 20.5
yy = body_lines(tx, yy, [
    (2, [("└ genre film", TEAL, "bold", False)]),
    (3, [("├ war film", INK, None, False)]),
    (3, [("└ action film", INK, None, False)]),
], line_h=20.5)
note(tx, y + h - 20, "final tree after all eight steps", w)

# ───────────────────────────── footer ─────────────────────────────
fy = H - 14
text(MARGIN, fy, "Every merge, reparent, and consolidation decision is adjudicated by a dedicated LLM call — never inferred from embedding similarity alone.",
     size=12.6, color=FAINT, family=FONT_SANS)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">' + "".join(parts) + '</svg>')

with open("hierarchy_induction.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote hierarchy_induction.svg")
