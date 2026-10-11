#!/usr/bin/env python3
"""Generates the AutoOntic hierarchy-placement algorithm schema as SVG --
the abstract companion to the worked example in hierarchy_induction.svg/pdf.
Matches the current two-call (PARENT CHECK / CHILD CHECK) + capacity-
triggered consolidation-call algorithm in src/ontodisco/hierarchy_induction.py,
with the SAME film-domain toy vocabulary and color system as the worked
example, so the two figures read as one visual series."""

import html

W = 640
MARGIN = 26
CONTENT_W = W - 2 * MARGIN

FONT_SANS = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"

INK = "#1B1F2B"
SUBINK = "#5B6474"
FAINT = "#9AA3B2"
PAGE_BG = "#FFFFFF"

VIOLET = "#6D4CE0"
VIOLET_FILL = "#F2EEFC"
GRAY = "#9099AA"
GRAY_FILL = "#F5F6FA"
ROSE = "#D6336C"
ROSE_FILL = "#FBEFF2"
BLUE = "#2E6FE0"
BLUE_FILL = "#ECF2FD"
TEAL = "#0E8C7F"
TEAL_FILL = "#E7F5F3"
PARENT_COL = "#4C5FD5"
PARENT_FILL = "#EEF0FC"
EXAMPLE_BG = "#F7F7FA"
EXAMPLE_BORDER = "#E4E7EE"

parts = []
cursor_y = 0  # running layout cursor


def esc(s):
    return html.escape(s, quote=False)


def add(s):
    parts.append(s)


def text(x, y, s, size=14, family=FONT_SANS, weight="normal", color=INK,
          anchor="start"):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')


def wrapped(x, y, s, size, color, max_chars, family=FONT_SANS, weight="normal",
            line_h=None, anchor="start"):
    """Greedy word-wrap by character budget (good enough for our fixed
    strings at a fixed font size -- avoids a real text-measurement pass)."""
    line_h = line_h or size * 1.35
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
    yy = y
    for ln in lines:
        text(x, yy, ln, size=size, family=family, weight=weight, color=color, anchor=anchor)
        yy += line_h
    return yy


def mono_block(x, y, w, lines, size=12.3, color=INK, pad=10, bg=EXAMPLE_BG,
                border=EXAMPLE_BORDER, line_h=17):
    """A code-style example chip: monospace lines on a faint tinted panel,
    visually distinct from the descriptive prose above it."""
    h = pad * 2 + len(lines) * line_h - (line_h - size)
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{bg}" stroke="{border}" stroke-width="1"/>')
    yy = y + pad + size * 0.8
    for ln in lines:
        segs = ln if isinstance(ln, list) else [(ln, color, "normal")]
        buf = [f'<text x="{x+pad}" y="{yy}" font-family="{FONT_MONO}" font-size="{size}">']
        for seg_text, seg_color, seg_w in segs:
            buf.append(f'<tspan fill="{seg_color}" font-weight="{seg_w}">{esc(seg_text)}</tspan>')
        buf.append('</text>')
        add("".join(buf))
        yy += line_h
    return y + h


def box(x, y, w, h, border, fill, rx=10, width_px=1.6):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{border}" stroke-width="{width_px}"/>')


def arrow_down(x, y1, y2, color=SUBINK, label=None, label_dx=12, dashed=False):
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    add(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="1.8"{dash} '
        f'marker-end="url(#arrow)"/>')
    if label:
        text(x + label_dx, (y1 + y2) / 2 + 4, label, size=12, color=SUBINK)


def outcome_chip(x, y, w, border, fill, title_segs, example_lines, tail=None,
                  title_size=13.6, desc=None):
    """One decision outcome: a colored label line, an optional one-line
    description, a monospace example chip, and an optional tail ('-> next
    node' etc). Returns the y just below the chip.

    The container box must be painted BEFORE its contents (SVG paints in
    document order, and the box is an opaque fill) -- so this lays out into
    a scratch buffer first to measure the needed height, then emits the box
    rect into the real output followed by the buffered content."""
    global parts
    outer_parts = parts
    parts = []
    pad = 12
    yy = y + pad + title_size * 0.8
    buf = [f'<text x="{x+pad}" y="{yy}">']
    for seg_text, seg_color, seg_w in title_segs:
        buf.append(f'<tspan fill="{seg_color}" font-family="{FONT_SANS}" font-size="{title_size}" '
                    f'font-weight="{seg_w}">{esc(seg_text)}</tspan>')
    buf.append('</text>')
    add("".join(buf))
    yy += 6
    if desc:
        yy = wrapped(x + pad, yy + 14, desc, size=12.4, color=SUBINK, max_chars=70, line_h=16)
    yy += 6
    yy = mono_block(x + pad, yy, w - 2 * pad, example_lines)
    if tail:
        yy += 18
        text(x + pad, yy, tail, size=12.2, color=SUBINK)
        yy += 6
    yy += pad
    inner_parts = parts
    parts = outer_parts
    box(x, y, w, yy - y, border, fill)
    parts.extend(inner_parts)
    return yy


# ═══════════════════════════════ canvas setup ═══════════════════════════════
add(f'<defs>'
    f'<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" '
    f'markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SUBINK}"/></marker>'
    f'<marker id="arrow-loop" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{VIOLET}"/></marker>'
    f'</defs>')

LOOP_GUTTER = 40  # reserved strip on the left for the "repeat" loop arrow + label
x0 = MARGIN + LOOP_GUTTER
w0 = W - x0 - MARGIN
y = MARGIN

# ── title ──
text(x0, y + 10, "Hierarchy placement, per focal type", size=19, weight="bold", color=INK)
y = wrapped(x0, y + 32, "Highest-priority type first; every merge, reparent, and consolidation decision below is a dedicated LLM call.",
            size=12, color=SUBINK, max_chars=78, line_h=16)
y += 12

# ── Box A: priority queue ──
ay_h = 66
box_a_mid_y = y + ay_h / 2  # the loop arrow at the bottom points back here
box(x0, y, w0, ay_h, VIOLET, VIOLET_FILL)
text(x0 + w0/2, y + 24, "Priority queue over T*", size=16, weight="bold", color=INK, anchor="middle")
wrapped(x0 + w0/2, y + 43, "types ranked by Weeds-precision in-degree (τ); ties within δ share a band — seeds, if any, pinned at the top",
        size=12.2, color=SUBINK, max_chars=66, anchor="middle", line_h=16)
y += ay_h

arrow_down(x0 + w0/2, y, y + 22, label="pop highest-priority node", label_dx=14)
y += 22

# ── Box B: level / capacity check ──
by_h = 58
box(x0, y, w0, by_h, GRAY, GRAY_FILL)
text(x0 + w0/2, y + 24, "Current level's candidates", size=15.5, weight="bold", color=INK, anchor="middle")
text(x0 + w0/2, y + 43, "siblings under this anchor (or the live root set)", size=12.2, color=SUBINK, anchor="middle")
y += by_h

arrow_down(x0 + w0/2, y, y + 20)
y += 20

# ── capacity branch: CONSOLIDATE (only when level exceeds B) ──
text(x0 + w0/2, y + 13, "level size > B (candidate batch size)?", size=12.6, weight="bold", color=INK, anchor="middle")
y += 24
yes_x = x0 + 16
arrow_down(yes_x, y, y + 18, color=TEAL)
text(yes_x + 10, y + 13, "yes", size=12, weight="bold", color=TEAL)
y += 18

cons_y_top = y
cons_bottom = outcome_chip(
    x0, y, w0, TEAL, TEAL_FILL,
    title_segs=[("CONSOLIDATE", TEAL, "bold"), ("  — fires BETWEEN types, never on one pair", SUBINK, "normal")],
    desc="LLM groups the TRUE siblings in this level under a synthesized (or reused) shared parent; everything else is left out.",
    example_lines=[
        [("film's children: ", SUBINK, "normal"), ("{documentary film, war film, action film}", INK, "bold")],
        [("3 ", INK, "bold"), ("> B(=2) ", SUBINK, "normal"), ("→ group ", SUBINK, "normal"),
         ("war film + action film", TEAL, "bold"), (" → ", SUBINK, "normal"), ("genre film", TEAL, "bold")],
    ],
    tail="→ level is now ≤ B; the focal type's own checks below run against the consolidated level",
)
y = cons_bottom + 10
text(x0 + w0 - 10, cons_y_top - 14, "no / now ≤ B → continue", size=11.6, color=FAINT, anchor="end")

arrow_down(x0 + w0/2, y, y + 20)
y += 20

# ── Box C: PARENT CHECK ──
pc_title_h = 40
box(x0, y, w0, pc_title_h, PARENT_COL, PARENT_FILL)
text(x0 + w0/2, y + 25, "PARENT CHECK — does focal belong under ANY ONE candidate?", size=14, weight="bold", color=INK, anchor="middle")
y += pc_title_h + 10

# outcome 1: merge
merge_bottom = outcome_chip(
    x0, y, w0, BLUE, BLUE_FILL,
    title_segs=[("match, same_concept = true", BLUE, "bold"), ("  →  ", SUBINK, "normal"), ("MERGE", BLUE, "bold")],
    desc="focal and the matched candidate are the identical concept, just worded differently.",
    example_lines=[
        [("movie", INK, "bold"), (" vs {film} → same_concept=true", SUBINK, "normal")],
        [("→ merge ", SUBINK, "normal"), ("movie", BLUE, "bold"), (" into ", SUBINK, "normal"), ("film", BLUE, "bold")],
    ],
    tail="→ next node",
)
y = merge_bottom + 10

# outcome 2: descend
descend_bottom = outcome_chip(
    x0, y, w0, PARENT_COL, PARENT_FILL,
    title_segs=[("match, not same concept", PARENT_COL, "bold"), ("  →  ", SUBINK, "normal"), ("DESCEND", PARENT_COL, "bold")],
    desc="focal IS-A the matched candidate; recurse into that candidate's own children.",
    example_lines=[
        [("documentary film", INK, "bold"), (" vs {audiovisual work} → match", SUBINK, "normal")],
        [("→ descend into ", SUBINK, "normal"), ("audiovisual work", PARENT_COL, "bold"), ("'s children", SUBINK, "normal")],
    ],
    tail="→ recurse: back to the level/capacity check, one level deeper",
)
y = descend_bottom + 10

# outcome 3: no match -> fall through
text(x0 + 4, y + 14, "no match on EITHER candidate", size=12.6, weight="bold", color=SUBINK)
text(x0 + 4, y + 30, "→ proceed to CHILD CHECK below", size=12.2, color=FAINT)
y += 44

arrow_down(x0 + w0/2, y, y + 20, color=ROSE)
y += 20

# ── Box D: CHILD CHECK ──
cc_title_h = 54
box(x0, y, w0, cc_title_h, ROSE, ROSE_FILL)
text(x0 + w0/2, y + 23, "CHILD CHECK — does focal subsume ONE OR MORE candidates?", size=14, weight="bold", color=INK, anchor="middle")
text(x0 + w0/2, y + 41, "only asked if PARENT CHECK matched nothing", size=11.8, color=SUBINK, anchor="middle")
y += cc_title_h + 10

reparent_bottom = outcome_chip(
    x0, y, w0, ROSE, ROSE_FILL,
    title_segs=[("match (one or more)", ROSE, "bold"), ("  →  ", SUBINK, "normal"), ("REPARENT", ROSE, "bold")],
    desc="insert focal above the matched candidate(s); corrects an earlier too-shallow attachment.",
    example_lines=[
        [("film", INK, "bold"), (" subsumes ", SUBINK, "normal"), ("documentary film", INK, "bold")],
        [("→ reparent ", SUBINK, "normal"), ("documentary film", ROSE, "bold"),
         (": audiovisual work ", SUBINK, "normal"), ("→", ROSE, "bold"), (" film", SUBINK, "normal")],
    ],
    tail="→ focal settles at this level; next node",
)
y = reparent_bottom + 10

none_bottom = outcome_chip(
    x0, y, w0, GRAY, GRAY_FILL,
    title_segs=[("no match either", GRAY, "bold"), ("  →  ", SUBINK, "normal"), ("NONE", GRAY, "bold")],
    desc="neither check found anything; focal settles right here.",
    example_lines=[
        [("war film", INK, "bold"), (" vs {documentary film} → no match in either check", SUBINK, "normal")],
        [("→ settle as a plain sibling under ", SUBINK, "normal"), ("film", INK, "bold")],
    ],
    tail="→ new root (top level) or plain child of the current anchor; next node",
)
y = none_bottom + 14

arrow_down(x0 + w0/2, y, y + 20)
y += 20

# ── Box E: depth bound ──
dy_h = 54
box(x0, y, w0, dy_h, VIOLET, VIOLET_FILL)
text(x0 + w0/2, y + 23, "Depth bound D", size=15.5, weight="bold", color=INK, anchor="middle")
text(x0 + w0/2, y + 41, "exceeds it ⇒ defer into unresolved_children, never forced or discarded", size=12.2, color=SUBINK, anchor="middle")
y += dy_h

# depth subscript patch (DejaVu Sans doesn't need special handling, draw a small "max")
text(x0 + w0/2 + 60, y - dy_h + 27, "max", size=10.5, color=INK)

y += 26

# ── loop arrow back to the top ──
loop_top_y = box_a_mid_y
loop_x = MARGIN + LOOP_GUTTER / 2
add(f'<path d="M {loop_x} {y} L {loop_x} {loop_top_y} L {x0} {loop_top_y}" '
    f'fill="none" stroke="{VIOLET}" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#arrow-loop)"/>')
add(f'<line x1="{x0}" y1="{y}" x2="{loop_x}" y2="{y}" stroke="{VIOLET}" stroke-width="1.8" stroke-dasharray="6,4"/>')
label_y = (y + loop_top_y) / 2
text(loop_x - 10, label_y, "repeat until the priority queue is empty", size=11.6,
     color=VIOLET, anchor="middle")
# rotate that label vertically
parts[-1] = parts[-1].replace(
    f'<text x="{loop_x - 10}" y="{label_y}"',
    f'<text x="{loop_x - 10}" y="{label_y}" transform="rotate(-90 {loop_x - 10} {label_y})"'
)

H = y + 20

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAGE_BG}"/>' + "".join(parts) + '</svg>')

with open("hierarchy_induction_example.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print(f"wrote hierarchy_induction_example.svg ({W}x{H})")
