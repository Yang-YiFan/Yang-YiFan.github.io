#!/usr/bin/env python3
"""Generate 9 PNG figures for the Blackwell tcgen05.mma SMEM descriptor blog.

Output directory: ../figures/

Visual conventions
==================
Three primitive blocks have a *fixed* visual signature across every figure:

  Swizzle atom   : fill = ATOM_FILL,    edge = ATOM_EDGE    (thick, ATOM_LW pt)
  MMA subtile    : fill = none,         edge = SUBTILE_EDGE (thicker, SUBTILE_LW)
  8x16 B chunk   : pastel (tab20), grey thin border

Every figure that shows atoms/subtiles also carries a small sidebar legend
showing a sample of each primitive with its shape.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "figure.facecolor": "white",
})

DPI = 150
OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Visual-convention constants
# ---------------------------------------------------------------------------
ATOM_FILL_A  = "#B6D7F4"     # denser light blue (even atoms)
ATOM_FILL_B  = "#FFD8A8"     # denser light peach (odd atoms)
ATOM_FILL    = ATOM_FILL_A   # legacy alias (used by sample swatches)
ATOM_EDGE    = "#3B7DC4"     # dark blue
ATOM_LW      = 3.0
SUBTILE_EDGE = "#C0392B"     # dark red
SUBTILE_LW   = 4.0
CHUNK_EDGE   = "#000000"     # plain black
CHUNK_LW     = 0.7


def _atom_fill(m, k):
    """Two-tone alternation: (m+k) even -> A, odd -> B."""
    return ATOM_FILL_A if ((m + k) % 2 == 0) else ATOM_FILL_B

ATOM_LABEL_KMAJOR     = ("Swizzle 128B atom (alternating fill)\n"
                          "M=8 × K=64 bf16 (8×128B = 1024 B)")
ATOM_LABEL_MNMAJOR    = ("Swizzle 128B atom (alternating fill)\n"
                          "M=64 × K=8 bf16 (128B×8 = 1024 B)")
SUBTILE_LABEL_KMAJOR  = "MMA subtile\nM=64 × K=16 bf16 (64×32B)"
SUBTILE_LABEL_MNMAJOR = "MMA subtile\nM=64 × K=16 bf16 (128B×16)"
SMEM_TILE_LABEL       = "SMEM tile\nM=128 × K=128 bf16 (256×128B = 32 KB)"
CHUNK_LABEL           = "8×16B chunk\n(one tcgen05.mma element block)"


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Sidebar legend helper
# ---------------------------------------------------------------------------
def _draw_legend_box(ax, x, y_top, items, width=4.6, line_h=1.05,
                     swatch_w=1.0, swatch_h=0.55):
    """Draw a vertical legend box.

    items: list of dicts each describing one swatch + label, with keys:
        kind  : "atom" | "subtile" | "chunk" | "smem_tile"
        label : multi-line text
        fill  : optional pastel fill colour (for "chunk")
    """
    n  = len(items)
    pad = 0.30
    box_h = n * line_h + pad
    box   = Rectangle((x, y_top - box_h), width, box_h,
                      facecolor="white", edgecolor="#666666",
                      linewidth=0.8, zorder=4)
    ax.add_patch(box)

    cur_y = y_top - pad
    for it in items:
        sx = x + 0.30
        sy = cur_y - swatch_h - 0.10
        if it["kind"] == "atom":
            # Two-cell strip: ATOM_FILL_A on left, ATOM_FILL_B on right.
            half = swatch_w / 2.0
            rA = Rectangle((sx, sy), half, swatch_h,
                           facecolor=ATOM_FILL_A, edgecolor="none",
                           zorder=5)
            rB = Rectangle((sx + half, sy), half, swatch_h,
                           facecolor=ATOM_FILL_B, edgecolor="none",
                           zorder=5)
            ax.add_patch(rA)
            ax.add_patch(rB)
            border = Rectangle((sx, sy), swatch_w, swatch_h,
                               facecolor="none", edgecolor=ATOM_EDGE,
                               linewidth=ATOM_LW, zorder=6)
            ax.add_patch(border)
        elif it["kind"] == "subtile":
            r = Rectangle((sx, sy), swatch_w, swatch_h,
                          facecolor="none", edgecolor=SUBTILE_EDGE,
                          linewidth=SUBTILE_LW, zorder=5)
            ax.add_patch(r)
        elif it["kind"] == "chunk":
            r = Rectangle((sx, sy), swatch_w, swatch_h,
                          facecolor=it.get("fill", "#FDD49E"),
                          edgecolor=CHUNK_EDGE,
                          linewidth=CHUNK_LW, zorder=5)
            ax.add_patch(r)
        elif it["kind"] == "smem_tile":
            r = Rectangle((sx, sy), swatch_w, swatch_h,
                          facecolor="#F5F5F5", edgecolor="black",
                          linewidth=1.2, zorder=5)
            ax.add_patch(r)
        else:
            raise ValueError(it["kind"])

        ax.text(sx + swatch_w + 0.20, sy + swatch_h / 2.0,
                it["label"], ha="left", va="center", fontsize=11,
                zorder=6)
        cur_y -= line_h
    return box_h


# ---------------------------------------------------------------------------
# Figure 1: descriptor_bits.png  (unchanged)
# ---------------------------------------------------------------------------
def make_descriptor_bits():
    fields = [
        (0,  14, "start_address",       "#cfe7ff", "addr >> 4, 16-B granularity", "down"),
        (14, 16, "",                    "#e6e6e6", "unused",                      None),
        (16, 30, "leading_byte_offset", "#d4edda", "LBO in 16-B units",           "down"),
        (30, 32, "",                    "#e6e6e6", "unused",                      None),
        (32, 46, "stride_byte_offset",  "#ffe0b3", "SBO in 16-B units",           "down"),
        (46, 48, "version",             "#e6e6e6", "version (= 1)",               "up"),
        (48, 49, "",                    "#e6e6e6", "unused",                      None),
        (49, 52, "base_offset",         "#e6e6e6", "base_offset",                 "down"),
        (52, 53, "lbo_mode",            "#e6e6e6", "lbo_mode",                    "up"),
        (53, 61, "",                    "#e6e6e6", "unused",                      None),
        (61, 64, "layout_type",         "#fadadd", "swizzle mode",                "down"),
    ]

    fig_w = 17.0
    fig_h = 4.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_y     = 1.9
    bar_h     = 1.0
    bit_total = 64

    down_short_y = bar_y - 0.20
    down_long_y  = bar_y - 1.05
    up_short_y   = bar_y + bar_h + 0.55
    up_long_y    = bar_y + bar_h + 1.25

    for lo, hi, name, color, meaning, leader in fields:
        w = hi - lo
        rect = Rectangle((lo, bar_y), w, bar_h,
                         facecolor=color, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)

        cx = lo + w / 2.0
        if not name:
            if w >= 3:
                ax.text(cx, bar_y + bar_h * 0.5, "unused",
                        ha="center", va="center", fontsize=11,
                        color="#555555")
            continue

        if w >= 6:
            ax.text(cx, bar_y + bar_h * 0.62, name,
                    ha="center", va="center", fontsize=13,
                    fontweight="bold")
            ax.text(cx, bar_y + bar_h * 0.30, f"[{lo}:{hi})",
                    ha="center", va="center", fontsize=11,
                    color="#333333")
            ax.text(cx, bar_y - 0.40, meaning,
                    ha="center", va="top", fontsize=12, color="#222222")
        else:
            if leader == "down":
                y_lbl = down_long_y if w == 1 else down_short_y
                ax.plot([cx, cx], [bar_y, y_lbl + 0.05],
                        color="#444444", linewidth=0.6)
                txt = f"{name}\n[{lo}:{hi})\n{meaning}"
                ax.text(cx, y_lbl, txt,
                        ha="center", va="top", fontsize=11,
                        color="#222222")
            else:
                y_lbl = up_long_y if w == 1 else up_short_y
                ax.plot([cx, cx], [bar_y + bar_h, y_lbl - 0.05],
                        color="#444444", linewidth=0.6)
                txt = f"{name}\n[{lo}:{hi})\n{meaning}"
                ax.text(cx, y_lbl, txt,
                        ha="center", va="bottom", fontsize=11,
                        color="#222222")

    for b in range(0, bit_total + 1, 4):
        ax.plot([b, b], [bar_y + bar_h, bar_y + bar_h + 0.06],
                color="black", linewidth=0.6)
        ax.text(b, bar_y + bar_h + 0.13, str(b),
                ha="center", va="bottom", fontsize=11)

    ax.text(bit_total / 2.0, up_long_y + 0.55,
            "Blackwell tcgen05.mma SMEM Descriptor (64 bits)",
            ha="center", va="bottom", fontsize=18, fontweight="bold")

    ax.set_xlim(-1, bit_total + 1)
    ax.set_ylim(down_long_y - 0.4, up_long_y + 1.1)
    ax.set_aspect("auto")
    ax.axis("off")

    _save(fig, "descriptor_bits.png")


# ---------------------------------------------------------------------------
# Figure 2: kmajor_tile.png
# K-major: 16 M-atoms (rows) x 2 K-atoms (cols), atom = M=8 x K=64 bf16 = 1KB
# ---------------------------------------------------------------------------
def make_kmajor_tile():
    n_m_atoms = 16
    n_k_atoms = 2
    atom_w = 4.0
    atom_h = 1.0

    fig, ax = plt.subplots(figsize=(15, 11))

    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = k * atom_w
            y = (n_m_atoms - 1 - m) * atom_h
            rect = Rectangle((x, y), atom_w, atom_h,
                             facecolor=_atom_fill(m, k), edgecolor=ATOM_EDGE,
                             linewidth=ATOM_LW)
            ax.add_patch(rect)
            ax.text(x + atom_w / 2.0, y + atom_h * 0.62, f"({m}, {k})",
                    ha="center", va="center", fontsize=12)
            storage_idx = k * n_m_atoms + m
            ax.text(x + atom_w / 2.0, y + atom_h * 0.30, f"#{storage_idx}",
                    ha="center", va="center", fontsize=11, color="#555555")

    total_w = n_k_atoms * atom_w
    total_h = n_m_atoms * atom_h

    # K arrow top
    ax.annotate("", xy=(total_w + 0.05, total_h + 1.0),
                xytext=(0, total_h + 1.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(total_w / 2.0, total_h + 1.2,
            "K  (K=0 → K=127, contiguous)",
            ha="center", va="bottom", fontsize=14)

    # M arrow left
    ax.annotate("", xy=(-3.0, 0), xytext=(-3.0, total_h),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(-3.25, total_h / 2.0, "M  (M=0 → M=127)",
            ha="center", va="center", fontsize=14, rotation=90)

    # M-atom stride arrow
    mstride_x = -1.5
    ax.annotate("",
                xy=(mstride_x, total_h - 1.5 * atom_h),
                xytext=(mstride_x, total_h - 0.5 * atom_h),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
    ax.text(mstride_x - 0.15, total_h - atom_h,
            "M-atom stride\n= 1024 B",
            ha="right", va="center", fontsize=14, color="#c0392b")

    # K-atom stride arrow
    ax.annotate("",
                xy=(atom_w + atom_w / 2.0, total_h + 0.05),
                xytext=(atom_w / 2.0,      total_h + 0.05),
                arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.4))
    ax.text(atom_w, total_h + 0.30,
            "K-atom stride = 16384 B",
            ha="center", va="bottom", fontsize=14, color="#2c7fb8")

    # Sidebar legend
    legend_x = total_w + 0.8
    _draw_legend_box(ax, legend_x, total_h - 0.1, [
        {"kind": "smem_tile", "label": SMEM_TILE_LABEL},
        {"kind": "atom",      "label": ATOM_LABEL_KMAJOR},
    ], width=5.0, line_h=1.15)

    # Storage order legend (below)
    ax.text(legend_x, total_h - 3.0,
            "Storage order in SMEM\n(column-major, M-first):\n"
            "#0=(0,0) → #1=(1,0) → ... → #15=(15,0)\n"
            "→ #16=(0,1) → ... → #31=(15,1)",
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#f0f0f0", edgecolor="black", linewidth=0.6))

    # Title
    ax.text(total_w / 2.0, total_h + 1.9,
            "(M=128, K=128) bf16 K-major Swizzle 128B SMEM tile\n"
            "16 M-atoms × 2 K-atoms = 32 atoms",
            ha="center", va="bottom", fontsize=18, fontweight="bold")

    ax.set_xlim(-4.5, total_w + 6.5)
    ax.set_ylim(-0.4, total_h + 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, "kmajor_tile.png")


# ---------------------------------------------------------------------------
# Figure 3 / 5: kmajor_subtiles.png and kmajor_advance.png
# Swizzle atoms underneath (light-blue with thick blue border), MMA subtile
# red borders overlaid on top. Pastel fill inside each subtile.
# ---------------------------------------------------------------------------
def make_kmajor_subtiles(advance_overlay=False):
    n_m_atoms = 16
    n_k_atoms = 2
    atom_w = 8.0
    atom_h = 1.0

    subtile_h = 8 * atom_h
    subtile_w = atom_w / 4.0

    fig_w = 20 if advance_overlay else 18
    fig, ax = plt.subplots(figsize=(fig_w, 11))

    byte_offsets = [
        [0,    32,    64,    96,    16384, 16416, 16448, 16480],
        [8192, 8224,  8256,  8288,  24576, 24608, 24640, 24672],
    ]

    total_w = n_k_atoms * atom_w
    total_h = n_m_atoms * atom_h

    # K-major chunk grid:
    #   each swizzle atom (M=8 rows × K=64 bf16) = 8 rows × 8 chunks-along-K = 64 chunks
    #   so the whole tile is 16 (M) × 16 (K) = 256 chunks
    n_chunks_k_per_atom = 8
    n_chunks_m_per_atom = 8
    chunk_w_k = atom_w / n_chunks_k_per_atom       # width in K direction
    chunk_h_m = atom_h / n_chunks_m_per_atom       # height in M direction

    # --- Layer 1: swizzle atoms with solid (denser) alternating fill ---
    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = k * atom_w
            y = (n_m_atoms - 1 - m) * atom_h
            rect = Rectangle((x, y), atom_w, atom_h,
                             facecolor=_atom_fill(m, k), edgecolor=ATOM_EDGE,
                             linewidth=ATOM_LW, zorder=2)
            ax.add_patch(rect)

    # --- Layer 2: thin black 8×16B chunk grid inside every atom ---
    # A K-major atom is M=8 rows × K=64 bf16; an 8×16B chunk is
    # `8 rows × 16 B`, so each atom holds 8 chunks-along-K and 1 chunk-along-M.
    # Draw ONLY the 7 internal vertical lines per atom that separate the 8
    # K-chunks — the atom border already provides the horizontal separation.
    for k_atom in range(n_k_atoms):
        x0 = k_atom * atom_w
        for j in range(1, n_chunks_k_per_atom):
            x = x0 + j * chunk_w_k
            ax.plot([x, x], [0, total_h], color="#000000",
                    linewidth=0.5, zorder=3)

    # --- Layer 3: redraw atom borders crisp on top of chunk grid ---
    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = k * atom_w
            y = (n_m_atoms - 1 - m) * atom_h
            edge_only = Rectangle((x, y), atom_w, atom_h,
                                  facecolor="none", edgecolor=ATOM_EDGE,
                                  linewidth=ATOM_LW, zorder=4)
            ax.add_patch(edge_only)

    # --- Layer 4: red MMA subtile borders on TOP of everything ---
    # No per-subtile fill; the contour is the only visual cue.
    for m_tile in range(2):
        for k_tile in range(8):
            k_atom_idx = k_tile // 4
            k_within   = k_tile %  4
            x = k_atom_idx * atom_w + k_within * subtile_w
            y = total_h - (m_tile + 1) * subtile_h
            rect = Rectangle((x, y), subtile_w, subtile_h,
                             facecolor="none", edgecolor=SUBTILE_EDGE,
                             linewidth=SUBTILE_LW, zorder=5)
            ax.add_patch(rect)

    # --- Layer 5: axis-style row/col labels (replaces per-subtile labels) ---
    # Row labels (m=0, m=1) on the LEFT side of each MMA-subtile row.
    for m_tile in range(2):
        y_center = total_h - (m_tile + 0.5) * subtile_h
        ax.text(-0.40, y_center, f"m={m_tile}",
                ha="right", va="center", fontsize=14,
                fontweight="bold", zorder=6)
    # Column labels (k=0..k=7) along the TOP of each MMA-subtile column.
    for k_tile in range(8):
        k_atom_idx = k_tile // 4
        k_within   = k_tile %  4
        x_center = k_atom_idx * atom_w + (k_within + 0.5) * subtile_w
        ax.text(x_center, total_h + 0.20, f"k={k_tile}",
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", zorder=6)

    # --- Layer 6: byte-offset labels inside top-left chunk (advance only) ---
    # The label sits inside the subtile's top-left chunk, and a short slanted
    # arrow points from the label down-and-to-the-left to the subtile's
    # top-left corner — communicating "this byte is the start_address of
    # this MMA subtile."
    if advance_overlay:
        for m_tile in range(2):
            for k_tile in range(8):
                k_atom_idx = k_tile // 4
                k_within   = k_tile %  4
                # Top-left corner of this subtile.
                x_tl = k_atom_idx * atom_w + k_within * subtile_w
                y_tl = total_h - m_tile * subtile_h
                bo = byte_offsets[m_tile][k_tile]
                # Place the label slightly inside the top region of the
                # subtile. Chunks are tiny in M (0.125 axes-units) so the
                # text necessarily spans multiple chunk rows visually —
                # the arrow makes the anchor unambiguous.
                lbl_x = x_tl + 0.60 * subtile_w
                lbl_y = y_tl - 0.45
                # Arrow target: just inside the top-left corner of the
                # subtile so the arrowhead doesn't sit on the red border.
                tgt_x = x_tl + 0.05
                tgt_y = y_tl - 0.05
                ax.annotate(
                    f"{bo}",
                    xy=(tgt_x, tgt_y),
                    xytext=(lbl_x, lbl_y),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="#000000",
                    arrowprops=dict(arrowstyle="->", color="#000000",
                                    lw=1.0, shrinkA=1, shrinkB=2),
                    zorder=7,
                )

    # Axes — placed slightly outside the k=/m= axis-style labels.
    ax.annotate("", xy=(total_w + 0.05, total_h + 0.75),
                xytext=(0, total_h + 0.75),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(total_w / 2.0, total_h + 0.95, "K  (contiguous)",
            ha="center", va="bottom", fontsize=14)
    ax.annotate("", xy=(-1.10, 0), xytext=(-1.10, total_h),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(-1.35, total_h / 2.0, "M",
            ha="center", va="center", fontsize=14, rotation=90)

    if advance_overlay:
        y_arrow = total_h + 1.9
        for k in range(7):
            k_atom_a = k // 4
            k_atom_b = (k + 1) // 4
            x_a = k_atom_a * atom_w + (k % 4) * subtile_w + subtile_w / 2.0
            x_b = k_atom_b * atom_w + ((k + 1) % 4) * subtile_w + subtile_w / 2.0
            ax.annotate("", xy=(x_b, y_arrow), xytext=(x_a, y_arrow),
                        arrowprops=dict(arrowstyle="->", color="#c0392b",
                                        lw=1.5))
        ax.text(total_w / 2.0, y_arrow + 0.3,
                "K-tile advance (non-affine): "
                "+32, +32, +32, +16288, +32, +32, +32 B",
                ha="center", va="bottom", fontsize=14, color="#c0392b",
                fontweight="bold")

        ax.annotate("",
                    xy=(-2.1, total_h - 1.5 * subtile_h),
                    xytext=(-2.1, total_h - 0.5 * subtile_h),
                    arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.8))
        ax.text(-2.3, total_h - subtile_h,
                "M-tile advance\n(+8192 B)",
                ha="right", va="center", fontsize=14, color="#2c7fb8",
                fontweight="bold")

        title = ("(M=128, K=128) K-major Swizzle 128B tile — "
                 "byte offsets of each tcgen05.mma subtile from subtile (0,0)")
    else:
        title = ("(M=128, K=128) K-major Swizzle 128B tile — "
                 "16 tcgen05.mma subtiles (2 M-tiles × 8 K-tiles)")

    title_y = total_h + (2.7 if advance_overlay else 1.7)
    ax.text(total_w / 2.0, title_y, title,
            ha="center", va="bottom", fontsize=18, fontweight="bold")

    # Sidebar legend (atom + subtile + smem tile)
    legend_x = total_w + 0.6
    _draw_legend_box(ax, legend_x, total_h - 0.1, [
        {"kind": "smem_tile", "label": SMEM_TILE_LABEL},
        {"kind": "atom",      "label": ATOM_LABEL_KMAJOR},
        {"kind": "subtile",   "label": SUBTILE_LABEL_KMAJOR},
        {"kind": "chunk",     "label": CHUNK_LABEL,
         "fill": "#FFFFFF"},
    ], width=5.0, line_h=1.15)

    ax.set_xlim(-3.4, total_w + 6.0)
    ax.set_ylim(-0.4, total_h + (3.6 if advance_overlay else 2.6))
    ax.set_aspect("equal")
    ax.axis("off")

    name = "kmajor_advance.png" if advance_overlay else "kmajor_subtiles.png"
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 4: kmajor_chunks.png
# One MMA subtile (red outer border) -> 8 x 2 = 16 pastel chunks.
# ---------------------------------------------------------------------------
def make_kmajor_chunks():
    n_m_chunks = 8
    n_k_chunks = 2
    chunk_w = 2.2
    chunk_h = 1.0

    fig, ax = plt.subplots(figsize=(14, 11))

    cmap = plt.get_cmap("tab20")
    total_w = n_k_chunks * chunk_w
    total_h = n_m_chunks * chunk_h

    for m in range(n_m_chunks):
        for k in range(n_k_chunks):
            x = k * chunk_w
            y = (n_m_chunks - 1 - m) * chunk_h
            idx = m * n_k_chunks + k
            color = cmap(idx % 20)
            r, g, b, _ = color
            light = (r * 0.40 + 0.60, g * 0.40 + 0.60, b * 0.40 + 0.60)
            rect = Rectangle((x, y), chunk_w, chunk_h,
                             facecolor=light, edgecolor=CHUNK_EDGE,
                             linewidth=CHUNK_LW, zorder=2)
            ax.add_patch(rect)
            ax.text(x + chunk_w / 2.0, y + chunk_h * 0.62,
                    f"({m}, {k})",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", zorder=3)
            ax.text(x + chunk_w / 2.0, y + chunk_h * 0.30,
                    "8×16B",
                    ha="center", va="center", fontsize=11, color="#444444",
                    zorder=3)

    # Outer MMA subtile border (red, thick)
    outer = Rectangle((0, 0), total_w, total_h,
                      facecolor="none", edgecolor=SUBTILE_EDGE,
                      linewidth=SUBTILE_LW, zorder=4)
    ax.add_patch(outer)

    # SBO arrow
    arrow_x = total_w + 0.4
    ax.annotate("",
                xy=(arrow_x, total_h - 1.5 * chunk_h),
                xytext=(arrow_x, total_h - 0.5 * chunk_h),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8))
    ax.text(arrow_x + 0.15, total_h - chunk_h,
            "SBO = 1024 B\n(M-atom stride)",
            ha="left", va="center", fontsize=14, color="#c0392b",
            fontweight="bold")

    # LBO arrow
    ax.annotate("",
                xy=(1.5 * chunk_w, total_h + 0.25),
                xytext=(0.5 * chunk_w, total_h + 0.25),
                arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.8))
    ax.text(chunk_w, total_h + 0.45,
            "LBO = 16 B",
            ha="center", va="bottom", fontsize=14, color="#2c7fb8",
            fontweight="bold")

    ax.text(total_w / 2.0, -0.6, "K (chunk_k = 0, 1)  contiguous",
            ha="center", va="top", fontsize=14)
    ax.text(-0.5, total_h / 2.0, "M (chunk_m = 0..7)",
            ha="center", va="center", fontsize=14, rotation=90)

    ax.text(total_w / 2.0, -1.4,
            "addr(m, k) = start_address + m·SBO + k·LBO + swizzle_XOR(m, k)",
            ha="center", va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#fff9d6", edgecolor="black", linewidth=0.6))

    ax.text(total_w / 2.0, total_h + 1.4,
            "First MMA subtile (M=64 × K=16 bf16 = 64×32B) — "
            "8×2 = 16 chunks of 8×16B each",
            ha="center", va="bottom", fontsize=17, fontweight="bold")

    # Legend
    legend_x = total_w + 3.0
    _draw_legend_box(ax, legend_x, total_h - 0.1, [
        {"kind": "subtile", "label": SUBTILE_LABEL_KMAJOR},
        {"kind": "chunk",   "label": CHUNK_LABEL,
         "fill": "#FDD49E"},
    ], width=4.4, line_h=1.15)

    ax.set_xlim(-1.5, total_w + 8.0)
    ax.set_ylim(-2.2, total_h + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, "kmajor_chunks.png")


# ---------------------------------------------------------------------------
# Figure 6: mnmajor_tile.png
# ---------------------------------------------------------------------------
def make_mnmajor_tile():
    n_m_atoms = 2
    n_k_atoms = 16
    atom_w = 1.0
    atom_h = 4.0

    fig, ax = plt.subplots(figsize=(18, 11))

    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = k * atom_w
            y = (n_m_atoms - 1 - m) * atom_h
            rect = Rectangle((x, y), atom_w, atom_h,
                             facecolor=_atom_fill(m, k), edgecolor=ATOM_EDGE,
                             linewidth=ATOM_LW)
            ax.add_patch(rect)
            ax.text(x + atom_w / 2.0, y + atom_h * 0.55,
                    f"({m},{k})",
                    ha="center", va="center", fontsize=11)
            storage_idx = k * n_m_atoms + m
            ax.text(x + atom_w / 2.0, y + atom_h * 0.42,
                    f"#{storage_idx}",
                    ha="center", va="center", fontsize=10, color="#555555")

    total_w = n_k_atoms * atom_w
    total_h = n_m_atoms * atom_h

    ax.annotate("", xy=(total_w + 0.05, total_h + 1.0),
                xytext=(0, total_h + 1.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(total_w / 2.0, total_h + 1.2, "K  (K=0 → K=127)",
            ha="center", va="bottom", fontsize=14)

    ax.annotate("", xy=(-3.5, 0), xytext=(-3.5, total_h),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(-3.75, total_h / 2.0, "M  (M=0 → M=127, contiguous)",
            ha="center", va="center", fontsize=14, rotation=90)

    ax.annotate("",
                xy=(1.5 * atom_w, total_h + 0.20),
                xytext=(0.5 * atom_w, total_h + 0.20),
                arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.4))
    ax.text(atom_w, total_h + 0.42,
            "K-atom stride = 2048 B",
            ha="center", va="bottom", fontsize=14, color="#2c7fb8")

    mstride_x = -1.4
    ax.annotate("",
                xy=(mstride_x, total_h - 1.5 * atom_h),
                xytext=(mstride_x, total_h - 0.5 * atom_h),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
    ax.text(mstride_x + 0.15, total_h - atom_h,
            "M-atom stride\n= 1024 B",
            ha="left", va="center", fontsize=14, color="#c0392b")

    # Sidebar legend
    legend_x = total_w + 0.6
    _draw_legend_box(ax, legend_x, total_h - 0.1, [
        {"kind": "smem_tile", "label": SMEM_TILE_LABEL},
        {"kind": "atom",      "label": ATOM_LABEL_MNMAJOR},
    ], width=5.0, line_h=1.15)

    ax.text(legend_x, total_h - 3.1,
            "Storage order in SMEM\n(column-major, M-first):\n"
            "#0=(0,0) → #1=(1,0) → #2=(0,1) → #3=(1,1) → ...\n"
            "→ #30=(0,15) → #31=(1,15)",
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#f0f0f0", edgecolor="black", linewidth=0.6))

    ax.text(total_w / 2.0, total_h + 1.9,
            "(M=128, K=128) bf16 MN-major Swizzle 128B SMEM tile\n"
            "2 M-atoms × 16 K-atoms = 32 atoms",
            ha="center", va="bottom", fontsize=18, fontweight="bold")

    ax.set_xlim(-4.5, total_w + 7.0)
    ax.set_ylim(-0.5, total_h + 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, "mnmajor_tile.png")


# ---------------------------------------------------------------------------
# Figure 7 / 9: mnmajor_subtiles.png and mnmajor_advance.png
# ---------------------------------------------------------------------------
def make_mnmajor_subtiles(advance_overlay=False):
    n_m_atoms = 2
    n_k_atoms = 16
    atom_w = 1.0
    atom_h = 4.0

    subtile_w = 2 * atom_w
    subtile_h = 1 * atom_h

    fig, ax = plt.subplots(figsize=(19, 11))

    byte_offsets = [
        [0,    4096,  8192,  12288, 16384, 20480, 24576, 28672],
        [1024, 5120,  9216,  13312, 17408, 21504, 25600, 29696],
    ]

    total_w = n_k_atoms * atom_w
    total_h = n_m_atoms * atom_h

    # MN-major chunk grid:
    #   each swizzle atom (M=64 × K=8 rows) = 8 chunks-along-M × 8 K-rows
    #   chunks-along-M: M is vertical here; chunk_h_m = atom_h/8 = 0.5
    #   chunks-along-K: K is horizontal here; one chunk per K-row, so 8 cols
    n_chunks_m_per_atom = 8
    n_chunks_k_per_atom = 8
    chunk_w_k = atom_w / n_chunks_k_per_atom   # 0.125
    chunk_h_m = atom_h / n_chunks_m_per_atom   # 0.5

    # Layer 1: swizzle atoms with solid alternating fill
    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = k * atom_w
            y = (n_m_atoms - 1 - m) * atom_h
            rect = Rectangle((x, y), atom_w, atom_h,
                             facecolor=_atom_fill(m, k), edgecolor=ATOM_EDGE,
                             linewidth=ATOM_LW, zorder=2)
            ax.add_patch(rect)

    # Layer 2: thin black 8×16B chunk grid inside every atom.
    # An MN-major atom is M=64 × K=8 rows; an 8×16B chunk is
    # `16 B along M × 8 rows along K`, so each atom holds 8 chunks-along-M
    # and 1 chunk-along-K. With M vertical and K horizontal in this figure,
    # the 8 M-chunks are horizontal stripes separated by 7 horizontal lines
    # per atom — the atom border already provides the vertical separation.
    for m_atom in range(n_m_atoms):
        y0 = (n_m_atoms - 1 - m_atom) * atom_h
        for j in range(1, n_chunks_m_per_atom):
            y = y0 + j * chunk_h_m
            ax.plot([0, total_w], [y, y], color="#000000",
                    linewidth=0.5, zorder=3)

    # Layer 3: redraw atom borders crisp on top of chunk grid
    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = k * atom_w
            y = (n_m_atoms - 1 - m) * atom_h
            edge_only = Rectangle((x, y), atom_w, atom_h,
                                  facecolor="none", edgecolor=ATOM_EDGE,
                                  linewidth=ATOM_LW, zorder=4)
            ax.add_patch(edge_only)

    # Layer 4: red MMA subtile borders on top (no fill).
    for m_tile in range(2):
        for k_tile in range(8):
            x = k_tile * subtile_w
            y = total_h - (m_tile + 1) * subtile_h
            rect = Rectangle((x, y), subtile_w, subtile_h,
                             facecolor="none", edgecolor=SUBTILE_EDGE,
                             linewidth=SUBTILE_LW, zorder=5)
            ax.add_patch(rect)

    # Layer 5: axis-style row/col labels.
    for m_tile in range(2):
        y_center = total_h - (m_tile + 0.5) * subtile_h
        ax.text(-0.30, y_center, f"m={m_tile}",
                ha="right", va="center", fontsize=14,
                fontweight="bold", zorder=6)
    for k_tile in range(8):
        x_center = (k_tile + 0.5) * subtile_w
        ax.text(x_center, total_h + 0.18, f"k={k_tile}",
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", zorder=6)

    # Layer 6: byte-offset labels inside top-left chunk (advance only).
    if advance_overlay:
        for m_tile in range(2):
            for k_tile in range(8):
                x_tl = k_tile * subtile_w
                y_tl = total_h - m_tile * subtile_h
                bo = byte_offsets[m_tile][k_tile]
                # Label sits inside the top portion of the subtile. The
                # top-left chunk is small (0.125 × 0.5 axes units) so we
                # offset slightly inward and anchor the label with a short
                # arrow back to the top-left corner.
                lbl_x = x_tl + 0.55 * subtile_w
                lbl_y = y_tl - 0.35
                tgt_x = x_tl + 0.03
                tgt_y = y_tl - 0.04
                ax.annotate(
                    f"{bo}",
                    xy=(tgt_x, tgt_y),
                    xytext=(lbl_x, lbl_y),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="#000000",
                    arrowprops=dict(arrowstyle="->", color="#000000",
                                    lw=1.0, shrinkA=1, shrinkB=2),
                    zorder=7,
                )

    # Axes — sit outside the k=/m= axis-style labels.
    ax.annotate("", xy=(total_w + 0.05, total_h + 0.65),
                xytext=(0, total_h + 0.65),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(total_w / 2.0, total_h + 0.85, "K",
            ha="center", va="bottom", fontsize=14)
    ax.annotate("", xy=(-1.0, 0), xytext=(-1.0, total_h),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.text(-1.20, total_h / 2.0, "M  (contiguous)",
            ha="center", va="center", fontsize=14, rotation=90)

    if advance_overlay:
        y_arrow = total_h + 1.45
        for k in range(7):
            x_a = (k + 0.5) * subtile_w
            x_b = (k + 1.5) * subtile_w
            ax.annotate("", xy=(x_b, y_arrow), xytext=(x_a, y_arrow),
                        arrowprops=dict(arrowstyle="->", color="#c0392b",
                                        lw=1.5))
        ax.text(total_w / 2.0, y_arrow + 0.25,
                "K-tile advance (uniform +4096 B)",
                ha="center", va="bottom", fontsize=14, color="#c0392b",
                fontweight="bold")

        ax.annotate("",
                    xy=(-1.95, total_h - 1.5 * subtile_h),
                    xytext=(-1.95, total_h - 0.5 * subtile_h),
                    arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.8))
        ax.text(-2.15, total_h - subtile_h,
                "M-tile advance\n(uniform +1024 B)",
                ha="right", va="center", fontsize=14, color="#2c7fb8",
                fontweight="bold")

        title = ("(M=128, K=128) MN-major Swizzle 128B tile — "
                 "byte offsets of each tcgen05.mma subtile from subtile (0,0)")
    else:
        title = ("(M=128, K=128) MN-major Swizzle 128B tile — "
                 "16 tcgen05.mma subtiles (2 M-tiles × 8 K-tiles)")

    title_y = total_h + (2.25 if advance_overlay else 1.75)
    ax.text(total_w / 2.0, title_y, title,
            ha="center", va="bottom", fontsize=18, fontweight="bold")

    # Sidebar legend
    legend_x = total_w + 0.6
    _draw_legend_box(ax, legend_x, total_h - 0.1, [
        {"kind": "smem_tile", "label": SMEM_TILE_LABEL},
        {"kind": "atom",      "label": ATOM_LABEL_MNMAJOR},
        {"kind": "subtile",   "label": SUBTILE_LABEL_MNMAJOR},
        {"kind": "chunk",     "label": CHUNK_LABEL,
         "fill": "#FFFFFF"},
    ], width=5.0, line_h=1.15)

    ax.set_xlim(-3.2, total_w + 6.5)
    ax.set_ylim(-0.5, total_h + (2.9 if advance_overlay else 2.6))
    ax.set_aspect("equal")
    ax.axis("off")

    name = "mnmajor_advance.png" if advance_overlay else "mnmajor_subtiles.png"
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 8: mnmajor_chunks.png
# One MN-major MMA subtile (red outer border) -> 8 x 2 = 16 pastel chunks.
# M is horizontal, K is vertical (to keep LBO/SBO arrow conventions parallel
# to kmajor_chunks). Chunks are 16B (along M-contiguous) x 8 (along K).
# ---------------------------------------------------------------------------
def make_mnmajor_chunks():
    n_m_chunks = 8
    n_k_chunks = 2
    chunk_w = 1.8
    chunk_h = 1.4

    fig, ax = plt.subplots(figsize=(18, 9))

    cmap = plt.get_cmap("tab20")
    total_w = n_m_chunks * chunk_w
    total_h = n_k_chunks * chunk_h

    for m in range(n_m_chunks):
        for k in range(n_k_chunks):
            x = m * chunk_w
            y = (n_k_chunks - 1 - k) * chunk_h
            idx = k * n_m_chunks + m
            color = cmap(idx % 20)
            r, g, b, _ = color
            light = (r * 0.40 + 0.60, g * 0.40 + 0.60, b * 0.40 + 0.60)
            rect = Rectangle((x, y), chunk_w, chunk_h,
                             facecolor=light, edgecolor=CHUNK_EDGE,
                             linewidth=CHUNK_LW, zorder=2)
            ax.add_patch(rect)
            ax.text(x + chunk_w / 2.0, y + chunk_h * 0.62,
                    f"({m}, {k})",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", zorder=3)
            ax.text(x + chunk_w / 2.0, y + chunk_h * 0.32,
                    "16B×8",
                    ha="center", va="center", fontsize=11, color="#444444",
                    zorder=3)

    # Outer MMA subtile border
    outer = Rectangle((0, 0), total_w, total_h,
                      facecolor="none", edgecolor=SUBTILE_EDGE,
                      linewidth=SUBTILE_LW, zorder=4)
    ax.add_patch(outer)

    # SBO arrow (vertical)
    arrow_x = total_w + 0.3
    ax.annotate("",
                xy=(arrow_x, total_h - 1.5 * chunk_h),
                xytext=(arrow_x, total_h - 0.5 * chunk_h),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8))
    ax.text(arrow_x + 0.15, total_h - chunk_h,
            "SBO = 2048 B\n(K-atom stride)",
            ha="left", va="center", fontsize=14, color="#c0392b",
            fontweight="bold")

    # LBO arrow (unused inside this subtile — only 1 M-atom along M)
    lbo_y = total_h + 0.30
    ax.annotate("",
                xy=(1.5 * chunk_w, lbo_y),
                xytext=(0.5 * chunk_w, lbo_y),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2,
                                linestyle="dashed"))
    ax.text(chunk_w, lbo_y + 0.22,
            "LBO = 1024 B  (M-atom stride — unused inside one subtile, "
            "only 1 M-atom along M)",
            ha="left", va="bottom", fontsize=12, color="#666666",
            style="italic")

    # M contiguous arrow at bottom
    ax.annotate("",
                xy=(total_w - 0.1, -0.45),
                xytext=(0.1, -0.45),
                arrowprops=dict(arrowstyle="->", color="#2c7fb8", lw=1.4))
    ax.text(total_w / 2.0, -0.7,
            "M contiguous  (within-atom, +16 B per chunk)",
            ha="center", va="top", fontsize=14, color="#2c7fb8",
            fontweight="bold")

    ax.text(-0.65, total_h / 2.0, "K (chunk_k = 0, 1)",
            ha="center", va="center", fontsize=14, rotation=90)

    ax.text(total_w / 2.0, -1.9,
            "addr(m, k) = start_address "
            "+ m·(within-atom M stride) + k·SBO + swizzle_XOR(m, k)",
            ha="center", va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#fff9d6", edgecolor="black", linewidth=0.6))

    ax.text(total_w / 2.0, total_h + 1.6,
            "First MMA subtile (M=64 × K=16 bf16 = 128B×16) — "
            "8×2 = 16 chunks of 16B×8 each",
            ha="center", va="bottom", fontsize=17, fontweight="bold")

    # Legend
    legend_x = total_w + 3.0
    _draw_legend_box(ax, legend_x, total_h - 0.1, [
        {"kind": "subtile", "label": SUBTILE_LABEL_MNMAJOR},
        {"kind": "chunk",   "label": CHUNK_LABEL,
         "fill": "#FDD49E"},
    ], width=4.4, line_h=1.15)

    ax.set_xlim(-1.5, total_w + 8.0)
    ax.set_ylim(-2.8, total_h + 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, "mnmajor_chunks.png")


# ---------------------------------------------------------------------------
# Figure 10: four_levels.png
# Four nested levels of granularity for the K-major Swizzle 128B
# (M=128, K=128) bf16 SMEM tile, laid out as a 2x2 grid of panels.
# ---------------------------------------------------------------------------
def make_four_levels():
    # Single shared axes so panel spacing and connecting arrows are fully
    # controllable (gridspec + equal-aspect + per-panel xlim padding made the
    # inter-panel arrows impossible to line up).
    fig, ax = plt.subplots(figsize=(22, 16))

    # --- shared tile geometry (Panels 1-3 draw the same SMEM tile) ---
    n_m_atoms = 16
    n_k_atoms = 2
    atom_w = 4.0
    atom_h = 0.6
    W = n_k_atoms * atom_w          # 8.0  tile width
    H = n_m_atoms * atom_h          # 9.6  tile height
    subtile_h = 8 * atom_h          # 4.8  (8 atoms in M)
    subtile_w = atom_w / 4.0        # 1.0  (4 subtiles per atom along K)

    BLUE  = "#2c7fb8"               # arrow / accent colour
    GREY  = "#555555"

    # --- panel origins (bottom-left corner of each tile) ---
    COL = 19.0                      # column pitch (gap = COL - W = 11.0)
    ROW = 14.5                      # row pitch
    P1 = (0.0,  ROW)                # top-left  : SMEM tile
    P2 = (COL,  ROW)                # top-right : swizzle atoms
    P3 = (0.0,  0.0)                # bot-left  : MMA subtiles
    # Panel 4 (chunks) drawn separately on the bottom-right.

    def dim_labels(x0, y0, w, h, m_txt, k_txt, size_txt, size_color="#7a3"):
        """Mark the outer box: M double-arrow on the left, K double-arrow on
        top, and a bold byte-size tag at the top-left corner."""
        # M (vertical, left)
        ax.annotate("", xy=(x0 - 0.55, y0 + h), xytext=(x0 - 0.55, y0),
                    arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.4))
        ax.text(x0 - 0.95, y0 + h / 2.0, m_txt, ha="center", va="center",
                rotation=90, fontsize=12, color=GREY, fontweight="bold")
        # K (horizontal, top)
        ax.annotate("", xy=(x0 + w, y0 + h + 0.45), xytext=(x0, y0 + h + 0.45),
                    arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.4))
        ax.text(x0 + w / 2.0, y0 + h + 0.75, k_txt, ha="center", va="bottom",
                fontsize=12, color=GREY, fontweight="bold")
        # byte-size tag
        ax.text(x0 + w / 2.0, y0 + h + 1.55, size_txt, ha="center",
                va="bottom", fontsize=13, color=size_color, fontweight="bold")

    def panel_title(cx, top_y, txt):
        ax.text(cx, top_y, txt, ha="center", va="bottom",
                fontsize=16, fontweight="bold", color="#222222")

    def caption(cx, bot_y, txt):
        ax.text(cx, bot_y, txt, ha="center", va="top", fontsize=12,
                style="italic", color="#444444")

    # ===================== Panel 1: SMEM tile =====================
    x0, y0 = P1
    ax.add_patch(Rectangle((x0, y0), W, H, facecolor="#FFF9DB",
                           edgecolor="black", linewidth=2.5, zorder=2))
    ax.text(x0 + W / 2.0, y0 + H / 2.0, "SMEM tile",
            ha="center", va="center", fontsize=18, fontweight="bold")
    dim_labels(x0, y0, W, H, "M=128", "K=128", "= 32 KB")
    panel_title(x0 + W / 2.0, y0 + H + 2.4, "1. SMEM tile")
    caption(x0 + W / 2.0, y0 - 0.9, "TMA loads the SMEM tile from GMEM")

    # ===================== Panel 2: Swizzle atoms =====================
    x0, y0 = P2
    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = x0 + k * atom_w
            y = y0 + (n_m_atoms - 1 - m) * atom_h
            ax.add_patch(Rectangle((x, y), atom_w, atom_h,
                                   facecolor=_atom_fill(m, k),
                                   edgecolor=ATOM_EDGE, linewidth=2.0,
                                   zorder=2))
    ax.add_patch(Rectangle((x0, y0), W, H, facecolor="none",
                           edgecolor="black", linewidth=2.0, zorder=3))
    dim_labels(x0, y0, W, H, "M=128", "K=128", "= 32 KB")
    panel_title(x0 + W / 2.0, y0 + H + 2.4, "2. Swizzle atoms")
    # pull out one atom to the right
    pull_w, pull_h = atom_w * 0.95, atom_h * 2.2
    pull_x, pull_y = x0 + W + 1.8, y0 + H - pull_h
    ax.add_patch(Rectangle((pull_x, pull_y), pull_w, pull_h,
                           facecolor=ATOM_FILL_A, edgecolor=ATOM_EDGE,
                           linewidth=2.0, zorder=4, clip_on=False))
    ax.annotate("", xy=(pull_x, pull_y + pull_h / 2.0),
                xytext=(x0 + W + 0.05, y0 + H - atom_h / 2.0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(pull_x + pull_w / 2.0, pull_y - 0.35,
            "one swizzle atom\nM=8 × K=64 bf16\n= 8×128B = 1024 B",
            ha="center", va="top", fontsize=11)
    caption(x0 + W / 2.0, y0 - 0.9,
            "SMEM tile = 16 × 2 = 32 swizzle atoms\n"
            "(stacking atoms is TMA-friendly)")

    # ===================== Panel 3: MMA subtiles =====================
    x0, y0 = P3
    for m in range(n_m_atoms):
        for k in range(n_k_atoms):
            x = x0 + k * atom_w
            y = y0 + (n_m_atoms - 1 - m) * atom_h
            ax.add_patch(Rectangle((x, y), atom_w, atom_h,
                                   facecolor=_atom_fill(m, k),
                                   edgecolor=ATOM_EDGE, linewidth=2.0,
                                   alpha=0.6, zorder=2))
            ax.add_patch(Rectangle((x, y), atom_w, atom_h, facecolor="none",
                                   edgecolor=ATOM_EDGE, linewidth=2.0,
                                   zorder=3))
    for m_tile in range(2):
        for k_tile in range(8):
            k_atom_idx = k_tile // 4
            k_within = k_tile % 4
            x = x0 + k_atom_idx * atom_w + k_within * subtile_w
            y = y0 + H - (m_tile + 1) * subtile_h
            ax.add_patch(Rectangle((x, y), subtile_w, subtile_h,
                                   facecolor="none", edgecolor=SUBTILE_EDGE,
                                   linewidth=SUBTILE_LW, zorder=5))
    dim_labels(x0, y0, W, H, "M=128", "K=128", "= 32 KB")
    panel_title(x0 + W / 2.0, y0 + H + 2.4, "3. MMA subtiles")
    # pull out one MMA subtile to the right (mirrors the atom pull-out),
    # vertically centred so the 3->4 arrow downstream is horizontal.
    msub_w, msub_h = 1.5, subtile_h
    msub_x = x0 + W + 1.8
    msub_y = y0 + (H - msub_h) / 2.0
    ax.add_patch(Rectangle((msub_x, msub_y), msub_w, msub_h,
                           facecolor="#F2F2F2", edgecolor=SUBTILE_EDGE,
                           linewidth=SUBTILE_LW, zorder=4))
    # arrow from the (m=0,k=0) MMA subtile (top-left of P3) to the pull-out
    ax.annotate("", xy=(msub_x, msub_y + msub_h / 2.0),
                xytext=(x0 + subtile_w, y0 + H - subtile_h / 2.0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(msub_x + msub_w / 2.0, msub_y - 0.35,
            "one MMA subtile\nM=64 × K=16 bf16\n= 64×32B",
            ha="center", va="top", fontsize=11)
    caption(x0 + W / 2.0, y0 - 0.9,
            "SMEM tile = 2 × 8 = 16 MMA subtiles\n"
            "(one tcgen05.mma per MMA subtile)")

    # ===================== Panel 4: 8x16B chunks (zoom) =====================
    n_m_chunks, n_k_chunks = 8, 2
    chunk_w, chunk_h = 2.0, 1.0
    Wc, Hc = n_k_chunks * chunk_w, n_m_chunks * chunk_h   # 4.0 x 8.0
    p4x = COL + 1.5
    p4y = (H - Hc) / 2.0                                  # vertically centre vs P3
    cmap = plt.get_cmap("tab20")
    for m in range(n_m_chunks):
        for k in range(n_k_chunks):
            x = p4x + k * chunk_w
            y = p4y + (n_m_chunks - 1 - m) * chunk_h
            idx = m * n_k_chunks + k
            r, g, b, _ = cmap(idx % 20)
            light = (r * 0.40 + 0.60, g * 0.40 + 0.60, b * 0.40 + 0.60)
            ax.add_patch(Rectangle((x, y), chunk_w, chunk_h, facecolor=light,
                                   edgecolor=CHUNK_EDGE, linewidth=CHUNK_LW,
                                   zorder=2))
            ax.text(x + chunk_w / 2.0, y + chunk_h / 2.0, f"({m},{k})",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    zorder=3)
    ax.add_patch(Rectangle((p4x, p4y), Wc, Hc, facecolor="none",
                           edgecolor=SUBTILE_EDGE, linewidth=SUBTILE_LW,
                           zorder=4))
    dim_labels(p4x, p4y, Wc, Hc, "M=64", "K=16", "= 64×32B")
    panel_title(p4x + Wc / 2.0, p4y + Hc + 2.4,
                "4. 8×16B chunks")
    caption(p4x + Wc / 2.0, p4y - 0.9,
            "MMA subtile = 8 × 2 = 16 chunks\n"
            "8×16B chunk = 16 B contiguous K × 8 rows M\n"
            "(what the descriptor describes)")

    # ===================== connecting arrows =====================
    def big_arrow(p0, p1, label, lx, ly):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.6,
                                    mutation_scale=26), zorder=6)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=12,
                color=BLUE, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=BLUE, linewidth=1.2))

    # 1 -> 2 : horizontal, top row. End short of P2 so the arrowhead clears
    #          P2's left-side M-dimension bracket (at P2[0]-0.95..-0.55).
    a_start = P1[0] + W + 0.4
    a_end   = P2[0] - 2.6
    big_arrow((a_start, P1[1] + H / 2.0),
              (a_end,   P2[1] + H / 2.0),
              "build with\nswizzle atoms\n(TMA partitioning)",
              (a_start + a_end) / 2.0, P1[1] + H / 2.0)
    # 2 -> 3 : diagonal, from P2's bottom edge down-left to P3's top edge
    s23 = (P2[0] + 1.5, P2[1] - 0.4)
    e23 = (P3[0] + W * 0.6, P3[1] + H + 0.6)
    big_arrow(s23, e23,
              "partition into\nMMA subtiles\n(partition_A)",
              (s23[0] + e23[0]) / 2.0 + 1.2, (s23[1] + e23[1]) / 2.0 + 0.4)
    # 3 -> 4 : horizontal, bottom row — from the MMA-subtile pull-out toward
    #          P4, ending short of P4's left-side M-dimension bracket.
    b_start = msub_x + msub_w + 0.3
    b_end   = p4x - 2.4
    big_arrow((b_start, msub_y + msub_h / 2.0),
              (b_end,   p4y + Hc / 2.0),
              "describe in\n8×16B chunks\n(the descriptor)",
              (b_start + b_end) / 2.0, msub_y + msub_h / 2.0)

    # ===================== title / framing =====================
    ax.set_title("Four nested levels of granularity "
                 "(K-major Swizzle 128B example)",
                 fontsize=20, fontweight="bold", pad=18)
    # right edge must include Panel 2's swizzle-atom pull-out, else it clips
    ax.set_xlim(-2.5, (COL + W + 1.8 + atom_w * 0.95) + 2.0)
    ax.set_ylim(-2.6, ROW + H + 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    out = os.path.join(OUT_DIR, "four_levels.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"writing figures to {OUT_DIR}")
    make_descriptor_bits()
    make_kmajor_tile()
    make_kmajor_subtiles(advance_overlay=False)
    make_kmajor_chunks()
    make_kmajor_subtiles(advance_overlay=True)
    make_mnmajor_tile()
    make_mnmajor_subtiles(advance_overlay=False)
    make_mnmajor_chunks()
    make_mnmajor_subtiles(advance_overlay=True)
    make_four_levels()
    print(f"done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
