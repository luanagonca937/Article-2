"""Graphical abstract for Article 2 (JCLP), version 2 (more visual).

All numbers come from the V11.20 manuscript and the archived result tables:
204 documents of 6,256 exported; composite score LDA 0.40 vs AviTech-Bayes 0.60;
criterion weights (C4 productivity/welfare 0.566, C2 0.151, C1 0.111, C6 0.062, four at 0.0275);
final rank 1 Sensors, 2 ML/AI, 3 Automation; Monte Carlo first place ML/AI 43.1%, sensors 34.0%.
Elsevier: landscape, readable at 13 cm wide. Drawn at 13 x 5 cm, 600 dpi.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Wedge, Circle, Rectangle
from PIL import Image

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
CM = 1 / 2.54
plt.rcParams.update({"font.family": "DejaVu Sans"})

INK, MUTED, GREY = "#17212b", "#55606b", "#c9d0d6"
BLUE, GREEN, AMBER, PURPLE = "#2a6f97", "#2d8a57", "#d08a1a", "#6a4c93"
LIGHT_B, LIGHT_G = "#e3eef6", "#e2f2e8"

fig = plt.figure(figsize=(13 * CM, 5 * CM), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 130)
ax.set_ylim(0, 50)
ax.set_aspect("equal")
ax.axis("off")


def stage_header(x, n, text, color):
    ax.add_patch(Circle((x - 10.6, 40.6), 2.1, fc=color, ec="none"))
    ax.text(x - 10.6, 40.5, str(n), ha="center", va="center", fontsize=6.8, fontweight="bold", color="white")
    ax.text(x - 7.9, 40.6, text, ha="left", va="center", fontsize=7.2, fontweight="bold", color=color)


def chevron(x):
    ax.add_patch(Polygon([(x, 24), (x + 2.2, 21.5), (x, 19), (x + 1.0, 21.5)], closed=True,
                         fc="#9aa5af", ec="none"))


# title band
ax.add_patch(FancyBboxPatch((1, 44.6), 128, 4.6, boxstyle="round,pad=0,rounding_size=2.3",
                            fc="#1f3b4d", ec="none"))
ax.text(65, 46.9, "Which digital poultry technologies have the strongest economic-feasibility evidence?",
        ha="center", va="center", fontsize=7.0, fontweight="bold", color="white")

# 1 corpus funnel
cx = 16
stage_header(cx, 1, "Literature", BLUE)
chips = [("SD", "#4f8fba", 5.0), ("Scopus", "#3c7ea9", 8.2), ("WoS", "#2e6d96", 6.0), ("IEEE", "#245d82", 6.4)]
gap = 0.6
total = sum(c[2] for c in chips) + gap * (len(chips) - 1)
x = cx - total / 2
for lab, col, wd in chips:
    ax.add_patch(FancyBboxPatch((x, 33.2), wd, 3.6, boxstyle="round,pad=0,rounding_size=1", fc=col, ec="none"))
    ax.text(x + wd / 2, 35.0, lab, ha="center", va="center", fontsize=5.4, color="white", fontweight="bold")
    x += wd + gap
ax.add_patch(Polygon([(cx - 13.6, 31.6), (cx + 13.6, 31.6), (cx + 4.2, 17.5), (cx - 4.2, 17.5)],
                     closed=True, fc=LIGHT_B, ec=BLUE, lw=0.7))
ax.text(cx, 28.2, "6,256 records", ha="center", va="center", fontsize=6.8, color=INK)
ax.text(cx, 22.6, "poultry\nfilter", ha="center", va="center", fontsize=6.0, color=MUTED, linespacing=1.0)
ax.add_patch(FancyBboxPatch((cx - 12.5, 9.6), 25.0, 6.0, boxstyle="round,pad=0,rounding_size=1.6",
                            fc=BLUE, ec="none"))
ax.text(cx, 12.6, "204 documents", ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
chevron(31.5)

# 2 topic model comparison
cx = 48
stage_header(cx, 2, "Topic models", GREEN)
ax.text(cx - 13.2, 35.4, "Composite score", ha="left", va="center", fontsize=6.0, color=MUTED, style="italic")
for y, lab, val, col, bold in [(27.8, "LDA", 0.40, GREY, False), (20.8, "AviTech-Bayes", 0.60, GREEN, True)]:
    ax.text(cx - 13.2, y + 3.3, lab, ha="left", va="center", fontsize=6.2,
            color=GREEN if bold else INK, fontweight="bold" if bold else "normal")
    ax.add_patch(Rectangle((cx - 13.2, y - 1.6), 26.4, 3.2, fc="#eef1f3", ec="none"))
    ax.add_patch(Rectangle((cx - 13.2, y - 1.6), 26.4 * val, 3.2, fc=col, ec="none"))
    ax.text(cx + 13.2, y, f"{val:.2f}", ha="right", va="center", fontsize=6.6,
            color=GREEN if bold else MUTED, fontweight="bold")
ax.add_patch(FancyBboxPatch((cx - 13.2, 7.4), 26.4, 9.0, boxstyle="round,pad=0,rounding_size=1.6",
                            fc=LIGHT_G, ec=GREEN, lw=0.6))
ax.text(cx, 11.9, "One topic links\npoultry, technology\nand economics", ha="center", va="center",
        fontsize=5.8, color=INK, linespacing=1.15)
chevron(63.5)

# 3 criteria donut
cx, cy = 80, 24.5
stage_header(cx, 3, "8 criteria", AMBER)
weights = [(0.5656, AMBER), (0.1512, "#e7b768"), (0.1114, "#efcd94"), (0.0617, "#f4ddb6"),
           (0.0275, GREY), (0.0275, "#d7dce0"), (0.0275, GREY), (0.0275, "#d7dce0")]
start = 90
for w, col in weights:
    ang = w * 360
    ax.add_patch(Wedge((cx, cy), 9.6, start - ang, start, width=3.6, fc=col, ec="white", lw=0.4))
    start -= ang
ax.text(cx, cy + 1.3, "56.6%", ha="center", va="center", fontsize=7.2, fontweight="bold", color=AMBER)
ax.text(cx, cy - 2.4, "weight", ha="center", va="center", fontsize=5.6, color=MUTED)
ax.text(cx, 12.4, "Productivity and welfare\nweigh most", ha="center", va="center", fontsize=6.0,
        color=INK, linespacing=1.1)
ax.text(cx, 7.9, "TOPSIS + VIKOR", ha="center", va="center", fontsize=6.4, fontweight="bold", color=AMBER)
chevron(95.5)

# 4 podium
cx = 112
stage_header(cx, 4, "Leading group", PURPLE)
base = 16.5
for x, hgt, rank, lab, col in [(cx - 10.0, 9.0, 2, "ML / AI", "#8a6fb3"),
                               (cx, 13.0, 1, "Sensors", PURPLE),
                               (cx + 10.0, 6.0, 3, "Automation", "#a891c9")]:
    ax.add_patch(FancyBboxPatch((x - 4.2, base), 8.4, hgt, boxstyle="round,pad=0,rounding_size=0.9",
                                fc=col, ec="none"))
    ax.text(x, base + hgt - 2.4, str(rank), ha="center", va="center", fontsize=8.0, fontweight="bold", color="white")
    ax.text(x, base + hgt + 2.0, lab, ha="center", va="center", fontsize=5.5, fontweight="bold", color=INK)
ax.plot([cx - 14.6, cx + 14.6], [base, base], color="#8f98a1", lw=0.6)
ax.text(cx, 12.2, "Monte Carlo, first place:", ha="center", va="center", fontsize=5.7, color=MUTED)
ax.text(cx, 7.9, "ML/AI 43.1% · Sensors 34.0%", ha="center", va="center", fontsize=5.6,
        fontweight="bold", color=PURPLE)

ax.text(65, 2.4, "A screening of the literature, not an investment recommendation: "
        "it shows where verified economic analysis should focus.",
        ha="center", va="center", fontsize=5.9, style="italic", color=MUTED)

fig.canvas.draw()
r = fig.canvas.get_renderer()
for t in ax.texts:
    bb = t.get_window_extent(r)
    if bb.x0 < 0 or bb.x1 > fig.bbox.width or bb.y0 < 0 or bb.y1 > fig.bbox.height:
        print("FORA:", t.get_text())

fig.savefig(out / "Graphical_Abstract.pdf")
fig.savefig(out / "Graphical_Abstract.png", dpi=600)
plt.close(fig)
img = Image.open(out / "Graphical_Abstract.png").convert("RGB")
img.save(out / "Graphical_Abstract.tif", compression="tiff_lzw", dpi=(600, 600))
print(img.size)
