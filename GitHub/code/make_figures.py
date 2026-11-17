"""Journal figures for the poultry-subset results. No titles on the figures; captions live in the manuscript."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, pandas as pd

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
T = os.path.join(R, "6_multicriteria", "without_expert_topic_weights", "TABLES")
BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#D55E00", "#222222", "#555555", "#DDDDDD"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def save(fig, n):
    fig.savefig(os.path.join(OUT, f"Figure_{n}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"Figure_{n}.png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# Figure 1
boxes = [("1  Corpus discovery and preparation", ["Boolean search in four databases", "6,256 exported; 214 with poultry terms", "204 documents after deduplication"]),
         ("2  Bibliometric characterization", ["Source, keyword, and author maps", "Coverage of poultry, PLF/digital, economic terms", "Source-overlap records"]),
         ("3  Topic modeling and alignment", ["LDA vs AviTech-Bayes on the same corpus", "K selected separately for each method", "Six topics labeled; T1 used for criterion weights"]),
         ("4  Weights and evidence mining", ["T1 terms converted into C1-C8 weights", "38 records with economic-evidence signals", "21 full texts screened; values checked manually"]),
         ("5  Decision matrix and ranking", ["Technology-by-criterion evidence matrix", "Seven technology families; TOPSIS and VIKOR", "Scores = evidence strength, not investment advice"]),
         ("6  Computational validation", ["73 weighting scenarios; 5,000 Monte Carlo vectors", "Ablation, epsilon floor, temporal holdout, bootstrap", "Full-corpus run reported for comparison"])]
pos = {0: (0, 2), 1: (1, 2), 2: (1, 1), 3: (0, 1), 4: (0, 0), 5: (1, 0)}
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.set_xlim(0, 2); ax.set_ylim(0, 3); ax.axis("off")
W, H = 0.9, 0.82
xy = {i: (0.05 + c, 0.09 + r) for i, (c, r) in pos.items()}
for i, (t, lines) in enumerate(boxes):
    x, y = xy[i]
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.03", fc="#F2F6FA", ec=MUTED, lw=0.9))
    ax.text(x + 0.03, y + H - 0.13, t, fontsize=10.5, fontweight="bold", color=INK)
    for j, l in enumerate(lines):
        ax.text(x + 0.03, y + H - 0.33 - j * 0.17, "• " + l, fontsize=7.6, color=INK)
for a, b in [((0.95, 2.5), (1.05, 2.5)), ((1.5, 2.09), (1.5, 1.91)), ((1.05, 1.5), (0.95, 1.5)), ((0.5, 1.09), (0.5, 0.91)), ((0.95, 0.5), (1.05, 0.5))]:
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
save(fig, 1)

# Figure 2
c = pd.read_excel(os.path.join(R, "5_LDA_BNMF_common_results", "05_method_comparison_metrics.xlsx")).set_index("method")
cols = [("objective_alignment", "Objective\nalignment"), ("topic_coherence", "Coherence"), ("word_exclusivity", "Exclusivity"), ("topic_balance", "Balance"),
        ("assignment_confidence", "Confidence"), ("uncertainty_quantification", "Uncertainty\nquantification"), ("composite_score", "Composite")]
x = np.arange(len(cols))
fig, ax = plt.subplots(figsize=(7.4, 3.5))
for off, m, col in ((-0.17, "AviTech-Bayes", BLUE), (0.17, "LDA", ORANGE)):
    vals = [float(c.loc[m, k]) for k, _ in cols]
    b = ax.bar(x + off, vals, 0.32, color=col, label=m)
    ax.bar_label(b, fmt="%.2f", fontsize=7, color=INK, padding=1)
ax.set_xticks(x, [l for _, l in cols]); ax.set_ylabel("Score"); ax.set_ylim(0, 1.12); grid(ax); ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
save(fig, 2)

# Figure 3
k = pd.read_excel(os.path.join(R, "3_BNMF_results", "04_bnmf_k_selection_metrics.xlsx"))
fig, ax = plt.subplots(figsize=(6.2, 3.3))
for col, lab, color, st, mk in (("composite_score", "Composite", INK, "-", "o"), ("reconstruction_norm", "Reconstruction (normalized)", BLUE, "--", "s"),
                                ("entropy_norm", "Entropy (normalized)", ORANGE, "-.", "^"), ("exclusivity_norm", "Exclusivity (normalized)", "#009E73", ":", "D")):
    ax.plot(k["k"], k[col], st, marker=mk, color=color, lw=1.8, ms=5, label=lab)
ax.set_xticks(k["k"]); ax.set_xlabel("Number of topics (K)"); ax.set_ylabel("Value"); ax.set_ylim(0, 1.05); grid(ax); ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, 3)

# Figure 4
LAB = {1: "T1 Economic and welfare evidence", 2: "T2 Broiler growth and imaging", 3: "T3 PLF adoption and readiness", 4: "T4 Thermal stress sensing",
       5: "T5 Vision-based chicken detection", 6: "T6 Laying hens and egg production"}
d = pd.read_csv(os.path.join(R, "3_BNMF_results", "04_bnmf_documents_with_topics.csv"), low_memory=False)["topic_bnmf"].value_counts()
al = pd.read_excel(os.path.join(R, "5_LDA_BNMF_common_results", "05_topic_alignment_by_method.xlsx"), sheet_name="AviTech_Bayes").set_index("topic")
order = sorted(LAB, key=lambda t: d[t])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3), sharey=True)
y = np.arange(6)
a1.barh(y, [d[t] for t in order], color=BLUE, height=0.6)
a1.set_yticks(y, [LAB[t] for t in order]); a1.set_xlabel("Documents"); grid(a1, "x")
for i, t in enumerate(order):
    a1.text(d[t] + 0.8, i, str(d[t]), va="center", fontsize=8)
a2.barh(y, [al.loc[t, "objective_alignment_score"] for t in order], color=[ORANGE if t == 1 else "#9AA5B1" for t in order], height=0.6)
a2.set_xlabel("Aggregate alignment A(k)"); a2.set_xlim(0, 1.12); grid(a2, "x")
for i, t in enumerate(order):
    a2.text(al.loc[t, "objective_alignment_score"] + 0.02, i, f"{al.loc[t, 'objective_alignment_score']:.2f}", va="center", fontsize=8)
save(fig, 4)

# Figure 5
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ts = sorted(LAB); x = np.arange(6)
for off, col, lab, color, hatch in ((-0.26, "poultry_hits", "Poultry", BLUE, ""), (0, "technology_hits", "Technology/PLF", "#9AA5B1", ""), (0.26, "economic_hits", "Economic feasibility", ORANGE, "")):
    b = ax.bar(x + off, [al.loc[t, col] for t in ts], 0.24, color=color, label=lab)
    ax.bar_label(b, fontsize=7, color=INK, padding=1)
ax.set_xticks(x, [f"T{t}" for t in ts]); ax.set_ylabel("Top-15 terms matched"); grid(ax); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
save(fig, 5)

# Figure 6
r = pd.read_excel(os.path.join(T, "10_no_expert_topsis_vikor_technology_ranking.xlsx")).sort_values("final_no_expert_rank", ascending=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3), sharey=True)
y = np.arange(len(r))
a1.barh(y, r["topsis_score"], color=BLUE, height=0.6); a1.set_yticks(y, r["technology_label"]); a1.set_xlabel("TOPSIS closeness (higher is better)"); a1.set_xlim(0, 1); grid(a1, "x")
a2.barh(y, r["vikor_q_compromise"], color=ORANGE, height=0.6); a2.set_xlabel("VIKOR Q (lower is better)"); a2.set_xlim(0, 1); grid(a2, "x")
for i, (s, q) in enumerate(zip(r["topsis_score"], r["vikor_q_compromise"])):
    a1.text(s + 0.015, i, f"{s:.3f}", va="center", fontsize=8); a2.text(q + 0.015, i, f"{q:.3f}", va="center", fontsize=8)
save(fig, 6)

# Figure 7
m = pd.read_excel(os.path.join(T, "07_no_expert_technology_decision_matrix.xlsx"))
cc = sorted(c_ for c_ in m.columns if c_[:2] in {f"C{i}" for i in range(1, 9)})
m = m.set_index("technology_label").loc[list(r["technology_label"][::-1])]
fig, ax = plt.subplots(figsize=(7, 3.2))
im = ax.imshow(m[cc].values.astype(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(8), [c_[:2] for c_ in cc]); ax.set_yticks(range(len(m)), m.index); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(len(m)):
    for j in range(8):
        v = float(m[cc].values[i, j]); ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.6 else INK)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Evidence strength")
save(fig, 7)

# Figure 8
p = pd.read_excel(os.path.join(T, "22_monte_carlo_rank_probability_matrix.xlsx"))
label = [k_ for k_ in p.columns if "label" in k_.lower()][0]
rank_cols = [k_ for k_ in p.columns if k_ not in (label, "technology")]
p = p.set_index(label).loc[list(r["technology_label"][::-1])]
fig, ax = plt.subplots(figsize=(7, 3.2))
im = ax.imshow(p[rank_cols].values.astype(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(rank_cols)), [f"R{i + 1}" for i in range(len(rank_cols))]); ax.set_yticks(range(len(p)), p.index); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(len(p)):
    for j in range(len(rank_cols)):
        v = float(p[rank_cols].values[i, j]); ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.6 else INK)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Rank probability")
save(fig, 8)
print("figures", sorted(os.listdir(OUT))[:4], list(p.columns)[:3])
