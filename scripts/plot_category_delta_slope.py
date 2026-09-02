"""
Slopegraph: x = {Thought, Answer}, y = category delta (mean similarity to
category minus avg. similarity to baseline categories, from
category_delta_summary.csv). One line per category, color-coded, so slope
direction/steepness shows thought-to-answer growth and the y-axis lets you
compare magnitudes across categories directly.
"""
import pandas as pd
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"

# Fresh 10-color qualitative palette (matplotlib tab10) -- the project's
# CATEGORY_COLORS reuses 2 hues (Family Duty/Grief), which is fine when
# they're folded into "Other" elsewhere but would clash here since all 10
# categories are on screen simultaneously.
CATEGORY_COLORS = {
    "Religion & Theology":                  "#1f77b4",
    "Law, Politics, Government & History":  "#ff7f0e",
    "Technology & Definitions":              "#2ca02c",
    "Applied Ethics & Moral Dilemmas":       "#d62728",
    "Marriage & Romantic Partnerships":      "#9467bd",
    "Human Nature & Philosophy":             "#8c564b",
    "Science, Physics & Math":               "#e377c2",
    "Depression, Addiction & Feeling Lost":  "#7f7f7f",
    "Family Duty & Obligations":             "#bcbd22",
    "Grief, Loss & Death":                   "#17becf",
}

df = pd.read_csv("category_delta_summary.csv")
piv = df.pivot(index="compared_category", columns="segment", values="mean")

fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

x = [0, 1]
for category, row in piv.iterrows():
    color = CATEGORY_COLORS[category]
    y = [row["thought"], row["answer"]]
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
             label=category, zorder=2)

ax.axhline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(["Thought", "Answer"], fontsize=10, color=TEXT_PRIMARY)
ax.set_xlim(-0.2, 1.2)
ax.set_ylabel("Category delta  (similarity to category − avg. similarity to baseline categories)",
              color=TEXT_SECONDARY, fontsize=9)
ax.set_title("Category delta: thought → answer (benchmark pilot, n=10)",
             color=TEXT_PRIMARY, fontsize=11.5, loc="left")
ax.grid(True, color=GRIDLINE, linewidth=0.8, axis="y", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(GRIDLINE)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, loc="center left",
          bbox_to_anchor=(1.02, 0.5))

fig.tight_layout()
fig.savefig("category_delta_slope.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("Wrote category_delta_slope.png")
