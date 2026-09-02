"""
Dumbbell chart: for each category, thought-delta and answer-delta (from
category_delta_summary.csv) as two points on the same row, connected by a
line -- so the line's direction/length shows the thought-to-answer growth
in that category's baseline-calibrated signal.
"""
import pandas as pd
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
THOUGHT_COLOR = "#2a78d6"
ANSWER_COLOR = "#eb6834"

df = pd.read_csv("category_delta_summary.csv")
piv = df.pivot(index="compared_category", columns="segment", values="mean").reset_index()
piv["growth"] = piv["answer"] - piv["thought"]
piv = piv.sort_values("growth")

fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

y_labels = piv["compared_category"].tolist()
for i, row in enumerate(piv.itertuples()):
    ax.plot([row.thought, row.answer], [i, i], color=GRIDLINE, linewidth=1.5, zorder=1)
    ax.scatter([row.thought], [i], color=THOUGHT_COLOR, s=70, zorder=2,
               label="Thought" if i == 0 else None)
    ax.scatter([row.answer], [i], color=ANSWER_COLOR, s=70, zorder=2,
               label="Answer" if i == 0 else None)

ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=0)
ax.set_yticks(range(len(piv)))
ax.set_yticklabels(y_labels, fontsize=9, color=TEXT_SECONDARY)
ax.set_xlabel("Category delta  (similarity to category − avg. similarity to baseline categories)",
              color=TEXT_SECONDARY, fontsize=9)
ax.set_title("Thought → answer: category delta growth (benchmark pilot, n=10)",
             color=TEXT_PRIMARY, fontsize=11.5, loc="left")
ax.grid(True, color=GRIDLINE, linewidth=0.8, axis="x", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(GRIDLINE)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, loc="lower right")

fig.tight_layout()
fig.savefig("category_delta_growth.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("Wrote category_delta_growth.png")
