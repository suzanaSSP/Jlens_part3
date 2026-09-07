"""
Pilot timeline for a single heldout question: sliding-window JS divergence
to every category's ground truth, across the thought -> answer segments.

A single token position's distribution is nearly a delta spike (the model's
one confident next-token guess), while a ground truth is an average over
hundreds of positions (diffuse). Comparing spike-to-diffuse swamps any real
category signal almost regardless of category (validated empirically before
building this). Pooling WINDOW_SIZE consecutive positions before comparing
brings the comparison back to "diffuse vs diffuse" -- confirmed on this same
question: a single position's top-1 prob was 0.947 over 10 nonzero tokens; a
10-position window's top-1 prob dropped to 0.164 over 111 nonzero tokens,
much closer to the ground truth's own shape (top-1 ~0.17 over thousands of
tokens). The window slides one position at a time so the timeline keeps
roughly one point per token, each representing its surrounding neighborhood
rather than itself alone.

Ground truths are computed fresh here for BOTH segments (thought and answer)
from the train_ids in ground_truth_pool.json. build_ground_truths.py only
persisted thought-segment ground truths so far (SEGMENT="thought" in
bootstrap_ground_truth.py) -- an answer-segment comparison against a
thought-only ground truth would confound "different segment" with "different
category", so answer-segment ground truths are built here too, matching the
same train_ids/layer, just not written to data/ground_truths/ yet.
"""
import json

import matplotlib.pyplot as plt

from scripts.bootstrap_ground_truth import (
    LAYER,
    average_distribution,
    js_divergence,
    load_output,
    position_distribution,
    question_distribution,
)

QUESTION_ID = "588"
HIGHLIGHT_CATEGORY = "Religion & Theology"
WINDOW_SIZE = 10

with open("ground_truths/ground_truth_pool.json") as f:
    POOL = json.load(f)


def build_ground_truths(segment):
    gts = {}
    for category, ids in POOL.items():
        dists = [question_distribution(load_output(qid), segment=segment) for qid in ids["train_ids"]]
        dists = [d for d in dists if d is not None]
        gts[category] = average_distribution(dists)
    return gts


def windowed_distributions(positions):
    """(center_norm_pos, pooled_distribution) for each WINDOW_SIZE-position
    window, sliding by 1. Pooling here uses the same average_distribution()
    the ground truths themselves are built from, so both sides of the later
    JS comparison are the same kind of thing: an average over positions."""
    n = len(positions)
    if n < WINDOW_SIZE:
        return []
    dists = [position_distribution(p) for p in positions]
    windows = []
    for start in range(n - WINDOW_SIZE + 1):
        pooled = average_distribution(dists[start:start + WINDOW_SIZE])
        center = start + (WINDOW_SIZE - 1) / 2.0
        norm_pos = 100.0 * center / (n - 1) if n > 1 else 0.0
        windows.append((norm_pos, pooled))
    return windows


def main():
    ground_truths = {seg: build_ground_truths(seg) for seg in ("thought", "answer")}

    output = load_output(QUESTION_ID)
    layer_data = next(l for l in output["jacobian_lens_per_layer"] if l["layer"] == LAYER)

    records = []
    for segment in ("thought", "answer"):
        positions = [t for t in layer_data["tokens"] if t["segment"] == segment]
        for norm_pos, dist in windowed_distributions(positions):
            for category, gt in ground_truths[segment].items():
                records.append({
                    "segment": segment,
                    "norm_pos": norm_pos,
                    "category": category,
                    "similarity": 1.0 - js_divergence(dist, gt),
                })

    plot(records, y_max=1.0, output_path="timeline_pilot_588.png")

    data_max = max(r["similarity"] for r in records)
    plot(records, y_max=data_max * 1.05, output_path="timeline_pilot_588_zoomed.png")


# 8 validated categorical hues (light mode, references/palette.md). Only 8
# hues are CVD-safe at this order -- with 10 categories, 2 must reuse a hue.
# Reused pairs get a dashed linestyle as a secondary encoding so they stay
# distinguishable (per the skill's "never cycle hues past 8 without secondary
# encoding" rule) rather than inventing two unvalidated extra colors.
CATEGORY_STYLES = {
    "Religion & Theology":                 ("#2a78d6", "-"),   # slot 1 blue -- highlight
    "Law, Politics, Government & History": ("#eb6834", "-"),   # slot 2 orange
    "Technology & Definitions":            ("#1baf7a", "-"),   # slot 3 aqua
    "Applied Ethics & Moral Dilemmas":     ("#eda100", "-"),   # slot 4 yellow
    "Marriage & Romantic Partnerships":    ("#e87ba4", "-"),   # slot 5 magenta
    "Human Nature & Philosophy":           ("#008300", "-"),   # slot 6 green
    "Science, Physics & Math":             ("#4a3aa7", "-"),   # slot 7 violet
    "Depression, Addiction & Feeling Lost": ("#e34948", "-"),  # slot 8 red
    "Family Duty & Obligations":           ("#eb6834", "--"),  # reuses orange, dashed
    "Grief, Loss & Death":                 ("#1baf7a", "--"),  # reuses aqua, dashed
}


def plot(records, y_max, output_path):
    categories = sorted({r["category"] for r in records})

    surface = "#fcfcfb"
    text_primary = "#0b0b0b"
    text_secondary = "#52514e"
    gridline = "#e1e0d9"
    baseline = "#c3c2b7"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, facecolor=surface)

    for ax, segment, title in zip(axes, ("thought", "answer"), ("Thought", "Answer")):
        ax.set_facecolor(surface)
        seg_records = [r for r in records if r["segment"] == segment]

        for category in categories:
            if category == HIGHLIGHT_CATEGORY:
                continue
            color, linestyle = CATEGORY_STYLES[category]
            xs = [r["norm_pos"] for r in seg_records if r["category"] == category]
            ys = [r["similarity"] for r in seg_records if r["category"] == category]
            ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.8, zorder=2)

        color, linestyle = CATEGORY_STYLES[HIGHLIGHT_CATEGORY]
        xs = [r["norm_pos"] for r in seg_records if r["category"] == HIGHLIGHT_CATEGORY]
        ys = [r["similarity"] for r in seg_records if r["category"] == HIGHLIGHT_CATEGORY]
        ax.plot(xs, ys, color=color, linewidth=3, zorder=3)

        ax.set_title(title, color=text_primary, fontsize=11, loc="left")
        ax.set_xlabel("Normalized token position (%)", color=text_secondary, fontsize=9)
        ax.grid(True, color=gridline, linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(baseline)
        ax.tick_params(colors=text_secondary, labelsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel("Similarity to category (1 − JS divergence)", color=text_secondary, fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=CATEGORY_STYLES[c][0], linestyle=CATEGORY_STYLES[c][1],
                   linewidth=3 if c == HIGHLIGHT_CATEGORY else 1.3, label=c)
        for c in categories
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3,
               frameon=False, fontsize=8, labelcolor=text_primary)
    fig.suptitle(
        f"Question {QUESTION_ID} ({HIGHLIGHT_CATEGORY}) — similarity to category ground truths "
        f"(window={WINDOW_SIZE}, y max={y_max:.2f})",
        color=text_primary, fontsize=12, y=1.16,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=surface, bbox_inches="tight")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
