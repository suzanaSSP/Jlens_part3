"""
Held-out validation: does a held-out question land closest to its own
category's ground truth, or to another category's?

For every category's held-out questions, pool each whole question into a
single distribution -- the same question_distribution() ground truths are
built from, not the sliding-window pooling in timeline_pilot.py, since this
asks "does this whole question belong near its category," not "how does
similarity evolve within it." Compare against every category's ground truth
(not just its own) for both the thought and answer segments.

Answer-segment ground truths aren't persisted anywhere (see
timeline_pilot.py's docstring) so this script builds both segments' ground
truths fresh from train_ids, matching timeline_pilot.py's approach, rather
than reusing the thought-only files under ground_truths/.

Two categories have thin held-out sets (Grief, Loss & Death: 7; Family Duty &
Obligations: 10). They're scored the same way but reported separately so
their noise doesn't dilute the main per-category accuracy numbers or
populate the main confusion heatmap.
"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from scripts.bootstrap_ground_truth import (
    average_distribution,
    js_divergence,
    load_output,
    question_distribution,
)

JSPACE_DIR = "../jspace_prompts"

with open("ground_truth_pool.json") as f:
    POOL = json.load(f)

# Fixed order, matching timeline_pilot.py's CATEGORY_STYLES slots.
CATEGORY_ORDER = [
    "Religion & Theology",
    "Law, Politics, Government & History",
    "Technology & Definitions",
    "Applied Ethics & Moral Dilemmas",
    "Marriage & Romantic Partnerships",
    "Human Nature & Philosophy",
    "Science, Physics & Math",
    "Depression, Addiction & Feeling Lost",
    "Family Duty & Obligations",
    "Grief, Loss & Death",
]
SMALL_CATEGORIES = {"Family Duty & Obligations", "Grief, Loss & Death"}

# Sequential blue ramp, steps 100->700 (references/palette.md).
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def build_ground_truths(segment):
    gts = {}
    for category, ids in POOL.items():
        dists = [question_distribution(load_output(qid), segment=segment) for qid in ids["train_ids"]]
        dists = [d for d in dists if d is not None]
        gts[category] = average_distribution(dists)
    return gts


def score_heldout(ground_truths):
    """One row per (segment, true_category, qid, compared_category).
    Skips held-out questions that haven't been run through get_jlens.py yet
    (most categories still have none -- see heldout_validation.py's
    docstring) rather than failing the whole run."""
    records = []
    for segment, gts in ground_truths.items():
        for true_category, ids in POOL.items():
            for qid in ids["heldout_ids"]:
                if not os.path.exists(f"{JSPACE_DIR}/jlens_output_prompt_{qid}.json.gz"):
                    continue
                dist = question_distribution(load_output(qid), segment=segment)
                if dist is None:
                    continue
                js_by_category = {cat: js_divergence(dist, gt) for cat, gt in gts.items()}
                ranked = sorted(js_by_category, key=js_by_category.get)
                predicted = ranked[0]
                true_rank = ranked.index(true_category) + 1
                for compared_category, js in js_by_category.items():
                    records.append({
                        "segment": segment,
                        "true_category": true_category,
                        "qid": qid,
                        "compared_category": compared_category,
                        "js_divergence": js,
                        "similarity": 1.0 - js,
                        "predicted_category": predicted,
                        "true_rank": true_rank,
                        "is_small_category": true_category in SMALL_CATEGORIES,
                    })
    return pd.DataFrame(records)


def summarize(df):
    own = df[df["compared_category"] == df["true_category"]].copy()
    own["correct"] = own["true_rank"] == 1
    summary = (
        own.groupby(["segment", "true_category"])
        .agg(n_scored=("qid", "nunique"), accuracy=("correct", "mean"), mean_rank=("true_rank", "mean"))
        .reset_index()
        .rename(columns={"true_category": "category"})
    )
    summary["n_heldout"] = summary["category"].map(lambda c: len(POOL[c]["heldout_ids"]))
    summary["is_small_category"] = summary["category"].isin(SMALL_CATEGORIES)
    return summary.sort_values(["segment", "is_small_category", "category"])


def confusion_matrix(df, segment, rows):
    seg = df[(df["segment"] == segment) & (df["compared_category"] == df["true_category"])]
    seg = seg.drop_duplicates(subset=["qid", "true_category"])
    mat = pd.crosstab(seg["true_category"], seg["predicted_category"], normalize="index")
    return mat.reindex(index=rows, columns=CATEGORY_ORDER, fill_value=0.0)


def plot_confusion(mat, segment, output_path):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

    surface = "#fcfcfb"
    text_primary = "#0b0b0b"
    text_secondary = "#52514e"

    fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=surface)
    ax.set_facecolor(surface)

    data = mat.values
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8, color=text_secondary)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8, color=text_secondary)

    for i, true_cat in enumerate(mat.index):
        for j, pred_cat in enumerate(mat.columns):
            val = data[i, j]
            text_color = "#ffffff" if val > 0.5 else text_primary
            weight = "bold" if true_cat == pred_cat else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5,
                     color=text_color, fontweight=weight)
            if true_cat == pred_cat:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor=text_primary, linewidth=1.5))

    ax.set_xlabel("Predicted category (closest ground truth)", color=text_secondary, fontsize=9)
    ax.set_ylabel("True category", color=text_secondary, fontsize=9)
    ax.set_title(f"Held-out validation — {segment.capitalize()} segment "
                 f"(diagonal = held-out question's own category)",
                 color=text_primary, fontsize=11.5, loc="left")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of held-out questions", color=text_secondary, fontsize=8)
    cbar.ax.tick_params(colors=text_secondary, labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=surface, bbox_inches="tight")
    print(f"Wrote {output_path}")


def main():
    ground_truths = {seg: build_ground_truths(seg) for seg in ("thought", "answer")}
    df = score_heldout(ground_truths)
    df.to_csv("heldout_validation.csv", index=False)
    print("Wrote heldout_validation.csv")

    summary = summarize(df)
    summary.to_csv("heldout_validation_summary.csv", index=False)
    print("Wrote heldout_validation_summary.csv\n")

    print("=== Main categories ===")
    print(summary[~summary["is_small_category"]].to_string(index=False))
    print("\n=== Small categories (low-confidence: thin held-out sets) ===")
    print(summary[summary["is_small_category"]].to_string(index=False))

    for segment in ("thought", "answer"):
        seg_df = df[df["segment"] == segment]
        # Only plot rows for categories that actually have scored held-out
        # data this round -- an all-zero row would otherwise look like a
        # real "never predicted correctly" result instead of "no data yet".
        scored_categories = set(seg_df["true_category"].unique())
        main_rows = [c for c in CATEGORY_ORDER if c not in SMALL_CATEGORIES and c in scored_categories]
        if not main_rows:
            print(f"\nNo main-category held-out data scored for segment={segment!r}; skipping heatmap.")
            continue
        mat = confusion_matrix(df, segment, main_rows)
        plot_confusion(mat, segment, f"heldout_confusion_{segment}.png")


if __name__ == "__main__":
    main()
