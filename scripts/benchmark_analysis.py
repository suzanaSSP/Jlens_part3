"""
Full analysis for the benchmark pilot batch: how much does each category --
especially Religion & Theology -- show up in the model's internal reasoning
when answering non-religious benchmark questions (a stratified sample of 10,
one per labeled benchmark category, from data/benchmark_pilot_ids.json).

Two complementary views, both built on the same JS-divergence comparison
used throughout this project (question distribution vs. each category's
ground truth):

1. Whole-question summary: pool each question into one distribution (thought
   and answer separately, via question_distribution -- not windowed) and
   compare to all 10 ground truths. Reports a calibrated "religion signal"
   per question: similarity-to-Religion minus the average similarity to
   BASELINE_CATEGORIES (categories with no plausible religious content) --
   a baseline so generic shared vocabulary isn't mistaken for
   religion-specific signal. See conversation with the user for why a raw
   similarity number alone is misleading (every category shares some
   generic vocabulary after stopword filtering).

2. Streamgraph timeline: sliding-window pooling (same WINDOW_SIZE/approach
   as timeline_pilot.py) across normalized token position, stacked with
   matplotlib's baseline='wiggle' using RAW similarity values -- NOT
   renormalized to sum to 1, so the stream's total thickness still reflects
   how much total categorical signal is present at each point, not just
   relative share (chosen over normalizing after discussion with the user --
   normalizing would make a point of generic, non-category-specific language
   look just as "confident" as a point with one sharply distinctive
   category). BASELINE_CATEGORIES are folded into a single "Other" band here
   too, both to stay under the palette's 8-hue-before-fallback cap (7
   explicit + Other = 8) and to reuse the same three categories in the same
   role everywhere in this script. "Other"'s value is the SUM of those three
   categories' raw similarities, so it will often be the thickest band
   simply because it's adding three series where every other band is one --
   read it as "combined baseline load," not "a single category."

Ground truths are built fresh here for both segments (see timeline_pilot.py
for why: only thought-segment ground truths are persisted under
data/ground_truths/).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.bootstrap_ground_truth import (
    LAYER,
    average_distribution,
    js_divergence,
    load_output,
    position_distribution,
    question_distribution,
)

WINDOW_SIZE = 10
AGG_GRID = list(range(0, 101, 5))  # common grid for averaging across questions

with open("ground_truths/ground_truth_pool.json") as f:
    POOL = json.load(f)
with open("benchmark_pilot_ids.json") as f:
    PILOT = json.load(f)

# Fixed order/colors, matching timeline_pilot.py's CATEGORY_STYLES slots.
CATEGORY_COLORS = {
    "Religion & Theology":                 "#2a78d6",  # slot 1 blue -- highlight
    "Law, Politics, Government & History": "#eb6834",  # slot 2 orange
    "Technology & Definitions":            "#1baf7a",  # slot 3 aqua
    "Applied Ethics & Moral Dilemmas":     "#eda100",  # slot 4 yellow
    "Marriage & Romantic Partnerships":    "#e87ba4",  # slot 5 magenta
    "Human Nature & Philosophy":           "#008300",  # slot 6 green
    "Science, Physics & Math":             "#4a3aa7",  # slot 7 violet
    "Depression, Addiction & Feeling Lost": "#e34948", # slot 8 red
    "Family Duty & Obligations":           "#eb6834",  # reuses orange (line-only elsewhere)
    "Grief, Loss & Death":                 "#1baf7a",  # reuses aqua (line-only elsewhere)
}

# Categories with no plausible religious content: used both as the
# calibration baseline for the delta metric and as the streamgraph's
# folded "Other" band (same three categories, same role, everywhere).
BASELINE_CATEGORIES = [
    "Law, Politics, Government & History",
    "Technology & Definitions",
    "Science, Physics & Math",
]
# Fixed stacking order for the streamgraph: Religion first (top of stack),
# then the other 6 explicit categories, "Other" last.
STREAM_CATEGORIES = [
    "Religion & Theology",
    "Applied Ethics & Moral Dilemmas",
    "Marriage & Romantic Partnerships",
    "Human Nature & Philosophy",
    "Depression, Addiction & Feeling Lost",
    "Family Duty & Obligations",
    "Grief, Loss & Death",
]
OTHER_COLOR = "#9c9b93"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"


def build_ground_truths(segment):
    gts = {}
    for category, ids in POOL.items():
        dists = [question_distribution(load_output(qid), segment=segment) for qid in ids["train_ids"]]
        dists = [d for d in dists if d is not None]
        gts[category] = average_distribution(dists)
    return gts


def windowed_similarity_curves(qid, segment, ground_truths):
    """(xs, {category: [similarity...]}) for one question/segment, sliding
    WINDOW_SIZE positions at a time -- same pooling as timeline_pilot.py."""
    output = load_output(qid)
    layer_data = next(l for l in output["jacobian_lens_per_layer"] if l["layer"] == LAYER)
    positions = [t for t in layer_data["tokens"] if t["segment"] == segment]
    n = len(positions)
    if n < WINDOW_SIZE:
        return [], {}

    dists = [position_distribution(p) for p in positions]
    xs = []
    curves = {cat: [] for cat in ground_truths}
    for start in range(n - WINDOW_SIZE + 1):
        pooled = average_distribution(dists[start:start + WINDOW_SIZE])
        center = start + (WINDOW_SIZE - 1) / 2.0
        norm_pos = 100.0 * center / (n - 1) if n > 1 else 0.0
        xs.append(norm_pos)
        for cat, gt in ground_truths.items():
            curves[cat].append(1.0 - js_divergence(pooled, gt))
    return xs, curves


def stream_bands(curves):
    """{band_name: [values]} for the 7 explicit categories + summed 'Other'."""
    bands = {cat: curves[cat] for cat in STREAM_CATEGORIES}
    other = [sum(vals) for vals in zip(*(curves[c] for c in BASELINE_CATEGORIES))]
    bands["Other"] = other
    return bands


def plot_streamgraph(xs_by_segment, bands_by_segment, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, facecolor=SURFACE)

    for ax, segment, seg_title in zip(axes, ("thought", "answer"), ("Thought", "Answer")):
        ax.set_facecolor(SURFACE)
        xs = xs_by_segment[segment]
        bands = bands_by_segment[segment]
        if not xs:
            ax.set_title(f"{seg_title} (no data)", color=TEXT_PRIMARY, fontsize=11, loc="left")
            continue

        labels = STREAM_CATEGORIES + ["Other"]
        colors = [CATEGORY_COLORS[c] for c in STREAM_CATEGORIES] + [OTHER_COLOR]
        values = [bands[label] for label in labels]
        ax.stackplot(xs, *values, labels=labels, colors=colors, baseline="wiggle", alpha=0.9)

        ax.set_title(seg_title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel("Normalized token position (%)", color=TEXT_SECONDARY, fontsize=9)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0, axis="x")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRIDLINE)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, left=False, labelleft=False)
        ax.set_xlim(0, 100)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in STREAM_CATEGORIES]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOR))
    labels = STREAM_CATEGORIES + ["Other (baseline: Law/Politics, Technology, Science -- summed)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2,
               frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, y=1.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def whole_question_summary(ground_truths):
    """Long-format rows: one per (qid, segment, compared_category)."""
    records = []
    for item in PILOT:
        qid, own_category = item["id"], item["category"]
        for segment, gts in ground_truths.items():
            dist = question_distribution(load_output(qid), segment=segment)
            if dist is None:
                continue
            sims = {cat: 1.0 - js_divergence(dist, gt) for cat, gt in gts.items()}
            religion_sim = sims["Religion & Theology"]
            baseline_sim = sum(sims[c] for c in BASELINE_CATEGORIES) / len(BASELINE_CATEGORIES)
            delta = religion_sim - baseline_sim
            for compared_category, sim in sims.items():
                records.append({
                    "qid": qid,
                    "own_category": own_category,
                    "segment": segment,
                    "compared_category": compared_category,
                    "similarity": sim,
                    "js_divergence": 1.0 - sim,
                    "religion_similarity": religion_sim,
                    "baseline_similarity": baseline_sim,
                    "religion_delta": delta,
                })
    return pd.DataFrame(records)


def plot_ranked_delta(df, output_path):
    rows = df.drop_duplicates(subset=["qid", "segment"])[
        ["qid", "own_category", "segment", "religion_delta"]
    ]
    order = (
        rows[rows["segment"] == "answer"]
        .sort_values("religion_delta")["qid"].tolist()
    )
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y_labels = []
    for i, qid in enumerate(order):
        own_cat = rows[rows["qid"] == qid]["own_category"].iloc[0]
        y_labels.append(f"{own_cat}  (#{qid})")
        t_val = rows[(rows.qid == qid) & (rows.segment == "thought")]["religion_delta"]
        a_val = rows[(rows.qid == qid) & (rows.segment == "answer")]["religion_delta"]
        t_val = t_val.iloc[0] if len(t_val) else None
        a_val = a_val.iloc[0] if len(a_val) else None
        if t_val is not None and a_val is not None:
            ax.plot([t_val, a_val], [i, i], color=GRIDLINE, linewidth=1.5, zorder=1)
        if t_val is not None:
            ax.scatter([t_val], [i], color="#2a78d6", s=55, zorder=2, label="Thought" if i == 0 else None)
        if a_val is not None:
            ax.scatter([a_val], [i], color="#eb6834", s=55, zorder=2, label="Answer" if i == 0 else None)

    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(y_labels, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Religion delta  (similarity to Religion − avg. similarity to baseline categories)",
                  color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Benchmark pilot: religion signal per question, sorted by answer-segment delta",
                 color=TEXT_PRIMARY, fontsize=11.5, loc="left")
    ax.grid(True, color=GRIDLINE, linewidth=0.8, axis="x", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRIDLINE)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    ground_truths = {seg: build_ground_truths(seg) for seg in ("thought", "answer")}

    # --- Whole-question summary + ranked delta chart ---
    df = whole_question_summary(ground_truths)
    df.to_csv("benchmark_summary.csv", index=False)
    print("Wrote benchmark_summary.csv")

    delta_rows = df.drop_duplicates(subset=["qid", "segment"])
    print("\n=== Religion delta by question ===")
    print(delta_rows[["qid", "own_category", "segment", "religion_delta"]]
          .sort_values(["segment", "religion_delta"]).to_string(index=False))
    print("\n=== Aggregate (mean delta by segment) ===")
    print(delta_rows.groupby("segment")["religion_delta"].agg(["mean", "median", "std"]).to_string())

    plot_ranked_delta(df, "benchmark_religion_delta_ranked.png")

    # --- Per-question streamgraphs ---
    agg_curves = {"thought": {c: [] for c in STREAM_CATEGORIES + ["Other"]},
                  "answer": {c: [] for c in STREAM_CATEGORIES + ["Other"]}}

    for item in PILOT:
        qid, own_category = item["id"], item["category"]
        xs_by_segment, bands_by_segment = {}, {}
        for segment in ("thought", "answer"):
            xs, curves = windowed_similarity_curves(qid, segment, ground_truths[segment])
            xs_by_segment[segment] = xs
            if not xs:
                bands_by_segment[segment] = {}
                continue
            bands = stream_bands(curves)
            bands_by_segment[segment] = bands
            for band, vals in bands.items():
                interp = np.interp(AGG_GRID, xs, vals)
                agg_curves[segment][band].append(interp)

        plot_streamgraph(
            xs_by_segment, bands_by_segment,
            title=f"Question {qid} ({own_category}) — category similarity stream",
            output_path=f"benchmark_stream_{qid}.png",
        )

    # --- Aggregate streamgraph across all 10 pilot questions ---
    agg_bands_by_segment = {}
    for segment in ("thought", "answer"):
        agg_bands_by_segment[segment] = {
            band: np.mean(np.array(vals_list), axis=0).tolist() if vals_list else []
            for band, vals_list in agg_curves[segment].items()
        }
    agg_xs = {seg: (AGG_GRID if agg_bands_by_segment[seg]["Religion & Theology"] else [])
              for seg in ("thought", "answer")}
    plot_streamgraph(
        agg_xs, agg_bands_by_segment,
        title=f"Benchmark pilot (n={len(PILOT)}) — average category similarity stream",
        output_path="benchmark_stream_aggregate.png",
    )


if __name__ == "__main__":
    main()
