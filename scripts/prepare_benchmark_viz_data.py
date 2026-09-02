"""
Precompute all data for the interactive full-benchmark visualization
(benchmark_viz.html): per-question thought/answer streamgraph curves for all
132 benchmark questions, plus the aggregate stream and the ranked religion-
delta chart -- same computation as benchmark_analysis.py's 10-question pilot,
just embedded as one inline JSON blob for a single self-contained HTML page
instead of one PNG per question.

Output: benchmark_viz_data.json (consumed by benchmark_viz.html at build
time -- see embed_data.py -- not fetched at runtime, so the page opens
directly via file://).
"""
import json

import numpy as np

from scripts.bootstrap_ground_truth import (
    LAYER,
    average_distribution,
    js_divergence,
    load_output,
    position_distribution,
    question_distribution,
)

WINDOW_SIZE = 10
AGG_GRID = list(range(0, 101, 5))

with open("ground_truth_pool.json") as f:
    POOL = json.load(f)
with open("benchmark_questions.json") as f:
    ALL_QUESTIONS = [q for q in json.load(f) if str(q["id"]) != "nan"]

CATEGORY_COLORS = {
    "Religion & Theology":                 "#2a78d6",
    "Law, Politics, Government & History": "#eb6834",
    "Technology & Definitions":            "#1baf7a",
    "Applied Ethics & Moral Dilemmas":     "#eda100",
    "Marriage & Romantic Partnerships":    "#e87ba4",
    "Human Nature & Philosophy":           "#008300",
    "Science, Physics & Math":             "#4a3aa7",
    "Depression, Addiction & Feeling Lost": "#e34948",
}
BASELINE_CATEGORIES = [
    "Law, Politics, Government & History",
    "Technology & Definitions",
    "Science, Physics & Math",
]
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


def norm_id(qid):
    return str(int(float(qid)))


def build_ground_truths(segment):
    gts = {}
    for category, ids in POOL.items():
        dists = [question_distribution(load_output(qid), segment=segment) for qid in ids["train_ids"]]
        dists = [d for d in dists if d is not None]
        gts[category] = average_distribution(dists)
    return gts


def windowed_similarity_curves(qid, segment, ground_truths):
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
        xs.append(round(norm_pos, 2))
        for cat, gt in ground_truths.items():
            curves[cat].append(round(1.0 - js_divergence(pooled, gt), 4))
    return xs, curves


def stream_bands(curves):
    bands = {cat: curves[cat] for cat in STREAM_CATEGORIES}
    other = [round(sum(vals), 4) for vals in zip(*(curves[c] for c in BASELINE_CATEGORIES))]
    bands["Other"] = other
    return bands


def diff_bands(curves):
    """Per-category similarity minus the mean baseline-category similarity at
    each position -- isolates category-specific signal from the shared
    generic-vocabulary drift that raw similarity curves are dominated by
    (all categories move together because the same per-position distribution
    is compared against every ground truth, and generic tokens overlap all
    of them)."""
    baseline_mean = [
        sum(vals) / len(BASELINE_CATEGORIES)
        for vals in zip(*(curves[c] for c in BASELINE_CATEGORIES))
    ]
    return {
        cat: [round(v - b, 4) for v, b in zip(curves[cat], baseline_mean)]
        for cat in STREAM_CATEGORIES
    }


def whole_question_deltas(ground_truths):
    rows = []
    for item in ALL_QUESTIONS:
        qid = norm_id(item["id"])
        row = {"qid": qid, "own_category": item["category"], "text": item["text"]}
        for segment, gts in ground_truths.items():
            dist = question_distribution(load_output(qid), segment=segment)
            if dist is None:
                row[f"{segment}_delta"] = None
                continue
            sims = {cat: 1.0 - js_divergence(dist, gt) for cat, gt in gts.items()}
            religion_sim = sims["Religion & Theology"]
            baseline_sim = sum(sims[c] for c in BASELINE_CATEGORIES) / len(BASELINE_CATEGORIES)
            row[f"{segment}_delta"] = round(religion_sim - baseline_sim, 4)
        rows.append(row)
    return rows


def main():
    ground_truths = {seg: build_ground_truths(seg) for seg in ("thought", "answer")}

    print("Computing per-question streams for", len(ALL_QUESTIONS), "questions...")
    questions_out = []
    agg_curves = {"thought": {c: [] for c in STREAM_CATEGORIES + ["Other"]},
                  "answer": {c: [] for c in STREAM_CATEGORIES + ["Other"]}}
    agg_diff_curves = {"thought": {c: [] for c in STREAM_CATEGORIES},
                        "answer": {c: [] for c in STREAM_CATEGORIES}}

    for i, item in enumerate(ALL_QUESTIONS):
        qid = norm_id(item["id"])
        entry = {"qid": qid, "category": item["category"], "text": item["text"], "segments": {}}
        for segment in ("thought", "answer"):
            xs, curves = windowed_similarity_curves(qid, segment, ground_truths[segment])
            if not xs:
                entry["segments"][segment] = None
                continue
            bands = stream_bands(curves)
            dbands = diff_bands(curves)
            entry["segments"][segment] = {"xs": xs, "bands": bands, "diff_bands": dbands}
            for band, vals in bands.items():
                interp = np.interp(AGG_GRID, xs, vals)
                agg_curves[segment][band].append(interp)
            for band, vals in dbands.items():
                interp = np.interp(AGG_GRID, xs, vals)
                agg_diff_curves[segment][band].append(interp)
        questions_out.append(entry)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(ALL_QUESTIONS)}")

    aggregate = {}
    for segment in ("thought", "answer"):
        bands = {
            band: np.mean(np.array(vals_list), axis=0).round(4).tolist() if vals_list else []
            for band, vals_list in agg_curves[segment].items()
        }
        dbands = {
            band: np.mean(np.array(vals_list), axis=0).round(4).tolist() if vals_list else []
            for band, vals_list in agg_diff_curves[segment].items()
        }
        aggregate[segment] = (
            {"xs": AGG_GRID, "bands": bands, "diff_bands": dbands} if bands["Religion & Theology"] else None
        )

    print("Computing whole-question religion deltas...")
    deltas = whole_question_deltas(ground_truths)

    out = {
        "n_questions": len(ALL_QUESTIONS),
        "category_colors": CATEGORY_COLORS,
        "baseline_categories": BASELINE_CATEGORIES,
        "stream_categories": STREAM_CATEGORIES,
        "other_color": OTHER_COLOR,
        "questions": questions_out,
        "aggregate": aggregate,
        "deltas": deltas,
    }
    with open("benchmark_viz_data.json", "w") as f:
        json.dump(out, f)
    print("Wrote benchmark_viz_data.json")


if __name__ == "__main__":
    main()
