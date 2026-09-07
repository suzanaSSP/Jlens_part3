"""
Per-window category presence and merging analysis across the neutral
benchmark questions.

For every benchmark question with jlens output, slides a WINDOW_SIZE window
(matching timeline_pilot.py's validated choice -- a single token position is
nearly a delta spike while a ground truth is diffuse, so comparing raw
positions swamps the category signal) across the thought and answer segments
separately. At each window, records the top-1 and top-2 closest category
ground truths by similarity and the gap between them: a small gap means the
window is hard to attribute to one category over the other -- the "merging"
signal. Comparing this gap's distribution in thought vs. answer is the first
cut at "do categories become harder to tell apart by the time the model
answers."

Raw similarity turned out to be dominated by 2-3 "attractor" categories
(Depression, Marriage) regardless of a question's actual topic -- 75% of all
thought-segment windows had one of those two as top-1, vs. Religion/Tech/Law
winning almost never, and overall top1-vs-source-category agreement was only
37% (vs. ~64% for whole-question pooling on the one category with real
held-out data). Likely cause: every benchmark question shares the same
first-person, personal-advice phrasing register, which structurally
resembles how the Depression/Marriage training questions are worded,
independent of topic. So each category's similarity is baseline-corrected
here: BASELINE[segment][category] = that category's own mean raw similarity
across every window in the benchmark set, and top1/top2/gap are computed
from (raw - baseline) instead of raw similarity -- the same delta-from-
baseline idea benchmark_analysis.py already uses for the religion-signal
calibration, generalized from one hand-picked baseline category set to each
category baselining against itself.

Ground truths are built fresh for both segments from train_ids (only
thought-segment ones are persisted under ground_truths/), same approach as
timeline_pilot.py and heldout_validation.py.

Similarity is computed against a fixed per-segment vocabulary (the union of
the 10 ground truths' own keys) via vectorized numpy rather than rebuilding a
key-union set per (window, category) pair -- a naive port of
bootstrap_ground_truth.js_divergence to ~800k calls (132 questions x 2
segments x ~300 windows x 10 categories, each rebuilding a several-thousand-
key set) didn't finish in 15+ minutes. Restricting each window's distribution
to that fixed vocabulary drops a window's own out-of-vocabulary mass, but
that mass contributes an identical additive constant to every category's JS
divergence (a key absent from all 10 ground truths' support affects each
category's KL(P||M) term the same way, and never appears in KL(Q||M) since
Q=0 there) -- so it shifts absolute similarity slightly but leaves the top-1
vs. top-2 ranking and the gap between them exactly unchanged.
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scripts.bootstrap_ground_truth import (
    LAYER,
    average_distribution,
    load_output,
    position_distribution,
    question_distribution,
)

WINDOW_SIZE = 10
N_LOAD_WORKERS = 16

with open("ground_truths/ground_truth_pool.json") as f:
    POOL = json.load(f)

with open("prompts_json/benchmark_questions.json") as f:
    BENCHMARK_QUESTIONS = json.load(f)

CATEGORIES = list(POOL.keys())


def _safe_load(qid):
    try:
        return qid, load_output(qid)
    except FileNotFoundError:
        return qid, None


def preload_outputs(qids):
    """Loads+decompresses each unique qid's jlens output once, in parallel --
    each file is ~9MB gzipped and takes ~1s to load, and both the ground
    truths (thought AND answer, per category) and the benchmark questions
    need one load per unique id. Loading serially, once per (qid, segment)
    use, was the actual bottleneck (not the similarity math)."""
    unique_qids = sorted(set(qids))
    outputs = {}
    with ProcessPoolExecutor(max_workers=N_LOAD_WORKERS) as pool:
        for qid, output in pool.map(_safe_load, unique_qids):
            if output is not None:
                outputs[qid] = output
    return outputs


def build_ground_truths(segment, cache):
    gts = {}
    for category, ids in POOL.items():
        dists = [question_distribution(cache[qid], segment=segment) for qid in ids["train_ids"] if qid in cache]
        dists = [d for d in dists if d is not None]
        gts[category] = average_distribution(dists)
    return gts


def build_vocab_matrix(gts):
    """Fixed vocabulary (union of all 10 ground truths' keys) and a dense
    (10, V) matrix of their distributions over it, built once per segment so
    every window's comparison is a vectorized lookup instead of rebuilding a
    key-union set from scratch."""
    vocab = sorted({k for gt in gts.values() for k in gt}, key=str)
    index = {k: i for i, k in enumerate(vocab)}
    matrix = np.zeros((len(CATEGORIES), len(vocab)))
    for row, category in enumerate(CATEGORIES):
        for k, v in gts[category].items():
            matrix[row, index[k]] = v
    return index, matrix


def to_dense(dist, index, size):
    vec = np.zeros(size)
    for k, v in dist.items():
        col = index.get(k)
        if col is not None:
            vec[col] = v
    return vec


def vectorized_js(p, q_matrix):
    """JS divergence (base-2) between one dense vector p (V,) and each row
    of q_matrix (K, V) -- same formula as bootstrap_ground_truth.js_divergence,
    vectorized across all K categories at once."""
    m = 0.5 * (p[None, :] + q_matrix)

    def kl_term(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a > 0, a / b, 1.0)
            terms = np.where(a > 0, a * np.log2(ratio), 0.0)
        return terms.sum(axis=-1)

    return 0.5 * kl_term(p[None, :], m) + 0.5 * kl_term(q_matrix, m)


def windowed_distributions(positions):
    """(center_norm_pos, pooled_distribution) per WINDOW_SIZE-position
    window, sliding by 1 -- see module docstring for why raw positions
    aren't compared directly."""
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


def benchmark_qids():
    """(qid, source_category) for benchmark questions with a parseable id.
    Skips 18 rows with 'nan' id/category -- broken metadata, separate
    cleanup needed before they can be matched to jlens output."""
    out = []
    for q in BENCHMARK_QUESTIONS:
        try:
            qid = str(int(float(q["id"])))
        except (ValueError, TypeError):
            continue
        out.append((qid, q.get("category")))
    return out


def main():
    train_ids = {qid for ids in POOL.values() for qid in ids["train_ids"]}
    qids = benchmark_qids()
    benchmark_ids = {qid for qid, _ in qids}

    print(f"Preloading {len(train_ids)} train + {len(benchmark_ids)} benchmark outputs "
          f"({N_LOAD_WORKERS} workers)...")
    cache = preload_outputs(train_ids | benchmark_ids)
    print(f"Loaded {len(cache)}/{len(train_ids | benchmark_ids)} files.")

    ground_truths = {seg: build_ground_truths(seg, cache) for seg in ("thought", "answer")}
    vocab = {}
    gt_matrix = {}
    for seg, gts in ground_truths.items():
        vocab[seg], gt_matrix[seg] = build_vocab_matrix(gts)
    print(f"Vocab sizes: thought={len(vocab['thought'])}, answer={len(vocab['answer'])}")

    meta = []
    sim_rows = []
    n_ok, n_skipped = 0, 0
    for i, (qid, source_category) in enumerate(qids, 1):
        output = cache.get(qid)
        if output is None:
            n_skipped += 1
            continue
        layer_data = next(l for l in output["jacobian_lens_per_layer"] if l["layer"] == LAYER)

        got_any = False
        for segment in ("thought", "answer"):
            positions = [t for t in layer_data["tokens"] if t["segment"] == segment]
            windows = windowed_distributions(positions)
            if not windows:
                continue
            got_any = True
            index, size = vocab[segment], len(vocab[segment])
            for norm_pos, dist in windows:
                p = to_dense(dist, index, size)
                sims = 1.0 - vectorized_js(p, gt_matrix[segment])
                meta.append((qid, source_category, segment, norm_pos))
                sim_rows.append(sims)
        if got_any:
            n_ok += 1
        else:
            n_skipped += 1

        if i % 20 == 0 or i == len(qids):
            print(f"  {i}/{len(qids)} questions processed...")

    meta_df = pd.DataFrame(meta, columns=["qid", "source_category", "segment", "norm_pos"])
    raw = np.array(sim_rows)  # (n_windows, 10), column order = CATEGORIES

    # Baseline-correct: subtract each category's own mean raw similarity
    # across all windows *of that segment* -- an "attractor" category with a
    # high baseline everywhere gets pulled back down to a comparable
    # footing, so top1/top2 reflect above/below-normal presence instead of
    # raw magnitude. See module docstring for why this was needed.
    adjusted = np.empty_like(raw)
    baseline = {}
    for segment in ("thought", "answer"):
        mask = (meta_df["segment"] == segment).to_numpy()
        baseline[segment] = raw[mask].mean(axis=0)
        adjusted[mask] = raw[mask] - baseline[segment]

    order = np.argsort(adjusted, axis=1)[:, ::-1]
    top1_idx, top2_idx = order[:, 0], order[:, 1]
    rows = np.arange(len(meta_df))

    df = meta_df.copy()
    df["top1_category"] = np.array(CATEGORIES)[top1_idx]
    df["top1_adjusted"] = adjusted[rows, top1_idx]
    df["top1_raw_similarity"] = raw[rows, top1_idx]
    df["top2_category"] = np.array(CATEGORIES)[top2_idx]
    df["top2_adjusted"] = adjusted[rows, top2_idx]
    df["gap"] = adjusted[rows, top1_idx] - adjusted[rows, top2_idx]

    # Full per-category matrix -- not just the top-2 -- so a category that's
    # elevated without actually winning (or "how well does ANY category
    # explain this window") can be checked later, e.g. for flagging
    # windows that don't fit any of the 10 well.
    for j, category in enumerate(CATEGORIES):
        df[f"raw__{category}"] = raw[:, j]
        df[f"adj__{category}"] = adjusted[:, j]

    df.to_csv("benchmark_category_presence.csv", index=False)
    print(f"\nProcessed {n_ok} questions ({len(df)} window-rows), skipped {n_skipped}")
    print("Wrote data/benchmark_category_presence.csv\n")

    print("Per-category baseline (mean raw similarity across all windows), by segment:")
    print(pd.DataFrame(baseline, index=CATEGORIES).round(4).to_string())
    print()

    print("Mean top1-vs-top2 gap (baseline-corrected) by segment (lower = more merged/ambiguous):")
    print(df.groupby("segment")["gap"].agg(["mean", "median", "std"]).to_string())
    print()

    print("top1_category distribution after correction, by segment:")
    for segment in ("thought", "answer"):
        print(f"\n{segment}:")
        print(df[df.segment == segment]["top1_category"].value_counts().to_string())

    match = (df["top1_category"] == df["source_category"]).mean()
    print(f"\nOverall top1 == source_category rate (post-correction): {match:.3f}")


if __name__ == "__main__":
    main()
