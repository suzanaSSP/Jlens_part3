"""
Bootstrap resampling check for category ground-truth stability.

For each category in ground_truth_pool.json, pools per-question j-space
distributions (thought segment, one layer) and answers two questions with a
single resampling procedure by sweeping sample size N over quarter-marks of
that category's train pool (e.g. 40 train questions -> N = 10, 20, 30, 40):
  - variance at N: how much does the ground truth wobble across B bootstrap
    resamples of size N drawn (with replacement) from the train questions?
  - saturation: does that variance keep shrinking as N grows, or has it
    flattened already?

A single incremental path (add 10, then 10 more...) would confound "still
converging" with "this particular draw was unusual" -- bootstrapping at each
N averages that ordering effect away.

Reference scale: JS divergence between each pair of categories' full train
ground truths -- within-category noise needs to be small relative to this to
trust a cross-category comparison.
"""
import gzip
import json
import random

import numpy as np
import pandas as pd

LAYER = 62  # last layer; layer axis deferred per plan, first-pass default
SEGMENT = "thought"
N_BOOTSTRAP = 500
SEED = 42

# Common English stopwords + punctuation-only tokens are excluded before
# pooling: once averaged over hundreds of positions, these dominate the top
# of every category's ground truth almost identically (",", ".", " the",
# " to", " of", ...), diluting the category-specific signal into the long
# tail. Filtering per-position, before pooling, keeps the comparison focused
# on content-bearing tokens.
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "would", "could", "also",
    "like", "one", "get", "got", "much", "many",
}


def is_stopword_or_punct(token_text):
    """True for punctuation-only tokens and common English stopwords
    (case/whitespace-insensitive -- BPE tokens carry a leading space)."""
    t = token_text.strip().lower()
    if not t:
        return True
    if not any(c.isalnum() for c in t):
        return True
    return t in STOPWORDS


def n_grid_for(pool_size):
    """Quarter-mark N grid scaled to however many train questions a category
    actually has (e.g. 40 -> [10,20,30,40], 20 -> [5,10,15,20])."""
    return sorted({round(pool_size * frac) for frac in (0.25, 0.5, 0.75, 1.0)})

with open("ground_truth_pool.json") as f:
    POOL = json.load(f)


def load_output(qid):
    with gzip.open(f"../jspace_prompts/jlens_output_prompt_{qid}.json.gz") as f:
        return json.load(f)


def position_distribution(pos, filter_stopwords=True):
    """Single-position sparse distribution: top-20 probs + a residual-mass
    bucket for probability outside the top-20 (always kept as-is -- we can't
    tell whether that untracked mass is stopwords or content). When
    filter_stopwords, common stopword/punctuation tokens are dropped from the
    top-20 and the remainder is renormalized back to sum to 1."""
    totals = {}
    top20_mass = 0.0
    dropped_mass = 0.0
    for entry in pos["top_k"]:
        top20_mass += entry["probability"]
        if filter_stopwords and is_stopword_or_punct(entry["token"]):
            dropped_mass += entry["probability"]
            continue
        totals[entry["token_id"]] = totals.get(entry["token_id"], 0.0) + entry["probability"]
    totals["__residual__"] = 1.0 - top20_mass

    kept_total = 1.0 - dropped_mass
    if kept_total <= 0:
        return None
    return {k: v / kept_total for k, v in totals.items()}


def question_distribution(output, layer=LAYER, segment=SEGMENT, filter_stopwords=True):
    """Pool one question into a single sparse distribution: average its
    per-position distributions (see position_distribution) at `layer`/`segment`."""
    layer_data = next(l for l in output["jacobian_lens_per_layer"] if l["layer"] == layer)
    positions = [t for t in layer_data["tokens"] if t["segment"] == segment]
    if not positions:
        return None

    dists = [position_distribution(pos, filter_stopwords) for pos in positions]
    dists = [d for d in dists if d is not None]
    if not dists:
        return None
    return average_distribution(dists)


def js_divergence(p, q):
    """Jensen-Shannon divergence (base-2, bounded [0,1]) between two sparse
    dict distributions with possibly different support."""
    keys = set(p) | set(q)
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])
    m = 0.5 * (p_vec + q_vec)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m)


def average_distribution(dists):
    """Elementwise average of a list of sparse dict distributions."""
    totals = {}
    for d in dists:
        for k, v in d.items():
            totals[k] = totals.get(k, 0.0) + v
    n = len(dists)
    return {k: v / n for k, v in totals.items()}


def bootstrap_curve(question_dists, rng):
    """For each N in this category's N grid, draw N_BOOTSTRAP resamples
    (with replacement) of size N, build the resampled ground truth, and
    measure its JS divergence to the full-pool reference ground truth."""
    reference = average_distribution(question_dists)
    pool_size = len(question_dists)
    records = []
    for n in n_grid_for(pool_size):
        for _ in range(N_BOOTSTRAP):
            sample = [question_dists[i] for i in rng.choices(range(pool_size), k=n)]
            gt_b = average_distribution(sample)
            records.append({"n": n, "js_to_reference": js_divergence(gt_b, reference)})
    return records


def main():
    rng = random.Random(SEED)
    category_dists = {}
    all_records = []

    for category, ids in POOL.items():
        train_ids = ids["train_ids"]
        dists = []
        for qid in train_ids:
            output = load_output(qid)
            d = question_distribution(output)
            if d is not None:
                dists.append(d)
        category_dists[category] = dists
        print(f"{category}: pooled {len(dists)}/{len(train_ids)} train questions")

        for rec in bootstrap_curve(dists, rng):
            rec["category"] = category
            all_records.append(rec)

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(["category", "n"])["js_to_reference"]
        .agg(mean="mean", std="std", p5=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95))
        .reset_index()
    )
    print()
    print(summary.to_string(index=False))
    summary.to_csv("bootstrap_variance_summary.csv", index=False)
    print("\nWrote data/bootstrap_variance_summary.csv")

    cats = list(category_dists.keys())
    full_gts = {c: average_distribution(category_dists[c]) for c in cats}
    print("\nBetween-category JS divergence (full train pool per category):")
    between_records = []
    for i, c1 in enumerate(cats):
        for c2 in cats[i + 1:]:
            d = js_divergence(full_gts[c1], full_gts[c2])
            print(f"  {c1}  vs  {c2}: {d:.4f}")
            between_records.append({"category_a": c1, "category_b": c2, "js_divergence": d})
    pd.DataFrame(between_records).to_csv("between_category_js.csv", index=False)
    print("\nWrote data/between_category_js.csv")
    print("(compare each pair above to the within-category 'mean' column at its")
    print(" max n in bootstrap_variance_summary.csv -- within-category noise should")
    print(" be small relative to the between-category number for that pair)")


if __name__ == "__main__":
    main()
