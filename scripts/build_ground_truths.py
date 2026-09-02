"""
Builds and saves the final per-category ground-truth j-space distribution.

Ground truth = plain average of all *train* questions' pooled distributions
(no bootstrapping -- bootstrapping in bootstrap_ground_truth.py was only a
diagnostic to confirm within-category noise is small relative to
between-category distances before trusting these).

None of the categories' bootstrap curves had flattened at their current
train pool size (still a ~20-25% relative noise drop per added batch), so
these ground truths aren't "converged" -- they're usable because
within-category noise is already ~5x smaller than every between-category
distance, not because more data wouldn't help.

Each category gets its own file under OUTPUT_DIR/ rather than one combined
JSON -- categories get added incrementally across sessions, and this way
adding/rebuilding one category never touches another's already-trusted file.
"""
import json
import os
import re

from scripts.bootstrap_ground_truth import (
    LAYER,
    POOL,
    SEGMENT,
    average_distribution,
    load_output,
    question_distribution,
)

OUTPUT_DIR = "ground_truths"


def slugify(category):
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for category, ids in POOL.items():
        train_ids = ids["train_ids"]
        dists = []
        for qid in train_ids:
            d = question_distribution(load_output(qid))
            if d is not None:
                dists.append(d)
        gt = average_distribution(dists)
        record = {
            "category": category,
            "layer": LAYER,
            "segment": SEGMENT,
            "n_questions": len(dists),
            "train_ids": train_ids,
            "distribution": gt,
        }

        output_path = os.path.join(OUTPUT_DIR, f"{slugify(category)}.json")
        with open(output_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"{category}: ground truth built from {len(dists)} questions -> {output_path}")


if __name__ == "__main__":
    main()
