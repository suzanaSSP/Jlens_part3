"""
Builds a fixed, reproducible question pool per category.

Design:
- Single random shuffle per category (seed=42, matching first_10_sample.csv's
  convention) defines a stable order. Nested batches (first N1, first N2, ...)
  are prefixes of this order, so GT(N2) always contains all of GT(N1)'s
  questions plus new ones -- required for the convergence/bootstrap check
  to isolate "still converging" from "different random draw."
- TRAIN_N questions per category go toward ground-truth building /
  convergence testing; the remainder is reserved as a held-out test set
  (validate later: do held-out questions land close to their own GT?).
- Merges into any existing ground_truth_pool.json rather than overwriting it
  -- re-running with the same seed/TRAIN_N for a category already in the
  pool reproduces the identical train_ids, so this is safe to re-run after
  adding a new category to CATEGORY_CONFIG.
"""
import json
import random

SEED = 42
# category -> TRAIN_N (upper bound on the nested batches).
# Law/Tech: pilot pair used to validate pipeline mechanics.
# Religion/Ethics: the actual comparison of interest -- smaller TRAIN_N
# because Religion & Theology only has 31 questions total (need to leave a
# real held-out set for later validation).
CATEGORY_CONFIG = {
    "Law, Politics, Government & History": 40,
    "Technology & Definitions": 40,
    "Religion & Theology": 20,
    "Applied Ethics & Moral Dilemmas": 20,
    "Marriage & Romantic Partnerships": 35,
    "Human Nature & Philosophy": 30,
    "Science, Physics & Math": 25,
    "Depression, Addiction & Feeling Lost": 20,
    "Family Duty & Obligations": 20,
    # Only 17 questions exist in this category total -- both train and
    # held-out are thin here, treat results involving it as low-confidence.
    "Grief, Loss & Death": 10,
}

with open("non_benchmark_961.json") as f:
    all_questions = json.load(f)

try:
    with open("ground_truths/ground_truth_pool.json") as f:
        pool = json.load(f)
except FileNotFoundError:
    pool = {}

for category, train_n in CATEGORY_CONFIG.items():
    cat_questions = [q for q in all_questions if q["category"] == category]
    rng = random.Random(SEED)
    rng.shuffle(cat_questions)

    train = cat_questions[:train_n]
    heldout = cat_questions[train_n:]

    pool[category] = {
        "train_ids": [q["id"] for q in train],
        "heldout_ids": [q["id"] for q in heldout],
    }

    print(f"{category}: {len(cat_questions)} total -> "
          f"{len(train)} train / {len(heldout)} held-out")

with open("ground_truths/ground_truth_pool.json", "w") as f:
    json.dump(pool, f, indent=2)

print("Wrote data/ground_truths/ground_truth_pool.json")
