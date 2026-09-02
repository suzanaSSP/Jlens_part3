# Category Timeline Analysis — Action Plan

Goal: for a benchmark of questions, track how close the model's j-space (per token
position, thought → answer) sits to each category's ground-truth distribution,
aggregated across the whole benchmark on a normalized timeline.

## 1. Ground truth distributions (per category)

- [ ] For each category, split available questions (e.g. 25/25 or k-fold if you
      want more robust estimates — see prior discussion).
- [ ] Restrict to `thought` (and decide separately whether to include `answer`)
      segment tokens only. **Exclude `prompt` segment** — it just reflects the
      input text and will trivially "match" its own category.
- [ ] Fix a pooling rule: normalize within-question first (average across a
      question's positions to get one distribution per question), *then*
      average across questions. Prevents long responses from dominating.
- [ ] Pick the layer(s) to build ground truths on. Recommend starting with
      whichever layer(s) came out as most discriminative in the earlier
      centroid-classifier validation, rather than all layers at once.
- [ ] Store this as a fixed, static reference per category — it does not get
      recomputed per position. It's the anchor the moving timeline is measured
      against.

## 2. Position normalization (for aggregating across questions)

- [ ] Normalize `thought` and `answer` segments **separately** — don't merge
      them into one 0–100% range. Thought:answer length ratios vary per
      question, so a combined scale would blur the transition point, which is
      itself one of the more interesting things to look at.
- [ ] Bucket each segment into N bins (e.g. 20) by normalized position, and
      interpolate/assign each token's divergence value into its bin.
- [ ] Handle short generations / sparse bins: some questions may not populate
      every bin cleanly — decide whether to interpolate, skip, or drop
      questions below a minimum length.

## 3. Per-position divergence computation

- [ ] For each test position's j-space distribution, compute divergence to
      **every** category's ground truth (not just its own) — you want the full
      set of curves per test question, not a single number.
- [ ] Use JS divergence by default (bounded, symmetric, more forgiving of
      top-20 support mismatch than KL). Add a smoothing / "residual mass"
      bucket for tokens outside the top-20 so divergence isn't undefined on
      support mismatches.
- [ ] Exclude or down-weight low-information tokens (punctuation, stopwords,
      connective tissue) — their top-20 is often near-deterministic and adds
      noise without carrying category signal.

## 4. Aggregation across the benchmark

- [ ] Per category-of-origin, per bin: average divergence-to-each-ground-truth
      across all benchmark questions in that category.
- [ ] Also compute variance / a confidence band per bin across questions —
      needed to tell whether a bump in the curve is a real effect or driven by
      one or two outlier questions.
- [ ] Apply a light smoothing pass (moving average across adjacent bins) on
      top of the binning if curves are still jumpy.

## 5. Visualization

- [ ] Line plot per category-of-origin: divergence to each of the K ground
      truths, over normalized position (two panels: thought, then answer).
- [ ] "Dominant category" strip (argmin divergence per bin) as a simpler
      categorical summary alongside the continuous curves.
- [ ] If time allows, the 2D version: position-bin × layer heatmap per
      category, to see whether the effect is depth-dependent as well as
      position-dependent (you said layer is deferred for now, but worth
      keeping the data shape compatible with adding this later).

## 6. Validation / sanity checks

- [ ] Confirm same-category curves behave as expected: a category's own
      held-out test questions should generally show the *lowest* divergence to
      their own ground truth relative to other categories, especially in later
      `answer` bins.
- [ ] Cross-check against the earlier centroid-classifier validation results
      (self vs. other-category divergence) for consistency.

## Open decisions to make before building

- Segment scope: `thought` only, or `thought` + `answer`?
- Number of bins per segment.
- Smoothing window size.
- Stopword / functional-token exclusion list.
- Which layer(s) to use for this first pass.
- Minimum question length / generation length to include in aggregation.

## Other thoughts

- **Isolate the category-specific effect from generic drift.** Raw divergence
  to the religious ground truth may rise and fall for reasons that have
  nothing to do with religion (e.g. general specificity increasing as
  reasoning progresses). Consider also plotting a *differential* curve:
  `divergence(test, religious_gt) − divergence(test, neutral_baseline_gt)`
  (or minus the average divergence to all non-target categories). This
  isolates the religion-specific signal from a generic "getting more
  concrete/more confident" trend that would otherwise show up in every
  category's curve.
- Keep an eye on how many numbers this produces: (bins) × (categories) ×
  (ground-truth targets) × (benchmark questions). Worth writing the
  aggregation step to produce a single tidy table (bin, source_category,
  target_ground_truth, mean_divergence, ci) before visualizing, rather than
  trying to plot straight from raw per-question data.
- Given the sample sizes discussed earlier (~50/category), treat any bump or
  dip in the aggregated timeline as suggestive rather than conclusive on the
  first pass — especially if it only shows up in a narrow range of bins.
