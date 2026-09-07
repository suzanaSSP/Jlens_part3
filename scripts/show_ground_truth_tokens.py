"""
Decode and display the top tokens from a ground-truth distribution file.

The ground-truth JSONs store token IDs as string keys. This script loads
the tokenizer, decodes each ID, and prints them sorted by probability so
you can see what vocabulary actually characterizes each category.

Usage (from project root):
    python -m scripts.show_ground_truth_tokens data/ground_truths/family_duty_obligations.json
    python -m scripts.show_ground_truth_tokens data/ground_truths/religion_theology.json --top 50

    # Compare two categories side by side:
    python -m scripts.show_ground_truth_tokens data/ground_truths/religion_theology.json \
        --compare data/ground_truths/science_physics_math.json

Options:
    --top N         How many top tokens to show (default: 40)
    --model         HuggingFace model id to load tokenizer from
                    (default: Qwen/Qwen2.5-32B-Instruct)
    --compare PATH  Second ground-truth file to display alongside the first
    --no-skip-residual  Also show the __residual__ mass bucket
"""

import argparse
import json


def load_gt(path):
    with open(path) as f:
        data = json.load(f)
    return data


def top_tokens(distribution, tokenizer, top_n, skip_residual=True):
    """Return list of (token_str, probability) sorted by probability desc."""
    rows = []
    for tok_id_str, prob in distribution.items():
        if tok_id_str == "__residual__":
            if not skip_residual:
                rows.append(("__residual__", prob))
            continue
        token_text = tokenizer.decode([int(tok_id_str)])
        rows.append((token_text, prob))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def print_table(rows, header):
    col_w = max(len(repr(r[0])) for r in rows) + 2
    print(f"\n{'─' * (col_w + 16)}")
    print(f"  {header}")
    print(f"{'─' * (col_w + 16)}")
    print(f"  {'Token':<{col_w}}  Probability")
    print(f"  {'─'*col_w}  {'─'*12}")
    for token_text, prob in rows:
        display = repr(token_text)   # repr so whitespace/newlines are visible
        print(f"  {display:<{col_w}}  {prob:.6f}")
    print(f"{'─' * (col_w + 16)}")


def print_comparison(rows_a, header_a, rows_b, header_b):
    """Print two token lists side by side."""
    col = 22
    prob_col = 10
    sep = "    "
    print(f"\n  {'─'*(col*2 + prob_col*2 + len(sep) + 4)}")
    print(f"  {header_a}")
    print(f"  vs.")
    print(f"  {header_b}")
    print(f"  {'─'*(col*2 + prob_col*2 + len(sep) + 4)}")
    print(f"  {'Token':<{col}}  {'Prob':>{prob_col}}{sep}{'Token':<{col}}  {'Prob':>{prob_col}}")
    print(f"  {'─'*col}  {'─'*prob_col}{sep}{'─'*col}  {'─'*prob_col}")
    for (ta, pa), (tb, pb) in zip(rows_a, rows_b):
        da = repr(ta)
        db = repr(tb)
        print(f"  {da:<{col}}  {pa:>{prob_col}.6f}{sep}{db:<{col}}  {pb:>{prob_col}.6f}")
    print(f"  {'─'*(col*2 + prob_col*2 + len(sep) + 4)}")


def main():
    parser = argparse.ArgumentParser(description="Decode ground-truth token distributions.")
    parser.add_argument("path", help="Path to a ground-truth JSON file")
    parser.add_argument("--top", type=int, default=40, help="Number of top tokens to show")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-32B-Instruct",
        help="HuggingFace model id for the tokenizer",
    )
    parser.add_argument("--compare", default=None, metavar="PATH",
                        help="Second ground-truth file to show alongside the first")
    parser.add_argument("--no-skip-residual", dest="skip_residual",
                        action="store_false", default=True,
                        help="Also show the __residual__ mass bucket")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    gt_a = load_gt(args.path)
    rows_a = top_tokens(gt_a["distribution"], tokenizer, args.top, args.skip_residual)
    header_a = (
        f"{gt_a['category']}  |  layer {gt_a['layer']}  "
        f"|  segment: {gt_a['segment']}  |  n={gt_a['n_questions']} questions"
    )

    if args.compare:
        gt_b = load_gt(args.compare)
        rows_b = top_tokens(gt_b["distribution"], tokenizer, args.top, args.skip_residual)
        header_b = (
            f"{gt_b['category']}  |  layer {gt_b['layer']}  "
            f"|  segment: {gt_b['segment']}  |  n={gt_b['n_questions']} questions"
        )
        print_comparison(rows_a, header_a, rows_b, header_b)
    else:
        print_table(rows_a, header_a)


if __name__ == "__main__":
    main()
