"""Reorders stream_categories to the CVD-validated adjacency (see chat: the
category-list order fixes which --hueN color a band gets, and the original
project order put red next to orange, which failed the accessibility
palette validator's normal-vision-floor check for this 7-color legend) and
embeds benchmark_viz_data.json into benchmark_viz_template.html, producing
the final self-contained benchmark_viz.html."""
import json

VALIDATED_ORDER = [
    "Religion & Theology",
    "Depression, Addiction & Feeling Lost",
    "Human Nature & Philosophy",
    "Marriage & Romantic Partnerships",
    "Applied Ethics & Moral Dilemmas",
    "Grief, Loss & Death",
    "Family Duty & Obligations",
]

with open("benchmark_viz_data.json") as f:
    data = json.load(f)

assert set(data["stream_categories"]) == set(VALIDATED_ORDER), "category set mismatch"
data["stream_categories"] = VALIDATED_ORDER

with open("benchmark_viz_template.html") as f:
    template = f.read()

# Guard against a question's text containing a literal "</script>" and
# prematurely closing the inline <script> tag on some future re-run.
data_json = json.dumps(data).replace("</", "<\\/")
out = template.replace("/*__DATA__*/", data_json)
with open("benchmark_viz.html", "w") as f:
    f.write(out)
print(f"Wrote benchmark_viz.html ({len(out) / 1e6:.2f} MB)")
