"""
Turns data/ground_truth_pool.json's train_ids into prompts.py, in the format
get_jlens.py expects (PROMPTS list of {id, category, text}).

Order is preserved from ground_truth_pool.json's train_ids (the fixed
shuffle), so slicing PROMPTS[:10], PROMPTS[:20]... per category later gives
the correct nested batches. get_jlens.py already skips prompt ids whose
output file exists, so this can be regenerated / re-run safely.
"""
import json

with open("ground_truths/ground_truth_pool.json") as f:
    pool = json.load(f)

with open("non_benchmark_961.json") as f:
    all_questions = json.load(f)

by_id = {q["id"]: q for q in all_questions}

prompts = []
for category, ids in pool.items():
    for qid in ids["train_ids"]:
        q = by_id[qid]
        prompts.append({"id": q["id"], "category": q["category"], "text": q["text"]})

with open("../prompts.py", "w") as f:
    f.write("PROMPTS = ")
    json.dump(prompts, f, indent=2)
    f.write("\n")

print(f"Wrote ../prompts.py with {len(prompts)} prompts")
