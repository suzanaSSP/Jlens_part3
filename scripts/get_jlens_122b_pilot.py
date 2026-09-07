"""Timed pilot run of the Qwen3.5-122B-A10B-FP8 J-Lens pipeline.

Runs the first N (default 2) religion-ground-truth training questions
through the model, extracting the same per-layer top-K J-Lens output as
get_jlens.py, but instrumented with wall-clock timing (model load,
per-prompt, total) so we can extrapolate to the full 20-question religious
ground-truth batch before committing to it.

GPU selection: before touching CUDA, queries `nvidia-smi` for per-GPU free
memory / utilization and restricts CUDA_VISIBLE_DEVICES to the least-loaded
ones (see --min-free-gb / --max-util-pct), so this run doesn't pile onto
GPUs another job is already using. Also skips any GPU marked unhealthy by a
`DONT_USE_GPU_<n>_ON_<HOST>` marker file next to the models dir (e.g. GPU 0
on this host has known ECC errors) -- nvidia-smi alone won't surface that.
Must happen before `import torch`.

CPU offload budget: accelerate's `max_memory["cpu"]` is sized off *actually
free* RAM at start (via psutil), minus a safety reserve, instead of a fixed
guess -- this machine is shared, and a previous run stalling for days with
swap fully saturated suggests the old fixed 200GB budget was oversubscribing
real headroom.

Usage:
    python -u -m scripts.get_jlens_122b_pilot [--n 2] [--out-dir jspace_prompts_122b]

Run with `python -u` (or PYTHONUNBUFFERED=1) and redirect to a log file to
watch progress live, e.g.:
    .venv_jl3/bin/python3 -u -m scripts.get_jlens_122b_pilot --n 1 \
        >> run_get_jlens_122b_pilot.log 2>&1
"""

import argparse
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import threading
import time


def _excluded_gpus_from_markers(models_dir: str = "/mnt/pccfs2/backed_up/models") -> dict[int, str]:
    """Return {gpu_index: reason} for GPUs marked unhealthy on this host via
    a `DONT_USE_GPU_<n>_ON_<HOST>` marker file (content is the reason)."""
    host = os.uname().nodename.upper()
    excluded = {}
    for path in glob.glob(os.path.join(models_dir, "DONT_USE_GPU_*_ON_*")):
        m = re.match(r"DONT_USE_GPU_(\d+)_ON_(.+)", os.path.basename(path))
        if m and m.group(2).upper() == host:
            try:
                reason = open(path).read().strip()
            except OSError:
                reason = "(marker file unreadable)"
            excluded[int(m.group(1))] = reason
    return excluded


def _select_gpus(min_free_gb: float, max_util_pct: float) -> list[int]:
    """Return indices of GPUs with enough free memory and low enough
    utilization, most-free-first. Sets CUDA_VISIBLE_DEVICES accordingly."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout

    rows = []
    for line in out.strip().splitlines():
        idx, free_mb, total_mb, util_pct = (x.strip() for x in line.split(","))
        rows.append((int(idx), int(free_mb), int(total_mb), int(util_pct)))

    marker_excluded = _excluded_gpus_from_markers()
    eligible = [
        r for r in rows
        if r[0] not in marker_excluded and r[1] >= min_free_gb * 1024 and r[3] <= max_util_pct
    ]
    eligible.sort(key=lambda r: r[1], reverse=True)  # most free memory first

    print("GPU status (index, free/total MiB, util%):")
    for idx, free_mb, total_mb, util_pct in rows:
        if idx in marker_excluded:
            tag = f"EXCLUDED ({marker_excluded[idx]})"
        elif any(idx == e[0] for e in eligible):
            tag = "SELECTED"
        else:
            tag = "skipped (busy)"
        print(f"  gpu{idx}: {free_mb}/{total_mb} MiB free, {util_pct}% util  -> {tag}")

    if not eligible:
        print(
            f"No GPUs met the bar (>= {min_free_gb} GB free, <= {max_util_pct}% util, "
            "not marked unhealthy). Refusing to guess -- pass --min-free-gb / "
            "--max-util-pct to relax."
        )
        sys.exit(1)

    total_free_gb = sum(r[1] for r in eligible) / 1024
    print(f"Selected {len(eligible)} GPU(s): {[r[0] for r in eligible]} "
          f"(~{total_free_gb:.0f} GB free total)")
    return [r[0] for r in eligible]


def _parse_gpu_args(argv: list[str]) -> tuple[float, float]:
    """Peek --min-free-gb / --max-util-pct out of argv before the real
    argparse pass, since GPU selection must happen before `import torch`."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--min-free-gb", type=float, default=28.0)
    p.add_argument("--max-util-pct", type=float, default=5.0)
    known, _ = p.parse_known_args(argv)
    return known.min_free_gb, known.max_util_pct


_min_free_gb, _max_util_pct = _parse_gpu_args(sys.argv[1:])
_selected_gpus = _select_gpus(_min_free_gb, _max_util_pct)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _selected_gpus)
# The checkpoint is FP8; V100 (compute capability 7.0) has no FP8 tensor
# cores, and transformers' FineGrainedFP8 quantizer hardcodes "dequantize to
# bf16" for compute capability < 8.9 with no override -- passing a different
# quantization_config (e.g. bitsandbytes) is silently ignored since the
# checkpoint's own config.json quantization_config takes precedence. So the
# ~119GB on-disk footprint becomes ~240GB in memory, plus a transient
# per-layer buffer during the fp8->bf16 conversion itself. That leaves ~0
# headroom across 32GB GPUs for activations/KV-cache. Capping per-GPU usage
# and letting accelerate overflow the rest onto CPU RAM (see max_memory
# below, sized off actually-free RAM at runtime) trades speed for actually
# completing.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

MODEL_PATH = "/mnt/pccfs2/backed_up/models/Qwen3.5-122B-A10B-FP8"
LENS_PATH = "trained_jlens/qwen35_122b_bf16_lens_fp32.pt"
GROUND_TRUTH_PATH = "data/ground_truths/religion_theology.json"
QUESTION_POOL_PATH = "data/prompts_json/non_benchmark_961.json"

TOP_K_VAL = 20
MAX_NEW_TOKENS = 512

# Reserve this much RAM for the rest of the (shared) machine when sizing the
# accelerate CPU-offload budget, rather than assuming a fixed pool is free.
CPU_RESERVE_GIB = 60.0


class _Heartbeat:
    """Background thread that prints elapsed time every `interval` seconds
    while a long blocking call (e.g. model load) is in flight, so a stalled
    run is visible in the log instead of going silent for hours/days."""

    def __init__(self, label: str, interval: float = 30.0):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - self._t0
            vm = psutil.virtual_memory()
            print(
                f"  ...[{self.label}] still running after {elapsed:.0f}s "
                f"(RAM available: {vm.available / 1024**3:.1f} GiB, "
                f"swap used: {psutil.swap_memory().used / 1024**3:.1f} GiB)",
                flush=True,
            )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)

system_instruction = (
    "Before answering, reflect on the user's situation.\n"
    "Put your step-by-step reasoning inside <thought>...</thought> tags, "
    "and your final response inside <answer>...</answer> tags."
)


def load_pilot_prompts(n):
    ground_truth = json.load(open(GROUND_TRUTH_PATH))
    train_ids = ground_truth["train_ids"][:n]

    pool = json.load(open(QUESTION_POOL_PATH))
    by_id = {str(p["id"]).split(".")[0]: p for p in pool}

    prompts = []
    for tid in train_ids:
        p = by_id.get(tid)
        if p is None:
            raise KeyError(f"train_id {tid!r} not found in {QUESTION_POOL_PATH}")
        prompts.append({"id": tid, "category": p["category"], "text": p["text"]})
    return prompts


def find_token_offset(full_text, substring, tokenizer, search_start=0):
    char_idx = full_text.find(substring, search_start)
    if char_idx == -1:
        return None
    prefix_text = full_text[:char_idx]
    return len(tokenizer.encode(prefix_text, add_special_tokens=False))


def process_prompt(model, tokenizer, lens, lens_model, raw_prompt, prompt_id, prompt_category, out_dir):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": raw_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    full_tokens = generated_ids[0]
    full_text = tokenizer.decode(full_tokens)

    assistant_marker = "<|im_start|>assistant\n"
    assistant_turn_char = full_text.rfind(assistant_marker)
    search_start = (
        assistant_turn_char + len(assistant_marker)
        if assistant_turn_char != -1
        else 0
    )

    cot_start = find_token_offset(full_text, "<thought>", tokenizer, search_start)
    cot_end = find_token_offset(full_text, "</thought>", tokenizer, search_start)
    ans_start = find_token_offset(full_text, "<answer>", tokenizer, search_start)
    ans_end = len(full_tokens)

    if None in (cot_start, cot_end, ans_start):
        print(
            f"[{prompt_id}] Warning: Failed to locate explicit "
            "<thought>/<answer> tags in generation."
        )

    with torch.no_grad():
        outputs = model(generated_ids, output_hidden_states=True)

    all_hidden_states = outputs.hidden_states
    seq_len = full_tokens.shape[0]

    layers_data = []
    for l in lens.source_layers:
        h_l = all_hidden_states[l + 1][0, :, :].float()
        transported = lens.transport(h_l, l)
        j_logits = lens_model.unembed(transported).float()
        j_probs = torch.softmax(j_logits, dim=-1)

        top_values, top_indices = torch.topk(j_probs, k=TOP_K_VAL, dim=-1)
        top_logits, _ = torch.topk(j_logits, k=TOP_K_VAL, dim=-1)

        layer_tokens_records = []
        for t in range(seq_len):
            if cot_start is not None and cot_start <= t < cot_end:
                segment_type = "thought"
            elif ans_start is not None and ans_start <= t < ans_end:
                segment_type = "answer"
            else:
                segment_type = "prompt"

            top_k_list = []
            for rank in range(TOP_K_VAL):
                t_id = top_indices[t, rank].item()
                top_k_list.append({
                    "rank": rank + 1,
                    "token_id": t_id,
                    "token": tokenizer.decode([t_id]),
                    "logit": round(top_logits[t, rank].item(), 4),
                    "probability": round(top_values[t, rank].item(), 6),
                })

            layer_tokens_records.append({
                "token_index": t,
                "generated_token_id": full_tokens[t].item(),
                "generated_token": tokenizer.decode([full_tokens[t].item()]),
                "segment": segment_type,
                "top_k": top_k_list,
            })

        layers_data.append({"layer": l, "tokens": layer_tokens_records})

    record = {
        "id": prompt_id,
        "category": prompt_category,
        "text": raw_prompt,
        "full_generated_text": full_text,
        "boundaries": {
            "cot_start_idx": cot_start,
            "cot_end_idx": cot_end,
            "answer_start_idx": ans_start,
            "answer_end_idx": ans_end,
        },
        "jacobian_lens_per_layer": layers_data,
    }

    output_filepath = os.path.join(out_dir, f"jlens_output_prompt_{prompt_id}.json.gz")
    with gzip.open(output_filepath, "wt", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return output_filepath, seq_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, help="Number of religion train_ids to run")
    parser.add_argument("--out-dir", default="jspace_prompts_122b")
    parser.add_argument(
        "--min-free-gb", type=float, default=28.0,
        help="[consumed pre-import] Minimum free VRAM (GB) for a GPU to be used",
    )
    parser.add_argument(
        "--max-util-pct", type=float, default=5.0,
        help="[consumed pre-import] Maximum current utilization pct for a GPU to be used",
    )
    parser.add_argument(
        "--max-gpu-gib", type=float, default=28.0,
        help="Cap per-GPU weight placement (GiB), leaving headroom for the fp8->bf16 "
             "conversion buffer, KV-cache, and hidden-state extraction activations",
    )
    parser.add_argument(
        "--cpu-gib", type=float, default=None,
        help="RAM budget (GiB) accelerate may use to offload weights that don't fit on GPU. "
             f"Default: actual free RAM at start minus a {CPU_RESERVE_GIB:.0f} GiB reserve "
             "for the rest of this shared machine (this machine has had swap saturate and "
             "stall for days when a fixed 200GiB guess oversubscribed real headroom).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = load_pilot_prompts(args.n)
    print(f"Pilot prompts ({len(prompts)}): {[p['id'] for p in prompts]}")

    if args.cpu_gib is None:
        available_gib = psutil.virtual_memory().available / 1024**3
        cpu_gib = max(0.0, available_gib - CPU_RESERVE_GIB)
        print(
            f"RAM available now: {available_gib:.1f} GiB -> cpu offload budget "
            f"{cpu_gib:.1f} GiB (reserving {CPU_RESERVE_GIB:.0f} GiB for the rest "
            "of the machine; pass --cpu-gib to override)"
        )
    else:
        cpu_gib = args.cpu_gib

    max_memory = {i: f"{args.max_gpu_gib}GiB" for i in range(len(_selected_gpus))}
    max_memory["cpu"] = f"{cpu_gib}GiB"
    print(f"max_memory: {max_memory}", flush=True)

    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Loading model (this is the step that has stalled before -- "
          "heartbeat below prints every 30s)...", flush=True)
    with _Heartbeat("model load"):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto", max_memory=max_memory,
        )
    offloaded = getattr(model, "hf_device_map", {})
    n_cpu_modules = sum(1 for v in offloaded.values() if v == "cpu")
    if n_cpu_modules:
        print(
            f"WARNING: {n_cpu_modules} module(s) offloaded to CPU RAM -- "
            "generation will be slower than a pure-GPU run, so per-prompt "
            "timing here is a pessimistic bound, not a clean extrapolation basis."
        )
    t_model_loaded = time.monotonic()
    print(f"Model load time: {t_model_loaded - t0:.1f}s")

    lens_model = jlens.from_hf(model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(LENS_PATH)
    t_lens_loaded = time.monotonic()
    print(f"Lens load time: {t_lens_loaded - t_model_loaded:.1f}s")
    print(f"Lens: {lens.source_layers[0]}..{lens.source_layers[-1]} "
          f"({len(lens.source_layers)} layers), d_model={lens.d_model}")

    per_prompt_times = []
    for p in prompts:
        output_filepath = os.path.join(args.out_dir, f"jlens_output_prompt_{p['id']}.json.gz")
        if os.path.exists(output_filepath):
            print(f"Skipping prompt {p['id']}: {output_filepath} already exists")
            continue

        t_start = time.monotonic()
        try:
            with _Heartbeat(f"prompt {p['id']}"):
                _, seq_len = process_prompt(
                    model, tokenizer, lens, lens_model,
                    p["text"], p["id"], p["category"], args.out_dir,
                )
            elapsed = time.monotonic() - t_start
            per_prompt_times.append({"id": p["id"], "seconds": elapsed, "seq_len": seq_len})
            print(f"[{p['id']}] done in {elapsed:.1f}s (seq_len={seq_len})")
        except Exception as e:
            elapsed = time.monotonic() - t_start
            print(f"[{p['id']}] FAILED after {elapsed:.1f}s: {e!r}")
            raise

    total_elapsed = time.monotonic() - t0
    summary = {
        "model_path": MODEL_PATH,
        "model_load_seconds": t_model_loaded - t0,
        "lens_load_seconds": t_lens_loaded - t_model_loaded,
        "per_prompt": per_prompt_times,
        "total_seconds": total_elapsed,
    }
    summary_path = os.path.join(args.out_dir, "pilot_timing.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if per_prompt_times:
        avg = sum(x["seconds"] for x in per_prompt_times) / len(per_prompt_times)
        print(f"\nAvg time/prompt (generation-only, excludes model load): {avg:.1f}s")
        print(f"Extrapolated to 20 prompts: {avg * 20 / 60:.1f} min "
              f"(+ {(t_model_loaded - t0) / 60:.1f} min one-time model load)")
    print(f"Timing summary written to {summary_path}")


if __name__ == "__main__":
    main()
