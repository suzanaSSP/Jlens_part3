import gzip
import json
import os
import torch
import jlens

OUTPUT_DIR = "jspace_prompts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# 1. Load Model, Tokenizer, and J-Lens (once, reused for every prompt below)
# ---------------------------------------------------------
model_id = "Qwen/Qwen2.5-32B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16, device_map="auto"
)

# Wrap the loaded HF model so the lens can locate its residual blocks / norm / lm_head
lens_model = jlens.from_hf(model, tokenizer)

# Load pre-fitted Jacobian Lens for Qwen (I trained them hehehe)
lens = jlens.JacobianLens.from_pretrained(
    ".", filename="Qwen_Qwen2.5-32B-Instruct_final_lens.pt"
)

system_instruction = (
    "Before answering, reflect on the user's situation.\n"
    "Put your step-by-step reasoning inside <thought>...</thought> tags, "
    "and your final response inside <answer>...</answer> tags."
)

TOP_K_VAL = 20

# ---------------------------------------------------------
# 2. Prompts to process -- add as many as you like
# ---------------------------------------------------------
from scripts.prompts import PROMPTS
from tqdm import tqdm


def find_token_offset(full_text, substring, tokenizer, search_start=0):
    char_idx = full_text.find(substring, search_start)
    if char_idx == -1:
        return None
    prefix_text = full_text[:char_idx]
    return len(tokenizer.encode(prefix_text, add_special_tokens=False))


def process_prompt(raw_prompt, prompt_id, prompt_category):
    # -------------------------------------------------------
    # Prompt Setup & Generation
    # -------------------------------------------------------
    formatted_prompt = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
    )

    full_tokens = generated_ids[0]
    full_text = tokenizer.decode(full_tokens)

    # -------------------------------------------------------
    # Locate Tag Boundaries
    # -------------------------------------------------------
    # The system prompt's instructions literally contain the strings
    # "<thought>", "</thought>", "<answer>" as an example, so searching from
    # char 0 finds those instead of the real tags in the assistant's
    # response. Only search after the assistant's turn actually starts.
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

    # -------------------------------------------------------
    # Extract Hidden States across Entire Sequence
    # -------------------------------------------------------
    with torch.no_grad():
        outputs = model(generated_ids, output_hidden_states=True)

    # hidden_states: tuple of length (Num_Layers + 1), shape: (1, Seq_Len, Hidden_Dim)
    all_hidden_states = outputs.hidden_states
    seq_len = full_tokens.shape[0]

    # -------------------------------------------------------
    # Apply J-Lens & Extract Top-K Tokens per Position
    # -------------------------------------------------------
    layers_data = []

    for l in lens.source_layers:
        # all_hidden_states[0] is the embedding output; the output of block
        # `l` (what the lens was fitted against) is all_hidden_states[l + 1].
        h_l = all_hidden_states[l + 1][0, :, :].float()  # (Seq_Len, Hidden_Dim)

        # Run Jacobian Lens transformation at layer l: transport into the
        # final-layer basis, then unembed (final norm + lm_head)
        transported = lens.transport(h_l, l)
        j_logits = lens_model.unembed(transported).float()  # (Seq_Len, Vocab_Size)
        j_probs = torch.softmax(j_logits, dim=-1)  # (Seq_Len, Vocab_Size)

        # Extract Top-K values along vocabulary dimension
        top_values, top_indices = torch.topk(j_probs, k=TOP_K_VAL, dim=-1)
        top_logits, _ = torch.topk(j_logits, k=TOP_K_VAL, dim=-1)

        layer_tokens_records = []

        # Loop continuously over the sequence timeline (t)
        for t in range(seq_len):
            # Assign segment tags
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

    # -------------------------------------------------------
    # Build Final JSON Record & Save Compressed Output
    # -------------------------------------------------------
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

    # Save as compressed .json.gz file to minimize disk footprint
    output_filepath = os.path.join(OUTPUT_DIR, f"jlens_output_prompt_{prompt_id}.json.gz")
    with gzip.open(output_filepath, "wt", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Successfully processed prompt {prompt_id} and saved to {output_filepath}")


pbar = tqdm(PROMPTS, desc="Processing prompts")
for p in pbar:
    pbar.set_postfix(id=p["id"], category=p["category"][:20])
    output_filepath = os.path.join(OUTPUT_DIR, f"jlens_output_prompt_{p['id']}.json.gz")
    if os.path.exists(output_filepath):
        print(f"Skipping prompt {p['id']}: {output_filepath} already exists")
        continue

    try:
        process_prompt(p["text"], p["id"], p["category"])
    except Exception as e:
        print(f"[{p['id']}] Failed: {e!r} -- skipping to next prompt")
