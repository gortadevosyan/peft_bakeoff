import re
import os
import json
import torch

ANSWER_RE = re.compile(r"####\s*(-?[\d,]+)")
FALLBACK_RE = re.compile(r"[Tt]he answer is\s*\$?\s*(-?[\d,]+)")

def extract_answer(text: str) -> int | None:
    """Pull the final numeric answer from a generation or gold answer.
    Tries '#### N' first (GSM8K gold format), then 'The answer is N' as fallback."""
    m = ANSWER_RE.search(text)
    if m:
        return int(m.group(1).replace(",", ""))
    m = FALLBACK_RE.search(text)
    if m:
        return int(m.group(1).replace(",", ""))
    return None

def evaluate(model, tokenizer, test_data, max_new_tokens=256):
    model.eval()                                 
    correct = 0
    total = 0
    for example in test_data:
        # 1. Build the prefill — SAME format you trained on, minus the answer.
        prefill = f"Question: {example['question']}\nAnswer:"
        
        # 2. Tokenize and move to GPU. Single example -> batch dim of 1.
        inputs = tokenizer(prefill, return_tensors="pt").to(model.device)
        
        # 3. Generate. Greedy decoding so it's reproducible and fast.
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                  # greedy
                pad_token_id=tokenizer.eos_token_id,  # silences a warning
            )
        
        # 4. out_ids contains the WHOLE thing: prefill + generation.
        #    Slice off the prefill so we only decode the new tokens.
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        # 5. Parse predicted number and gold number.
        pred = extract_answer(gen_text)
        gold = extract_answer(example["answer"])
        
        # 6. Compare. Both have to be present and equal.
        if pred is not None and pred == gold:
            correct += 1
        total += 1
    
    model.train()                                 # back to training mode
    return correct / total


def evaluate_and_dump(model, tokenizer, test_data, output_path, max_new_tokens=256):
    """Run accuracy eval AND save per-example results to a JSONL file.
    Use this for final post-training eval — slower than evaluate() because it
    records every example. Each line of the JSONL is one example's full result.
    """
    model.eval()
    correct = 0
    total = 0
    results = []

    for example in test_data:
        prefill = f"Question: {example['question']}\nAnswer:"
        inputs = tokenizer(prefill, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        pred = extract_answer(gen_text)
        gold = extract_answer(example["answer"])
        is_correct = pred is not None and pred == gold

        results.append({
            "question": example["question"],
            "gold_answer": example["answer"],
            "gold_number": gold,
            "generated_text": gen_text,
            "predicted_number": pred,
            "correct": is_correct,
        })

        if is_correct:
            correct += 1
        total += 1

    acc = correct / total
    results.insert(0, {"accuracy":acc})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    model.train()
    return acc
