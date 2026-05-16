import re
import torch

ANSWER_RE = re.compile(r"####\s*(-?\d+)")

def extract_answer(text: str) -> int | None:
    """Pull the integer after '####' out of a generation or gold answer.
    Returns None if not found."""
    m = ANSWER_RE.search(text)
    return int(m.group(1)) if m else None

def evaluate(model, tokenizer, test_data, max_new_tokens=256):
    model.eval()                                  # disable dropout etc.
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
