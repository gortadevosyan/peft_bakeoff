from datasets import load_dataset

def load_gsm8k(partition):
    gsm8k = load_dataset("openai/gsm8k", "main")
    data = gsm8k[partition]
    return data


def preprocess_example(example, tokenizer, max_length=512):
    """example: dict{"question": '...', 'answer': '...'}"""

    prompt = f"Question: {example['question']}\nAnswer: {example['answer']}" + tokenizer.eos_token
    prefill = f"Question: {example['question']}"

    prefill_len = len(tokenizer(prefill)["input_ids"])

    tokenized = tokenizer(prompt)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    labels = list(input_ids)
    labels[:prefill_len] = [-100] * prefill_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}