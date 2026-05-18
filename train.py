import os
import torch
os.environ["WANDB_PROJECT"] = "peft-bakeoff"

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import data_loader
import eval
import utils
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
from lora import apply_lora, freeze_model

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="full_ft",
                    choices=["full_ft", "lora", "ia3", "prefix", "qlora"])
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--prefix_tokens", type=int, default=16)
parser.add_argument("--flash_attn", action="store_true")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--run_name", type=str, default=None)
args_cli = parser.parse_args()

METHOD = args_cli.method
RUN_NAME = args_cli.run_name or f"{METHOD}_v1"
RUN_DIR = f"runs/{RUN_NAME}"
LR = args_cli.lr or utils.LR[METHOD]

model_name = 'Qwen/Qwen2.5-1.5B'
attn_impl = "flash_attention_2" if args_cli.flash_attn else None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    attn_implementation=attn_impl,
)

# ── PEFT surgery (before moving to GPU, so new params are on same device) ──
if METHOD == "lora":
    freeze_model(model)
    model = apply_lora(model, rank=args_cli.lora_rank, alpha=args_cli.lora_alpha)
elif METHOD == "ia3":
    from lora import freeze_model
    from ia3 import apply_ia3
    freeze_model(model)
    model = apply_ia3(model)
elif METHOD == "prefix":
    from prefix_tuning import apply_prefix_tuning
    freeze_model(model)
    model = apply_prefix_tuning(model, num_prefix_tokens=args_cli.prefix_tokens)
elif METHOD == "qlora":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",                    # NormalFloat-4, the QLoRA paper's default
        bnb_4bit_compute_dtype=torch.bfloat16,        # matmuls run in bf16, weights stored 4-bit
        bnb_4bit_use_double_quant=True,               # quantize the quant constants too; ~0.4 bit/param savings
    )
    # Re-load the model in 4-bit (this REPLACES the load above)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    freeze_model(model)
    model = apply_lora(model, rank=args_cli.lora_rank, alpha=args_cli.lora_alpha)
    model._hf_peft_config_loaded = True


model = model.to('cuda')

trainable, total = utils.count_params(model)
print(f"[{METHOD}] trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
utils.save_config(RUN_DIR, METHOD, model_name, LR, trainable, total, args_cli)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

train_data = data_loader.load_gsm8k("train")
test_data_full = data_loader.load_gsm8k("test")

eval_test_slice  = test_data_full.select(range(300))  # 300 examples for fast during-training eval_loss
final_test_slice = test_data_full                      # full 1319-example test set for headline accuracy

print("pre train (quick 50-example accuracy check)")
print(eval.evaluate(model, tokenizer, final_test_slice.select(range(50))))

train_ds = train_data.map(
    lambda ex: data_loader.preprocess_example(ex, tokenizer),
    remove_columns=train_data.column_names,
)
eval_ds = eval_test_slice.map(
    lambda ex: data_loader.preprocess_example(ex, tokenizer),
    remove_columns=eval_test_slice.column_names,
)

training_args = TrainingArguments(
    output_dir=RUN_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    save_only_model=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    report_to="wandb",
    run_name=RUN_NAME,
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_ds,
                  eval_dataset=eval_ds,
                  data_collator=collator,
                  processing_class=tokenizer)

# Flag was set for QLoRA to pass Trainer's quantization validation;
# remove it now so checkpoint saving doesn't try HF PEFT save logic.
model._hf_peft_config_loaded = False

trainer.train()

print("post train — running detailed eval on full 1319-example test set (~30-60 min)...")
acc = eval.evaluate_and_dump(
    model, tokenizer, final_test_slice,
    output_path=f"{RUN_DIR}/inference_results.jsonl",
)
print(f"final accuracy: {acc:.3f}")