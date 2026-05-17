import os
os.environ["WANDB_PROJECT"] = "peft-bakeoff"

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import data_loader
import eval
import utils
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="full_ft",
                    choices=["full_ft", "lora", "ia3", "prefix", "qlora"])
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lr", type=float, default=None)
args_cli = parser.parse_args()

METHOD = args_cli.method
RUN_DIR = f"runs/{METHOD}"
RUN_NAME = f"{METHOD}_v1"
LR = args_cli.lr or utils.LR[METHOD]

model_name = 'Qwen/Qwen2.5-1.5B'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
)

# ── PEFT surgery (before moving to GPU, so new params are on same device) ──
if METHOD == "lora":
    from lora import apply_lora, freeze_model
    freeze_model(model)
    model = apply_lora(model, rank=args_cli.lora_rank, alpha=args_cli.lora_alpha)
elif METHOD == "ia3":
    from lora import freeze_model
    from ia3 import apply_ia3
    freeze_model(model)
    model = apply_ia3(model)
elif METHOD == "prefix":
    raise NotImplementedError("prefix")
elif METHOD == "qlora":
    raise NotImplementedError("qlora")

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

trainer.train()

print("post train — running detailed eval on full 1319-example test set (~30-60 min)...")
acc = eval.evaluate_and_dump(
    model, tokenizer, final_test_slice,
    output_path=f"{RUN_DIR}/inference_results.jsonl",
)
print(f"final accuracy: {acc:.3f}")