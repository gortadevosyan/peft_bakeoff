import os
os.environ["WANDB_PROJECT"] = "peft-bakeoff"

from transformers import AutoModelForCausalLM, AutoTokenizer
import data_loader
import eval
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments

model_name = 'Qwen/Qwen2.5-1.5B'


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    # device_map='auto',
)

model = model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

train_data = data_loader.load_gsm8k("train")
test_data_full = data_loader.load_gsm8k("test")

# Two test slices:
#   - eval_test_slice: used by Trainer during training for fast eval_loss
#   - final_test_slice: used after training for accuracy via generation (slow)
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

args = TrainingArguments(
    output_dir="runs/full_ft_baseline",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,         # effective batch = 8
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=250,                        # save_steps must be a multiple of eval_steps
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,                    # keep last 1; best is preserved by load_best_model_at_end
    save_only_model=True,                  # skip optimizer states to save disk
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    report_to="wandb",
    run_name="full_ft_baseline_v1",
)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds,
                  eval_dataset=eval_ds,
                  data_collator=collator,
                  processing_class=tokenizer)

trainer.train()

print("post train — running detailed eval on full 1319-example test set (~30-60 min)...")
acc = eval.evaluate_and_dump(
    model, tokenizer, final_test_slice,
    output_path="runs/full_ft_baseline/inference_results.jsonl",
)
print(f"final accuracy: {acc:.3f}")