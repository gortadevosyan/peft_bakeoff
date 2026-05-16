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
test_data = data_loader.load_gsm8k("test").select(range(20))

print("pre train")
print(eval.evaluate(model, tokenizer, test_data))

train_ds = train_data.map(
    lambda ex: data_loader.preprocess_example(ex, tokenizer),
    remove_columns=train_data.column_names,
)

args = TrainingArguments(
    output_dir="runs/smoketest",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_steps=50,
    learning_rate=2e-5,            # standard for full FT of small LMs
    logging_steps=5,
    save_strategy="no",
    bf16=True,                     # A6000 supports bf16; this halves memory
    report_to="none",              # no wandb yet
)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds.select(range(64)),
                  data_collator=collator,
                  processing_class=tokenizer)

trainer.train()


print("post train")
print(eval.evaluate(model, tokenizer, test_data))