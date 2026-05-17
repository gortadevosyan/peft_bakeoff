import os
import json


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_config(run_dir, method, model_name, lr, trainable, total, cli_args):
    config = {
        "method": method,
        "model": model_name,
        "learning_rate": lr,
        "epochs": 3,
        "batch_size": 4,
        "grad_accum": 2,
        "bf16": True,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2),
    }

    if method == "lora":
        config["lora_rank"] = cli_args.lora_rank
        config["lora_alpha"] = cli_args.lora_alpha
        config["lora_targets"] = ["q_proj", "v_proj"]

    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config

LR = {
    "full_ft": 2e-5,   
    "lora":    2e-4,   
    "ia3":     5e-3,  
    "prefix":  2e-3,   
    "qlora":   2e-4,   
}
