# PEFT Bake-Off

A systematic comparison of parameter-efficient fine-tuning methods on Qwen-2.5-1.5B (Base) using GSM8K math word problems. All PEFT methods (LoRA, IA3, Prefix Tuning) are implemented from scratch as custom PyTorch modules — no HuggingFace `peft` library.

## Methods

- **Full Fine-Tuning** — all 1.5B parameters trained (baseline)
- **LoRA** — low-rank A/B matrices injected into Q/V attention projections
- **IA3** — learned scaling vectors on K/V outputs and FFN inputs
- **Prefix Tuning** — learned KV-cache prefixes prepended at every attention layer
- **QLoRA** — LoRA on a 4-bit NF4-quantized base (via bitsandbytes)

## Results

| Method        | Trainable Params | % of Model | GSM8K Accuracy |
|---------------|-----------------|------------|----------------|
| Full FT       | 1,543.7M        | 100.00%    | 58.0%          |
| LoRA (r=16)   | 2.2M            | 0.14%      | 60.3%          |
| IA3           | 0.27M           | 0.02%      | 59.7%          |
| Prefix Tuning | 0.23M           | 0.01%      | 60.9%*         |
| QLoRA (4-bit) | 2.2M            | 0.24%      | 54.4%          |

\*Prefix Tuning uses relaxed answer parsing (see [RESULTS.md](RESULTS.md) for details).

PEFT methods match or exceed full fine-tuning while training <0.3% of parameters. Full FT likely overfits on this small dataset. QLoRA trades ~6 points of accuracy for significantly lower memory via 4-bit quantization. Full results, ablations, and analysis in [RESULTS.md](RESULTS.md).

## Setup

```bash
conda create -n peft python=3.11
conda activate peft
# Install PyTorch for your CUDA version from https://pytorch.org
pip install transformers datasets accelerate wandb matplotlib
pip install bitsandbytes  # for QLoRA
```

## Usage

```bash
# Full fine-tuning
python -u train.py --method full_ft

# LoRA
python -u train.py --method lora --lora_rank 16

# IA3
python -u train.py --method ia3

# Prefix Tuning
python -u train.py --method prefix --prefix_tokens 16

# QLoRA
python -u train.py --method qlora

# Ablations
python -u train.py --method lora --lora_rank 32 --run_name lora_r32
python -u train.py --method lora --flash_attn --run_name lora_flash
```

Results are saved to `runs/<run_name>/` with config, checkpoints, and per-example evaluation output. Training metrics are logged to Weights & Biases.

## Project Structure

```
train.py            — main training script (method selection, Trainer loop)
lora.py             — LoraLinear module + apply_lora surgery + freeze_model
ia3.py              — Ia3Linear module + apply_ia3 surgery
prefix_tuning.py    — PrefixEncoder + forward/generate patching
data_loader.py      — GSM8K loading and preprocessing
eval.py             — generation-based accuracy evaluation
utils.py            — param counting, config saving, LR defaults
RESULTS.md          — full results tables and analysis
```
