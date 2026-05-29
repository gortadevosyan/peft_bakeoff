import torch
import torch.nn as nn

class LoraLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: int=16):
        super().__init__()
        self.base = base_linear
        fin  = base_linear.in_features
        fout = base_linear.out_features
        # can't carry gradients. Fall back to bfloat16 for the trainable LoRA matrices.
        base_dtype = self.base.weight.dtype
        lora_dtype = base_dtype if base_dtype.is_floating_point else torch.bfloat16
        device = self.base.weight.device
        self.a = nn.Linear(fin, rank, bias=False, dtype=lora_dtype, device=device)
        self.b = nn.Linear(rank, fout, bias=False, dtype=lora_dtype, device=device)
        self.scale = alpha / rank
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.b(self.a(x)) 



def apply_lora(model, rank=16, alpha=16, target_modules=("q_proj", "v_proj")):
    """Walk model tree, replace matching nn.Linear layers with LoraLinear."""
    for name, module in model.named_modules():
        for attr in target_modules:
            if hasattr(module, attr):
                old = getattr(module, attr)
                if isinstance(old, nn.Linear):
                    setattr(module, attr, LoraLinear(old, rank=rank, alpha=alpha))
    return model

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False