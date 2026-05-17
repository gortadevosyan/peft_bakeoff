import torch
import torch.nn as nn

class LoraLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: int=16):
        super().__init__()
        self.base = base_linear
        fout, fin = base_linear.weight.shape
        self.a = nn.Linear(fin, rank, bias=False, dtype=self.base.weight.dtype)
        self.b = nn.Linear(rank, fout, bias=False, dtype=self.base.weight.dtype)
        self.scale = alpha / rank
        nn.init.zeros_(self.b.weight)

        self.a.requires_grad_(True)
        self.b.requires_grad_(True)

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