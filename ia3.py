import torch
import torch.nn as nn


class Ia3Linear(nn.Module):
    def __init__(self, base_linear: nn.Linear, is_feedforward: bool = False):
        super().__init__()
        self.base = base_linear
        self.is_feedforward = is_feedforward
        # Attention K/V: scale OUTPUT (size = out_features).
        # FFN down_proj: scale INPUT  (size = in_features) — this is the IA3 lff vector.
        size = base_linear.in_features if is_feedforward else base_linear.out_features
        self.l = nn.Parameter(torch.ones(
            size,
            dtype=base_linear.weight.dtype,
            device=base_linear.weight.device,
        ))

    def forward(self, x):
        if self.is_feedforward:
            return self.base(x * self.l)
        return self.base(x) * self.l


def apply_ia3(
    model,
    target_modules=("k_proj", "v_proj"),     
    feedforward_modules=("down_proj",),
):
    """Walk model tree, replace matching nn.Linear layers with Ia3Linear."""
    all_targets = set(target_modules) | set(feedforward_modules)
    for _, module in model.named_modules():
        for attr in all_targets:
            if hasattr(module, attr):
                old = getattr(module, attr)
                if isinstance(old, nn.Linear):
                    is_ff = attr in feedforward_modules
                    setattr(module, attr, Ia3Linear(old, is_feedforward=is_ff))
    return model
