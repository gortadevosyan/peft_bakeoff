import torch
import torch.nn as nn
from transformers import DynamicCache


class PrefixEncoder(nn.Module):
    def __init__(self, num_layers, num_kv_heads, head_dim, num_prefix_tokens, dtype):
        super().__init__()
        self.num_layers = num_layers
        self.num_prefix_tokens = num_prefix_tokens
        # [num_layers, 2(k+v), num_kv_heads, num_prefix_tokens, head_dim]
        self.prefix = nn.Parameter(torch.randn(
            num_layers, 2, num_kv_heads, num_prefix_tokens, head_dim, dtype=dtype,
        ) * 0.02)

    def forward(self, batch_size):
        cache = DynamicCache()
        for i in range(self.num_layers):
            k = self.prefix[i, 0].unsqueeze(0).expand(batch_size, -1, -1, -1)
            v = self.prefix[i, 1].unsqueeze(0).expand(batch_size, -1, -1, -1)
            cache.update(k.contiguous(), v.contiguous(), i)
        return cache


def apply_prefix_tuning(model, num_prefix_tokens=16):
    """Freeze base model, attach learned KV prefixes to every attention layer.

    Works by monkey-patching model.forward and model.generate so that:
      - Training: prefix past_key_values are injected, attention mask extended.
      - Generation step 1: generate() extends the mask upfront; forward injects the prefix KV.
      - Generation step 2+: prefix is already in the KV cache, nothing extra needed.
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    n = num_prefix_tokens

    encoder = PrefixEncoder(num_layers, num_kv_heads, head_dim, n, model.dtype)
    model.prefix_encoder = encoder

    _orig_forward = model.forward
    _orig_generate = model.generate

    def forward(input_ids=None, attention_mask=None, past_key_values=None, labels=None, **kw):
        if past_key_values is None:
            bs = input_ids.shape[0]
            past_key_values = model.prefix_encoder(bs)
            # Extend mask only if not already extended by generate() wrapper
            if attention_mask is not None and attention_mask.shape[1] == input_ids.shape[1]:
                attention_mask = torch.cat(
                    [attention_mask.new_ones(bs, n), attention_mask], dim=1)
        return _orig_forward(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, labels=labels, **kw)

    def generate(input_ids=None, attention_mask=None, **kw):
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask.new_ones(attention_mask.shape[0], n), attention_mask], dim=1)
        return _orig_generate(input_ids=input_ids, attention_mask=attention_mask, **kw)

    model.forward = forward
    model.generate = generate
    return model
