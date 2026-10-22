"""A small decoder-only (causal) Transformer, trained from scratch, for the
modular-arithmetic grokking line. No pretrained weights, no HF `transformers`
dependency -- this is exactly the kind of tiny model (Nanda et al. 2023-style)
that grokking papers use, reimplemented directly on top of torch.nn so the
whole forward pass fits on one screen and is easy to audit.

Sequences are short (<= modp_common.MAX_LEN = 10 tokens) and always
right-padded; because attention is causal, padding at the end never
contaminates any real position, so no padding mask is needed anywhere in
this file (see modp_common.encode_row).
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int
    max_len: int = 10
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                  nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.cfg.max_len, (T, self.cfg.max_len)
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, n_new: int) -> torch.Tensor:
        """Greedy decode `n_new` tokens after `prompt_ids` (B, prompt_len).
        Returns (B, n_new) of argmax token ids, one full forward pass per
        new token (n_new is always 1 or 2 here, so this is cheap)."""
        cur = prompt_ids
        out = []
        for _ in range(n_new):
            logits = self(cur)[:, -1, :]
            nxt = logits.argmax(dim=-1)
            out.append(nxt)
            cur = torch.cat([cur, nxt[:, None]], dim=1)
        return torch.stack(out, dim=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
