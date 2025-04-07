import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from .layer import Block, RMSNorm, precompute_rope_emb
from .utils import check_tuple


class ExtractCls(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x[:, 0]


class VIT(nn.Module):

    def __init__(self,
                 # common
                 img_shape: int = 224,
                 patch_size: int = 32,
                 channel: int = 3,
                 # mla
                 dim: int = 512,
                 inter_dim: int = 1344,
                 n_head: int = 8,
                 q_lora_rank: int = 0,
                 kv_lora_rank: int = 128,
                 qk_nope_head_dim: int = 32,
                 qk_rope_head_dim: int = 16,
                 v_head_dim: int = 32,
                 # Transformer
                 n_layer: int = 8,
                 # out
                 num_classes: int = 1000
                 ):
        super().__init__()

        self.img_h, self.img_w = check_tuple(img_shape)
        self.p_h, self.p_w = check_tuple(patch_size)
        self.seq_len = (self.img_h // self.p_h) * (self.img_w // self.p_w) + 1
        self.patch_dim = channel * self.p_h * self.p_w
        self.num_classes = num_classes

        self.to_patch = nn.Sequential(
            Rearrange('b c (h p_h) (w p_w) -> b (h w) (c p_h p_w)', p_h = self.p_h, p_w = self.p_w),
            RMSNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.ModuleList([
            Block(dim = dim, inter_dim = inter_dim, n_head = n_head, q_lora_rank = q_lora_rank,
                  kv_lora_rank = kv_lora_rank, qk_nope_head_dim = qk_nope_head_dim, qk_rope_head_dim = qk_rope_head_dim,
                  v_head_dim = v_head_dim, attn_name = 'mla', mlp_name = 'mlp')
            for _ in range(n_layer)
        ])
        self.to_out = nn.Sequential(
            ExtractCls(),
            nn.RMSNorm(dim),
            nn.Linear(dim, self.num_classes)
        ) if num_classes is not None else ExtractCls()

        self.register_buffer('rope_emb',
                             precompute_rope_emb(dim = qk_rope_head_dim, seq_len = self.seq_len),
                             persistent=False
                             )

    def forward(self, x: torch.Tensor):
        # x:[b, c, h, w]
        b, _, _, _ = x.shape
        x = self.to_patch(x)

        # transformer
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_token, x), dim = 1)
        for layer in self.transformer:
            x, _ = layer(x, start_pos = 0, rope_emb = self.rope_emb)
        out = self.to_out(x)
        return out
