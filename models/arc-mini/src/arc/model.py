from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ArcConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (freqs / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cached", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", angles.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seqlen > self.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds RoPE cache length {self.max_seq_len}")
        return self.cos_cached[:, :, :seqlen], self.sin_cached[:, :, :seqlen]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ArcConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(seqlen, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(bsz, seqlen, dim))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ArcConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionBlock(nn.Module):
    def __init__(self, width: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = nn.MultiheadAttention(width, n_heads, dropout=dropout, batch_first=True, bias=False)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, 4 * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: ArcConfig):
        super().__init__()
        patch_dim = 3 * cfg.patch_size * cfg.patch_size
        self.patch_size = cfg.patch_size
        self.patch_proj = nn.Linear(patch_dim, cfg.vision_width, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_image_patches, cfg.vision_width))
        self.blocks = nn.ModuleList(
            [VisionBlock(cfg.vision_width, cfg.vision_heads, cfg.dropout) for _ in range(cfg.vision_layers)]
        )
        self.norm = RMSNorm(cfg.vision_width)
        self.query = nn.Parameter(torch.randn(1, cfg.vision_resampler_tokens, cfg.vision_width) * 0.02)
        self.resampler = nn.MultiheadAttention(
            cfg.vision_width,
            cfg.vision_heads,
            dropout=cfg.dropout,
            batch_first=True,
            bias=False,
        )
        self.proj = nn.Linear(cfg.vision_width, cfg.d_model, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(images, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch_proj(patches) + self.pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        query = self.query.expand(images.shape[0], -1, -1)
        tokens, _ = self.resampler(query, x, x, need_weights=False)
        return self.proj(tokens)


class ArcModel(nn.Module):
    def __init__(self, cfg: ArcConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.vision = VisionEncoder(cfg)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._embedding_mask_hook = None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def clear_embedding_gradient_mask(self) -> None:
        if self._embedding_mask_hook is not None:
            self._embedding_mask_hook.remove()
            self._embedding_mask_hook = None

    def allow_only_embedding_rows(self, token_ids: list[int]) -> None:
        self.clear_embedding_gradient_mask()
        allowed = sorted({int(token_id) for token_id in token_ids if token_id >= 0})
        if not allowed:
            self.tok_emb.weight.requires_grad_(False)
            return
        self.tok_emb.weight.requires_grad_(True)

        def mask_grad(grad: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(grad)
            masked[allowed] = grad[allowed]
            return masked

        self._embedding_mask_hook = self.tok_emb.weight.register_hook(mask_grad)

    def freeze_all(self) -> None:
        self.clear_embedding_gradient_mask()
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze_all(self) -> None:
        self.clear_embedding_gradient_mask()
        for param in self.parameters():
            param.requires_grad_(True)

    def configure_stage2a_trainable(self, image_start_id: int, image_end_id: int) -> None:
        self.freeze_all()
        for param in self.vision.parameters():
            param.requires_grad_(True)
        self.allow_only_embedding_rows([image_start_id, image_end_id])

    def configure_stage2b_trainable(self, image_start_id: int, image_end_id: int, final_layers: int = 2) -> None:
        self.configure_stage2a_trainable(image_start_id, image_end_id)
        for block in self.blocks[-final_layers:]:
            for param in block.parameters():
                param.requires_grad_(True)
        for param in self.norm.parameters():
            param.requires_grad_(True)

    def configure_stage2c_trainable(self) -> None:
        self.unfreeze_all()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_insert_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(input_ids)
        if images is not None and image_insert_positions is not None:
            visual_tokens = self.vision(images)
            if bool((image_insert_positions == image_insert_positions[0]).all()) and int(image_insert_positions[0]) >= 0:
                pos = int(image_insert_positions[0])
                x[:, pos : pos + visual_tokens.shape[1]] = visual_tokens
            else:
                for i, pos in enumerate(image_insert_positions.tolist()):
                    if pos >= 0:
                        end = pos + visual_tokens.shape[1]
                        x[i, pos:end] = visual_tokens[i]

        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                labels.contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_parameters(cfg: ArcConfig) -> dict[str, int]:
    model = ArcModel(cfg)
    total = count_parameters(model)
    vision = count_parameters(model.vision)
    return {
        "total": total,
        "text_plus_embeddings": total - vision,
        "vision": vision,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
