"""P-JiT 21K's DiT (Linum-AI/pjit-21k, pjit_21k/model.py) in MLX.

The reference runs fp32 master weights under CUDA bf16 autocast, and every rounding point here follows that flow:
  - Linear / Conv layers take bf16 inputs and bf16 weights (autocast casts both) and return bf16;
  - RMSNorm normalises in fp32, rounds to the input dtype, then multiplies by the fp32 gain (so it returns fp32);
  - LayerNorm (the output head) runs in fp32 (autocast's fp32 list);
  - RoPE is applied in fp32 (autocast is disabled there);
  - attention runs on bf16 q/k/v (FlashAttention-3 packs to bf16) and is cast back to the query's dtype (fp32);
  - elementwise nonlinearities on bf16 tensors are evaluated in fp32 and rounded once, as torch's kernels do;
  - the residual stream starts bf16 (patch embed) and becomes fp32 after the first residual add.
`set_compute_dtype(mx.float32)` runs everything in fp32 instead (for exactness tests against torch fp32).

Caption padding is not modelled: the reference pads captions to 256 tokens and masks the pads out of attention,
which leaves every valid token's value unchanged, so each caption runs at its true length (cond and uncond as two
batch-1 forwards, or batched with a key mask).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

F32, BF16 = mx.float32, mx.bfloat16
_COMPUTE = {"dtype": BF16}


def set_compute_dtype(dtype) -> None:
    """bf16 (the reference's autocast flow, default) or fp32 (exactness tests)."""
    _COMPUTE["dtype"] = dtype


def cdt():
    return _COMPUTE["dtype"]


def _f32_then(fn, x):
    """Evaluate an elementwise op in fp32 and round once to x's dtype (torch's bf16 kernels)."""
    return fn(x.astype(F32)).astype(x.dtype)


def silu(x):
    return _f32_then(lambda t: t * mx.sigmoid(t), x)


def gelu_tanh(x):
    return _f32_then(lambda t: 0.5 * t * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (t + 0.044715 * t * t * t))), x)


@dataclass(frozen=True)
class Config:
    dim: int = 2944
    num_heads: int = 23
    num_layers: int = 22
    ffn_dim: int = 7936
    refiner_blocks: int = 2
    refiner_ffn_dim: int = 4096
    adaln_rank: int = 256
    text_dim: int = 7680
    text_len: int = 256
    in_channels: int = 3
    out_channels: int = 3
    freq_dim: int = 256
    eps: float = 1e-6
    bottleneck_dim: int = 1024
    patch: tuple = (1, 32, 32)
    rope_max_positions: int = 1024
    rope_spatial_scale: float = 0.5
    text_layers: tuple = (7, 15, 27)

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


class Linear(nn.Module):
    """Autocast-style linear: the input is cast to the compute dtype; weight (and bias) are stored in it."""

    def __init__(self, i: int, o: int, bias: bool = True):
        super().__init__()
        self.weight = mx.zeros((o, i))
        if bias:
            self.bias = mx.zeros((o,))

    def __call__(self, x):
        x = x.astype(cdt())
        if "bias" in self:
            return mx.addmm(self.bias, x, self.weight.T)
        return x @ self.weight.T


class QLinear(nn.QuantizedLinear):
    """Quantized counterpart of Linear (same input cast)."""

    def __call__(self, x):
        return super().__call__(x.astype(cdt()))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=F32)

    def __call__(self, x):
        n = mx.fast.rms_norm(x.astype(F32), None, self.eps)
        return n.astype(x.dtype).astype(F32) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=F32)
        self.bias = mx.zeros((dim,), dtype=F32)

    def __call__(self, x):
        return mx.fast.layer_norm(x.astype(F32), self.weight, self.bias, self.eps)


def apply_rope(x, cos, sin):
    """x (B, L, n, d) rotated by per-token (L, d/2) tables over interleaved pairs, in fp32 (cos=1, sin=0 rows are the
    identity and are passed as None-free tables)."""
    b, l, n, d = x.shape
    p = x.astype(F32).reshape(b, l, n, d // 2, 2)
    xe, xo = p[..., 0], p[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return mx.stack([xe * c - xo * s, xe * s + xo * c], axis=-1).reshape(b, l, n, d)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = Linear(dim, dim), Linear(dim, dim), Linear(dim, dim), Linear(dim, dim)
        self.norm_q, self.norm_k = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.gate_proj = Linear(dim, num_heads)

    def __call__(self, x, rope, key_mask=None):
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).reshape(b, s, n, d)
        k = self.norm_k(self.k(x)).reshape(b, s, n, d)
        v = self.v(x).reshape(b, s, n, d)
        if rope is not None:
            cos, sin, n_rot = rope  # rotate the first n_rot tokens (image); the rest (caption) sit at identity
            if n_rot == s:
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            else:
                q = mx.concatenate([apply_rope(q[:, :n_rot], cos, sin), q[:, n_rot:]], axis=1)
                k = mx.concatenate([apply_rope(k[:, :n_rot], cos, sin), k[:, n_rot:]], axis=1)
        qa, ka, va = (t.astype(cdt()).transpose(0, 2, 1, 3) for t in (q, k, v))
        att = mx.fast.scaled_dot_product_attention(qa, ka, va, scale=d ** -0.5, mask=key_mask)
        att = att.transpose(0, 2, 1, 3).astype(q.dtype)
        gate = _f32_then(mx.sigmoid, self.gate_proj(x))
        att = (att * gate[..., None]).reshape(b, s, n * d)
        return self.o(att)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        h = int(2 * hidden_dim / 3)
        h = multiple_of * ((h + multiple_of - 1) // multiple_of)
        self.w13 = Linear(dim, 2 * h, bias=False)
        self.w2 = Linear(h, dim, bias=False)

    def __call__(self, x):
        x1, x3 = mx.split(self.w13(x), 2, axis=-1)
        return self.w2(silu(x1) * x3)


class Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float, adaln_rank: int):
        super().__init__()
        self.norm1, self.post_norm1 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.norm2, self.post_norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.attn = SelfAttention(dim, num_heads, eps)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.adaln_up = Linear(adaln_rank, 4 * dim)

    def __call__(self, x, c, rope, key_mask=None, valid=None):
        s1, g1, s2, g2 = mx.split(self.adaln_up(c)[:, None, :], 4, axis=-1)
        g1, g2 = _f32_then(mx.tanh, g1), _f32_then(mx.tanh, g2)
        s1, s2 = 1.0 + s1, 1.0 + s2
        y = self.post_norm1(self.attn(self.norm1(x) * s1, rope, key_mask))
        if valid is not None:
            y = y * valid
        x = x + g1 * y
        y = self.post_norm2(self.ffn(self.norm2(x) * s2))
        if valid is not None:
            y = y * valid
        return x + g2 * y


class OutputHead(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.proj = Linear(dim, out_dim)
        self.modulation = mx.zeros((1, 2, dim), dtype=F32)

    def __call__(self, x, e):
        m = self.modulation + e[:, None, :].astype(F32) if e.dtype != F32 else self.modulation + e[:, None, :]
        shift, scale = m[:, 0:1], m[:, 1:2]
        return self.proj(self.norm(x) * (1 + scale) + shift)


def sinusoidal_embedding(dim: int, t: np.ndarray) -> np.ndarray:
    """[cos | sin] timestep embedding computed in float64 (as the reference), shape (B, dim)."""
    half = dim // 2
    pos = np.asarray(t, dtype=np.float64)
    inv = np.power(10000.0, -np.arange(half, dtype=np.float64) / half)
    ang = np.outer(pos, inv)
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)


def rope_tables(cfg: Config, f_p: int, h_p: int, w_p: int):
    """fp32 (cos, sin) for the image tokens in (f, h, w) order: per-axis tables computed in float64, T at integer
    positions and H/W at positions * rope_spatial_scale, then rounded to fp32 (what apply_rope reads)."""
    hd = cfg.head_dim
    dims = (hd - 4 * (hd // 6), 2 * (hd // 6), 2 * (hd // 6))

    def axis(dim, positions):
        inv = 1.0 / np.power(10000.0, np.arange(0, dim, 2, dtype=np.float64) / dim)
        return np.outer(positions, inv)

    n = np.arange(cfg.rope_max_positions, dtype=np.float64)
    at, ah, aw = axis(dims[0], n[:f_p]), axis(dims[1], (n * cfg.rope_spatial_scale)[:h_p]), axis(dims[2], (n * cfg.rope_spatial_scale)[:w_p])
    grid = np.concatenate([
        np.broadcast_to(at[:, None, None, :], (f_p, h_p, w_p, at.shape[1])),
        np.broadcast_to(ah[None, :, None, :], (f_p, h_p, w_p, ah.shape[1])),
        np.broadcast_to(aw[None, None, :, :], (f_p, h_p, w_p, aw.shape[1]))], axis=-1).reshape(f_p * h_p * w_p, -1)
    return mx.array(np.cos(grid).astype(np.float32)), mx.array(np.sin(grid).astype(np.float32))


class PJiT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, pt, ph, pw = cfg.dim, *cfg.patch
        self.patch_in = Linear(cfg.in_channels * pt * ph * pw, cfg.bottleneck_dim, bias=False)  # Conv3d k=stride=patch
        self.patch_up = Linear(cfg.bottleneck_dim, d)                                            # 1x1x1 Conv3d
        self.time_in, self.time_out = Linear(cfg.freq_dim, d), Linear(d, d)
        self.image_adaln, self.text_adaln = Linear(d, cfg.adaln_rank), Linear(d, cfg.adaln_rank)
        self.text_in, self.text_out = Linear(cfg.text_dim, d, bias=False), Linear(d, d, bias=False)
        kw = dict(dim=d, num_heads=cfg.num_heads, eps=cfg.eps, adaln_rank=cfg.adaln_rank)
        self.image_refiner = [Block(ffn_dim=cfg.refiner_ffn_dim, **kw) for _ in range(cfg.refiner_blocks)]
        self.text_refiner = [Block(ffn_dim=cfg.refiner_ffn_dim, **kw) for _ in range(cfg.refiner_blocks)]
        self.blocks = [Block(ffn_dim=cfg.ffn_dim, **kw) for _ in range(cfg.num_layers)]
        self.head = OutputHead(d, pt * ph * pw * cfg.out_channels)
        self._rope = {}
        self._fns = {}
        self.compile = True  # mx.compile each block (bit-identical, ~6% faster); masked/padded calls stay eager

    def _block(self, blk, rope: bool):
        """A (compiled) call of one block; RoPE rotates the first len(cos) tokens."""
        key = id(blk)
        if key not in self._fns:
            if rope:
                fn = lambda s, c, co, si, b=blk: b(s, c, (co, si, co.shape[0]))  # noqa: E731
            else:
                fn = lambda s, c, b=blk: b(s, c, None)  # noqa: E731
            self._fns[key] = mx.compile(fn) if self.compile else fn
        return self._fns[key]

    def rope(self, f_p, h_p, w_p):
        key = (f_p, h_p, w_p)
        if key not in self._rope:
            self._rope = {key: rope_tables(self.cfg, f_p, h_p, w_p)}
        return self._rope[key]

    def embed_text(self, text):
        """(B, L, 7680) caption features -> (B, L, dim) projected tokens (bf16)."""
        return self.text_out(gelu_tanh(self.text_in(text)))

    def __call__(self, x_t, t: np.ndarray, text, key_mask=None, valid=None, text_tokens=None):
        """x_t (B, 3, 1, H, W) (any float dtype; patchified in the compute dtype), t (B,) integer timesteps,
        text (B, Lt, 7680) caption features at their true length (or padded, with key_mask/valid for the padded
        layout). Returns the clean-image prediction (B, 3, 1, H, W) in the compute dtype."""
        cfg = self.cfg
        pt, ph, pw = cfg.patch
        b, ch, f, hh, ww = x_t.shape
        pads = [(p - s % p) % p for s, p in zip((f, hh, ww), cfg.patch)]
        if any(pads):
            x_t = mx.pad(x_t, [(0, 0), (0, 0), (0, pads[0]), (0, pads[1]), (0, pads[2])])
        fp, hp, wp = (f + pads[0]) // pt, (hh + pads[1]) // ph, (ww + pads[2]) // pw
        patches = x_t.astype(cdt()).reshape(b, ch, fp, pt, hp, ph, wp, pw).transpose(0, 2, 4, 6, 1, 3, 5, 7)
        x_img = self.patch_up(self.patch_in(patches.reshape(b, fp * hp * wp, ch * pt * ph * pw)))
        l_img = fp * hp * wp

        emb = mx.array(sinusoidal_embedding(cfg.freq_dim, t)).astype(x_img.dtype)
        e = self.time_out(silu(self.time_in(emb)))
        c_image, c_text = self.image_adaln(silu(e)), self.text_adaln(silu(e))

        cos, sin = self.rope(fp, hp, wp)
        for blk in self.image_refiner:
            x_img = self._block(blk, True)(x_img, c_image, cos, sin)
        txt = text_tokens if text_tokens is not None else self.embed_text(text)
        if key_mask is None:
            for blk in self.text_refiner:
                txt = self._block(blk, False)(txt, c_text)
        else:  # padded batch: eager, with key masks and zeroed pad rows
            for blk in self.text_refiner:
                txt = blk(txt, c_text, None, key_mask[1], valid[1])
        seq = mx.concatenate([x_img, txt.astype(x_img.dtype) if txt.dtype != x_img.dtype else txt], axis=1)
        if key_mask is None:
            for blk in self.blocks:
                seq = self._block(blk, True)(seq, c_image, cos, sin)
        else:
            for blk in self.blocks:
                seq = blk(seq, c_image, (cos, sin, l_img), key_mask[0], valid[0])

        out = self.head(seq[:, :l_img], e)
        u = out.reshape(b, fp, hp, wp, pt, ph, pw, cfg.out_channels).transpose(0, 7, 1, 4, 2, 5, 3, 6)
        u = u.reshape(b, cfg.out_channels, fp * pt, hp * ph, wp * pw)
        return u[:, :, :f, :hh, :ww]
