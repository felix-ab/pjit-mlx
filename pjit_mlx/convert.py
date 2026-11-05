"""Convert the released fp32 checkpoint (Linum-AI/pjit-21k model.safetensors) to the MLX DiT's layout.

Linear and conv weights (and their biases) are what autocast feeds the bf16 GEMMs, so storing them as bf16 is exact
for the reference's dtype flow; norm gains, the LayerNorm affine and the head modulation stay fp32 (they are applied
in fp32). `dtype="float32"` keeps everything fp32 (exactness tests).
"""

from __future__ import annotations

import mlx.core as mx

FP32_SUFFIXES = ("norm1.weight", "norm2.weight", "post_norm1.weight", "post_norm2.weight", "norm_q.weight",
                 "norm_k.weight", "head.norm.weight", "head.norm.bias", "head.modulation")
RENAME = {
    "patch_embed.proj1.weight": "patch_in.weight",
    "patch_embed.proj2.weight": "patch_up.weight",
    "patch_embed.proj2.bias": "patch_up.bias",
    "time_embed.0.": "time_in.", "time_embed.2.": "time_out.",
    "image_adaln.1.": "image_adaln.", "text_adaln.1.": "text_adaln.",
    "text_proj.0.": "text_in.", "text_proj.2.": "text_out.",
}


def convert(torch_weights: dict, dtype: str = "bfloat16") -> dict:
    lin = mx.bfloat16 if dtype == "bfloat16" else mx.float32
    out = {}
    for key, v in torch_weights.items():
        new = key
        for a, b in RENAME.items():
            if new.startswith(a):
                new = b + new[len(a):]
                break
        if new == "patch_in.weight":       # Conv3d (1024, 3, 1, 32, 32), kernel = stride = patch -> Linear (1024, 3072)
            v = v.reshape(v.shape[0], -1)
        elif new == "patch_up.weight":     # 1x1x1 Conv3d (2944, 1024, 1, 1, 1) -> Linear
            v = v.reshape(v.shape[0], v.shape[1])
        out[new] = v.astype(mx.float32 if new.endswith(FP32_SUFFIXES) else lin)
    return out


def load_reference(path: str, dtype: str = "bfloat16") -> dict:
    return convert(mx.load(path), dtype)
