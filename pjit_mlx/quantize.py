"""Weight-only quantization tiers (MLX affine, per-group scales and biases).

DiT: every linear inside the 22 trunk blocks and 4 refiner blocks (q/k/v/o, SwiGLU, AdaLN up-projection) plus the
caption projection; the per-head attention gates, patch embed, timestep MLP, AdaLN roots and output head stay bf16
(small, and closest to pixels or to the conditioning). Text encoder: the decoder blocks' projections.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .dit import Linear, QLinear

DIT_SKIP = ("gate_proj",)


def _swap(module: nn.Module, name: str, bits: int, group_size: int, from_weights: bool):
    lin = module[name]
    q = QLinear(lin.weight.shape[1], lin.weight.shape[0], "bias" in lin, group_size=group_size, bits=bits)
    if from_weights:
        q.weight, q.scales, q.biases = mx.quantize(lin.weight, group_size, bits)
        if "bias" in lin:
            q.bias = lin.bias
    module[name] = q


def _linears(module: nn.Module):
    for name, child in module.items():
        if isinstance(child, Linear):
            yield module, name
        elif isinstance(child, nn.Module):
            yield from _linears(child)
        elif isinstance(child, list):
            for c in child:
                if isinstance(c, nn.Module):
                    yield from _linears(c)


def quantize_dit(model, spec: dict, from_weights: bool = True) -> None:
    """In place. spec: {"bits": 8, "group_size": 64}. from_weights=True quantizes the loaded bf16 weights; False only
    builds the quantized structure (before loading an exported quantized checkpoint)."""
    bits, gs = int(spec["bits"]), int(spec.get("group_size", 64))
    for blk in model.image_refiner + model.text_refiner + model.blocks:
        for mod, name in list(_linears(blk)):
            if name not in DIT_SKIP:
                _swap(mod, name, bits, gs, from_weights)
    for name in ("text_in", "text_out"):
        _swap(model, name, bits, gs, from_weights)
    model._fns = {}  # recompile against the new layers


def quantize_text_encoder(model, spec: dict) -> None:
    """The decoder blocks' linear projections (the embedding is read row by row and stays bf16)."""
    bits, gs = int(spec["bits"]), int(spec.get("group_size", 64))
    nn.quantize(model, group_size=gs, bits=bits, class_predicate=lambda p, m: isinstance(m, nn.Linear))
