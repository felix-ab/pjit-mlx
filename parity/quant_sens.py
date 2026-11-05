"""Which layers make a quantized DiT fork? Step-0 x-prediction (t=1000, the gallery caption) of each quantization
recipe vs bf16, as PSNR over [-1, 1]. bf16 itself sits ~63.5 dB from fp32 (and torch's bf16 at 64.5), so a recipe near
that level should keep seed parity.
Usage: venv-mlx/bin/python -I parity/quant_sens.py
"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from mlx_check import R, stats  # noqa: E402
from pjit_mlx import quantize as Q  # noqa: E402
from pjit_mlx.pipeline import load_dit  # noqa: E402

x = mx.array(np.load(R / "x_in.npy"))
cond = mx.array(np.load(R / "text_cond.npy"))[None].astype(mx.bfloat16)
t = np.array([1000])
ref32 = np.load(R / "fp32_pred_cond.npy")


def pred(model):
    p = model(x, t, cond)
    mx.eval(p)
    return np.array(p.astype(mx.float32))


def quantize_where(model, bits, gs, keep):
    """Quantize like quantize_dit, but leave layers for which keep(block_kind, block_index, name) is True."""
    groups = [("img", model.image_refiner), ("txt", model.text_refiner), ("trunk", model.blocks)]
    for kind, blocks in groups:
        for i, blk in enumerate(blocks):
            for mod, name in list(Q._linears(blk)):
                if name in Q.DIT_SKIP or keep(kind, i, name):
                    continue
                Q._swap(mod, name, bits, gs, True)
    if not keep("proj", 0, "text_in"):
        for name in ("text_in", "text_out"):
            Q._swap(model, name, bits, gs, True)
    model._fns = {}


RECIPES = {
    "q8g64_all": (8, 64, lambda k, i, n: False),
    "q8g32_all": (8, 32, lambda k, i, n: False),
    "q8_keep_adaln": (8, 64, lambda k, i, n: n == "adaln_up"),
    "q8_keep_refiners": (8, 64, lambda k, i, n: k in ("img", "txt")),
    "q8_keep_textproj": (8, 64, lambda k, i, n: k == "proj"),
    "q8_keep_qk": (8, 64, lambda k, i, n: n in ("q", "k")),
    "q8_keep_attn": (8, 64, lambda k, i, n: n in ("q", "k", "v", "o")),
    "q8_keep_ffn": (8, 64, lambda k, i, n: n in ("w13", "w2")),
    "q8_keep_first_last2": (8, 64, lambda k, i, n: k != "trunk" or i < 2 or i >= 20),
    "q8_ffn_only": (8, 64, lambda k, i, n: n not in ("w13", "w2")),
}


def main():
    base = load_dit()
    pb = pred(base)
    res = {"bf16_vs_fp32": stats(pb, ref32)}
    del base
    for name, (bits, gs, keep) in RECIPES.items():
        model = load_dit()
        quantize_where(model, bits, gs, keep)
        p = pred(model)
        res[name] = {"vs_bf16": round(stats(p, pb)["psnr_db"], 2), "vs_fp32": round(stats(p, ref32)["psnr_db"], 2)}
        print(name, res[name], flush=True)
        del model
        mx.clear_cache()
    (W / "parity/quant_sens.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
