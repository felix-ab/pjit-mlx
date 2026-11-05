"""Batched cond+uncond forward (uncond caption padded to the cond length, key-masked) vs two batch-1 forwards:
speed, and whether the valid outputs match. Also mx.compile over whole blocks (via a module-level wrapper)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from bench_forward import R, timeit  # noqa: E402
from mlx_check import stats  # noqa: E402
from pjit_mlx import dit  # noqa: E402
from pjit_mlx.pipeline import load_dit  # noqa: E402

model = load_dit()
x = mx.array(np.load(R / "x_in.npy"))
cond = mx.array(np.load(R / "text_cond.npy"))[None].astype(mx.bfloat16)
unc = mx.array(np.load(R / "text_uncond.npy"))[None].astype(mx.bfloat16)
ct, ut = model.embed_text(cond), model.embed_text(unc)
t = np.array([1000])
pc = np.array(model(x, t, None, text_tokens=ct).astype(mx.float32))
pu = np.array(model(x, t, None, text_tokens=ut).astype(mx.float32))
Lc, Lu = ct.shape[1], ut.shape[1]
tt = mx.concatenate([ct, mx.pad(ut, [(0, 0), (0, Lc - Lu), (0, 0)])], 0)
l_img = (864 // 32) * (1152 // 32)
tvalid = mx.array(np.array([[1] * Lc, [1] * Lu + [0] * (Lc - Lu)], dtype=bool))
svalid = mx.concatenate([mx.ones((2, l_img), dtype=mx.bool_), tvalid], 1)
km = (svalid[:, None, None, :], tvalid[:, None, None, :])
va = (svalid[..., None].astype(mx.float32), tvalid[..., None].astype(mx.float32))
x2, t2 = mx.concatenate([x, x], 0), np.array([1000, 1000])
pb = np.array(model(x2, t2, None, key_mask=km, valid=va, text_tokens=tt).astype(mx.float32))
out = {"batched_vs_b1_cond": stats(pb[0:1], pc), "batched_vs_b1_uncond": stats(pb[1:2], pu)}
out["two_b1"] = timeit(lambda: [model(x, t, None, text_tokens=ct), model(x, t, None, text_tokens=ut)])
out["batched_b2"] = timeit(lambda: model(x2, t2, None, key_mask=km, valid=va, text_tokens=tt))
print(json.dumps(out), flush=True)
(W / "parity/bench_batch.json").write_text(json.dumps(out, indent=1))
