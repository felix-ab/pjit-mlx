"""Trunk (22 blocks) eager vs mx.compile'd per block, on the step-0 sequence."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from bench_forward import R, timeit  # noqa: E402
from mlx_check import stats  # noqa: E402
from pjit_mlx.pipeline import load_dit  # noqa: E402

model = load_dit()
cos, sin = model.rope(1, 27, 36)
seq = (mx.random.normal((1, 972 + 67, 2944)) * 0.5).astype(mx.float32)
c = (mx.random.normal((1, 256)) * 0.1).astype(mx.bfloat16)


def trunk(blocks, s):
    for b in blocks:
        s = b(s, c, (cos, sin, 972))
    return s


fns = [mx.compile(lambda s, cc, co, si, b=b: b(s, cc, (co, si, co.shape[0]))) for b in model.blocks]


def trunk_c(s):
    for f in fns:
        s = f(s, c, cos, sin)
    return s


e, k = np.array(trunk(model.blocks, seq)), np.array(trunk_c(seq))
out = {"eager": timeit(lambda: trunk(model.blocks, seq)), "compiled": timeit(lambda: trunk_c(seq)), "diff": stats(k, e)}
print(json.dumps(out))
