"""Fused q/k/v/gate projection (one GEMM over the concatenated weights) vs four GEMMs: speed of the 22-block trunk,
and whether the outputs stay bit-identical (each output column has the same reduction either way)."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from bench_forward import timeit  # noqa: E402
from mlx_check import stats  # noqa: E402
from pjit_mlx import dit  # noqa: E402
from pjit_mlx.pipeline import load_dit  # noqa: E402

model = load_dit()
cos, sin = model.rope(1, 27, 36)
mx.random.seed(0)
seq = (mx.random.normal((1, 972 + 67, 2944)) * 0.5).astype(mx.float32)
c = (mx.random.normal((1, 256)) * 0.1).astype(mx.bfloat16)


def trunk():
    s = seq
    for b in model.blocks:
        s = model._block(b, True)(s, c, cos, sin)
    return s


base = np.array(trunk())
t_base = timeit(trunk)


def fused_call(self, x, rope, key_mask=None):
    b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim
    xin = x.astype(dit.cdt())
    y = mx.addmm(self.qkvg_b, xin, self.qkvg_w.T)
    D = n * d
    q = self.norm_q(y[..., :D]).reshape(b, s, n, d)
    k = self.norm_k(y[..., D:2 * D]).reshape(b, s, n, d)
    v = y[..., 2 * D:3 * D].reshape(b, s, n, d)
    gl = y[..., 3 * D:]
    cs, sn, n_rot = rope
    q = mx.concatenate([dit.apply_rope(q[:, :n_rot], cs, sn), q[:, n_rot:]], axis=1)
    k = mx.concatenate([dit.apply_rope(k[:, :n_rot], cs, sn), k[:, n_rot:]], axis=1)
    qa, ka, va = (t.astype(dit.cdt()).transpose(0, 2, 1, 3) for t in (q, k, v))
    att = mx.fast.scaled_dot_product_attention(qa, ka, va, scale=d ** -0.5).transpose(0, 2, 1, 3).astype(q.dtype)
    gate = dit._f32_then(mx.sigmoid, gl)
    return self.o((att * gate[..., None]).reshape(b, s, n * d))


for blk in model.blocks:
    a = blk.attn
    a.qkvg_w = mx.concatenate([a.q.weight, a.k.weight, a.v.weight, a.gate_proj.weight], 0)
    a.qkvg_b = mx.concatenate([a.q.bias, a.k.bias, a.v.bias, a.gate_proj.bias], 0)
mx.eval([blk.attn.qkvg_w for blk in model.blocks])
dit.SelfAttention.__call__ = fused_call
model._fns = {}
fused = np.array(trunk())
out = {"base": t_base, "fused": timeit(trunk), "fused_vs_base": stats(fused, base)}
print(json.dumps(out))
