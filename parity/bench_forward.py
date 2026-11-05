"""Time one DiT forward (B=1, the gallery caption, 864x1152) in a few execution modes and check each against eager.
Usage: venv-mlx/bin/python -I parity/bench_forward.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from mlx_check import stats  # noqa: E402
from pjit_mlx import dit  # noqa: E402
from pjit_mlx.pipeline import load_dit  # noqa: E402

R = W / "parity/ref"


def timeit(fn, n=4):
    mx.eval(fn())
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return min(ts), np.median(ts)


def main():
    model = load_dit()
    x = mx.array(np.load(R / "x_in.npy"))
    cond = mx.array(np.load(R / "text_cond.npy"))[None].astype(mx.bfloat16)
    unc = mx.array(np.load(R / "text_uncond.npy"))[None].astype(mx.bfloat16)
    ct, ut = model.embed_text(cond), model.embed_text(unc)
    t = np.array([1000])
    eager = np.array(model(x, t, None, text_tokens=ct).astype(mx.float32))
    out = {"eager_b1": timeit(lambda: model(x, t, None, text_tokens=ct))}
    out["eager_pair"] = timeit(lambda: [model(x, t, None, text_tokens=ct), model(x, t, None, text_tokens=ut)])
    print(json.dumps(out), flush=True)

    # compiled blocks
    for blk in model.image_refiner + model.text_refiner + model.blocks:
        blk.__call__ = mx.compile(blk.__call__)
    comp = np.array(model(x, t, None, text_tokens=ct).astype(mx.float32))
    out["compiled_b1"] = timeit(lambda: model(x, t, None, text_tokens=ct))
    out["compiled_vs_eager"] = stats(comp, eager)
    print(json.dumps(out), flush=True)
    (W / "parity/bench_forward.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
