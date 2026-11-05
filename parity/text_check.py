"""Text-encoder parity floor: MLX (fp32 and bf16) and torch bf16 against torch fp32 hidden states."""
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "mlx"))
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from mlx_check import PROMPT, cos_rows, stats  # noqa: E402
from pjit_mlx.text_encoder import TextEncoder, sanitize  # noqa: E402

R = W / "parity/ref"
qwen = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
raw = {}
for f in sorted(Path(qwen).glob("*.safetensors")):
    raw.update(mx.load(str(f)))
for mode in ("float32", "bfloat16"):
    w = sanitize(raw, 28)
    if mode == "float32":
        w = {k: v.astype(mx.float32) for k, v in w.items()}
    te = TextEncoder(qwen, weights=w)
    for name, cap in (("cond", PROMPT), ("uncond", "")):
        ids = mx.array(te.token_ids(cap))[None]
        h = te.embed(te.token_ids(cap))[None]
        if mode == "float32":
            h = h.astype(mx.float32)  # the rows are stored bf16; run the fp32 check on fp32 activations
        outs = []
        for i, layer in enumerate(te.model.layers):
            h = layer(h, None, None)
            if i in (7, 15, 27):
                outs.append(h[0])
        mine = np.array(mx.concatenate(outs, -1).astype(mx.float32))
        ref = np.load(R / f"text32raw_{name}.npy")
        s = stats(mine, ref)
        print(mode, name, {k: round(v, 6) for k, v in s.items()}, "min cos", round(cos_rows(mine, ref), 7), flush=True)
    del te
    mx.clear_cache()
for name in ("cond", "uncond"):
    s = stats(np.load(R / f"text_{name}.npy"), np.load(R / f"text32raw_{name}.npy"))
    print("torch bf16", name, {k: round(v, 6) for k, v in s.items()}, "min cos",
          round(cos_rows(np.load(R / f"text_{name}.npy"), np.load(R / f"text32raw_{name}.npy")), 7))
