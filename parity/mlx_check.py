"""MLX port vs the torch reference dumps (parity/ref_dump.py).

  text   MLX Qwen3.5-4B features vs pjit_21k.text_encoder (bf16 both): max |diff|, worst-token cosine
  fp32   MLX DiT in fp32 vs torch fp32 (CPU), the same caption features: rel L2, PSNR over the [-1, 1] range
  bf16   MLX DiT with the autocast flow vs torch bf16 autocast (MPS), and each vs the fp32 reference
Usage: venv-mlx/bin/python -I parity/mlx_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "mlx"))
import mlx.core as mx  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from pjit_mlx import dit  # noqa: E402
from pjit_mlx.convert import load_reference  # noqa: E402
from pjit_mlx.text_encoder import TextEncoder  # noqa: E402

R = W / "parity/ref"
PROMPT = ("Bright midday sun lights a white man in his thirties in a head-and-shoulders portrait, squinting slightly "
          "with a relaxed half smile. His faded blue chambray shirt is open at the collar, and a field of tall grass "
          "blurs behind him.")


def stats(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    err = a - b
    return {"rel_l2": float(np.linalg.norm(err) / np.linalg.norm(b)), "max_abs": float(np.abs(err).max()),
            "psnr_db": float(10 * np.log10(4.0 / np.mean(err ** 2)))}


def cos_rows(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float((np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).min())


def main():
    res = {}
    qwen = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
    te = TextEncoder(qwen)
    for name, cap in (("cond", PROMPT), ("uncond", "")):
        mine = np.array(te(cap).astype(mx.float32))
        ref = np.load(R / f"text_{name}.npy")
        res[f"text_{name}"] = {"tokens": [mine.shape[0], ref.shape[0]], **stats(mine, ref), "min_token_cos": cos_rows(mine, ref)}
        print(name, res[f"text_{name}"], flush=True)
    del te
    mx.clear_cache()

    wd = snapshot_download("Linum-AI/pjit-21k", local_files_only=True, allow_patterns=["config.json", "model.safetensors"])
    cfg = dit.Config.from_dict(json.load(open(wd + "/config.json"))["architecture"])
    x_in = mx.array(np.load(R / "x_in.npy"))
    t = np.array([1000])
    texts = {n: mx.array(np.load(R / f"text_{n}.npy"))[None].astype(mx.bfloat16) for n in ("cond", "uncond")}
    for mode in ("float32", "bfloat16"):
        dit.set_compute_dtype(mx.float32 if mode == "float32" else mx.bfloat16)
        model = dit.PJiT(cfg)
        model.load_weights(list(load_reference(wd + "/model.safetensors", mode).items()), strict=True)
        for n in ("cond", "uncond"):
            text = texts[n].astype(mx.float32) if mode == "float32" else texts[n]
            pred = model(x_in, t, text)
            mx.eval(pred)
            pred = np.array(pred.astype(mx.float32))
            res[f"{mode}_{n}_vs_fp32ref"] = stats(pred, np.load(R / f"fp32_pred_{n}.npy"))
            if mode == "bfloat16":
                res[f"{mode}_{n}_vs_bf16ref"] = stats(pred, np.load(R / f"bf16_pred_{n}.npy"))
            print(mode, n, {k: v for k, v in res.items() if k.startswith(f"{mode}_{n}")}, flush=True)
        del model
        mx.clear_cache()
    for n in ("cond", "uncond"):
        res[f"torch_bf16_{n}_vs_fp32ref"] = stats(np.load(R / f"bf16_pred_{n}.npy"), np.load(R / f"fp32_pred_{n}.npy"))
    (W / "parity/results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
