"""MLX side of the end-to-end fp32 trajectory check (parity/traj_ref.py): pjit_mlx.sampler with the DiT in fp32, the
same caption features, 512x512, seed 4. Compares the final state with the torch fp32 run, and the integer timesteps.
Usage: venv-mlx/bin/python -I parity/traj_mlx.py
"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from mlx_check import stats  # noqa: E402
from pjit_mlx import dit  # noqa: E402
from pjit_mlx import sampler as S  # noqa: E402
from pjit_mlx.convert import load_reference  # noqa: E402

R = W / "parity/ref"


def main():
    dit.set_compute_dtype(mx.float32)
    wd = snapshot_download("Linum-AI/pjit-21k", local_files_only=True, allow_patterns=["config.json", "model.safetensors"])
    model = dit.PJiT(dit.Config.from_dict(json.load(open(wd + "/config.json"))["architecture"]))
    model.load_weights(list(load_reference(wd + "/model.safetensors", "float32").items()), strict=True)
    cond = mx.array(np.load(R / "text_cond.npy"))[None]
    unc = mx.array(np.load(R / "text_uncond.npy"))[None]
    s = S.SamplerConfig(height=512, width=512)
    img, info = S.sample(model, cond, unc, 4, s)
    ref = np.load(R / "traj_fp32.npy")
    ref_img = np.round((np.clip(ref, -1, 1) + 1) / 2 * 255).astype(np.uint8)[0, :, 0].transpose(1, 2, 0)
    ts = S.linspace_f32(1.0, 0.0, 51)
    t_int = (ts[:-1] * np.float32(1000)).astype(np.int64)
    out = {"timesteps_equal": bool(np.array_equal(t_int, np.load(R / "traj_timesteps.npy"))),
           "image_psnr_db": float(10 * np.log10(255 ** 2 / max(np.mean((img.astype(np.float64) - ref_img) ** 2), 1e-12))),
           "image_max_abs": int(np.abs(img.astype(int) - ref_img.astype(int)).max()),
           "image_identical_pixels_pct": float(100 * np.mean(img == ref_img)), "sample_s": info["sample_s"]}
    print(json.dumps(out))
    (W / "parity/traj_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
