"""Torch reference dumps for the MLX port's parity tests (the repo's own code, ext/repo @95403933).

Writes parity/ref/:
  text_cond.npy, text_uncond.npy   Qwen3.5-4B features (bf16 -> fp32) from pjit_21k.text_encoder, CPU
  x_in.npy                         the step-0 model input: bf16-rounded H100 noise x 4, seed 4, 864 x 1152
  fp32_pred_{cond,uncond}.npy      DiT x-prediction at t=1000, everything fp32 on the CPU, padded captions + mask
  bf16_pred_{cond,uncond}.npy      the same under bf16 autocast on MPS (run_mps.py's patches)
  x_mid.npy, fp32_mid_cond.npy     a mid-trajectory state (t index 25, from the MPS run) and its fp32 prediction
Usage: venv/bin/python -I parity/ref_dump.py
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "ext/repo"))
sys.path.insert(0, str(W))
import run_mps  # noqa: E402  (installs the MPS patches: RoPE autocast off, fp64 timestep embedding on the CPU)
from h100_noise import randn_h100  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from pjit_21k import PyramidJiT, QwenTextEncoder, set_attention_backend  # noqa: E402
from pjit_21k.sampler import _text_kwargs  # noqa: E402

PROMPT = ("Bright midday sun lights a white man in his thirties in a head-and-shoulders portrait, squinting slightly "
          "with a relaxed half smile. His faded blue chambray shirt is open at the collar, and a field of tall grass "
          "blurs behind him.")
OUT = W / "parity/ref"


def bf16(x):
    return x.to(torch.bfloat16).float()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    set_attention_backend(backend="sdpa")
    qwen = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
    weights = snapshot_download("Linum-AI/pjit-21k", local_files_only=True, allow_patterns=["config.json", "model.safetensors"])
    te = QwenTextEncoder(model_path=qwen, extraction_layers=(7, 15, 27), max_length=256, device="cpu")
    cond, uncond = te([PROMPT])[0], te([""])[0]
    np.save(OUT / "text_cond.npy", cond.float().numpy())
    np.save(OUT / "text_uncond.npy", uncond.float().numpy())
    print("text", tuple(cond.shape), tuple(uncond.shape), flush=True)
    del te
    gc.collect()

    x_in = bf16(4.0 * torch.from_numpy(randn_h100((3, 1, 864, 1152), seed=4))[None])
    np.save(OUT / "x_in.npy", x_in.numpy())

    def text_args(dev):
        c = _text_kwargs(text=cond.to(dev), batch_size=1, text_len=256)
        u = _text_kwargs(text=uncond.to(dev), batch_size=1, text_len=256)
        return c, u

    # fp32 on the CPU, no autocast
    model = PyramidJiT.from_pretrained(weights_dir=weights, device="cpu")
    c, u = text_args("cpu")
    t = torch.tensor([1000])
    for name, ta in (("cond", c), ("uncond", u)):
        t0 = time.time()
        with torch.no_grad():
            pred = model(x_t=x_in, t=t, text=ta["text"].float(), text_lens=ta["text_lens"])
        np.save(OUT / f"fp32_pred_{name}.npy", pred.float().numpy())
        print("fp32", name, round(time.time() - t0, 1), "s", flush=True)
    mid = W / "parity/ref/x_mid.npy"
    if mid.exists():
        xm = torch.from_numpy(np.load(mid))
        with torch.no_grad():
            pred = model(x_t=xm, t=torch.tensor([500]), text=c["text"].float(), text_lens=c["text_lens"])
        np.save(OUT / "fp32_mid_cond.npy", pred.float().numpy())
        print("fp32 mid done", flush=True)
    del model
    gc.collect()

    # bf16 autocast on MPS (fp32 masters, as on CUDA)
    model = PyramidJiT.from_pretrained(weights_dir=weights, device="mps")
    model.freqs = model.freqs.to(torch.complex64)
    c, u = text_args("mps")
    for name, ta in (("cond", c), ("uncond", u)):
        with torch.no_grad(), torch.autocast(device_type="mps", dtype=torch.bfloat16):
            pred = model(x_t=x_in.to("mps"), t=torch.tensor([1000], device="mps"), text=ta["text"], text_lens=ta["text_lens"])
        np.save(OUT / f"bf16_pred_{name}.npy", pred.float().cpu().numpy())
    print("bf16 done", flush=True)


if __name__ == "__main__":
    main()
