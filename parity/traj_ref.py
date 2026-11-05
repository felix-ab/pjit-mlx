"""End-to-end fp32 reference trajectory: the repo's sampler loop (pjit_21k/sampler.generate) with fp32 everywhere on
the CPU (no autocast), the H100 noise, the dumped caption features; 512x512, seed 4. Writes parity/ref/traj_fp32.npy.
Usage: venv/bin/python -I parity/traj_ref.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "ext/repo"))
sys.path.insert(0, str(W))
from h100_noise import randn_h100  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from pjit_21k import PyramidJiT, SamplerConfig, set_attention_backend  # noqa: E402
from pjit_21k.sampler import MomentumBuffer, _text_kwargs, adaptive_projected_guidance  # noqa: E402

R = W / "parity/ref"
H = Wd = 512


def main():
    set_attention_backend(backend="sdpa")
    weights = snapshot_download("Linum-AI/pjit-21k", local_files_only=True, allow_patterns=["config.json", "model.safetensors"])
    model = PyramidJiT.from_pretrained(weights_dir=weights, device="cpu")
    s = SamplerConfig(height=H, width=Wd)
    cond = _text_kwargs(text=torch.from_numpy(np.load(R / "text_cond.npy")), batch_size=1, text_len=256)
    unc = _text_kwargs(text=torch.from_numpy(np.load(R / "text_uncond.npy")), batch_size=1, text_len=256)
    text = torch.cat([cond["text"], unc["text"]])
    lens = torch.cat([cond["text_lens"], unc["text_lens"]])
    x = s.noise_scale * torch.from_numpy(randn_h100((3, 1, H, Wd), seed=4))[None]
    ts = torch.linspace(1.0, 0.0, s.sampling_steps + 1)
    ti = (ts[:-1] * s.num_timesteps).long()
    momentum = torch.zeros_like(x)
    t0 = time.time()
    with torch.no_grad():
        for i in range(s.sampling_steps):
            t, tn = ts[i], ts[i + 1]
            x_in = x.to(torch.bfloat16).float()
            pc, pu = model(x_t=torch.cat([x_in, x_in]), t=ti[i].expand(2), text=text, text_lens=lens).chunk(2)
            vc, vu = (x_in - pc.float()) / t, (x_in - pu.float()) / t
            mb = MomentumBuffer(momentum=s.apg_momentum)
            mb.running_average = momentum.to(torch.bfloat16).float()
            vg = adaptive_projected_guidance(pred_cond=vc, pred_other=vu, guidance_scale=s.guidance_scale,
                                             momentum_buffer=mb, eta=s.apg_eta, rescale=s.apg_rescale)
            momentum = mb.running_average
            xp = x_in - t * vg
            x = x + (tn - t) * (x - xp) / t
    np.save(R / "traj_fp32.npy", x.numpy())
    np.save(R / "traj_timesteps.npy", ti.numpy())
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
