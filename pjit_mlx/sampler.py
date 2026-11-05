"""P-JiT 21K's sampler (pjit_21k/sampler.py): 50 uniform Euler steps from t=1 to 0 on the x-prediction model, with
adaptive projected guidance (APG, Sadat et al. 2025) in velocity space against the empty caption.

The state, the APG momentum and projection, and the Euler update run on the CPU in NumPy with the reference's
dtypes (fp32 state, bf16-rounded model input and momentum, float64 projection); only the DiT runs on the GPU.
That keeps every non-network step bit-faithful at a cost of milliseconds per step.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np

from .noise import randn_h100


@dataclass
class SamplerConfig:
    height: int = 1024
    width: int = 1024
    sampling_steps: int = 50
    guidance_scale: float = 15.0
    apg_momentum: float = -0.75
    apg_eta: float = 0.0
    apg_rescale: float = 10.0
    noise_scale: float = 4.0
    negative_prompt: str = ""
    noise_sm_count: int | None = 132
    num_timesteps: int = 1000

    def to_dict(self):
        return asdict(self)


def linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
    """torch.linspace in fp32 as its kernels compute it: step = (end - start) / (steps - 1), the first half as
    start + step * i and the second half as end - step * (steps - 1 - i), each a fused multiply-add (one rounding)."""
    s, e = np.float32(start), np.float32(end)
    step = np.float32((e - s) / np.float32(steps - 1))
    i = np.arange(steps, dtype=np.float64)
    half = steps // 2
    lo = (np.float64(s) + np.float64(step) * i).astype(np.float32)
    hi = (np.float64(e) - np.float64(step) * (steps - 1 - i)).astype(np.float32)
    return np.where(i < half, lo, hi).astype(np.float32)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round fp32 to bf16 (round-to-nearest-even) and back, as tensor.to(torch.bfloat16).float()."""
    u = x.astype(np.float32).view(np.uint32)
    rounded = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000))
    nan = np.isnan(x)
    out = rounded.view(np.float32)
    if nan.any():
        out = np.where(nan, x, out)
    return out


def apg(v_cond, v_uncond, running_prev, s: SamplerConfig):
    """One APG step on (B, C, T, H, W) fp32 velocities. Returns (guided velocity, new running average)."""
    diff = v_cond - v_uncond
    running = diff + np.float32(s.apg_momentum) * running_prev
    diff = running
    if s.apg_rescale > 0:
        norm = np.sqrt(np.sum(diff.astype(np.float64) ** 2, axis=(1, 3, 4), keepdims=True)).astype(np.float32)
        diff = diff * np.minimum(np.float32(1.0), np.float32(s.apg_rescale) / norm)
    v0, v1 = diff.astype(np.float64), v_cond.astype(np.float64)
    dims = tuple(range(1, v1.ndim))
    v1 = v1 / np.maximum(np.sqrt(np.sum(v1 * v1, axis=dims, keepdims=True)), 1e-12)
    par = np.sum(v0 * v1, axis=dims, keepdims=True) * v1
    orth = (v0 - par).astype(np.float32)
    par = par.astype(np.float32)
    guided = v_cond + np.float32(s.guidance_scale) * (orth + np.float32(s.apg_eta) * par)
    return guided.astype(np.float32), running.astype(np.float32)


def sample(model, cond, uncond, seed: int, s: SamplerConfig, progress=None):
    """One image. cond / uncond: (1, L, 7680) caption features (MLX, bf16). Returns (H, W, 3) uint8 and timings."""
    noise = randn_h100((3, 1, s.height, s.width), seed=seed, sm_count=s.noise_sm_count or 132)
    x = (np.float32(s.noise_scale) * noise)[None].astype(np.float32)
    ts = linspace_f32(1.0, 0.0, s.sampling_steps + 1)
    t_int = (ts[:-1] * np.float32(s.num_timesteps)).astype(np.int64)
    running = np.zeros_like(x)
    cond_tok, uncond_tok = model.embed_text(cond), model.embed_text(uncond)  # constant over steps
    t0 = time.perf_counter()
    for i in range(s.sampling_steps):
        t, t_next = ts[i], ts[i + 1]
        x_in = to_bf16(x)
        xm = mx.array(x_in)
        tt = np.array([t_int[i]])
        pc = model(xm, tt, None, text_tokens=cond_tok)
        pu = model(xm, tt, None, text_tokens=uncond_tok)
        mx.eval(pc, pu)
        pc, pu = np.array(pc.astype(mx.float32)), np.array(pu.astype(mx.float32))
        v_cond = (x_in - pc) / t
        v_uncond = (x_in - pu) / t
        guided, running = apg(v_cond, v_uncond, to_bf16(running), s)
        x_pred = x_in - t * guided
        x = (x + (t_next - t) * ((x - x_pred) / t)).astype(np.float32)
        if progress:
            progress(i + 1, s.sampling_steps, time.perf_counter() - t0)
    img = np.round((np.clip(x, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)[0, :, 0].transpose(1, 2, 0)
    return img, {"sample_s": round(time.perf_counter() - t0, 2)}
