"""Run P-JiT 21K (Linum-AI/pjit-21k, code ext/repo @95403933) on the M4 Pro: the repo's model, text encoder and APG
unchanged; its sampler loop re-done for MPS (2026-11-05, Felix: "try downloading this and recreating this image").

Changes from pjit_21k/sampler.generate, which is CUDA-only:
  - initial noise from h100_noise.randn_h100 (NumPy port of the repo's H100 SXM draw), so seeds match the gallery;
  - the DiT forward on MPS under bf16 autocast (as on CUDA), with RoPE kept out of autocast (the repo disables CUDA
    autocast there; the same is done for MPS);
  - the state, the APG projection (float64, which MPS lacks) and the Euler step on the CPU in fp32/fp64, exactly as
    the repo computes them;
  - the fp64 timestep embedding computed on the CPU;
  - the RoPE table as complex64 instead of complex128 (same fp32 cos/sin);
  - attention by PyTorch SDPA (FlashAttention-3 is Hopper-only; the repo says this differs by kernel rounding only).
Run under frontier scripts/gpu_cmd_long.sh (heavy-accelerator lock + 6 GB free-memory guard).
Usage: run_mps.py --key b_512_s4 --seed 4 --width 1152 --height 864 --prompt "..." [--te_device cpu|mps]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / "ext/repo"))
sys.path.insert(0, str(W))
import pjit_21k.model as pm  # noqa: E402
from h100_noise import randn_h100  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from pjit_21k import PyramidJiT, QwenTextEncoder, SamplerConfig, set_attention_backend  # noqa: E402
from pjit_21k.sampler import MomentumBuffer, _text_kwargs, adaptive_projected_guidance  # noqa: E402

for _name in ("compute_rotary_frequencies", "apply_rope"):  # fp32 RoPE, as the repo has it on CUDA
    setattr(pm, _name, torch.autocast(device_type="mps", enabled=False)(getattr(pm, _name)))
_sin_emb = pm.sinusoidal_embedding_1d  # the timestep embedding is computed in fp64: do it on the CPU, as exactly


def _sin_emb_cpu(dim, position, dtype):
    return _sin_emb(dim=dim, position=position.cpu(), dtype=dtype).to(position.device)


pm.sinusoidal_embedding_1d = _sin_emb_cpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--te_device", default="cpu")
    a = ap.parse_args()
    s = SamplerConfig(height=a.height, width=a.width)
    dev = "mps"
    set_attention_backend(backend="sdpa")
    t0 = time.perf_counter()

    qwen = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
    weights = snapshot_download("Linum-AI/pjit-21k", local_files_only=True, allow_patterns=["config.json", "model.safetensors"])
    te = QwenTextEncoder(model_path=qwen, extraction_layers=(7, 15, 27), max_length=256, device=a.te_device)
    cond_t, uncond_t = te([a.prompt])[0], te([s.negative_prompt])[0]
    n_tok = int(cond_t.shape[0])
    del te
    gc.collect()
    t_text = time.perf_counter() - t0

    model = PyramidJiT.from_pretrained(weights_dir=weights, device=dev)
    # the RoPE table is complex128 (fp64), which MPS lacks; complex64 gives bit-identical fp32 cos/sin, which is all
    # apply_rope uses (checked 11-05)
    model.freqs = model.freqs.to(torch.complex64)
    cond = _text_kwargs(text=cond_t.to(dev), batch_size=1, text_len=model.text_len)
    uncond = _text_kwargs(text=uncond_t.to(dev), batch_size=1, text_len=model.text_len)
    text = torch.cat([cond["text"], uncond["text"]])
    text_lens = torch.cat([cond["text_lens"], uncond["text_lens"]])
    t_load = time.perf_counter() - t0 - t_text

    noise = randn_h100((3, 1, s.height, s.width), seed=a.seed, sm_count=s.noise_sm_count)
    x = s.noise_scale * torch.from_numpy(noise)[None]                       # (1, 3, 1, H, W) fp32, CPU
    timesteps = torch.linspace(1.0, 0.0, s.sampling_steps + 1)
    timesteps_int = (timesteps[:-1] * s.num_timesteps).long()
    momentum = torch.zeros_like(x)
    t1 = time.perf_counter()
    with torch.no_grad():
        for i in range(s.sampling_steps):
            t, t_next = timesteps[i], timesteps[i + 1]
            x_in = x.to(torch.bfloat16).float()
            xm = x_in.to(dev)
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                pc, pu = model(x_t=torch.cat([xm, xm]), t=timesteps_int[i].expand(2).to(dev), text=text,
                               text_lens=text_lens).chunk(2)
            v_cond = (x_in - pc.float().cpu()) / t
            v_uncond = (x_in - pu.float().cpu()) / t
            mb = MomentumBuffer(momentum=s.apg_momentum)
            mb.running_average = momentum.to(torch.bfloat16).float()
            v_guided = adaptive_projected_guidance(pred_cond=v_cond, pred_other=v_uncond, guidance_scale=s.guidance_scale,
                                                   momentum_buffer=mb, eta=s.apg_eta, rescale=s.apg_rescale)
            momentum = mb.running_average
            x_pred = x_in - t * v_guided
            x = x + (t_next - t) * (x - x_pred) / t
            if i % 10 == 9:
                print(f"step {i + 1}/{s.sampling_steps}  {time.perf_counter() - t1:.0f} s", flush=True)
    t_sample = time.perf_counter() - t1
    img = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)[0, :, 0].permute(1, 2, 0).numpy()
    out = W / "out"
    out.mkdir(exist_ok=True)
    Image.fromarray(img).save(out / f"{a.key}_mps.png")
    meta = dict(key=a.key, prompt=a.prompt, seed=a.seed, width=a.width, height=a.height, caption_tokens=n_tok,
                sampler=s.to_dict(), te_device=a.te_device, attention="sdpa", torch=torch.__version__,
                seconds=dict(text=round(t_text, 1), load=round(t_load, 1), sample=round(t_sample, 1)))
    (out / f"{a.key}_mps.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["seconds"]), out / f"{a.key}_mps.png", flush=True)


if __name__ == "__main__":
    main()
