# Parity and benchmark scripts

These are the scripts behind the numbers in the top-level README, kept as they were run. They expect a workspace laid
out like this (paths are relative to the workspace root, which is `parents[1]` of each script):

```
workspace/
  mlx/                this repository (pip install -e mlx into venv-mlx)
  ext/repo/           github.com/Linum-AI/pjit-21k @ 95403933, unmodified
  venv/               torch 2.15 + transformers 5.22, for the reference side
  venv-mlx/           mlx 0.32.4 + this package
  parity/             these scripts (copy this folder up one level)
  run_mps.py          the PyTorch-on-MPS runner (copy from this folder)
  h100_noise.py       the NumPy H100 noise draw, identical to pjit_mlx/noise.py (copy from this folder)
```

| Script | Side | What it does |
|---|---|---|
| `ref_dump.py` | torch | Caption features, step-0 input and the DiT's predictions in fp32 and bf16 autocast, saved to `parity/ref/` |
| `text_fp32_ref.py` | torch | The transformers text encoder in fp32 |
| `mlx_check.py` | MLX | DiT and text encoder against the dumps (`results.json`) |
| `text_check.py` | MLX | Text encoder per layer, fp32 and bf16 |
| `traj_ref.py`, `traj_mlx.py` | both | A full 50-step fp32 run of each sampler loop, compared pixel by pixel (`traj_results.json`) |
| `gallery.py` | MLX | Renders Linum's gallery entries from their prompt, seed and size and scores them against the originals (`gallery_results.json`) |
| `quant_sens.py` | MLX | Step-0 error for each quantization recipe (`quant_sens.json`) |
| `tiers.py`, `verify_export.py` | MLX | Speed, peak memory and bit-identity of the exported tiers (`export_check.json`) |
| `bench_*.py` | MLX | Forward-pass timings, batching, `mx.compile` and fused projections |
| `footprint.py` | any | Runs a command and reports its peak physical footprint (macOS) |

`gallery.py` reads `parity/gallery_pick.json` (key, category, seed, size and prompt per entry) and the
full-resolution originals in `ref/gallery/`. Neither is included: the prompts and images are Linum's, published at
[linum.ai/field-notes/pjit-21k](https://www.linum.ai/field-notes/pjit-21k). `gallery_results.json` lists the
entries tested with their PSNR; `b_512_s4` was tested separately (38.4 dB at bf16).

**Memory:** the README's figures are the process's peak physical footprint, the kernel's
`ri_lifetime_max_phys_footprint` from `proc_pid_rusage`. `footprint.py` measures them, wrapping one fresh process per
image. The `peak_gb` in `export_check.json` is MLX's own allocator peak (`mx.get_peak_memory()`). It reads lower
because it leaves out MLX's buffer cache and everything outside MLX. The CLI prints both, as `mlx_peak_gb` and
`footprint_gb`.
