# pjit-mlx

[P-JiT 21K](https://huggingface.co/Linum-AI/pjit-21k) on Apple silicon with [MLX](https://github.com/ml-explore/mlx).
P-JiT 21K is Linum's 2.18B-parameter pixel-space text-to-image DiT, with a frozen Qwen3.5-4B caption encoder.

The port is exact against the released PyTorch code. It reproduces Linum's own gallery images from their seeds on a
Mac, and it runs about twice as fast as PyTorch on the same machine, in under half the memory.

| M4 Pro (48 GB), one 1152×864 image, 50 steps | PyTorch 2.15 (MPS) | pjit-mlx bf16 | pjit-mlx 8-bit |
|---|---|---|---|
| Caption encoding | 17.5 s | 1-2 s | ~1 s |
| Sampling | 118 s | 69 s | 77 s |
| Total, including loading | 139 s | 72 s | 79 s |
| Peak memory, whole process | 14.9 GB | 6.7 GB | 5.0 GB |
| Peak memory with `--low-memory` | | 5.6 GB | **3.6 GB** |
| Match to Linum's H100 image (PSNR) | 37.8 dB | 38.4 dB | 18.1 dB (this seed forks) |
| Download | 18.0 GB | 11.9 GB | 6.9 GB |

Peak memory is the process's peak physical footprint as macOS reports it (`proc_pid_rusage`), the same meter in
every column; it includes MLX's buffer cache, Python and the tokenizer. MLX's own counter (`mx.get_peak_memory()`)
reads lower and is not used for these numbers. `--low-memory` gives identical pixels: see Memory below.

## Install and run

```bash
pip install git+https://github.com/felix-ab/pjit-mlx
pjit-mlx --tier bf16 --seeds 4 --aspect 4:3 --out portrait.png \
  --prompt "Bright midday sun lights a white man in his thirties in a head-and-shoulders portrait, squinting slightly with a relaxed half smile. His faded blue chambray shirt is open at the collar, and a field of tall grass blurs behind him."
```

That command reproduces gallery image [`b_512_s4`](https://www.linum.ai/field-notes/pjit-21k#img-b_512_s4).

- **Weights:** `--tier` downloads an MLX export from Hugging Face on first use.
  - [`bf16`](https://huggingface.co/VisualInference/P-JiT-21K-MLX-bf16): 6.7 GB peak (5.6 GB with `--low-memory`),
    11.9 GB download. It is the reference precision.
  - [`8bit`](https://huggingface.co/VisualInference/P-JiT-21K-MLX-8bit): 5.0 GB peak (3.6 GB with `--low-memory`),
    6.9 GB download. It forks on some seeds, including this one (see Tiers).
  - Without `--tier`, the released checkpoints (`Linum-AI/pjit-21k`, `Qwen/Qwen3.5-4B`, 18 GB) are converted on load.
  - `--dit DIR/dit --text-encoder DIR/text_encoder` uses a local export. Make one with
    `python -m pjit_mlx.export --out DIR [--dit-bits 8 --te-bits 8]`.
- **Sizes:** the model was trained at five ~1 MP sizes: `square` 1024², `tall` 864×1216, `4:3` 1152×864,
  `3:2` 1248×832 and `16:9` 1280×720.
- **Captions:** it was trained on dense 50-80 word captions and cannot draw text.

```python
from pjit_mlx.pipeline import Pipeline

pipe = Pipeline(tier="8bit", low_memory=True)   # or Pipeline() for the released checkpoints, or Pipeline("DIR/dit", "DIR/text_encoder")
image, info = pipe.generate(caption, seed=4, aspect="4:3")
image.save("out.png")
```

## Exactness

**Parity against the reference code** (`parity/`) uses the same caption features, the step-0 input and t = 1000:

| | Relative L2 error | PSNR |
|---|---|---|
| DiT, MLX fp32 vs PyTorch fp32 | 2.8e-6 | 135 dB |
| DiT, MLX bf16 flow vs PyTorch fp32 | 1.1e-2 | 63.5 dB |
| DiT, PyTorch bf16 autocast vs PyTorch fp32 | 1.0e-2 | 64.5 dB |
| Text encoder, MLX fp32 vs transformers fp32 | 3e-6 | 123 dB |
| Text encoder, MLX bf16 vs transformers fp32 | 1.08e-2 | 52.5 dB |
| Text encoder, transformers bf16 vs transformers fp32 | 1.06e-2 | 52.7 dB |

In bf16 both components sit at the reference's own rounding noise.

**End to end**, a full 50-step sampling run in fp32 (512², seed 4) with the same caption features goes through the
model, the APG sampler, the H100 noise and the timestep schedule. Against the repo's own sampler loop in PyTorch, MLX
gives **99.5% identical pixels**; the rest differ by 1/255 at most (70.8 dB), and the integer timesteps agree
exactly.

**Seed parity with Linum's gallery** covers 15 images, across all five sizes. Twelve were drawn two per category
(person, animal, building, product, landscape, stylized), two more at 3:2, plus `b_512_s4`. Each was rendered from
its published prompt, seed and size at the bf16 tier. All 15 come out as the same picture as the H100 original:

- PSNR against the full-resolution AVIF originals has a median of 34.8 dB and ranges from 24.5 to 43.1 dB.
- The low end is where fine detail drifts, such as clay texture and grass, not a different composition.

**How it stays faithful:**
- **Noise:** the starting noise is the H100 SXM draw that Linum's images used, computed in NumPy.
  - It is curand's Philox4x32-10 and Box-Muller laid out as torch's CUDA kernel lays them out (`pjit_mlx/noise.py`).
  - It passes the Random123 known-answer tests.
  - The only difference from the GPU is the hardware `sin.approx` and `cos.approx`, about 1e-6.
- **Precision flow:** the DiT follows the reference's bf16-autocast dtypes op for op.
  - Linears run in bf16.
  - RMSNorm runs in fp32 and returns fp32; the output LayerNorm also runs in fp32.
  - RoPE is fp32, with tables from float64.
  - Attention runs in bf16 as FlashAttention-3 does.
  - The residual stream is fp32.
  - Nonlinearities on bf16 tensors are evaluated in fp32 and rounded once.
- **Text encoder:** it uses the transformers-faithful Qwen3.5 layers (gated delta rule in fp32, fp32 `1 + w` norms),
  first validated in the MOSS-Image MLX port.
- **Sampler:** the APG momentum, the float64 projection and the Euler update run in NumPy with the reference's
  dtypes. That includes torch's fused-multiply-add `linspace` and bf16 round-to-nearest-even.
- **Captions:** they run at their true length instead of padded to 256. The reference masks the padding, so every
  valid token's value is unchanged.

## Tiers

- **bf16** is the reference precision. Use it whenever ~7 GB fits.
- **8-bit** quantizes every linear inside the transformer blocks and the caption projection to 8 bits (group 64). It
  also quantizes the text encoder's block projections.
  - It keeps the token embedding, patch embed, timestep MLP, AdaLN roots, attention gates and output head in bf16.
  - It peaks at 5.0 GB for the whole process, or 3.6 GB with `--low-memory`.
  - It looks as clean as bf16 at 100%: no added grain or mesh.
  - It keeps the gallery picture on 14 of the 15 test prompts, with more fine-detail drift than bf16 (median 29.6 dB
    against 34.8 dB).
  - It does not guarantee seed parity. P-JiT samples with strong adaptive projected guidance (scale 15), and 8-bit
    weights move the first prediction to ~58 dB from fp32, against 63.5 dB for bf16. On some prompts the trajectory
    forks: `b_512_s4` becomes a different man.
  - Keeping attention, the FFNs, the refiners, the AdaLN or the first and last blocks in bf16 does not close that gap
    (57-61 dB).
  - A 4-bit DiT is possible (`--dit-bits 4`), but it is not exported: it forks more.

## Memory

All numbers are the whole process's peak physical footprint on macOS, for one image in a fresh process.

| 8-bit tier | Default | `--low-memory` |
|---|---|---|
| 1152×864, 67-token caption | 5.04 GB | 3.55-3.59 GB |
| 864×1216, 102-token caption | | 3.50 GB |
| 864×1216, 256-token caption (the maximum) | 5.02 GB | 3.51 GB |
| bf16 tier, 1152×864 | 6.65 GB | 5.56 GB |

- **Text encoder:** it always runs layer by layer. Each block's weights are read from disk when that block runs and
  dropped right after (`mx.load` is lazy). Caption encoding peaks at ~0.6 GB with bit-identical features, and the
  encoder and the DiT are never resident together.
- **Default sampling:** the DiT's weights (2.35 GB at 8-bit) stay resident, and each forward is one lazy graph. Every
  block still pending in that graph holds ~0.26 GB, about 1.3 GB per forward, and MLX's buffer cache keeps about as
  much again between steps.
- **`--low-memory`:** it evaluates the DiT block by block and caps MLX's buffer cache at 256 MB while sampling.
  - The pixels are identical: every test image was bit-identical to the default run.
  - In 50-step end-to-end runs it cost under 1% in time; a 12-step microbenchmark showed up to 8%.
- `parity/footprint.py` measures a command's peak footprint the same way.

## Speed notes

Sampling time per image (50 steps) on an M4 Pro, averaged over the gallery set:

| Size | bf16 | 8-bit |
|---|---|---|
| 1024×1024 | 71.7 s | 76.5 s |
| 864×1216 | 72.0 s | 76.7 s |
| 1152×864 | 68.3 s | 73.1 s |
| 1248×832 | 70.9 s | 75.7 s |
| 1280×720 | 65.7 s | 70.1 s |

- One DiT forward at 1152×864 (1,039 tokens) takes 0.74 s on an M4 Pro, about 6 TFLOPS of bf16 GEMM.
- Each block is `mx.compile`d, which is bit-identical and 6% faster.
- Some changes were measured and are not used:
  - Batching the conditional and unconditional passes with padding is slower (1.49 s vs 1.45 s per step).
  - Fusing the q/k/v/gate projections gives no gain.
  - Quantized GEMMs are not faster at this token count.
- The token embedding (1.27 GB) is memory-mapped and read row by row.

## Limits

- Prompt upsampling is opt-in. `--upsample` runs Linum's upsampler unchanged (`pjit_mlx/upsampler.py`): it rewrites a
  short idea into a dense caption with Claude through OpenRouter. It needs `OPENROUTER_API_KEY` and sends the prompt
  to OpenRouter. Linum's CLI upsamples by default; here nothing leaves the machine unless asked.
- The model itself cannot render text, and it struggles with full-body figures and crowds (see Linum's model card).

## License

- **This code:** Apache-2.0.
- **P-JiT 21K weights:** Apache-2.0, © Linum Inc.
- **Qwen3.5-4B:** Apache-2.0, © Alibaba Cloud.
- See `NOTICE`.
