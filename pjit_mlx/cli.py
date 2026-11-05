"""pjit-mlx: P-JiT 21K text-to-image on Apple silicon.

  pjit-mlx --tier 8bit --prompt "A dense 50-80 word caption ..." --seeds 4 --aspect 4:3 --out out.png
  pjit-mlx --prompt "..." --dit path/to/export/dit --text-encoder path/to/export/text_encoder --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import ASPECT_RATIOS, TIERS, Pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pjit-mlx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="caption (the model was trained on dense 50-80 word captions)")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds (noise matches the H100 SXM draw)")
    ap.add_argument("--aspect", default="square", choices=sorted(ASPECT_RATIOS))
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=15.0)
    ap.add_argument("--negative", default="")
    ap.add_argument("--tier", choices=sorted(TIERS),
                    help="download and use a published MLX tier: 8bit (3.6 GB peak, 6.9 GB download) or bf16 "
                         "(6.4 GB peak, 11.9 GB download)")
    ap.add_argument("--dit", help="exported MLX DiT directory (default: convert the released checkpoint)")
    ap.add_argument("--text-encoder", help="exported MLX text-encoder directory (default: Qwen/Qwen3.5-4B)")
    ap.add_argument("--dit-bits", type=int, choices=(4, 6, 8), help="quantize the reference DiT on the fly")
    ap.add_argument("--te-bits", type=int, choices=(4, 6, 8), help="quantize the text encoder on the fly")
    ap.add_argument("--upsample", action="store_true",
                    help="rewrite a short prompt into a dense caption with Linum's upsampler (Claude via OpenRouter; "
                         "needs OPENROUTER_API_KEY; sends the prompt to OpenRouter)")
    ap.add_argument("--out", default="pjit.png", help="output path; with several seeds, _s<seed> is added")
    a = ap.parse_args(argv)
    if (a.height is None) != (a.width is None):
        ap.error("--height and --width go together")
    if a.tier and (a.dit or a.text_encoder or a.dit_bits or a.te_bits):
        ap.error("--tier replaces --dit, --text-encoder, --dit-bits and --te-bits")
    prompt = a.prompt
    if a.upsample:
        from .pipeline import ASPECT_RATIOS as AR
        from .upsampler import resolve_api_key, upsample_prompt

        h, w = (a.height, a.width) if a.height else AR[a.aspect]
        res = upsample_prompt(prompt=a.prompt, height=h, width=w, api_key=resolve_api_key())
        print(json.dumps(res.to_dict(), indent=2), file=sys.stderr)
        if res.error:
            raise SystemExit(f"upsampler error: {res.error}; no image generated")
        prompt = res.upsampled_prompt
    pipe = Pipeline(a.dit, a.text_encoder, a.dit_bits, a.te_bits, tier=a.tier)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out = Path(a.out)
    for seed in seeds:
        path = out if len(seeds) == 1 else out.with_name(f"{out.stem}_s{seed}{out.suffix}")
        img, info = pipe.generate(prompt, seed, a.aspect, a.height, a.width, sampling_steps=a.steps,
                                  guidance_scale=a.guidance, negative_prompt=a.negative,
                                  progress=lambda i, n, t: print(f"\r{i}/{n} {t:.0f} s", end="", file=sys.stderr))
        img.save(path)
        print(f"\n{path} {json.dumps(info)}", file=sys.stderr)


if __name__ == "__main__":
    main()
