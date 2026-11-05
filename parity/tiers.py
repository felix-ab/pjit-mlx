"""Render the gallery image b_512_s4 in each precision tier and score it against the bf16 render and the gallery.
Usage: venv-mlx/bin/python -I parity/tiers.py [TIER...]   (tiers: bf16, d8, d6, d4, d8t8, d6t8, t8, t6)
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
import mlx.core as mx  # noqa: E402
from PIL import Image  # noqa: E402
from pjit_mlx.pipeline import Pipeline  # noqa: E402

PROMPT = ("Bright midday sun lights a white man in his thirties in a head-and-shoulders portrait, squinting slightly "
          "with a relaxed half smile. His faded blue chambray shirt is open at the collar, and a field of tall grass "
          "blurs behind him.")
TIERS = {"bf16": (None, None), "d8": (8, None), "d6": (6, None), "d4": (4, None), "d8t8": (8, 8), "d6t8": (6, 8),
         "t8": (None, 8), "t6": (None, 6)}
OUT = W / "out_mlx/tiers"


def psnr(a, b):
    a, b = (np.asarray(x, dtype=np.float64) for x in (a, b))
    return round(float(10 * np.log10(255 ** 2 / np.mean((a - b) ** 2))), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gallery = Image.open(W / "ref/b_512_s4-full.png").convert("RGB")
    res_path = OUT / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for tier in sys.argv[1:] or list(TIERS):
        db, tb = TIERS[tier]
        mx.reset_peak_memory()
        pipe = Pipeline(dit_bits=db, te_bits=tb)
        img, info = pipe.generate(PROMPT, 4, "4:3")
        img.save(OUT / f"b_512_s4_{tier}.png")
        info["psnr_vs_gallery"] = psnr(img, gallery)
        if (OUT / "b_512_s4_bf16.png").exists():
            info["psnr_vs_bf16"] = psnr(img, Image.open(OUT / "b_512_s4_bf16.png"))
        res[tier] = info
        res_path.write_text(json.dumps(res, indent=1))
        print(tier, json.dumps(info), flush=True)
        del pipe
        gc.collect()
        mx.clear_cache()


if __name__ == "__main__":
    main()
