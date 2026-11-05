"""Seed parity with Linum's gallery: render the 12 picked gallery entries (parity/gallery_pick.json: 2 per category)
from their own prompt, seed and size, and score each against the full-resolution original (AVIF, ref/gallery/).
Usage: venv-mlx/bin/python -I parity/gallery.py [--dit-bits 8] [--te-bits 8] [--tag bf16]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

W = Path(__file__).resolve().parents[1]
from pjit_mlx.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-bits", type=int)
    ap.add_argument("--te-bits", type=int)
    ap.add_argument("--tag", default="bf16")
    a = ap.parse_args()
    pick = json.loads((W / "parity/gallery_pick.json").read_text())
    out = W / "out_mlx/gallery" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    pipe = Pipeline(dit_bits=a.dit_bits, te_bits=a.te_bits)
    for e in pick:
        if e["key"] in res:
            continue
        img, info = pipe.generate(e["prompt"], e["seed"], height=e["height"], width=e["width"])
        img.save(out / f"{e['key']}.png")
        ref = np.asarray(Image.open(W / "ref/gallery" / f"{e['key']}-full.png").convert("RGB"), dtype=np.float64)
        mine = np.asarray(img, dtype=np.float64)
        info["mad"] = round(float(np.abs(ref - mine).mean()), 2)
        info["psnr"] = round(float(10 * np.log10(255 ** 2 / np.mean((ref - mine) ** 2))), 2)
        res[e["key"]] = {"category": e["category"], **info}
        res_path.write_text(json.dumps(res, indent=1))
        print(e["key"], e["category"], info["psnr"], "dB", info["sample_s"], "s", flush=True)


if __name__ == "__main__":
    main()
