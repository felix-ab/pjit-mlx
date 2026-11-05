"""Exported tiers must reproduce the on-the-fly renders bit for bit (out_mlx/tiers/b_512_s4_{bf16,d8t8}.png)."""
import gc
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "parity"))
import mlx.core as mx  # noqa: E402
from PIL import Image  # noqa: E402
from pjit_mlx.pipeline import Pipeline  # noqa: E402
from tiers import PROMPT  # noqa: E402

res = {}
for tier, ref in (("bf16", "bf16"), ("8bit", "d8t8")):
    d = W / f"export/P-JiT-21K-MLX-{tier}"
    mx.reset_peak_memory()
    pipe = Pipeline(d / "dit", d / "text_encoder")
    img, info = pipe.generate(PROMPT, 4, "4:3")
    img.save(W / f"out_mlx/tiers/b_512_s4_export_{tier}.png")
    a = np.asarray(img)
    b = np.asarray(Image.open(W / f"out_mlx/tiers/b_512_s4_{ref}.png").convert("RGB"))
    info["identical_to_on_the_fly"] = bool(np.array_equal(a, b))
    info["max_abs_diff"] = int(np.abs(a.astype(int) - b.astype(int)).max())
    res[tier] = info
    print(tier, json.dumps(info), flush=True)
    del pipe
    gc.collect()
    mx.clear_cache()
(W / "out_mlx/tiers/export_check.json").write_text(json.dumps(res, indent=1))
