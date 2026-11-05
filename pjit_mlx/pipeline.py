"""P-JiT 21K text-to-image in MLX: caption features (Qwen3.5-4B blocks 7/15/27), then the APG Euler sampler on the
x-prediction DiT. Loads either the released torch checkpoints (converted on the fly) or an exported MLX layout
(pjit_mlx.export), and frees the text encoder before the DiT is loaded so the two never share memory.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from . import dit
from .sampler import SamplerConfig, sample
from .text_encoder import TextEncoder

ASPECT_RATIOS = {"square": (1024, 1024), "tall": (1216, 864), "4:3": (864, 1152), "3:2": (832, 1248), "16:9": (720, 1280)}
REF_DIT, REF_QWEN = "Linum-AI/pjit-21k", "Qwen/Qwen3.5-4B"
TIERS = {"bf16": "VisualInference/P-JiT-21K-MLX-bf16", "8bit": "VisualInference/P-JiT-21K-MLX-8bit"}


def _snapshot(repo: str, patterns=None) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo, local_files_only=True, allow_patterns=patterns))
    except Exception:
        return Path(snapshot_download(repo, allow_patterns=patterns))


def tier_paths(tier: str) -> tuple[Path, Path]:
    """The dit/ and text_encoder/ directories of a published MLX tier, downloaded on first use."""
    root = _snapshot(TIERS[tier], ["dit/*", "text_encoder/*"])
    return root / "dit", root / "text_encoder"


def load_dit(path: str | Path | None = None, bits: int | None = None, group_size: int = 64) -> dit.PJiT:
    """An exported MLX DiT directory (config.json with a "pjit_mlx" block), or the reference checkpoint; bits
    quantizes the reference on the fly."""
    if path is None:
        path = _snapshot(REF_DIT, ["config.json", "model.safetensors"])
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = dit.PJiT(dit.Config.from_dict(config["architecture"]))
    exported = config.get("pjit_mlx")
    if exported:
        if exported.get("quantization"):
            from .quantize import quantize_dit

            quantize_dit(model, exported["quantization"], from_weights=False)
        weights = mx.load(str(path / "model.safetensors"))
    else:
        from .convert import load_reference

        weights = load_reference(str(path / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=True)
    if bits and not exported:
        from .quantize import quantize_dit

        quantize_dit(model, {"bits": bits, "group_size": group_size})
    model.eval()
    mx.eval(model.parameters())
    return model


class Pipeline:
    def __init__(self, dit_path=None, text_encoder_path=None, dit_bits=None, te_bits=None, tier=None):
        if tier:
            dit_path, text_encoder_path = tier_paths(tier)
        self.dit_path, self.te_path = dit_path, text_encoder_path
        self.dit_bits, self.te_bits = dit_bits, te_bits
        self.model = None
        self._features = {}  # caption -> (1, L, 7680) bf16, so further seeds of a prompt skip the encoder

    def encode(self, captions: list[str]):
        todo = [c for c in dict.fromkeys(captions) if c not in self._features]
        if todo:
            if self.model is not None:  # never hold the encoder and the DiT at once (the DiT reloads in ~0.3 s)
                self.model = None
                gc.collect()
                mx.clear_cache()
            te = TextEncoder(self.te_path or _snapshot(REF_QWEN))
            if self.te_bits:
                from .quantize import quantize_text_encoder

                quantize_text_encoder(te.model, {"bits": self.te_bits})
            for c in todo:
                self._features[c] = te(c)[None]
            mx.eval(list(self._features.values()))
            del te
            gc.collect()
            mx.clear_cache()
        return [self._features[c] for c in captions]

    def generate(self, prompt: str, seed: int, aspect: str = "square", height: int | None = None,
                 width: int | None = None, progress=None, **sampler_kw):
        h, w = (height, width) if height else ASPECT_RATIOS[aspect]
        s = SamplerConfig(height=h, width=w, **sampler_kw)
        t0 = time.perf_counter()
        cond, uncond = self.encode([prompt, s.negative_prompt])
        t_text = time.perf_counter() - t0
        if self.model is None:
            self.model = load_dit(self.dit_path, self.dit_bits)
        t_load = time.perf_counter() - t0 - t_text
        img, info = sample(self.model, cond, uncond, seed, s, progress)
        info.update(text_s=round(t_text, 2), load_s=round(t_load, 2), tokens=int(cond.shape[1]),
                    peak_gb=round(mx.get_peak_memory() / 1e9, 2))
        return Image.fromarray(img), info
