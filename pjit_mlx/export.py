"""Export MLX checkpoints: the DiT (bf16, or 8/6/4-bit) and a trimmed Qwen3.5-4B text encoder (blocks 0..27 and the
token embedding only: the vision tower, the MTP head and blocks 28..31 are never used), with its tokenizer files.

  python -m pjit_mlx.export --out OUT [--dit-bits 8] [--te-bits 8]
writes OUT/dit/{config.json, model.safetensors} and OUT/text_encoder/{config.json, model.safetensors, tokenizer
files}; each config.json carries a "pjit_mlx" block (format version, dtype, quantization, source revisions).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from . import dit
from .pipeline import REF_DIT, REF_QWEN, _snapshot, load_dit
from .text_encoder import EMBED_KEY, LAYERS, TextEncoder, sanitize

FORMAT = 1
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")


def _revision(path: Path) -> str:
    return path.name  # snapshot directory = commit hash


def export_dit(out: Path, bits: int | None = None, group_size: int = 64) -> dict:
    src = _snapshot(REF_DIT, ["config.json", "model.safetensors"])
    model = load_dit(src, bits, group_size)
    config = json.loads((src / "config.json").read_text())
    config["pjit_mlx"] = {"format": FORMAT, "dtype": "bfloat16",
                          "quantization": {"bits": bits, "group_size": group_size} if bits else None,
                          "source": {"repo": REF_DIT, "revision": _revision(src)}}
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), {"format": "mlx"})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return config["pjit_mlx"]


def export_text_encoder(out: Path, bits: int | None = None, group_size: int = 64) -> dict:
    src = _snapshot(REF_QWEN)
    raw = {}
    for f in sorted(src.glob("*.safetensors")):
        raw.update(mx.load(str(f)))
    embed = raw[EMBED_KEY]
    weights = sanitize(raw, max(LAYERS) + 1)
    del raw
    te = TextEncoder(src, weights=weights)
    if bits:
        from .quantize import quantize_text_encoder

        quantize_text_encoder(te.model, {"bits": bits, "group_size": group_size})
    config = json.loads((src / "config.json").read_text())
    config.pop("vision_config", None)
    config["text_config"]["num_hidden_layers"] = max(LAYERS) + 1
    config["text_config"]["layer_types"] = config["text_config"]["layer_types"][: max(LAYERS) + 1]
    config["pjit_mlx"] = {"format": FORMAT, "layers": list(LAYERS), "dtype": "bfloat16",
                          "quantization": {"bits": bits, "group_size": group_size} if bits else None,
                          "embedding": "embed_tokens.safetensors (bf16, read row by row)",
                          "source": {"repo": REF_QWEN, "revision": _revision(src)}}
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(te.model.parameters())), {"format": "mlx"})
    mx.save_safetensors(str(out / "embed_tokens.safetensors"), {"weight": embed}, {"format": "mlx"})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    for name in ("LICENSE",):
        if (src / name).exists():
            shutil.copy(src / name, out / "LICENSE.qwen")
    return config["pjit_mlx"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pjit_mlx.export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dit-bits", type=int, choices=(4, 6, 8))
    ap.add_argument("--te-bits", type=int, choices=(4, 6, 8))
    a = ap.parse_args(argv)
    out = Path(a.out)
    print("dit", export_dit(out / "dit", a.dit_bits))
    print("text_encoder", export_text_encoder(out / "text_encoder", a.te_bits))
    for p in sorted(out.rglob("*.safetensors")):
        print(p.relative_to(out), round(p.stat().st_size / 1e9, 2), "GB")


if __name__ == "__main__":
    main()
