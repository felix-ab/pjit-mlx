"""Torch Qwen3.5-4B features in fp32 (CPU) for the text-encoder parity floor: parity/ref/text32_{cond,uncond}.npy."""
import sys
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "ext/repo"))
import pjit_21k.text_encoder as tem  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

sys.path.insert(0, str(W / "parity"))
from ref_dump import PROMPT  # noqa: E402

qwen = snapshot_download("Qwen/Qwen3.5-4B", local_files_only=True)
orig = tem.AutoModelForCausalLM.from_pretrained
tem.AutoModelForCausalLM.from_pretrained = lambda p, torch_dtype=None, **k: orig(p, torch_dtype=torch.float32, **k)
te = tem.QwenTextEncoder(model_path=qwen, device="cpu")
for name, cap in (("cond", PROMPT), ("uncond", "")):
    np.save(W / f"parity/ref/text32_{name}.npy", te([cap])[0].float().numpy())  # note: the encoder casts outputs to bf16
    with torch.inference_mode():
        f = te.tokenizer.apply_chat_template([{"role": "user", "content": cap}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = te.tokenizer([f], return_tensors="pt")
        hs = te.model(**ids, output_hidden_states=True).hidden_states
    np.save(W / f"parity/ref/text32raw_{name}.npy", torch.cat([hs[i + 1][0] for i in (7, 15, 27)], -1).float().numpy())
print("ok")
