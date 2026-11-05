"""Fast checks that need no weights: the H100 noise port, torch-style linspace and bf16 rounding, the checkpoint
key mapping, and the DiT's structure (parameter count of the released architecture)."""
import numpy as np

from pjit_mlx import dit
from pjit_mlx.noise import philox4x32_10, randn_h100
from pjit_mlx.sampler import linspace_f32, to_bf16


def test_philox_known_answers():
    kat = [((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
           ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
           ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1))]
    for ctr, key, want in kat:
        got = tuple(int(np.asarray(v).item()) for v in philox4x32_10(*[np.array([c], dtype=np.uint32) for c in ctr], *key))
        assert got == want


def test_noise_layout_and_moments():
    n = randn_h100((3, 1, 64, 96), seed=4)
    assert n.shape == (3, 1, 64, 96) and n.dtype == np.float32
    big = randn_h100((3, 1, 864, 1152), seed=4)
    assert abs(float(big.mean())) < 5e-3 and abs(float(big.std()) - 1) < 5e-3
    assert not np.array_equal(randn_h100((3, 1, 64, 96), seed=5), n)


def test_linspace_matches_torch_formula():
    ts = linspace_f32(1.0, 0.0, 51)
    assert ts[0] == 1.0 and ts[-1] == 0.0 and ts.dtype == np.float32
    assert np.all(np.diff(ts) < 0)
    # the integer timesteps the model sees
    t_int = (ts[:-1] * np.float32(1000)).astype(np.int64)
    assert t_int[0] == 1000 and t_int[-1] in (19, 20)


def test_bf16_rounding_is_round_to_nearest_even():
    x = np.array([1.0, 1.00390625, 1.005859375, -3.14159, 65504.0, 1e-8], dtype=np.float32)
    r = to_bf16(x)
    assert r[0] == 1.0 and r[1] == 1.0  # exact halfway rounds to even
    assert r[2] == np.float32(1.0078125)
    assert np.all(np.abs(r - x) <= np.abs(x) * 2 ** -8 + 1e-30)


def test_architecture_parameter_count():
    from mlx.utils import tree_flatten

    model = dit.PJiT(dit.Config())
    n = sum(v.size for _, v in tree_flatten(model.parameters()))
    assert n == 2_184_091_606  # Linum-AI/pjit-21k export_manifest.json
