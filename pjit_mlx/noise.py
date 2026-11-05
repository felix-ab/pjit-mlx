"""NumPy port of pjit_21k/noise.py's randn_as_on_sm_count: torch.randn as an H100 SXM (132 SMs) draws it, so a seed
gives the gallery's noise field on the Mac (the repo's version is a Triton kernel and needs an NVIDIA GPU).

torch's CUDA normal kernel launches min(ceil(numel/256), 8*SMs) blocks of 256 threads; thread i runs
curand Philox4x32-10 with key (seed_lo, seed_hi) at counter (offset/4 + iteration, 0, i_lo, i_hi), turns each 4 x u32
into 4 normals with curand's Box-Muller, and writes them to elements i + T*k (T = total threads), iteration after
iteration. Exact except for the GPU's sin.approx/cos.approx and logf, which NumPy replaces with float32 sin/cos/log
(differences of order 1e-6 in a unit normal).
Self-test: python -m pjit_mlx.noise --test   (Random123 known-answer vectors)
"""
import math
import sys

import numpy as np

M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
MASK = np.uint64(0xFFFFFFFF)
INV_2POW32 = np.float32(2.3283064e-10)
INV_2POW32_2PI = np.float32(np.float32(2.3283064e-10) * np.float32(6.2831855))
INV_2POW32_HALF = np.float32(INV_2POW32 / np.float32(2))
INV_2POW32_2PI_HALF = np.float32(INV_2POW32_2PI / np.float32(2))


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on uint32 arrays (Random123 / curand / Triton tl.philox)."""
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint32) for c in (c0, c1, c2, c3))
    k0, k1 = np.uint32(k0), np.uint32(k1)
    for r in range(10):
        if r:
            k0, k1 = np.uint32((int(k0) + int(W0)) & 0xFFFFFFFF), np.uint32((int(k1) + int(W1)) & 0xFFFFFFFF)
        p0 = c0.astype(np.uint64) * M0
        p1 = c2.astype(np.uint64) * M1
        hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), (p0 & MASK).astype(np.uint32)
        hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), (p1 & MASK).astype(np.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def box_muller(x, y):
    """curand's device Box-Muller: (sin(v) * s, cos(v) * s)."""
    xf = x.astype(np.float32).astype(np.float64)
    yf = y.astype(np.float32).astype(np.float64)
    u = (xf * np.float64(INV_2POW32) + np.float64(INV_2POW32_HALF)).astype(np.float32)  # fma, one rounding
    v = (yf * np.float64(INV_2POW32_2PI) + np.float64(INV_2POW32_2PI_HALF)).astype(np.float32)
    s = np.sqrt(np.float32(-2.0) * np.log(u))
    return np.sin(v) * s, np.cos(v) * s


def randn_h100(size, seed, sm_count=132, max_threads_per_sm=2048, offset=0):
    """The fp32 tensor torch.randn(size, generator=cuda_gen.manual_seed(seed)) gives on a GPU with sm_count SMs."""
    numel = math.prod(size)
    grid = min(math.ceil(numel / 256), max_threads_per_sm // 256 * sm_count)
    threads = 256 * grid
    iterations = (numel - 1) // (threads * 4) + 1
    seed = int(seed) % (1 << 32)
    idx = np.arange(threads, dtype=np.uint64)
    lo, hi = (idx & MASK).astype(np.uint32), (idx >> np.uint64(32)).astype(np.uint32)
    out = np.empty(threads * 4 * iterations, dtype=np.float32)
    zero = np.zeros(threads, dtype=np.uint32)
    for it in range(iterations):
        r0, r1, r2, r3 = philox4x32_10(zero + np.uint32(offset // 4 + it), zero, lo, hi, seed & 0xFFFFFFFF, seed >> 32)
        n0, n1 = box_muller(r0, r1)
        n2, n3 = box_muller(r2, r3)
        base = threads * 4 * it
        for k, n in enumerate((n0, n1, n2, n3)):
            out[base + k * threads: base + (k + 1) * threads] = n
    return out[:numel].reshape(size)


def _test():
    kat = [((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
           ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
           ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1))]
    for ctr, key, want in kat:
        got = tuple(int(np.asarray(v).item()) for v in philox4x32_10(*[np.array([c], dtype=np.uint32) for c in ctr], *key))
        print("KAT", "ok" if got == want else "FAIL", [hex(g) for g in got])
    n = randn_h100((3, 1, 864, 1152), seed=4)
    print("seed 4 noise: mean %.4f std %.4f, first %s" % (n.mean(), n.std(), n.reshape(-1)[:4]))


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
