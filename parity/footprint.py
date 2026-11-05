"""Run a command and report its peak physical footprint on macOS: the kernel's lifetime maximum for the process
(rusage_info_v4.ri_lifetime_max_phys_footprint), the meter behind every memory figure in the README.
Usage: python parity/footprint.py -- pjit-mlx --tier 8bit --low-memory --prompt "..."
"""
import ctypes
import subprocess
import sys
import time

LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")


def lifetime_max_footprint(pid: int) -> int | None:
    buf = (ctypes.c_uint64 * 64)()
    if LIBPROC.proc_pid_rusage(pid, 4, ctypes.cast(buf, ctypes.c_void_p)) != 0:
        return None
    return buf[2 + 28]  # after the 16-byte uuid


def main():
    cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    t0 = time.time()
    child = subprocess.Popen(cmd)
    peak = 0
    while child.poll() is None:  # the value is monotonic, so polling until exit keeps the peak
        peak = max(peak, lifetime_max_footprint(child.pid) or 0)
        time.sleep(0.05)
    print(f"exit {child.returncode}, wall {time.time() - t0:.1f} s, peak footprint {peak / 1e9:.2f} GB", file=sys.stderr)
    sys.exit(child.returncode)


if __name__ == "__main__":
    main()
