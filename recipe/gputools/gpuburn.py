#!/usr/bin/env python3
"""gpuburn.py — GB10 clock-latch gate (idea: tonyd2wild docs/gpu-clock-latch.md / tools/prelaunch-quick.sh).
Runs ~10 s of fp16 4096x4096 matmuls and prints achieved TFLOPS; a latched GPU sits at ~700-950 MHz and reads far below 50 TFLOPS
(healthy GB10: ~75-90). The hidden fast/slow *bandwidth* state is a different failure and is covered by gpuflip.py (gemv GB/s).
Exit 0 always; the caller gates on the printed number. Usage: python3 gpuburn.py [seconds] [min_tflops]
"""
import sys, time, torch
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
n = 4096
a = torch.randn(n, n, dtype=torch.float16, device="cuda"); b = torch.randn(n, n, dtype=torch.float16, device="cuda")
for _ in range(5):
    torch.matmul(a, b)
torch.cuda.synchronize()
t0 = time.time(); it = 0
while time.time() - t0 < secs:
    for _ in range(20):
        torch.matmul(a, b)
    torch.cuda.synchronize(); it += 20
dt = time.time() - t0
tflops = it * 2 * n ** 3 / dt / 1e12
print(f"gpuburn: {it} matmuls in {dt:.1f}s = {tflops:.1f} TFLOPS fp16 (4096^2)")
