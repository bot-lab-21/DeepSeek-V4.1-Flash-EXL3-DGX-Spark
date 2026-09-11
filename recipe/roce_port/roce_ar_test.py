#!/usr/bin/env python3
"""roce_ar_test.py — 2..4-node unit test of the b12x RoCE one-shot all-reduce inside the DSV41 vehicle (overlay6), outside vLLM.
Each rank: torch.distributed (nccl device group + gloo cpu group) → B12xRoceAllReduce(shim) → correctness vs NCCL for bf16 sizes
4 KB..2 MB → latency of 88 all-reduces of a DSV41 decode step (36 tokens × 5120 bf16 = 360 KB) RoCE vs NCCL. Env: RANK, WORLD_SIZE,
MASTER_ADDR, MASTER_PORT, VLLM_ENABLE_ROCE_ALLREDUCE=1 (+ the NCCL/IB env of dsv41_tp4_launch.sh)."""
import os, time, torch, torch.distributed as dist
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]); dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev); cpu = dist.new_group(backend="gloo"); dg = dist.group.WORLD
def log(m):
    if rank == 0: print(f"[rank0] {m}", flush=True)
import vllm.envs as envs
log(f"VLLM_ENABLE_ROCE_ALLREDUCE={envs.VLLM_ENABLE_ROCE_ALLREDUCE} max={envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE} world={world}")
from vllm.distributed.device_communicators.b12x_roce_all_reduce import B12xRoceAllReduce
ar = B12xRoceAllReduce(group=cpu, device_group=dg, device=dev)
print(f"[rank{rank}] roce disabled={ar.disabled}", flush=True); dist.barrier()
if ar.disabled: raise SystemExit("ROCE-AR-DISABLED")
ok = True
for n in (2048, 16384, 65536, 184320, 262144, 1048576):
    x = (torch.randn(n, device=dev) * (rank + 1)).to(torch.bfloat16)
    ref = x.clone(); dist.all_reduce(ref)
    if not ar.should_custom_ar(x): log(f"n={n} ({n*2/1024:.0f} KB): not eligible (size gate)"); continue
    y = ar.custom_all_reduce(x); torch.cuda.synchronize(); ar.check_health()
    err = (y.float() - ref.float()).abs().max().item(); rel = err / (ref.float().abs().max().item() + 1e-6)
    ok &= rel < 2e-2; log(f"n={n} ({n*2/1024:.0f} KB): max abs err {err:.4f} rel {rel:.2e} {'OK' if rel < 2e-2 else 'MISMATCH'}")
def bench(fn, x, iters=88, reps=5):
    best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
        for _ in range(iters): fn(x)
        torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
    return best * 1e3
x = torch.randn(36 * 5120, device=dev, dtype=torch.bfloat16)
t_nccl = bench(lambda t: dist.all_reduce(t), x.clone())
if ar.should_custom_ar(x):
    t_roce = bench(lambda t: ar.custom_all_reduce(t), x.clone()); ar.check_health()
    log(f"88 all-reduces of 36x5120 bf16 (360 KB): NCCL {t_nccl:.2f} ms  RoCE one-shot {t_roce:.2f} ms  ({t_nccl / t_roce:.2f}x)")
else: log(f"88 all-reduces of 360 KB: NCCL {t_nccl:.2f} ms; RoCE not eligible at this size")
x = torch.randn(6 * 5120, device=dev, dtype=torch.bfloat16)   # 6 tokens (single stream, k=5) = 60 KB
t_nccl = bench(lambda t: dist.all_reduce(t), x.clone()); t_roce = bench(lambda t: ar.custom_all_reduce(t), x.clone()) if ar.should_custom_ar(x) else float("nan"); ar.check_health()
log(f"88 all-reduces of 6x5120 bf16 (60 KB): NCCL {t_nccl:.2f} ms  RoCE {t_roce:.2f} ms")
ar.close(); dist.barrier(); log("ROCE-AR-TEST " + ("PASS" if ok else "FAIL")); dist.destroy_process_group()
