# recipe/ — how the 4× DGX Spark EXL3 line is built, booted and gated

Everything here is the actual tooling we run, with site identifiers replaced by placeholders. It is published as a *record* of the recipe, not as a turnkey installer: expect to edit the node map and paths.

> **Built on two people's work above all.** The quantization follows the **Pollard method** as framed and documented in [WestWaters/pollard-weights](https://github.com/WestWaters/pollard-weights) (Hessian-aware, sensitivity-allocated expert quantization; our ledgers and tools are contributed back there). The serving recipe is **[tonyd2wild's DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)** — the patch set, Engram-on-NVMe staging (with Kai), worker-first boot, image chain and bench protocol; we changed the expert bytes and added a few levers on top. If you use this, cite them first.

| placeholder | meaning |
|---|---|
| `user@RANK_LAN_IP` | ssh target of a rank node (we use one Linux user per Spark) |
| `LAN_IP` | a 1/10 GbE management address (head API, hubs) |
| `FABRIC_IP` | the node's address on the 200 GbE ConnectX fabric (`VLLM_HOST_IP`, NCCL) |
| `ARCHIVE_ROOT`, `user@ARCHIVE_HOST` | the NAS that holds archives and the HF staging copy |
| `vault-get NAME` | our secrets helper (SOPS+age); replace with your own secret source |
| `~/bin/tg-send`, `mbx` | our notification / mailbox helpers; safe to delete |

## Files

| file | role |
|---|---|
| `RECIPE_TP4_EXL3.md` | the serving recipe: memory ritual, knobs, gate rows, accept rule, the drafter fix |
| `dsv41_tp4_launch.sh` | per-rank launcher (`docker run` of the vehicle image): patch mounts, env knobs, **per-rank memory gate** right before launch |
| `dsv41_boot_tp4.sh` | one-line boot orchestration: preflight (checkpoint signature parity, image + patches, MemAvailable), node prep, worker-first launch, health wait, warm-up, smokes, bench |
| `dsv41_node_prep.sh` | the memory ritual on every node: sysctls, compaction, drop caches, cache-flusher sidecar, `compaction_proactiveness=0`, earlyoom off, fabric MTU 9000 + jumbo ping, GPU fast/slow probe, CUDA-free gate, checkpoint gate |
| `dsv41_gate_run.sh`, `dsv41_ladder.sh` | gate rows (G0, ROCE, ET64, CH8, K7, ASYNC, HOT, SERVE1M …) and the cumulative tuning ladder with the accept rule |
| `patches/mounts.txt` | which files are bind-mounted over the vehicle image (`<local file> <path under site-packages/vllm>`; `../cuda_exl3/config.py` for the plugin) |
| `patches/engram.py` | tonyd2wild's Engram-on-NVMe module + our P6 resident hot rows (`DSV41_ENGRAM_HOT_DIR/ROWS`, inert unless set) |
| `patches/exl3_config.py` | cuda-exl3 1.0.3 `config.py` + the worker-side model-path fallback needed for DSpark drafters (see recipe doc) |
| `patches/exl3_moe.py` | cuda-exl3 1.0.3 `moe.py` + uneven 128-aligned TP split of the expert intermediate dim (V4.1-Flash: 2304/4 = 576 is not a multiple of the kernel block; ranks get 512/640/640/512) |
| `patches/test_*.py` | in-image smoke tests for the two patches |
| `images/Dockerfile.overlay6/7/8` | vehicle image chain: tonyd2wild overlay5 → + RoCE/PCIe one-shot all-reduce shim (b12x) → + cuda-exl3 with the V4.1 overlay → + spin-wait fix |
| `roce_port/*.diff`, `roce_ar_*.{py,sh}` | the vLLM shim diffs for the b12x one-shot collectives and the 2-node bit-exactness / latency test |


## Tuning chain (v2) and helpers
| file | role |
|---|---|
| `dsv41_ladder3.sh` → `dsv41_gate_run_v2.sh` → `dsv41_boot_tp4_v2.sh` → `dsv41_node_prep_v2.sh` | the fixed ladder (row-scoped smoke check), lever rows (ROCE, CH8, ET64, K7, ASYNC, HOT, SERVE1M, MAXB16K, FLUSH8, NCCLTREE, NTH256, RUST, BLOCKM16/32, ADAPT), boot with knob passthrough, prep with the GPU slow-state probe **and** the clock-latch burn gate (`gputools/gpuburn.py`) |
| `ops/stack_next.sh`, `ops/ladder_relaunch.sh`, `ops/boot_prod.sh` | hand-off between ladders (re-derive verdict, cross-apply the other line's wins, re-measure the base), relaunch by PID tree, boot a line on its accepted base |
| `ops/accepted-{A,B}.example.txt` | the accepted bases we serve (pinned KV, GMU guard, MemFree floor, NCCL channels 8, RoCE; A adds async + 16K batch) |
| `ops/publish_best.py`, `ops/publish_best_loop.sh` | rebuild the "best serving configuration" block from the bench rows and push the card |
| `ops/TUNING_BACKLOG.md` | every lever tried or queued, with source and credit |
| `patches/roce_cuda_communicator.py` | vLLM CUDA communicator with the b12x RoCE shim (fixes the unbound-list bug in the backend logger) |
| `dsv41_quality_gate.sh` | ppl probe, needle, tool-call, evalplus HumanEval+/MBPP+ battery |

## Boot in one line

```
LINE=A PROFILE=exl3 EXP_NAME=g0 bash dsv41_boot_tp4.sh          # prep + launch + health + smokes + bench
bash dsv41_ladder.sh A ROCE K7 ASYNC HOT SERVE1M                # cumulative tuning ladder, accept rule inside
```

## Third-party code in this folder

`patches/engram.py` derives from the vLLM project (Apache-2.0) and tonyd2wild's DGX Spark patches (MIT). `patches/exl3_config.py` derives from cuda-exl3 (MIT). `roce_port/*.diff` patch vLLM files (Apache-2.0) to call the b12x runtime (Apache-2.0). `images/Dockerfile.*` build on tonyd2wild's image chain (MIT) and eugr's spark-vllm-docker lineage (MIT). Our own scripts are MIT. See `../THIRD_PARTY_NOTICES.md`.
