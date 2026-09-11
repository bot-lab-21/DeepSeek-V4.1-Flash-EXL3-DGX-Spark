# DSV41 EXL3 TP4 — tuning backlog (living; ranked by expected impact on served decode/prefill at the pinned base)

Goal (user 2026-09-11): outperform every published 4×DGX-Spark serving of DeepSeek-V4.1-Flash, crediting every borrowed lever in
`CREDITS.md` the day it is adopted. Accept rule: +3 % C1/stream or C6 with ≤5 % prefill@47K loss and smokes passing; levers stack per
line, winners cross-applied, base re-measured at every hand-off (`ops/stack_next.sh`).

## Where we stand (line B, pin + NCCL_CH=8): C1 56.4 agg / 62.2 per-stream, C4 142.3, C6 199.0, TTFT 0.24 s, prefill 1337/1243/1423/1460, KV 2.56 M tokens

## A. Our own levers (from what we measured tonight)
| # | lever | why it should help | how to test | status |
|---|---|---|---|---|
| A1 | **Reclaim non-torch memory on the 640-col ranks → raise the KV pin** | profile shows non-torch 4–11 GiB/rank (NCCL buffers per channel ×8 now, Engram staging buffers, compile pools). Every GiB recovered on the worst rank is +200K tokens KV | log `non_torch` per rank at boot; try `NCCL_BUFFSIZE=2097152`, `NCCL_CUMEM_ENABLE=0/1`, Engram chunk 8 vs 16, `VLLM_CUDA_GRAPH pool` size; then re-pin | measure next |
| A2 | **ET64 as a prefill-profile win** | ENGRAM_THREADS=64 gave prefill@47K +12 % with decode flat (+1.6 %); the accept rule ignores prefill gains | re-test stacked on CH8; adopt under "decode ≥ −1 %, prefill ≥ +5 %" | re-test |
| A3 | **DSpark adaptive verification** (`enable_adaptive_verification=true`, launcher `SPEC_ADAPT`) | prose/narrative accept ~2 of 5 drafted tokens; adaptive k should stop wasting verification on low-acceptance streams | ladder lever `ADAPT` | queue |
| A4 | **DSpark k=7 / k=4** | code/math accept high (k=7 wins there), prose low (k=4 wins) — pick by measured mean acceptance; K7 queued on A | ladder K7 (queued), K4 after | running |
| A5 | **Async scheduling** (`--async-scheduling`) | overlaps CPU scheduling with the GPU step; TP4 decode is step-latency bound | ladder ASYNC (queued A) | queued |
| A6 | **RoCE one-shot all-reduce** (b12x) + `ROCE_MAX` threshold sweep (512 KB / 2 MB / 8 MB) | 40 all-reduces per token at TP4; unit test 2.6× at 360 KB | ladder ROCE (queued both lines), then ROCEMAX | queued |
| A7 | **MoE kernel block_m forcing** (`CUDA_EXL3_MOE_BLOCK_M=16/32`) | decode routes ~1 row/expert; the auto tier may pick 32 at C4–C6 and pad | needs generic env passthrough (`EXTRA_ENV`) in boot/launcher v2 | build v2 chain |
| A8 | **`--max-num-batched-tokens` 16384** for prefill | shipped 12K prefill dipped to 558 tok/s (chunk boundary); ours 1082–1243 | lever MAXB16K | queue |
| A9 | **Resident Engram rows on the GPU side** | 20 M CPU-pinned rows regressed decode −5.5 % (CPU hit/miss split + gather). A GPU gather over a device-resident fp8 table slice (6 GiB/rank per table at 100 M rows) removes the NVMe read *and* the CPU work; needs the KV pin lowered or A1 first | prototype after A1 | later |
| A10 | **Move the 640-col shards to the two nodes with the lowest baseline usage** | rank memory imbalance sets the KV pin; the head (API server) should stay 512 | P7 assignment order is [4,5,5,4]; verify per-node idle free and swap rank↔node mapping if useful | check |
| A11 | **Text-only profile** (`TEXT_ONLY=1`) for the API line that never sees images | vision tower + mm processor memory → KV; keep one line with vision | measure memory delta | later |
| A12 | **Quality battery timeout 7200 s** so MBPP+ completes | 90 min cut MBPP mid-run | v2 gate_run | build v2 |

## B. Borrowed levers (filled from the research pass; each row must have a source and a CREDITS.md entry when adopted)
(pending — see agents' reports appended below)

### B1. From tonyd2wild / tenaiaiai / alexellis (research pass 2026-09-11 01:10; credit each on adoption)
| # | lever | setting | source | their effect | us |
|---|---|---|---|---|---|
| B1.1 | GPU **clock-latch** burn gate before launch (distinct from the hidden slow state our gemv probe catches) | 15 s fp16 4096² matmul; healthy 75–90 TFLOPS / 2.2–2.4 GHz; refuse < 50 TFLOPS; fix = AC unplug 30–60 s | tonyd2wild docs/gpu-clock-latch.md, tools/prelaunch-quick.sh | two latched nodes: count 41.5→60.8, code 32.9→57.1 tok/s | ADD to node_prep_v2 (gputools/gpuburn.py) |
| B1.2 | Keep-warm ping when a line idles > ~10 min (hidden slow state appears after ~13 min idle, not 45 s) | tiny completion every ≤45 s while idle | tonyd2wild issue #1 | 63 vs 94 ms/step (92 vs 62 tok/s) | ADD to the production watchdog (not during ladder rows) |
| B1.3 | **Unconditional page-cache flush during load** (threshold flusher not enough) | `sync; echo 3 > drop_caches` every 60 s for the whole boot window | tonyd2wild GLM-5.3-Flash-4x flusher-unconditional.sh | KV pin 16 → 24 GiB/rank (+55 % pool) | = our lever A1/FLUSH8 (threshold 8 GiB) — corroborated, run next |
| B1.4 | `--max-num-batched-tokens 16384` | launcher MAX_BATCHED=16384 | alexellis (+11 % cold prefill, 0 decode cost); tonyd2wild GLM 4x (prefill +52–79 %, agg −3 %) | prefill profile | lever MAXB16K (v2 chain) |
| B1.5 | Concurrency cliff C6 → C8 (178–187 → 104 agg) | `--max-num-seqs 6` for latency lines | tenaiaiai next-en #3 | cliff between 6 and 8 streams | measure our C8 once; consider SEQS=6 on the API line |
| B1.6 | DSpark k=3 is −40 %; k=7 untested upstream (mean acceptance 3.57; ~6 on code, ~2 prose) | SPEC_K=7 | tenaiaiai #2/#17, tonyd2wild acceptance stats | — | lever K7 (queued both lines); ADAPT is risky: padded spec batches hang SM120 sparse MLA (FlashInfer #5015) → keep adaptive OFF unless proven |
| B1.7 | NCCL env set (`NCCL_IB_MERGE_NICS=0 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_CROSS_NIC=0 NCCL_IGNORE_CPU_AFFINITY=1`, GID 3, RoCE v2, addr range) + JIT caps (`MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1`) | — | tonyd2wild launch/dsv41-tp4.sh | collectives ≈5 ms of a 63 ms step | HAVE (all present in our launcher) |
| B1.8 | `nccl_lat.py` baseline (60 KB all-reduce p50 60 µs, 88 AR/step) | tools/nccl_lat.py | tonyd2wild results/boot10/prelaunch | — | run once as the A/B reference for the b12x one-shot path |
| B1.9 | Gates: `GPU KV cache size` token count as a regression signal; `completion_tokens>0 && text==""` garble check; per-rank `Engram DISK mode … rows [start,end)` must differ | — | tonyd2wild issue #2, field notes | silent regressions with /health 200 | ADD empty-text check to smokes; KV tokens already logged |
| B1.10 | Regular (non-breakable) CUDA graphs were faster on a qualified pair | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | alexellis deepseek-v4-flash-0731 docs/tuning.md | qualitative | only if our Engram prestage tolerates it (tony needs =1) — low priority |

### B2. From cuda-exl3 (Zeuss5; GB10 numbers by NNNtrance) / exllamav3 (turboderp) / MiaAI-Lab / brandonmusic (Festr) / FujitsuPolycom sparkring
| # | lever | setting | source | their effect | us |
|---|---|---|---|---|---|
| B2.1 | Live row count on the unsplit MoE launch (`n_rows`) | cuda-exl3 commit a95e809 | Zeuss5 / NNNtrance (#1) | 3×GB10 e2e C1 40.8→49.7, C4 59.5→99.6, prefill 1025→1257 | **HAVE** (our gemm.cu carries the fix and its comment) |
| B2.2 | Skip padding rows in gemm + had_in | `CUDA_EXL3_MOE_SKIP_PAD` (on by default) | cuda-exl3 #4/#5 | +1.6–4.5 % decode on GB10 | HAVE (skip_pad=True; had_in e<0 early return present) |
| B2.3 | block_m tier from the global expert count; pin with `CUDA_EXL3_MOE_BLOCK_M` | ladder rows/E_global: <16→16, <48→32, <96→64, else 128 | cuda-exl3 moe.py / #1 | ladder = measured optimum within 2 % | HAVE; levers BLOCKM16/BLOCKM32 for a sanity sweep on our shapes |
| B2.4 | MoE split-k accumulator | `CUDA_EXL3_MOE_ACC_MAX_ELEMS=0` frees the 32 MiB reservation | cuda-exl3 gemm.cu | split effectively off at serving M | try when memory-tight (tiny) |
| B2.5 | **Draft KV group page size 256** for the DSpark KV group (vLLM picks 16 via `_largest_kernel_block_within`) | vLLM kv_cache_utils patch | cuda-exl3 #2 (NNNtrance) | KV pool +82 % (2.43→4.41 M), C8 +6 %, TTFT −20–30 % | **investigate**: our 1M-context row reports 3.41 M tokens vs 2.56 M at 300K on the same 12 GiB pin → page geometry matters; check the group layout in the boot log |
| B2.6 | `--max-num-batched-tokens` multiple of block size | 8192 = 64×128 ✓ | cuda-exl3 #5 | 12.5 % wasted budget otherwise | HAVE |
| B2.7 | Indexer workspace right-size (vLLM default max_model_len×40 entries ≈ 5 GB @1M) | MiaAI `GLM53_INDEXER_WORKSPACE=rightsize` idea | MiaAI-Lab GLM-5.3 kit (AGPL — idea only) | ~26 % KV pool recovered | try for SERVE1M (our own patch, credit the idea) |
| B2.8 | Graph capture sizes must include seqs×(k+1) exactly (42 truncating to 40 = −12 % at c=6) | our CG_SIZES list is explicit (k·n, (k+1)·n) | MiaAI DSpark-2x | −12 % if padded | HAVE (verify the captured list in the log once) |
| B2.9 | PDL gate + tpurtell's top-k fixes on sm_12x | vLLM platforms/cuda.py `is_arch_support_pdl` | cuda-exl3 #6 | wash ±1 %; insurance | low |
| B2.10 | K-mix quality: flat K3 vs tiered mix KLD 0.0375 vs 0.0240 — a thin tail of damaged experts | — | brandonmusic GLM-5.3 3bpw card | quality | next cook: ledger-driven per-expert K (we already allocate per layer/matrix) |
| B2.11 | Merged shared gate+up (one small-M GEMM) | — | brandonmusic TR3v4 (Festr) | C1 +6–9 % | check vLLM's shared-expert path for V4.1 (MXFP4 shipped) |
| B2.12 | Do NOT: fused cooperative MoE decode kernel, expert re-read tricks, fp16 accumulation | — | cuda-exl3 related-work; brandonmusic | dead ends measured | skip |

### B3. From vLLM upstream / DeepSeek recipes / b12x (Luke Alonso) / r0b0tlab / vladimir-voinea / 0xSero / drowzeys
| # | lever | setting | source | their effect | us |
|---|---|---|---|---|---|
| B3.1 | b12x RoCE one-shot knobs | `VLLM_ROCE_ALLREDUCE_MAX_SIZE` (2 MB), `VLLM_ROCE_ALLGATHER_MAX_SIZE` (16 MB), `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0` (stripe both functions), `B12X_ROCE_GID_INDEX=3`, `B12X_ROCE_SPIN_LIMIT` | b12x docs/rocenante.md, local-inference-lab/vllm#597 | 4-Spark 48 KB AR 65.5→23.6 µs; GLM TP4 step 64→59 ms c1, c4 +6–22 %, c8 +5–14 % | ROCE rows queued; then ROCEMAX rows; HCA striping needs the 2nd rail at MTU 4096 (currently 1024, unused) — later |
| B3.2 | Adaptive verification | `enable_adaptive_verification:true` (+`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN`) | vLLM recipes YAML, vLLM blog (benchislett) | Pareto edge at concurrency; SM120 padded-batch hang root cause fixed in vLLM #51538 for the V4 tree | **OFF**: our V4.1 flashinfer_sparse.py has no seq_len clamp → port #51538 first, then lever ADAPT |
| B3.3 | Engram lookup overlap / side-stream prefetch | `--engram-config lookup_overlap` (vLLM #56220, 0z5a; #56357 Juntian777) | +2.2 % c4 (H200) | not in our build; port the idea onto the disk stager (overlap layer-14 rows with layers 1–13) — our own patch, credit the idea |
| B3.4 | Rust frontend | `VLLM_USE_RUST_FRONTEND=1` (recipe default; we run 0) | vLLM recipes | frontend CPU at concurrency | lever RUST |
| B3.5 | NCCL GID index pin (driver 580.159 reordered the GID table → silent TCP fallback, AR 22→424 µs) | `NCCL_IB_GID_INDEX=3` | r0b0tlab, tonyd2wild, eugr NETWORKING.md | 3.5× regression avoided | HAVE |
| B3.6 | Prefix caching once collapsed DSpark acceptance (vLLM #47930) | A/B `--no-enable-prefix-caching` once | vLLM issue | acceptance | try once (bench prompts are unique anyway) |
| B3.7 | Track DSpark acceptance length, not just tok/s (a V4 perf PR cost −10.6 % acceptance with all gates green, #49927) | parse SpecDecoding metrics per row | vLLM issue | — | ADD to gate summary |
| B3.8 | Upstream DSpark kernel PRs (mHC-pre broadcast #53972, Markov addmm #50737, stacked WKV #54674, KV-only context insert #55654/#56320) | cherry-pick onto deepseek_v4_1/nvidia/dspark.py | liuyao0322, yzyDavid, TeloySXH, 0z5a | +3–4 % e2e, TTFT wins | later |
| B3.9 | NVFP4 experts + b12x FP4 MoE on GB10 is NEGATIVE (80→44 tok/s) | — | drowzeys ds4-nvfp4-dspark-gb10 | — | skip the A/B (confirms EXL3 route) |
| B3.10 | `--async-scheduling` with FULL graphs: ROCm segfault on 3rd token (#56347) — smoke a 3+ token greedy ladder first | — | vLLM issue | — | ASYNC rung has smokes; add a 3-token greedy check |

### B4. GB10 system level (NVIDIA forums, NCCL docs, Sangiorgi, hazyumps, ShiningMeUp, agjs, Petronella, GaelicThunder, ArgentAIOS)
| # | lever | setting | source | their effect | us |
|---|---|---|---|---|---|
| B4.1 | Comms ≈ 8–10 % of the 63 ms step (88 ARs × ~57 µs) → NCCL env tuning caps at single digits at C1; node state (clock latch, thermal, host copies) is the big lever | — | Sangiorgi, hazyumps, tonyd2wild | framing | prioritise B1.1/B4.4 over env sweeps |
| B4.2 | NCCL Tree/LL for ≤2 MB messages on 4 nodes (Tree = 4 latencies vs Ring 6); never force LL128 over the net (nccl#831 corruption) | `NCCL_ALGO=Tree NCCL_PROTO=LL` | NVIDIA NCCL tuning blog | small-message latency | lever NCCLTREE (added) |
| B4.3 | NCCL_NTHREADS 128/256 (smaller CTAs pipeline tiny messages better) | `NCCL_NTHREADS=256` | NCCL docs | unmeasured on GB10 | levers NTH256/NTH128 (added) |
| B4.4 | Pageable small H2D copies are 50× slower than pinned on GB10; removing H2D staging gave +63 % | audit Engram row staging buffers (pinned?) | GaelicThunder gb10-uma-inference-notes, forum 354506 | large | **audit** Tony's stager + our P6 buffers |
| B4.5 | Bounded Engram page cache via `posix_fadvise(RANDOM/DONTNEED)` on the row fds instead of drop_caches rituals | fadvise in the disk reader | posix_fadvise(2); 0xSero io_uring reader | bounded cache, fewer flushes | try (our own patch) |
| B4.6 | Both NIC rails for NCCL: +9 % prefill but −5–9 % decode | keep single HCA for NCCL | Petronella CRS812 guide | — | HAVE (single) |
| B4.7 | GDR/DMA-BUF (`NCCL_NET_GDR_LEVEL=5 NCCL_DMABUF_ENABLE=1`): +bw for training, init hangs on GB10 | — | ArgentAIOS; NVIDIA: unsupported | risk | skip on live lines |
| B4.8 | IOMMU passthrough=1: lab-only (NVIDIA ships 0 deliberately) | — | Ampere SMMU wiki | risk | skip |
| B4.9 | Node identity: firmware/driver/kernel must match across ranks (a mismatched rank silently bottlenecks the head) | checked 01:35: UEFI 0x516 / 0x2009b0b / EC 0x3000508, driver 580.173.02, kernel 6.17.0-1029 on all 8 ✓ | tonyd2wild TROUBLESHOOTING | +140 % prefill when fixed | HAVE (verified) |
| B4.10 | zram: 1.8–3.7 GB swapped on every serving node (swappiness 1) → memory tightness signal; `vfs_cache_pressure=10000` is extreme (→100) | watch; consider `vm.swappiness=0` on serving nodes | memory-creep forum thread | — | note; FLUSH8 + pin discipline first |
| B4.11 | Clock cap 2100 for hard-off units costs −1 % decode; EC clock-latch fix = AC-cut power cycle | have on sp1/sp2 | forums 370304/376239/379389, ShiningMeUp, agjs | — | HAVE; burn gate added (B1.1) |

### B5. From MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks (SGLang, TP3/EP3 on 3 Sparks; AGPL-3.0 → ideas only, credit on adoption; checked 2026-09-11)
| # | lever | setting | their effect | us |
|---|---|---|---|---|
| B5.1 | **NCCL connection buffers in pinned host RAM = GPU memory on GB10** | `NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144 NCCL_PROTO=^LL128 NCCL_MAX_NCHANNELS=8` | 4.7 GiB → 0.14 GiB pinned per node; ended their boot-time "KV lottery" | **= our lever A1 exactly** (non-torch 4–12 GiB/rank). Add BUFFSIZE/LL128/PROTO knobs to the launcher; measure non-torch before/after → raise the KV pin |
| B5.2 | FP8 dense projections routed to FlashInfer's b12x warp-level MXFP8 kernel (32×32 ue8m0 blocks fall to Triton otherwise) | `--fp8-gemm-backend flashinfer_*` + adapter | dense GEMMs 52 → 17 ms per step (of 118 → 82 total) | **CHECK what vLLM uses for the 32×32-block FP8 dense on SM121** (profile one step); if Triton/CUTLASS, a b12x route is a large lever |
| B5.3 | `wo_a` einsum bf16 fallback + lm_head bf16 ≈ 13 ms/step | — | no FP8 kernel on SM121 | check our step profile for the same |
| B5.4 | KV bytes/token: 1,670.75 B/token/rank (only the 4 `kv_source` layers store KV) → 1M tokens = 1.67 GB | SGLang accounting | 750K pool in ~1.3 GB | **our vLLM pool: 12 GiB → 2.56 M tokens ≈ 4.7 KB/token** — 2.8× theirs. Investigate vLLM's KV group accounting for the encoder/decoder KV-sharing layers (+ indexer pages, block 128); ties to B2.5 |
| B5.5 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → NaN logits for prefills > 64 tokens on their stack | never set it | — | we don't set it; keep it that way |
| B5.6 | Engram: O_DIRECT 4 KiB reads, 96 IO threads, row cache 0 ("~0 % reuse") | `DSV41_IO_THREADS=96 DSV41_CACHE_GIB=0` | 1–3 ms/step | consistent with our HOT result (resident rows didn't pay) |

### C. Found on our own stack 2026-09-11 (kernel + KV accounting)
| # | lever | status |
|---|---|---|
| C1 | B12XLIN — force `B12xMxfp8LinearKernel` for MXFP8 dense linears (cutlass 3× slower at decode M) | row running on B |
| C2 | WOPROJ — b12x fused `wo_projection` for `_o_proj` (wo_a was bf16 emulation); patches/woproj | unit-tested 6–9× at M=6; served row + quality battery queued on B |
| C3 | KVGROUP — vLLM KV grouping heuristic (`DSV41_KV_GROUPING=fine`): +63 % tokens @1M, +43 % @300K (simulated) | patch being drafted (patches/kvgroup) |
| C4 | in-flight reservation: async + 16K batched costs −12 % KV → on A drop MAXB16K (or keep 8K batched with async) | apply at next A re-base |
| C5 | MAXB16K accepted on A (+4.6 % C6) but rejected on B (−0.7 %) and costs KV → treat as neutral; prefer KV | decide with C4 |
