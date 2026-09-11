# DeepSeek-V4.1-Flash on four DGX Sparks with EXL3 routed experts (3.5 bpw) — recipe and receipts

**Status: work in progress (2026-09-11) — the checkpoint serves on two 4-node lines; cells marked TBD are being filled from the tuning gate. Every number below was measured on our cluster.**

> **Built on two people's work above all.** The quantization follows the **Pollard method** as framed and documented in [WestWaters/pollard-weights](https://github.com/WestWaters/pollard-weights) (Hessian-aware, sensitivity-allocated expert quantization; our ledgers and tools are contributed back there). The serving recipe is **[tonyd2wild's DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)** — the patch set, Engram-on-NVMe staging (with Kai), worker-first boot, image chain and bench protocol; we changed the expert bytes and added a few levers on top. If you use this, cite them first.

DeepSeek-V4.1-Flash is a 552B-parameter mixture-of-experts model (40 layers × 384 routed experts, two Engram n-gram tables, DSpark speculative head) shipped as MXFP4 experts + FP8 dense, 510 GB. It fits four 128 GB DGX Spark (GB10) nodes only because tonyd2wild's recipe keeps the Engram tables on NVMe, and even then leaves about 7 GiB per rank for KV cache. This repository is the recipe that re-quantizes the routed experts to an EXL3 trellis at **3.5 bits per weight** with the Pollard method (Hessian-aware, per-expert, allocated by measured sensitivity), keeps everything else bit-identical, and serves it with vLLM + cuda-exl3 on the same four nodes with the freed memory going to KV cache.

The benchmark script (`v41bench.py`) and prompt set (`prompts-v1.json`) are the upstream repo's, byte-identical, so the columns are the same protocol on different hardware instances.

Weights: **[bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard)** (HF, includes this recipe under `recipe/`).

This is not our model, our serving stack, or our quantizer. It is our measurements and the glue. See **Credits**.

<!-- BEST-SERVING-START -->
## Best serving configuration so far (auto-updated 2026-09-11 10:56 Pacific)
Line A, row `exl3-ROCE-A` (accepted rung: ROCE=1). Settings on top of the recipe defaults: KV cache pinned = 12884901888, gpu-memory-utilization = 0.78, launcher MemFree floor (GiB) = 113, NCCL channels = 8, b12x RoCE one-shot all-reduce = 1, async scheduling = 1, max context = 1000000, max concurrent seqs = 8, DISABLED_KERNELS = FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel,MarlinMxfp8LinearKernel.

| | best so far |
|---|---|
| single stream, aggregate / per-stream tok/s | 56.1 / 62.2 |
| 4 streams aggregate tok/s | 151.4 |
| 6 streams aggregate tok/s | 199.7 |
| mean TTFT at C1 | 0.26 s |
| cold prefill 3K / 12K / 47K / 93K tok/s | 1313 / 1390 / 1440 / 1443 |

Tuning ladder (accept = +3 % single-stream or 6-stream aggregate with ≤5 % prefill loss at 47K and smokes passing; levers stack per line, winners cross-applied):

| line | row | lever | C1/stream | C6 agg | prefill@47K | verdict |
|---|---|---|---|---|---|---|
| A | exl3-G0-A-base0052 | base | 61.1 | 178.7 | 1363 | base |
| A | exl3-G0-A-base0250 | base | 62.5 | 193.4 | 1440 | base |
| A | exl3-ROCE-A | ROCE=1 | 62.2 | 199.7 | 1440 | ACCEPT |
| A | exl3-ASYNC-A | ASYNC=1 | 68.0 | 193.6 | 1440 | ACCEPT |
| A | exl3-SERVE1M-A | MAXLEN=1000000 SEQS=8 | 64.4 | 189.9 | 1419 | ACCEPT |
| A | exl3-G0-A | base | 65.1 | 184.7 | 1403 | base |
| A | exl3-MAXB16K-A | MAX_BATCHED=16384 | 65.5 | 193.2 | 1405 | ACCEPT |
| B | exl3-G0-B-base0116 | base | 61.2 | 178.6 | 1323 | base |
| B | exl3-CH8-B | NCCL_CH=8 | 62.2 | 199.0 | 1423 | ACCEPT |
| B | exl3-SERVE1M-B | MAXLEN=1000000 SEQS=8 | 59.9 | 184.9 | 1475 | ACCEPT |
| B | exl3-G0-B-base09110801 | base | 61.5 | 185.6 | 1417 | base |
| B | exl3-ROCE-B | ROCE=1 | 64.9 | 189.8 | 1436 | ACCEPT |
| B | exl3-G0-B-base09110837 | base | 65.2 | 185.1 | 1452 | base |
| B | exl3-G0-B | base | 64.9 | 185.6 | 1413 | base |
| B | exl3-B12XLIN-B | DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel,MarlinMxfp8LinearKernel | 65.0 | 192.5 | 1406 | ACCEPT |
| A | exl3-HOT-A | HOT_DIR=/mnt/glm52/dsv41engram/hot90 HOT_ROWS=20000000 | 63.63 | 198.07 | 1354.5 | reject |
| A | exl3-FLUSH8-A | FLUSH_GIB=8 | 64.83 | 181.74 | 1422.5 | reject |
| A | exl3-NTH256-A | NCCL_NTHREADS=256 | 63.95 | 184.97 | 1416.9 | reject |
| B | exl3-ET64-B | ENGRAM_THREADS=64 | 62.22 | 176.68 | 1480.5 | reject |
| B | exl3-HOT-B | HOT_DIR=/mnt/glm52/dsv41engram/hot90 HOT_ROWS=20000000 | 59.9 | 188.0 | 1424 | reject |
| B | exl3-FLUSH8-B | FLUSH_GIB=8 | 62.07 | 185.05 | 1436.4 | reject |
| B | exl3-MAXB16K-B | MAX_BATCHED=16384 | 64.88 | 188.53 | 1403.1 | reject |
| B | exl3-NCCLBUF-B | NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144 NCCL_PROTO=^LL128 | 63.4 | 193.67 | 1412.1 | reject |
| B | exl3-NCCLBUF2-B | NCCL_BUFFSIZE=1048576 NCCL_LL128_BUFFSIZE=262144 | 61.68 | 182.41 | 1390.8 | reject |
| B | exl3-WOPROJ-B | WOPROJ=1 | 65.03 | 185.91 | 1402.8 | reject |
| B | exl3-KVGROUP-B | KVGROUP=fine | 60.92 | 177.68 | 1388.7 | reject |

Levers in the served configurations and where they come from (full ledger in `CREDITS.md`):

- pinned KV cache — our fix for vLLM sizing KV from rank 0 only under the uneven expert split
- per-rank memory gate — ours; unconditional load-time flush idea from tonyd2wild
- NCCL channels 8 — lever from tenaiaiai's dsv41-flash-4x-dgx-spark-ja recipe (https://github.com/tenaiaiai/dsv41-flash-4x-dgx-spark-ja)
- b12x one-shot RoCE all-reduce — Luke Alonso and the b12x contributors (https://github.com/local-inference-lab/b12x), vLLM shim ported from local-inference-lab/vllm
- async scheduling — vLLM project
- b12x MXFP8 dense GEMM kernel (`B12xMxfp8LinearKernel`, b12x by Luke Alonso and contributors) selected over the CUTLASS path — idea from MiaAI-Lab's DeepSeek-v4.1-Flash-DGX-Sparks write-up (SGLang), re-measured on our stack
<!-- BEST-SERVING-END -->

## Results

| measurement | [upstream recipe, published boot 10](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/main/results/boot10) (its hardware) | shipped MXFP4/FP8 checkpoint, same recipe, our 4 nodes | EXL3 3.5 bpw experts (this build, our 4 nodes) |
|---|---|---|---|
| weights loaded per rank | 81.6 GiB | ~81 GiB | **62.8 GiB** (measured at load) |
| calibration NLL, 10-node held-out rows, vs bf16 reference | — | 1.3209 | 1.3177 ± 0.006 (flat; all-K3 fallback 1.3152 ± 0.003) |
| per-matrix weight relative error | — | — | 0.167 |
| single stream, aggregate / per-stream tok/s | 37.95 / 43.12 | 42.3 / 50.5 | 54.7 / 61.1 |
| 4 streams aggregate tok/s | 85.72 | 102.7 | 133.1 |
| 6 streams aggregate tok/s | 131.86 | 130.7 | 178.7 |
| cold prefill 3K / 12K / 47K / 93K tok/s | 902 / 1026 / 1539 / 1194 | 1094 / 558 / 1310 / 1292 | 1263 / 1082 / 1363 / 1375 |
| KV capacity at gmu 0.80 | 1,078,380 tokens (boot 7, 1M ctx) | 6.99 GiB/rank = 1.16 M tokens | 2.56 M tokens |
| ppl probe (6 held-out texts, 4,210 tokens) | — | TBD (pending) | 4.043 |
| HumanEval / HumanEval+ pass@1 (greedy, evalplus) | — | TBD (pending) | 0.951 / 0.921 |
| MBPP+ (greedy) | — | TBD | pending |
| needle at 219K tokens (2 keys) | — | — | PASS / PASS |
| 1M-context serving row (pin + NCCL channels 8, `--max-model-len 1000000`) | 1M ctx: KV 1,078,380 tokens (boot 7) | — | C1 54.8 / 59.9, C4 141.8, C6 184.9, prefill 673 / 1007 / 1475 / 1449; **3.41 M tokens KV** |
| **production (1M context), line A: pin + NCCL ch 8 + RoCE + async + 16K batch** | — | — | C1 58.4 / 65.1, C4 148.4, C6 183.1, TTFT 0.26, prefill 1312 / 1412 / 1411 / 1391; KV 2.99 M tokens |
| **production (1M context), line B: pin + NCCL ch 8 + RoCE** | — | — | C1 55.5 / 61.8, C4 149.4, C6 192.7, TTFT 0.25, prefill 1343 / 1176 / 1459 / 1463; KV 3.41 M tokens |
| **optional KV grouping fix** (`recipe/patches/kvgroup`, env `DSV41_KV_GROUPING=fine`) | — | — | KV 3.41 M → **5.57 M tokens at 1M** (+63 %) for −6 % single-stream / −8 % C6 (47 KV groups of scheduler work); off in our served bases |

Engram: the top 100 M rows of each n-gram table (by frequency over 1.12 B tokens of real assistant traffic) cover **92.7 %** of held-out lookups (43 % at 1 M, 74 % at 20 M). Quantizing Engram rows to fp4 (mxfp4 or nvfp4) was NLL-neutral within noise. A 20 M-row resident set (CPU hit/miss split) measured −4 to −6 % single-stream on both lines, so the ids are not shipped; a GPU-side gather is the open follow-up.

## Hardware

- 10 × NVIDIA DGX Spark (GB10, 121.7 GiB unified memory), 200 GbE ConnectX fabric (MTU 9000). All ten cook; four serve one line (we run two lines behind an nginx front door).
- Serving: TP=4, fp8 DS-MLA KV, `--gpu-memory-utilization 0.80`, DSpark speculative decoding k=5, FULL_AND_PIECEWISE CUDA graphs, Engram rows read from NVMe before each forward.

## The recipe in one paragraph

Exact bf16 upscale of the MXFP4 checkpoint → DeepSeek's reference forward over a calibration corpus, dumping per-expert input Hessians on all ten nodes (node-local, served over nginx) → per-expert `quantize_exl3` (exllamav3 LDLQ) at K=3, then targeted K=4 re-cooks for the `down` and `gate/up` matrices where a gain-weighted, token-weighted proxy error says they matter (layers 0–14 mostly; the last four layers barely register) → measured 3.51 bpw recipe → splice the EXL3 experts into the shipped shards (dense, attention, Engram, DSpark head untouched) → serve with the cuda-exl3 vLLM plugin. Pre-splice NLL gate on the reference forward before any serving. Details, ledgers and the allocator are in `recipe/RECIPE_TP4_EXL3.md` and in our data contribution to the Pollard Weights repository.

## Two things you will hit

1. **cuda-exl3 + a DSpark drafter:** the plugin finds the standalone `quantization_config.json` through a hint set only in the main process; the drafter's quant config is rebuilt in the spawned worker and fails with "could not find tensor_storage". `recipe/patches/exl3_config.py` adds an env fallback (`CUDA_EXL3_MODEL_PATH`).
2. **EXL3 MoE kernel block width:** `exl3_moe_gemm` needs each per-rank shard of the expert intermediate dim to be a multiple of 128; this model has 2304 → 576 per rank at TP4. `recipe/patches/exl3_moe.py` splits the dim unevenly on 128-column boundaries (512/640/640/512) — the TP reduction does not care.
3. **Memory gates drift:** vLLM refuses to start a rank when free memory at its init snapshot is below `gmu × total`; a check minutes earlier is not enough on unified memory. `recipe/dsv41_tp4_launch.sh` gates every rank right before `docker run` (probe → drop caches → re-probe → fail loudly).

## Layout

- `recipe/` — launcher, boot, node prep, gate/ladder scripts, patches with mount map, image Dockerfiles, RoCE shim diffs and test (site identifiers replaced by placeholders; see `recipe/README.md`).
- `docs/HF_MODEL_CARD.md` — the model card as uploaded.
- `CREDITS.md`, `THIRD_PARTY_NOTICES.md` — who this stands on and under which licenses.

## Credits

DeepSeek-AI (model, reference code) · turboderp / exllamav3 (EXL3, LDLQ) · Zeuss5 and cuda-exl3 contributors (plugin, kernels) · tonyd2wild and Kai (the 4× Spark recipe, Engram-on-NVMe, bench protocol) · vLLM project · Luke Alonso / b12x and local-inference-lab (one-shot RoCE collectives) · eugr (Spark image lineage) · drowzeys (spin-wait fix) · WestWaters / Pollard Weights (method framing) · tenaiaiai, 0xSero, MiaAI-Lab, kishida, alexellis (ideas we measured or borrowed levers from). Full ledger: `CREDITS.md`.

## License

Our scripts and documents: MIT. Third-party code retains its licenses (`THIRD_PARTY_NOTICES.md`). The weights are a derivative of DeepSeek-V4.1-Flash (MIT).
