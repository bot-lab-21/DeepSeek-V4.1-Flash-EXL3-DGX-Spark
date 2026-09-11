---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: vllm
tags: [deepseek, deepseek-v4.1-flash, exl3, exllamav3, mixture-of-experts, dgx-spark, gb10, vllm, cuda-exl3, pollard-method, quantization]
---

# DeepSeek-V4.1-Flash — EXL3 routed experts (3.5 bpw) for a 4× DGX Spark tensor-parallel line

**Work in progress (2026-09-11): the checkpoint is complete and serving; cells marked TBD are being filled from the served tuning gate over the next hours. Every number here was measured on our hardware.**

> **Built on two people's work above all.** The quantization follows the **Pollard method** as framed and documented in [WestWaters/pollard-weights](https://github.com/WestWaters/pollard-weights) (Hessian-aware, sensitivity-allocated expert quantization; our ledgers and tools are contributed back there). The serving recipe is **[tonyd2wild's DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)** — the patch set, Engram-on-NVMe staging (with Kai), worker-first boot, image chain and bench protocol; we changed the expert bytes and added a few levers on top. If you use this, cite them first.

A hybrid checkpoint of [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) for four DGX Spark (GB10) nodes in tensor parallel. The 40 × 384 routed experts are EXL3 trellis-quantized (K=3 base, K=4 where a measured ledger put the extra bit; 3.51 bpw average). Everything else is byte-identical to the shipped release: fp8 dense and attention, MXFP4 shared experts and DSpark drafter, fp8 Engram n-gram tables, native fp4 KV. Served with vLLM + the cuda-exl3 plugin (with our DeepSeek-V4.1 overlay) inside tonyd2wild's 4× Spark recipe.

This is not our model, our serving stack, or our quantizer. It is our measurements and the glue. See **Credits** — please cite the people whose work this stands on.

## Why this exists
The shipped 510 GB checkpoint fits four GB10s only with the Engram tables on NVMe and ~7 GiB per rank left for KV (1.16 M tokens at 80 % memory utilization). Shrinking the routed experts from 4.25 to ~3.5 bpw frees ~15 GiB per rank for KV and concurrency on the same four nodes, with no measurable loss on our calibration rows.

## What is in the repository
| file | content |
|---|---|
| `model-000NN-of-00048.safetensors` | same shard count and numbering as the release. Body shards (3–42) carry EXL3 expert tensors (`…experts.E.w1/w3/w2.{trellis,suh,svh,mul1}`) next to the untouched non-expert tensors; shards 1, 2, 43–48 are byte-identical to the release (vision, embeddings, norm/head, drafter, Engram tables). |
| `config.json`, `quantization_config.json` | `quant_method: "exl3"`, per-layer `expert_bits` (gate/up and down K), per-module `tensor_storage`; DeepSeek's original quantization block kept as `original_quantization_config`. Read by cuda-exl3's `Exl3Config`; no `--quantization` flag needed. |
| `tokenizer*`, `chat_template`, `generation_config.json`, `*.py`, `README` | copied from the release |
| Engram hot-row ids | **not shipped** (removed 2026-09-11): a 20 M-row resident set measured −4 to −6 % single-stream on both lines (CPU hit/miss split); the frequency ledger itself (top 100 M ids per table = 92.7 % held-out coverage) is available on request and described below |

<!-- BEST-SERVING-START -->
## Best serving configuration so far (auto-updated 2026-09-11 11:38 Pacific)
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

## Measured — pre-serving (DeepSeek's reference forward with every routed expert replaced by its EXL3 reconstruction; 39 in-domain calibration rows × 2048 tokens, paired per row against bf16)
| recipe | experts bpw | NLL bf16 → EXL3 | paired ΔNLL (sem) | rows worse |
|---|---|---|---|---|
| all layers K=3 | 3.01 | 1.3209 → 1.3152 | −0.0057 (0.0031) | 16 / 39 |
| **this build** (K=3 / K=4 mix: 19 layers gu4/down4, 3 gu3/down4, 18 gu3/down3) | 3.51 | 1.3209 → 1.3177 | −0.0032 (0.0055) | 19 / 39 |

The per-matrix weight relative error at K=3 is 0.167 and the output does not move: expert-output errors average out through top-6 routing and the residual stream. Caveat: these rows are the Hessian calibration set (in-domain chat). The held-out checks are the served numbers below.

## Measured — served (TP4, vLLM + cuda-exl3, DSpark k=5, FULL_AND_PIECEWISE graphs, 300K context, gmu 0.80; bench script and prompt set v1 are the upstream recipe repo's, byte-identical) — TBD
| | [upstream recipe, published boot 10](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/main/results/boot10) (its hardware) | shipped MXFP4/FP8 checkpoint (same recipe, our 4 nodes) | this build (our 4 nodes) |
|---|---|---|---|
| single stream, aggregate / per-stream tok/s | 37.95 / 43.12 | 42.3 / 50.5 | 54.7 / 61.1 |
| 4 streams aggregate tok/s | 85.72 | 102.7 | 133.1 |
| 6 streams aggregate tok/s | 131.86 | 130.7 | 178.7 |
| cold prefill 3K / 12K / 47K / 93K tok/s | 902 / 1026 / 1539 / 1194 | 1094 / 558 / 1310 / 1292 | 1263 / 1082 / 1363 / 1375 |
| KV capacity at gmu 0.80 | 1,078,380 tokens (boot 7, 1M ctx) | 1.16 M tokens | 2.56 M tokens |
| ppl probe (6 held-out texts, 4,210 tokens) | — | TBD (pending) | 4.043 |
| HumanEval / HumanEval+ pass@1 (greedy, evalplus) | — | TBD (pending) | 0.951 / 0.921 |
| MBPP+ (greedy) | — | TBD | pending (battery timed out mid-run; re-running) |
| needle at 219K tokens (2 keys) | — | — | PASS / PASS |
| 1M-context serving row (line B, pin + NCCL channels 8, `--max-model-len 1000000`) | — | — | C1 54.8 / 59.9, C4 141.8, C6 184.9, prefill 673 / 1007 / 1475 / 1449; **3.41 M tokens KV** (3.4 full-length requests) |
| **production (1M context), line A: pin + NCCL ch 8 + RoCE + async + b12x MXFP8 dense kernel** | — | — | C1 55.9 / 61.4, C4 147.2, C6 192.0, TTFT 0.25, prefill 1142 / 1326 / 1385 / 1406; KV 3.41 M tokens (11 Sep 11:36) |
| **production (1M context), line B: pin + NCCL ch 8 + RoCE + b12x MXFP8 dense kernel** | — | — | C1 56.3 / 62.3, C4 149.6, C6 178.7, TTFT 0.25, prefill 1067 / 1174 / 1396 / 1418; KV 3.41 M tokens (11 Sep 11:15). Run-to-run spread on this fleet is about ±5 %: the same config benched C6 192.7 two hours earlier |
| **optional KV grouping fix** (`recipe/patches/kvgroup`, env `DSV41_KV_GROUPING=fine`) | — | — | KV 3.41 M → **5.57 M tokens at 1M** (+63 %) for −6 % single-stream / −8 % C6 (47 KV groups of scheduler work); off in our served bases |
| needle 300K, tool-call integrity, image probe | TBD | TBD |

## How it was made (Pollard-method, "route B")
1. **Exact bf16 upscale** of the release (fp8 · 2^(ue8m0−127) 32×32 blocks; MXFP4 e2m1 LUT × per-32 ue8m0 scale), round-trip checked. There is no native bf16 release; the source is a 4.25-bit QAT checkpoint. Gate-0 measured that EXL3 on these FP4-grid weights behaves exactly like a Gaussian control, so the grid neither helps nor hurts the trellis quantizer.
2. **Calibration forward** = DeepSeek's reference `model.py`/`engram.py` (no framework loads this architecture), streamed layer by layer on ten GB10s: 384 × 2048 in-domain rows rendered with DeepSeek's DSML encoder; per-layer FFN input + routing dumps; Engram rows gathered by hash id; reference NLL per node.
3. **Per-expert Hessians** (H shared by gate/up; H_down from silu(xW1)·xW3 of the routed tokens) → exllamav3's `quantize_exl3` per expert (K=3 all layers, 57 min/layer per GB10; K=4 down-only 16 min; K=4 gate/up 30 min), with a JSON ledger per layer.
4. **Allocation**: cost(layer, matrix, K) = gain² · Σ_experts tokens · proxy_err, where gain is the layer's measured hyper-connection write gain into the residual (5–7.5× at layers 0–13, ≈1 at 19–30, 0.03–0.2 at 36–39). Greedy K=3→4 to 3.5 bpw; K=4/K=3 error ratio measured 0.254. Result: down-proj K=4 on layers 0–14, 16–18, 20–27; gate/up K=4 on layers 0–14, 16–18, 20–22, 24, 25; layers 28–39 stay K=3.
5. **Splice**: expert tensors rewritten per body shard, everything else hardlinked from the release; `quantization_config` in the cuda-exl3 layout.
Measurements and (sanitized) tooling are contributed to the Pollard Weights repository (link TBD when the PR is open).

## Engram
The two n-gram tables (203 GB fp8) are unchanged; the serving recipe reads their rows from NVMe before each forward. Two measured facts for anyone making them resident: fp4 rows (MXFP4 or NVFP4) cost no NLL (Δ −0.003 ± 0.003 while 80–100 % of elements change), and row accesses are Zipfian (top 1 M of 384 M rows = 62 % of lookups in-sample, top 5 M = 89 %); our frequency ledger (not shipped; ask) lists the top 100 M ids per table = 92.7 % held-out coverage (24.6 GiB fp8 per table = ~6.1 GiB per TP4 rank per table, half at fp4; coverage curve: 1 M 43 %, 5 M 59 %, 10 M 67 %, 20 M 74 %, 50 M 84 %, 100 M 93 %); a reference implementation (resident hits by binary search, misses to disk) is in our recipe notes.

## Serving
- **Recipe**: tonyd2wild's *DeepSeek-V4.1-Flash-vLLM-DGX-Spark* (worker-first launch, Engram on disk staged before the forward, SM12x sparse-MLA page size, DSpark k=5, CUDA graphs, node-local Engram rows). Our additions: cuda-exl3 1.0.3 + a V4.1 overlay (expert naming w1/w3/w2, hybrid config delegating non-EXL3 modules to the fp8 path, clamped SwiGLU limit 10, shared experts kept fp8), the b12x one-shot RoCE all-reduce (2.5× faster than NCCL at the decode-step size on two nodes; served effect TBD), the GB10 spin-wait fix, and a boot ritual for GB10 unified memory (drop caches, GPU reclaim, cache-flusher during load, min_free 1 GiB, MTU 9000 jumbo check, GPU fast/slow-state probe, CUDA-visible free-memory gate, image/patch parity across ranks).
- Container path must be identical on every rank; `quantization_config.json` must sit next to the weights.
- Known GB10 caveat (tonyd2wild issue #1): a hidden GPU slow state can move any single measurement by ~1.5×; probe before you bench.

## Credits — this is assembled on other people's work
- **DeepSeek-AI** — DeepSeek-V4.1-Flash (model, weights, MIT), the reference inference code (`model.py`, `engram.py`, `kernel.py`), the DSML chat encoder and [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe), DeepGEMM, and the architecture itself (hyper-connections, CSA2 sparse attention, Engram, DSpark).
- **turboderp** — [ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 trellis format and the `quantize_exl3` LDLQ quantizer every expert here went through.
- **cuda-exl3 contributors** — [cuda-exl3](https://github.com/Zeuss5/cuda-exl3): the vLLM plugin and CUDA kernels (EXL3 GEMM/MoE, Hadamard, MLA decode) that serve these experts; our overlay (expert naming, hybrid config, SwiGLU clamp, shared-expert fp8) is a small layer on top and is offered upstream.
- **tonyd2wild** — [DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) (MIT): the 4× Spark recipe this build targets, the seven patches, the launch/boot protocol, the bench protocol and prompt set our numbers use, the GB10 slow-state probe, and the node-local Engram rows idea.
- **Kai** — the first Engram-on-disk patch and the SM12x page-size patches (credited in tonyd2wild's recipe).
- **vLLM** ([vllm-project](https://github.com/vllm-project/vllm), Apache-2.0) — the engine and the DeepSeek-V4.1 model tree.
- **b12x** (Luke Alonso and contributors, Apache-2.0) — the GB10 kernel and collectives package; the one-shot RoCE all-reduce adapter comes from the [local-inference-lab vLLM fork](https://github.com/local-inference-lab/vllm) delta merged in our [vLLM 0.29.0+b12x port](https://github.com/bot-lab-21/vllm).
- **eugr** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT): the DGX Spark build pipeline our images descend from.
- **drowzeys** — [vllm-gb10-spin-wait-fix](https://github.com/drowzeys/vllm-gb10-spin-wait-fix): the GB10 spin-wait heat fix baked into the image.
- **WestWaters** — [Pollard Weights](https://github.com/WestWaters/pollard-weights): the method framing (measured Hessian sensitivity → budgeted allocation → MoE-aware export) these notes follow; our ledgers and tooling are contributed there.
- **FlashInfer**, **NVIDIA CUTLASS DSL**, **NCCL**, **PyTorch**, **safetensors** — the kernels and plumbing underneath.
- **tenaiaiai** ([dsv41-flash-4x-dgx-spark-ja](https://github.com/tenaiaiai/dsv41-flash-4x-dgx-spark-ja)) and **0xSero** ([deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000)) — parallel recipes whose NCCL/Engram-thread knobs and Engram-outside-memory ideas we tested against; **alexellis** (switchless NCCL notes) via tenaiaiai.
- **IST-DASLab** (GPTQ, Marlin) and **QuantTrio** — the lineage of our earlier GLM Pollard-method builds that this recipe grew out of.

Built by bot-labs-21 with AI assistance (Claude Code); every number was measured on our own hardware. Corrections to any attribution are welcome — open an issue.

## License
MIT for our glue, following the base model's license; the model weights remain under DeepSeek's terms.
