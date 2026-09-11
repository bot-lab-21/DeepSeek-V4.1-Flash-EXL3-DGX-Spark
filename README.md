# DeepSeek-V4.1-Flash on four DGX Sparks with EXL3 routed experts (3.5 bpw) — recipe and receipts

**Status: work in progress (2026-09-11) — the checkpoint serves on two 4-node lines; cells marked TBD are being filled from the tuning gate. Every number below was measured on our cluster.**

> **Built on two people's work above all.** The quantization follows the **Pollard method** as framed and documented in [WestWaters/pollard-weights](https://github.com/WestWaters/pollard-weights) (Hessian-aware, sensitivity-allocated expert quantization; our ledgers and tools are contributed back there). The serving recipe is **[tonyd2wild's DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)** — the patch set, Engram-on-NVMe staging (with Kai), worker-first boot, image chain and bench protocol; we changed the expert bytes and added a few levers on top. If you use this, cite them first.

DeepSeek-V4.1-Flash is a 552B-parameter mixture-of-experts model (40 layers × 384 routed experts, two Engram n-gram tables, DSpark speculative head) shipped as MXFP4 experts + FP8 dense, 510 GB. It fits four 128 GB DGX Spark (GB10) nodes only because tonyd2wild's recipe keeps the Engram tables on NVMe, and even then leaves about 7 GiB per rank for KV cache. This repository is the recipe that re-quantizes the routed experts to an EXL3 trellis at **3.5 bits per weight** with the Pollard method (Hessian-aware, per-expert, allocated by measured sensitivity), keeps everything else bit-identical, and serves it with vLLM + cuda-exl3 on the same four nodes with the freed memory going to KV cache.

Weights: **[bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard)** (HF, includes this recipe under `recipe/` and the Engram hot-row ids).

This is not our model, our serving stack, or our quantizer. It is our measurements and the glue. See **Credits**.

## Results

| measurement | shipped MXFP4/FP8 checkpoint, same serving recipe and settings, same 4 nodes | EXL3 3.5 bpw experts (this build) |
|---|---|---|
| weights loaded per rank | ~81 GiB | **62.8 GiB** (measured at load) |
| calibration NLL, 10-node held-out rows, vs bf16 reference | 1.3209 | 1.3177 ± 0.006 (flat; all-K3 fallback 1.3152 ± 0.003) |
| per-matrix weight relative error | — | 0.167 |
| single stream, aggregate / per-stream tok/s | 42.3 / 50.5 | 54.7 / 61.1 |
| 4 streams aggregate tok/s | 102.7 | 133.1 |
| 6 streams aggregate tok/s | 130.7 | 178.7 |
| cold prefill 3K / 12K / 47K / 93K tok/s | 1094 / 558 / 1310 / 1292 | 1263 / 1082 / 1363 / 1375 |
| KV capacity at gmu 0.80 | 6.99 GiB/rank = 1.16 M tokens | TBD |
| ppl probe (24 held-out texts), HumanEval+ / MBPP+ | TBD | TBD |

Engram: the top 100 M rows of each n-gram table (by frequency over 1.12 B tokens of real assistant traffic) cover **92.7 %** of held-out lookups (43 % at 1 M, 74 % at 20 M). Quantizing Engram rows to fp4 (mxfp4 or nvfp4) was NLL-neutral within noise. The ids are published with the weights; the serving patch can keep them resident.

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
- `docs/HF_MODEL_CARD.md` — the model card as uploaded; `docs/engram_hot90_README.md` — the hot-row files.
- `CREDITS.md`, `THIRD_PARTY_NOTICES.md` — who this stands on and under which licenses.

## Credits

DeepSeek-AI (model, reference code) · turboderp / exllamav3 (EXL3, LDLQ) · Zeuss5 and cuda-exl3 contributors (plugin, kernels) · tonyd2wild and Kai (the 4× Spark recipe, Engram-on-NVMe, bench protocol) · vLLM project · Luke Alonso / b12x and local-inference-lab (one-shot RoCE collectives) · eugr (Spark image lineage) · drowzeys (spin-wait fix) · WestWaters / Pollard Weights (method framing) · tenaiaiai, 0xSero, MiaAI-Lab, kishida, alexellis (ideas we measured or borrowed levers from). Full ledger: `CREDITS.md`.

## License

Our scripts and documents: MIT. Third-party code retains its licenses (`THIRD_PARTY_NOTICES.md`). The weights are a derivative of DeepSeek-V4.1-Flash (MIT).
