# DeepSeek-V4.1-Flash (552B MoE) on four DGX Sparks — EXL3 3.5-bit experts, Pollard method

**Full 1M-context model, 4 × 128 GB desktop boxes, 60+ tok/s single stream, ~200 tok/s at six streams, 3.4 M tokens of KV.**

| same 4 nodes, same bench script + prompt set | shipped MXFP4/FP8 checkpoint | **this build (EXL3 3.5 bpw experts + tuning stack)** | Δ |
|---|---|---|---|
| single stream, per-stream tok/s | 50.5 | **68.0** | **+35 %** |
| single stream, aggregate tok/s | 42.3 | **60.8** | +44 % |
| 2 / 3 / 4 streams aggregate | 57.8 / 77.3 / 102.7 | **93.1 / 119.2 / 154.4** | +61 % / +54 % / **+50 %** |
| 6 streams aggregate | 130.7 | **199.7** | **+53 %** |
| time to first token (C1) | 0.62 s | **0.25 s** | 2.5× faster |
| cold prefill 3K / 12K / 47K / 93K tok/s | 1094 / 558 / 1310 / 1292 | **1358 / 1404 / 1440 / 1443** | +24 % / **+152 %** / +10 % / +12 % |
| weights per node | ~81 GiB | **57–69 GiB** | −12 to −24 GiB |
| KV cache capacity | 1.16 M tokens | **2.56 M (300K profile) · 3.41 M (1M profile)** | **2.2× / 2.9×** |
| quality: HumanEval / HumanEval+ (greedy) | — | **0.951 / 0.921** | |
| ppl probe (held-out), needle @219K | — | **4.043, PASS/PASS** | |

**The stack:** DeepSeek-V4.1-Flash routed experts re-quantized per expert to an EXL3 trellis at 3.5 bits (K=3 base, K=4 where a measured sensitivity ledger says it matters — Pollard method), everything else byte-identical to the release; vLLM + cuda-exl3, tensor-parallel 4 with an uneven 128-aligned expert split, DSpark speculative decoding k=5, Engram n-gram tables streamed from NVMe, pinned KV cache, NCCL 8 channels, b12x RoCE one-shot all-reduce, async scheduling, FULL+PIECEWISE CUDA graphs. Every lever measured against the same prompt set and accepted only on a +3 % rule; rejects are published too.

**Open weights + recipe + receipts:** https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard · https://github.com/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-DGX-Spark

**Standing on:** DeepSeek-AI (model) · WestWaters / Pollard Weights (method) · tonyd2wild + Kai (the 4× Spark serving recipe, Engram-on-NVMe, bench protocol) · turboderp / exllamav3 (EXL3) · Zeuss5 / cuda-exl3 (kernels, plugin) · vLLM · Luke Alonso / b12x (RoCE collectives) · eugr, drowzeys, tenaiaiai and the DGX Spark community. Full ledger in CREDITS.md.

*Measured 2026-09-10/11 on ten NVIDIA DGX Spark (GB10) — four serve, all ten cooked the quant in ~7 h.*
