# Credits ledger — DeepSeek-V4.1-Flash EXL3 TP4 work (living; update on every adopted lever, borrowed idea, or copied code)

Rule (user, 2026-09-10): keep track of who to credit. Every row here must appear in the HF card, the Pollard notes and the public write-up before anything is published.

**Special credit (user, 2026-09-10: "give special credit to the Pollard repo and Tony in all forms"):** WestWaters/pollard-weights (the method and the home of our data) and tonyd2wild (the serving recipe we run) get a named call-out at the top of every artifact — HF card, GitHub README, recipe README, gists, PR text — not just a row in the table.

| who | what we used | where it shows up | status |
|---|---|---|---|
| **DeepSeek-AI** | DeepSeek-V4.1-Flash weights (MIT); reference `inference/` code (model.py, engram.py, kernel.py, convert.py) used as our calibration forward; DSML `encoding.py`; deepseek-recipe (renderer parity check); DeepGEMM | card, notes, gist | in use |
| **turboderp** (turboderp-org/exllamav3) | EXL3 format; `quantize_exl3`/`quantize_exl3_batch` (LDLQ), had/out-scale/seed rules, `reconstruct` path | card, notes, Pollard PR README | in use |
| **cuda-exl3 contributors** (Zeuss5/cuda-exl3, 1.0.3) | vLLM plugin + CUDA kernels serving the EXL3 experts; our V4.1 overlay P1–P5 sits on top (held for upstream until user go) | card, notes | in use |
| **tonyd2wild** (DeepSeek-V4.1-Flash-vLLM-DGX-Spark, MIT) | 4× Spark recipe: 7 patches, worker-first boot, Engram-on-disk staging before forward, node-local Engram rows, bench script `v41bench.py` + prompt set `prompts-v1.json` used VERBATIM (byte-identical md5 bfedc7f15661…), GPU slow-state probe (gpuflip) + clock-latch burn gate idea, unconditional load-time flusher idea, NCCL/RoCE env, image chain baseline, bench script + prompt set verbatim | card, recipe doc, notes, gist | in use — our boot3 reproduces boot-10 |
| **Kai** | first Engram-on-disk patch; SM12x page-size patches (credited via tonyd2wild) | card | in use |
| **vLLM project** | engine, DeepSeek-V4.1 model tree (nightly 8a728663c + dsv41-feat) | card | in use |
| **b12x — Luke Alonso and contributors** (Apache-2.0); **local-inference-lab/vllm** fork (`dev/jovian-judgement`) | one-shot RoCE/PCIe collectives runtime (`b12x.comm.roce/pcie`) + the vLLM shim we ported into overlay6 (via our bot-lab-21/vllm 0.29 port) | card, gist, recipe | in use (ladder lever ROCE) |
| **eugr** (spark-vllm-docker, MIT) | DGX Spark build pipeline lineage of our images | card | lineage |
| **drowzeys** (vllm-gb10-spin-wait-fix) | SpinCondition busy_loop_s fix baked into overlay8 | card | in use |
| **WestWaters** (Pollard Weights) | method framing; EXL3 lane choice; MoE export policy spirit; SKILL guidance; contribution format | card, notes, PR | in use |
| **tenaiaiai** (dsv41-flash-4x-dgx-spark-ja, MIT) | NCCL channels 8 lever, Engram disk threads 64 lever, concurrency-8 cliff, k=3 −40 % finding, failure table | card (referenced), ladder levers CH8/ET64; C8 cliff, k=3 finding | **NCCL_CH=8 ADOPTED 2026-09-11 (+11 % C6)**; ET64 = prefill-only win (profile lever) |
| **0xSero** (deepseek-v4.1-flash-4x-rtx-pro-6000, MIT; @OxSero on X) | Engram-outside-memory idea and the "fp4 Engram" post that triggered our fp4 NLL test; io_uring NVMe reader (not adopted yet) | card, gist | referenced; credit if fp4/resident Engram ships |
| **alexellis** (glm-5.3-flash-4x-dgx-spark-switchless, deepseek-v4-flash-0731) | switchless NCCL notes; `--max-num-batched-tokens 16384` prefill lever; non-breakable graphs note | card, backlog B1.4/B1.10 | testing (MAXB16K) |
| **kishida** (webdemos/llkvapprox) | Late-Layer KV Approximation (CED prefill) idea | PLAN (parked research) | not used yet |
| **FlashInfer / NVIDIA CUTLASS DSL / NCCL / PyTorch / safetensors** | kernels and plumbing in the vehicle | card | in use |
| **IST-DASLab (GPTQ, Marlin), QuantTrio** | lineage of our GLM Pollard-method builds | card (lineage) | lineage |
| **MiaAI-Lab** (GLM-5.3-Flash-EXL3-2x-DGX-Sparks, DeepSeek-v4-Flash-One-DGX-Spark, exllamav3 fork) | reviewed 2026-09-10: fused `exl3_moe` per-layer MoE kernel idea, padded slot-share draft/target KV co-allocation, DFlash2 k=7, KLD quality method; nothing adopted yet | PLAN (candidates for a later A/B) | referenced |
| **brandonmusic** (TR3 EXL3 quants MiaAI-Lab mirrors) · **Fujitsu-Polycom sparkring** (GLM-5.2 EXL3 3.5 bpw TP4/DCP4) · **Anemll** (dspark-vllm-gx10 image) · **local-inference-lab/b12x** (sparkinfer) | context from the MiaAI-Lab review | — | referenced |
| **NNNtrance** (via Zeuss5/cuda-exl3 issues #1/#2/#4–#6) | GB10 measurements behind cuda-exl3's MoE fixes (live-row bound on the unsplit launch, skip-pad, block_m tiers, draft KV page size 256, NCCL channel bands) | backlog B2 | referenced (fixes already in our overlay) |
| **Luke Alonso / local-inference-lab (b12x docs)** | RoCE one-shot knobs (`VLLM_ROCE_*_MAX_SIZE`, `B12X_ROCE_HCA` striping, GID, spin limit) and GLM TP4 gains | backlog B3.1, ladder ROCE/ROCEMAX rows | testing |
| **vLLM contributors** benchislett (adaptive verification blog), 0z5a / Juntian777 (Engram lookup overlap), liuyao0322 / yzyDavid / TeloySXH (DSpark kernel PRs), WoosukKwon (indexer logits), tpurtell (sm_12x top-k fixes) | ideas in backlog B3 | backlog | referenced |
| **r0b0tlab**, **vladimir-voinea** (dspark-vllm-gb10), **ecohash-co**, **hazyumps**, **Alessandro Sangiorgi** (NCCL latency on Spark), **ShiningMeUp** / **agjs** (GB10 clock latch & cap), **Petronella Technology** (CRS812 fabric guide), **GaelicThunder** (GB10 UMA H2D notes), **ArgentAIOS** | GB10 system-level findings in backlog B4 | backlog | referenced |
| **Festr** (via brandonmusic TR3v4 card) | merged shared gate+up idea | backlog B2.11 | referenced |
| **MiaAI-Lab** (DeepSeek-v4.1-Flash-DGX-Sparks, SGLang TP3; AGPL-3.0 → ideas only) | NCCL pinned-buffer knobs (BUFFSIZE / LL128_BUFFSIZE / ^LL128) to reclaim GPU memory; b12x MXFP8 dense-GEMM routing; KV bytes/token accounting; decode step breakdown | backlog B5; ladder rows NCCLBUF/NCCLBUF2 (buffer knobs), B12XLIN (b12x dense kernel), WOPROJ (b12x wo_projection routing), KVGROUP (KV accounting) | **testing on line B 09-11**; each becomes an adopted-lever credit on the card + README the moment its row is accepted (ops/lever_credits.json) |
| **Z.ai (zai-org), ciprianveg, humming (inclusionAI), yichengj0, Light Foundry** | GLM-side lineage (0.29 port report) — not part of the DSV41 line | 0.29 report | n/a here |

Process: when a ladder rung is ACCEPTED, add the lever's source here and in the card's Serving section the same day.
