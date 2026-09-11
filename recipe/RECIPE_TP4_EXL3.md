# DeepSeek-V4.1-Flash — TP4 serving recipe (sp4/sp6/sp7/sp10) for the bot-labs-21 EXL3 build

Living recipe. Vehicle = Tony's boot-10 config (tonyd2wild, MIT) + Kai's SM12x patches, in OUR image chain, with every GB10 memory/fabric
ritual we run on the GLM brain baked into the boot path. Baseline row to beat: `results/boot3` (shipped MXFP4/FP8, same nodes, 2026-09-10).

## 1. Pieces
| piece | value |
|---|---|
| nodes / ranks | rank0 **sp4** FABRIC_IP (head, API :8000) · rank1 sp6 FABRIC_IP · rank2 sp7 FABRIC_IP · rank3 sp10 FABRIC_IP; RoCE `rocep1s0f1`, IF `enp1s0f1np1`, GID 3, TC 106, MTU 9000 |
| image | `vllm-dsv41:overlay8` = nightly 8a728663c + dsv41-feat + sm_121a ext + FlashInfer 0.7.0rc1 caches (overlay5) + b12x RoCE/PCIe one-shot collectives shim (overlay6) + cuda-exl3 1.0.3 + V4.1 plugin compiled (overlay7) + GB10 spin-wait fix busy_loop_s 0.002 (overlay8). **Same image ID on all 4 nodes** (docker save|load from sp7; a per-node build = different digest = rank mismatch). |
| patches (bind-mounted) | `~/patches/dsv41-boot3/` = Tony ca662ac: attention, engram (node-local rows), flashinfer_sparse, model_state (Engram lookup in prepare_inputs), sparse_attn_indexer, sparse_swa, weight_utils — md5 parity across nodes is checked by the prep |
| model, PROFILE=shipped | `/mnt/glm52/hub/DeepSeek-V4.1-Flash` on all 4 (48 shards, fp8 dense + MXFP4 experts + fp8 Engram) |
| model, PROFILE=exl3 | sp4/sp6/sp7 `/mnt/glm52/hub/DeepSeek-V4.1-Flash-EXL3-3p5` (spliced: body shards rewritten, others hardlinked into the shipped dir); sp10 converted **in place** under the shipped name (user decision 09-10; shipped copies survive on sp4/sp6/sp7 after the 19:45 archive/delete freed sp7). Container path is `/models/DeepSeek-V4.1-Flash` on every rank either way (`MODEL_HOST` per node). |
| quantization | `config.json quantization_config.quant_method = "exl3"` + `quantization_config.json` (tensor_storage) → cuda-exl3 `Exl3Config` (P2 hybrid: experts EXL3, dense/shared/MTP delegated to `DeepseekV4FP8Config`); no `--quantization` flag needed |
| Engram | on NVMe: `DSV41_ENGRAM_DISK=1`, threads 32 (A/B 64), chunk 16; rows read from the node's own full checkpoint copy (= node-local); Engram fp4 reserved for TP8 residency |
| KV | model-native FP4 KV; gmu 0.80; pin `KV_BYTES=<vLLM's suggested value from the first boot>` on later boots (memory rule 4: never a bigger value) |
| spec decode | DSpark k=5 (`SPEC=dspark SPEC_K=5`), k=3 is −40 %, k=7 untested |
| graphs | `FULL_AND_PIECEWISE`, capture sizes k·seqs and (k+1)·seqs, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| context / seqs | gate profile = Tony: `MAXLEN=300000 SEQS=8`; serve profile (EXL3 headroom ~22 GiB KV/rank ≈ 3.7 M tokens): `MAXLEN=1000000 SEQS=8` or `MAXLEN=300000 SEQS=12`; cap concurrency at 6-8 (tenaiaiai: agg cliff at 8) |
| vision / tools | on: `--limit-mm-per-prompt {"image":4}`, tool parser + reasoning parser `deepseek_v41`, thinking off by default |

## 2. Boot = `dsv41_boot_tp4.sh` (ops-host), which runs in order
1. **Preflight**: no capture/cook containers on the 4 nodes; checkpoint signature identical (exl3: MD5SUMS.body identity, `VERIFY_MD5=1` for the full 240 GB read); image + patches present; MemAvailable ≥ 100 GiB.
2. **Node prep = `dsv41_node_prep.sh sp4 sp6 sp7 sp10`** (the memory playbook, all of it):
   `mem_playbook.sh` (sysctls min_free 1G, watermark 200, vfs_cache_pressure, swappiness 1, dirty ratios; `compact_memory`; `drop_caches`; `glm-reclaim` — skipped if any vllm/cook runs) · **cache_flusher sidecar** installed + started (drop when Cached > 40 GiB, every 5 s; killed after /health) · compaction_proactiveness 0 · earlyoom inactive · fabric MTU 9000 + `glm-fabric-mtu-guard.timer` active + **jumbo ping (8972 B, -M do) to the other 3 fabric IPs** · **GPU fast/slow-state probe** (gemv p50 ≥ 150 GB/s else refuse — 1.5× swing) · **CUDA-visible free ≥ 100 GiB** in the vehicle (`torch.cuda.mem_get_info`; MemAvailable lies after a failed boot) · model dir gate (48 shards, quant_method) · **config parity** (image ID + patch md5 + launcher md5 identical on all 4). Any FAIL aborts the boot: **clean and verify, never lower gmu**.
3. Workers first (ranks 3,2,1), then head; `/health` wait ≤ 45 min (weights ~4 min local, graphs ~1 min).
4. After health: flushers stopped; KV line logged (`Available KV cache memory` → `KV_BYTES` for the next boot); **warm-up** request; count smoke; **tool-call smoke**; then Tony's bench (levels 1-6 + cold prefill 3K/12K/47K/93K) → `results/<EXP>/`.

## 3. Knobs (env to `dsv41_boot_tp4.sh`)
`PROFILE=exl3|shipped` · `IMAGE` · `EXP_NAME` · `GMU` (0.80) · `MAXLEN` · `SEQS` · `SPEC_K` · `ROCE=1` (b12x RoCE one-shot all-reduce, `ROCE_MAX` 2MB) · `NCCL_CH=8` · `ENGRAM_THREADS=64` · `KV_BYTES` · `TEXT_ONLY` · `THINKING` · `SKIP_PREP=1` · `VERIFY_MD5=1` · `SERVED_ALIAS="--served-model-name x"`.

## 4. Gate plan for the EXL3 build (each row = one boot, Tony bench + smokes; ~35 min/row)
| row | env | question |
|---|---|---|
| G0 | `PROFILE=exl3` (Tony config) | fits? KV tokens vs boot3 1.16 M; C1/C6 vs 42.3/130.7; prefill vs 1.1-1.3k |
| G1 | + `ROCE=1` | 2-node test: 2.6× at 360 KB, ≈ at 60 KB → 4-node decode effect |
| G2 | + `ENGRAM_THREADS=64` | tenaiaiai knob |
| G3 | + `NCCL_CH=8` | expect ≈ 0 on our switch (our 0.29 port: −5 % prefill) |
| G4 | `--async-scheduling` (add to VLLM_EXTRA) | our 0.29 recipe lever |
| Q | quality: ppl probe + golden + needle 300K + tool-call + vision (bench_quality_body.sh style) | no regression vs shipped |
Adopt only after the full battery; then the serve profile (1 M ctx) and the ops wiring (watchdog, warm-up cron, dashboard).

## 5. Restore paths
Shipped TP4 on all 4 nodes = copy the 40 body shards back to sp7/sp10 from sp4's shipped dir (`*.shipped` index/config kept in place; ~25 min over the fabric). Brain (GLM TP8) restore = `brain_down_v4.sh up` (needs sp4/6/7/10 back → TP4 line and TP8 brain are exclusive on this fleet).

## 6. Failure patterns already met (see PLAN.md + tenaiaiai failure-log #0-20)
`${VAR:-…}` swallowing `}` in JSON defaults · image ENTRYPOINT already `vllm serve` · rank config mismatch → silent hang (parity check) · 120 s ssh timeouts on image sync · MTU 1500 after a link flap → hang · slow GPU state (1.5×) · MemAvailable vs CUDA-free divergence after a failed boot · pgrep self-match in `ssh bash -c`.

## Required for EXL3 + DSpark: worker-side quantization_config path (P6, 2026-09-10)
cuda-exl3 locates the standalone `quantization_config.json` (the per-tensor `tensor_storage` table; the block inlined into config.json is only a summary) through a class attribute set when vLLM builds the *target* ModelConfig in the main process. The DSpark drafter's quant config is rebuilt inside each spawned worker (`load_dspark_model → get_draft_quant_config → Exl3Config.from_config`), where that attribute is empty → `ValueError: EXL3: could not find tensor_storage` about 4–5 min into the boot, after "Loading weights took …". Symptom at the head: "Engine core initialization failed … Failed core proc(s): {}".
Fix shipped in this recipe: `patches/exl3_config.py` (mounted over `cuda_exl3/config.py` via mounts.txt line `exl3_config.py ../cuda_exl3/config.py`) falls back to env `CUDA_EXL3_MODEL_PATH` (the launcher sets it to `/models/$MODEL_DIR`) or a `quantization_config_file` key in the summary, and remembers the path for `_augment_from_checkpoint`. Smoke: `patches/test_exl3_config_hint.py` inside the image builds all 46080 EXL3 modules from the summary in a fresh process. Candidate for the held cuda-exl3 upstream overlay. Credit: cuda-exl3 (Zeuss5) for the plugin; fix is ours.
