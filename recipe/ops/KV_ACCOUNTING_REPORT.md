# DeepSeek-V4.1-Flash TP4 — vLLM KV cache accounting (why 12 GiB/rank = 2.56 M "tokens" @300K, 3.41 M @1M)

Date: 2026-09-11. Image `vllm-dsv41:overlay8` (vLLM 0.28.1rc1.dev388+g8a728663c) read on sp4 only, plus the
files bind-mounted over it by `dsv41_tp4_launch.sh:44-50` from `~/patches/dsv41-boot3/` (`mounts.txt`).
All numbers below were reproduced by running the real `vllm.v1.core.kv_cache_utils` grouping/allocation code
CPU-only inside the image (script in Appendix A). Three live data points reproduce **exactly**:

| boot | max_model_len | in-flight tokens T | logged "GPU KV cache size" | simulated |
|---|---|---|---|---|
| G0 rungs (async 0, batched 16384) | 300,000 | 16,384 | 2,561,073 | 2,561,073 |
| SERVE1M / prod-B (async 0, batched 16384) | 1,000,000 | 16,384 | 3,411,140 | 3,411,140 |
| MAXB16K-A (async 1, batched 16384) | 1,000,000 | 32,768 | 2,990,822 | 2,990,822 |

## 0. TL;DR

1. The number is **not** `12 GiB / bytes-per-token`. `GPU KV cache size = int(max_concurrency × max_model_len)`
   where `max_concurrency = num_blocks / Σ_groups per_request_blocks(group)` (`kv_cache_utils.py:2300-2327`,
   `:1049-1071`). That is why a larger `max_model_len` prints *more* tokens: the per-request cost has a
   constant part (sliding-window + in-flight reservation, 3,886 blocks) that gets amortised over L.
2. The intrinsic per-token state of this model per rank is **1,790 B unpadded / 1,800 B padded**
   (3 ratio-2 compressed layers × 292 B + 1 ratio-1 layer × 584 B + indexer K caches 3×66 B + 132 B).
   That is within 7 % of the "1,671 B/token/rank" figure of the other stack. The model is not the problem.
3. The loss is structural in vLLM's hybrid allocator, made worse by the SM12x page-size patch that production
   needs (Kai/Tech2Wild `sm12x-pages`):
   * **All 17 KV groups draw blocks from one pool whose block is sized by the widest group
     (138,240 B = the 3 ratio-2 layers' compressed+indexer pages)** (`kv_cache_utils.py:1559-1576`, `:1697-1705`).
   * The ratio-1 layer (layer 20) had to move to **64-token blocks** (64-state pages for the SM120 kernels), so
     it can no longer share a group with the ratio-2 layers (128-token blocks). Its group only fills
     **46,080 of every 138,240-byte block (33 %)** yet it needs `cdiv(L,64)` blocks — **57 % of all blocks a
     1 M request takes**. Net: **1.34 GiB per 1 M-token request is padding** (38 % of the KV pool).
   * Per-request pool utilisation is therefore 63.4 % @300K and 58.7 % @1M (Appendix B).
   * The 43 SWA layers (40 target + 3 DSpark draft) reserve `cdiv(127 + T, 64) + 1 = 259` blocks **per group ×
     15 groups = 3,886 blocks (537 MB of pool) per request** for in-flight prefill tokens
     (`kv_cache_interface.py:718-757`, `config/vllm.py:577-598`). At 300K that is 36 % of a request's blocks;
     at 1M 14 %. Async scheduling doubles T (observed: 3.41 M → 2.99 M).
4. Best lever (simulated): a one-line grouping-heuristic change in `kv_cache_utils.py` so the pool block is one
   (compressed + indexer) layer pair = 46,080 B: **+63 % tokens @1M (3.41 M → 5.57 M), +43 % @300K
   (2.56 M → 3.67 M)**, pool utilisation 91-96 %. Cost: 47 KV groups instead of 17 (scheduler CPU).
   Second: keep T at 8-16K and do not let async double it (+7-12 %). Kernel-side 128-token blocks for layer 20
   would give the same +66 % with only 9 groups but is a kernel-facing change.

---

## 1. What the model registers and the groups vLLM builds (TP4, live patch set)

### 1.1 Layer inventory (from `/mnt/glm52/hub/DeepSeek-V4.1-Flash/config.json`)

`num_hidden_layers=40`, `compress_ratios=[0,0, 2×18 (L2-19), 1×20 (L20-39), 0,0,0 (MTP)]`,
`kv_source_layer_ids=[2,8,14,20]`, `index_source_layer_ids=[2,8,14,20,24,28,32,36]`, `sliding_window=128`,
`head_dim=512`, `index_head_dim=128`, `num_nextn_predict_layers=3`, `dspark_target_layer_ids=[37,38,39]`.

Per attention layer the v4.1 module registers up to three `AttentionLayerBase` caches
(`models/deepseek_v4_1/attention.py`):

| cache | who owns one | spec | source |
|---|---|---|---|
| compressed MLA KV (`…self_attn`) | **only kv-source layers 2, 8, 14, 20** — consumers read the source's cache through the forward context | `MLAAttentionSpec(tokens_per_state=compress_ratio, state_content_bytes=584, alignment=576)` | `attention.py:930-955` (returns `None` when `not self.is_kv_source`, :935-936) |
| indexer K cache (`…self_attn.indexer.k_cache`) | **only kv-source layers** (index-source layers 24/28/32/36 share the K cache of the kv source below them) | `MLAAttentionSpec(head_size=132, tokens_per_state=compress_ratio, alignment=576)` | `attention.py:370-395`, `:991-1009`; 132 B = 128 fp8 + 4 B fp32 scale, `:82-91` |
| SWA cache (`…self_attn.swa_cache`) | **every layer** incl. the 3 DSpark draft layers (ids 40-42, `compress_ratio=0`) | `SlidingWindowMLASpec(sliding_window=128, state_content_bytes=584, alignment=576)` | `attention.py:449-457`; `sparse_swa.py:107-127`; draft layers: `nvidia/dspark.py:110-120`, `:339-341` |

Not KV caches: the candidate-block buffer is a plain `torch.empty` (`nvidia/model.py:428-437`);
`HiddenStateCacheSpec` is only produced by `extract_hidden_states.py` (not enabled); the vision tower registers
no KV spec (`gpu_model_runner.py:7694-7698` skips modules returning `None`).

### 1.2 Page geometry — image vs. what actually runs

| spec | image (`attention.py:456`, `:942`, `:996`) | **live** (`~/patches/dsv41-boot3/{attention,flashinfer_sparse,sparse_swa}.py`, `sm12x-pages/*.diff`) |
|---|---|---|
| SWA block | 32 tokens | **64** — `DeepseekSparseSWAFlashInferSM120Backend.get_swa_block_size()=64` (flashinfer_sparse.py.diff) because FlashInfer SM120 sparse kernels are instantiated for 64-token pages |
| compressed block | `cache_config.block_size` = 128 for all | **64 × compress_ratio** → ratio-2: 128 tok (64 states), ratio-1: **64 tok** (64 states) — `get_compressed_block_size()` |
| indexer block | 128 | **64 × compress_ratio** (indexer-64state.diff; DeepGEMM paged MQA logits take 32/64 states) |
| attention class | `DeepseekV4FlashInferSM120Attention` on SM12x (`nvidia/model.py:114-150`), `use_fp8_ds_mla_layout=True` (`flashinfer_sparse.py:542-547`) | same |

Page bytes (`kv_cache_interface.py:421-435`: `num_heads × num_states × state_content_size_bytes`, then
`_apply_alignment_padding` :543-549 rounds up to 576):

| page | states | unpadded | padded (576) | bytes/token |
|---|---|---|---|---|
| compressed, any ratio (64 states × 584 B) | 64 | 37,376 | **37,440** | 292.5 (ratio 2) / 585 (ratio 1) |
| indexer K (64 states × 132 B) | 64 | 8,448 | **8,640** | 67.5 (ratio 2) / 135 (ratio 1) |
| SWA (64 tokens × 584 B) | 64 | 37,376 | **37,440** | 585 (only inside window) |

### 1.3 Groups actually built (packed path, layout BLHNC)

`DeepseekV4IndexerBackend.supported_kv_cache_layouts = (BLHNC, BLNHC)` (`indexer.py:286-292`) → block-outermost
→ `get_kv_cache_groups` takes `_get_packed_kv_cache_groups` (`kv_cache_utils.py:2229-2235`, `:1936-2071`).
Buckets are formed by `UniformTypeKVCacheSpecs.is_uniform_type`, which requires the **same block_size**
(`kv_cache_interface.py:1083-1092`), so the 64-token layer-20 caches cannot join the 128-token ratio-2 caches.
`repeats_per_group = _approximate_gcd([3, 1, 43], lower_bound=3) = 3` (`:1985-2000`, `:1901-1935`), so the
ratio-2 bucket stays whole, the SWA bucket is cut into groups of 3 layers, and
`bytes_per_block = max(Σ pages in a group) = 138,240` (`:1559-1576`).

| group(s) | layers | spec | block (tokens) | window / tps | group page (B) | fill of 138,240 | per-request blocks @300K | @1M |
|---|---|---|---|---|---|---|---|---|
| g0-g11 (12) | 3 SWA each (36 target layers) | `SlidingWindowMLASpec` | 64 | win 128, tps 1 | 112,320 | 81 % | 259 each | 259 |
| g12-g13 (2) | 2 SWA each (4 target layers) | `SlidingWindowMLASpec` | 64 | win 128 | 74,880 | 54 % | 259 each | 259 |
| g14 | L2, L8, L14 compressed + 3 indexer K | `MLAAttentionSpec` ×6 | **128** | tps 2 | **138,240** (widest) | 100 % | 2,344 | 7,813 |
| g15 | L20 compressed + indexer K | `MLAAttentionSpec` ×2 | **64** | tps 1 | 46,080 | **33 %** | 4,688 | 15,625 |
| g16 (`is_eagle_group`) | 3 DSpark draft SWA (ids 40-42) | `SlidingWindowMLASpec` | 64 | win >128 (see note) | 112,320 | 81 % | 260 | 260 |
| **total 17 groups** | 48 cache tensors | | | | | | **10,918** | **27,324** |

Per-request block formulas: full/MLA `cdiv(max_model_len, block)` (`kv_cache_interface.py:471-476`);
sliding window `cdiv(min(window-1 + extra_retained + max_in_flight_tokens, L), block) + 1`
(`:718-757`), `extra_retained_tokens = 0` for dspark (`kv_cache_utils.py:2570-2582`,
`use_multi_module_mtp` is mtp-only, `config/speculative.py:1908-1914`);
`max_in_flight_tokens = max_concurrent_batches × max_num_batched_tokens` (`config/vllm.py:590-598`),
`max_concurrent_batches = 2` with async scheduling else 1 (`:577-587`). With batched 16,384: T=16,384 → 259;
async: T=32,768 → 515.

Note on g16: the exact numbers only reproduce if the 3 draft SWA layers sit in their own group with a
window in (129, 193] (any value gives 260 blocks). The DSpark non-causal index width is
`cdiv(128+5, 64)×64 = 192` (`compressor_utils.py:19-28`); I did not locate the line that writes it into the
draft spec — it changes 1 block/request (0.01 %) and nothing else.

## 2. Bytes per token and how the token count is derived

* Pin: `--kv-cache-memory-bytes 12884901888` is used verbatim (`gpu_worker.py:534-556`;
  `reserve_mm_ipc_gpu_memory` returns it unchanged because `mm_ipc_gpu_memory_gb` defaults to 0,
  `multimodal/gpu_ipc_memory.py:186-200`, `config/multimodal.py:256`).
* `num_blocks = 12,884,901,888 // 138,240 = 93,206` (`kv_cache_utils.py:1703-1705`). Every group aliases the
  same pool from byte 0; a block used by a small-page group still consumes 138,240 B.
* `max_concurrency = 93,206 / Σ per-request blocks` → 93,206/10,918 = 8.5369 @300K, 93,206/27,324 = 3.4111 @1M
  (`:1049-1071`). `tokens = int(max_concurrency × max_model_len)` = **2,561,073** and **3,411,140** (`:2300-2327`).

Decomposition of the "bytes/token" that the pin implies:

| | @300K | @1M | @4M (sim) | L→∞ |
|---|---|---|---|---|
| pool bytes charged per token (12 GiB / tokens) | 5,031 | 3,777 | 3,374 | → 3,254 |
| bytes actually written in the charged blocks / token | 3,190 | 2,217 | 1,904 | → 1,800 |
| of which intrinsic compressed+indexer state | 1,800 | 1,800 | 1,800 | 1,800 |
| of which SWA window + in-flight reservation (3,886 blocks ≈ 398 MiB/req, constant) | 1,390 | 417 | 104 | 0 |
| pool utilisation (written / charged) | 63.4 % | 58.7 % | 56.4 % | 55.3 % |

So versus the other stack's 1,671 B/token: our intrinsic state is 1,800 B (+7.7 %, the 8 B per-token fp8 scale
block and the 4 B fp32 indexer scale plus 576-B alignment account for it); the remaining ×2.1 (@1M) is
1/0.587 packing waste × (2,217/1,800) SWA reservation. The 1M profile prints *more* tokens than 300K purely
because the constant 3,886-block SWA term is divided by a larger L; physical capacity is identical.

## 3. What dominates and what is over-allocated

Per 1 M-token request (27,324 blocks × 138,240 B = 3,602 MiB charged; 2,114 MiB written):

| component | blocks | share of blocks | charged MiB | written MiB | wasted MiB | verdict |
|---|---|---|---|---|---|---|
| g15 layer-20 ratio-1 compressed+indexer (64-tok blocks) | 15,625 | **57.2 %** | 2,060 | 687 | **1,373** | **over-allocated 3×** — needs 46,080 B/block, charged 138,240 |
| g14 ratio-2 compressed+indexer ×3 (128-tok blocks) | 7,813 | 28.6 % | 1,030 | 1,030 | 0 | sets the pool block; fine |
| 14 target SWA groups + 1 draft group | 3,886 | 14.2 % | 512 | 398 | 115 | reservation-dominated: 3,886 blocks = 43 layers × (window 128 + T 16,384 in-flight + 1 slack); steady-state decode only needs ~3 blocks/group |
| total | 27,324 | | 3,602 | 2,114 | 1,488 (41 %) | |

Answers to the specific suspicions:

* **Compressed MLA layers**: 4 caches only (not 38) — the consumer layers correctly share the kv-source cache
  (`attention.py:262-283`, `:935-936`). Not over-allocated per se; the ratio-1 one is charged 3× because of the
  block-size split (above).
* **Indexer K cache**: 4 caches (only kv-source layers own one, `:370-395`), 337 B/token total. Small. The
  MXFP4 variant (68 B/state, `:82-87`) is refused on sm_120 (`indexer.py:54-66`).
* **SWA windows**: never full-length (`SlidingWindowMLASpec` is admitted with `min(window-1+T, L)`), so no
  "full cache for a sliding layer" problem. The cost is (a) in-flight reservation T×43 layers, (b) group page
  padding (81 %/54 % fill), (c) `+1` slack block per group.
* **Hidden-state / candidate cache**: none exists in this configuration.
* **DSpark draft group**: 3 SWA layers, 260 blocks/request (≈35 MB charged). Removing the drafter does not
  help — the gcd heuristic then picks d=4 and the pool block grows to 149,760 (sim: 3.31 M, −3 %).
* **Page padding across groups**: this is the big one — 138,240-B blocks for a 46,080-B group. Alignment
  padding to 576 B itself is 0.17 %.
* **Block size flag**: `--block-size 128` only reaches the ratio-2 caches now (the patch overrides the rest);
  sweeping it in the image geometry changes tokens by <1 % (Appendix B). Irrelevant as a lever.

## 4. Options to raise tokens per GiB (ranked; all simulated with the real grouping code, 12 GiB, T=16,384)

| # | change | tokens @300K | tokens @1M | gain | risk / cost |
|---|---|---|---|---|---|
| 0 | as-is (async 0) | 2,561,073 | 3,411,140 | — | — |
| 0' | as-is with `ASYNC=1` + batched 16,384 (line A `accepted-A.env`) | 1,894,687 | 2,990,822 (observed) | **−12 % / −26 %** | metric+max-concurrency only; physical SWA use during decode is unchanged |
| **1** | **Grouping-heuristic patch**: in `_get_packed_kv_cache_groups` let balanced mixed-page buckets split to one repeat (`kv_cache_utils.py:1985-2000`: `_approximate_gcd(..., lower_bound=1)` or pick `d` minimising charged bytes). Pool block → 46,080 B; groups → 47 (43 SWA×1, 4 MLA×1) | **3,669,553** | **5,569,675** | **+43 % / +63 %** (util 91-96 %) | pure Python, mountable via `mounts.txt` like the other patches; 47 groups → 47 block tables/slot-mapping builds per step (scheduler CPU), `HybridKVCacheCoordinator` already in use with 17 groups. Needs a canary boot + the `find_longest_cache_hit` path exercised (prefix caching on). No kernel change. |
| 1b | #1 + `MAX_BATCHED=8192` (T=8,192) | 4,833,256 | 6,255,480 | +89 % / +83 % | halves prefill chunk; TTFT on long prompts ↑ |
| 1c | #1 with `ASYNC=1`, batched 16,384 (T=32,768) | 2,476,851 | 4,568,058 | −3 % / +34 % vs as-is | keeps async decode gains |
| 2 | **Kernel-side**: give layer 20 (and its indexer) 128-token manager blocks viewed as 2×64-state kernel pages (`storage_block_size` / kernel-block splitting, as `models/glm5next` and `qwen4_exp` do). Requires dense pages: drop the 576 alignment to 64/512 so 37,376 stays unpadded (`create_kv_cache_views` refuses padded pages for splitting, `kv_cache_interface.py:326-345`), and the SM120 sparse decode/prefill + DeepGEMM indexer paths must accept the split view | 3,798,324 | 5,656,888 | +48 % / +66 % (9 groups, util 97 %) | kernel-facing; Kai's patch exists because these kernels are fixed at 64 states — high effort, correctness testing on the sparse path |
| 3 | Lower `MAX_BATCHED` only: 8,192 / 4,096 / 2,048 | — | 3,668,949 / 3,813,042 / 3,889,417 | +7.6 % / +11.8 % / +14 % | prefill throughput ↓; largely an accounting gain (in-flight reservation) |
| 4 | Force d=2 (SWA pairs, pool block 92,160) | 2,782,104 | 3,783,659 | +8.6 % / +11 % | worse than #1; 25 groups |
| 5 | Indexer K cache MXFP4 (68 B/state) | — | 3,738,252 | +9.6 % | not available on sm_120 (`indexer.py:63-66`) |
| 6 | `--block-size` 64/256/512 | ±1 % | ±1 % | none | only affects ratio-2 caches after the patch |
| 7 | Raise `max_model_len` | — | 4M: 3,818,509 | metric only | no physical change; do not compare the token count across max_model_len values |

Recommendation: implement #1 as an additional bind-mounted file (`v1/core/kv_cache_utils.py`) on a canary line,
verify the boot log shows 47 groups / `GPU KV cache size ≈ 5.57 M` at 1M, then A/B decode/prefill throughput
against the 17-group boot (the only cost is scheduler-side per-group work). Keep T at ≤16,384 (async with
batched 8,192, or async off with 16,384) so the SWA reservation does not eat the gain. Treat #2 as the
long-term fix and raise it with the sm12x-pages author (it is the natural completion of that patch).

Hold per `feedback-hold-upstream-patches-until-confirmed`: do not send #1/#2 upstream until a boot confirms.

---

## Appendix A — CPU-only reproduction (run inside the image on sp4, no GPU)

```python
# ssh user@RANK_LAN_IP 'docker run --rm -i --entrypoint python3 vllm-dsv41:overlay8 -' < this.py
import types, torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec, get_kv_quant_mode
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.core import kv_cache_utils as U
from vllm.utils.math_utils import cdiv
MEM = 12884901888
CR = [0,0]+[2]*18+[1]*20+[0,0,0]; KV_SRC = [2,8,14,20]
def specs(swa_bs=64, comp_bs=lambda cr: 64*cr, idx_bs=lambda cr: 64*cr, n_draft=3, draft_win=192):
    d = {}; q = get_kv_quant_mode("fp8_ds_mla")
    for i in range(40+n_draft):
        cr = CR[i]; p = f"model.layers.{i}.self_attn"
        if i < 40 and i in KV_SRC:
            d[p] = MLAAttentionSpec(block_size=comp_bs(cr), num_kv_heads=1, head_size=512, dtype=torch.uint8,
                tokens_per_state=cr, cache_dtype_str="fp8_ds_mla", alignment=576, model_version="deepseek_v4",
                kv_quant_mode=q, state_content_bytes=584)
            d[p+".indexer.k_cache"] = MLAAttentionSpec(block_size=idx_bs(cr), num_kv_heads=1, head_size=132,
                dtype=torch.uint8, tokens_per_state=cr, alignment=576)
        d[p+".swa_cache"] = SlidingWindowMLASpec(block_size=swa_bs, num_kv_heads=1, head_size=512, dtype=torch.uint8,
            sliding_window=(draft_win if i >= 40 else 128), cache_dtype_str="fp8_ds_mla", state_content_bytes=584,
            alignment=576, model_version="deepseek_v4", kv_quant_mode=q)
    return d
def cfg(L, T):
    ns = types.SimpleNamespace
    return ns(scheduler_config=ns(disable_hybrid_kv_cache_manager=False, async_scheduling=False, max_num_batched_tokens=T),
        model_config=ns(max_model_len=L, original_max_model_len=L, hf_config=ns(model_type="deepseek_v41")),
        parallel_config=ns(decode_context_parallel_size=1, pipeline_parallel_size=1), max_in_flight_tokens=T,
        cache_config=ns(block_size=128, num_gpu_blocks_override=None, prefix_cache_retention_interval=None,
                        cache_dtype="fp8_ds_mla", get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC),
        speculative_config=ns(method="dspark", num_speculative_tokens=5, use_eagle=lambda: True,
                              use_eagle_block_drop=lambda: True, use_multi_module_mtp=lambda: False))
def run(L, T=16384, **kw):
    c = cfg(L, T); g = U.get_kv_cache_groups(c, specs(**kw)); k = U.get_kv_cache_config_from_groups(c, g, MEM)
    tok, conc = U.get_kv_cache_capacity(c, k)
    print(L, T, "groups", len(g), "bytes/block", U._get_kv_cache_bytes_per_block(g), "blocks", k.num_blocks, "tokens", f"{tok:,}")
for L in (300000, 1000000): run(L)                 # 2,561,073 / 3,411,140
run(1000000, T=32768)                              # 2,990,822 (async on)
orig = U._approximate_gcd; U._approximate_gcd = lambda v, lower_bound=None: orig(v, lower_bound=1)
for L in (300000, 1000000): run(L)                 # option 1: 3,669,553 / 5,569,675
```

## Appendix B — other simulated variants (12 GiB, T=16,384 unless noted)

| variant | groups | bytes/block | tokens @1M | pool util @1M |
|---|---|---|---|---|
| live geometry | 17 | 138,240 | 3,411,140 | 58.7 % |
| live, SWA block 32 (image default; SM120 kernel rejects) | 17 | 138,240 | 2,987,850 | 51.5 % |
| image geometry (no sm12x patch: all 128-tok, SWA 32) | 5 | 230,400 | 4,688,071 | — |
| image geometry, `--block-size` 64 / 256 / 512 | 9 / 3 / 2 | 116,928 / 460,224 / 918,720 | 4,618,979 / 4,693,545 / 4,701,307 | — |
| live, no drafter | 12 | 149,760 | 3,305,555 | 56.1 % |
| live, T = 8,192 / 4,096 / 2,048 | 17 | 138,240 | 3,668,949 / 3,813,042 / 3,889,417 | 57 % / 56 % / 56 % |
| live, indexer MXFP4 (68 B) | 17 | 126,144 | 3,738,252 | 59.8 % |
| option 1 (d=1) | 47 | 46,080 | 5,569,675 | 95.8 % |
| option 1 + T=8,192 | 47 | 46,080 | 6,255,480 | 97.6 % |
| option 1 + async (T=32,768) | 47 | 46,080 | 4,568,058 | 93.2 % |
| option 2 (all MLA at 128-tok blocks) | 9 | 230,400 | 5,656,888 | 97.3 % |
| option 4 (d=2) | 25 | 92,160 | 3,783,659 | 65.1 % |

## Appendix C — file:line index (image unless marked live)

* Spec construction: `models/deepseek_v4_1/attention.py:82-91` (indexer 132 B), `:236` (window), `:251-256`
  (compress_ratio), `:262-283` (kv/index source), `:370-395` (indexer K cache only on kv-source),
  `:449-457` (SWA cache, block 32 in image), `:930-955` (compressed spec), `:966-1009` (indexer spec).
  Live overrides: `~/patches/dsv41-boot3/sm12x-pages/{attention.py.diff, flashinfer_sparse.py.diff,
  sparse_swa.py.diff, indexer-64state.diff}`, mounted per `~/patches/dsv41-boot3/mounts.txt`.
* Attention class on SM12x: `models/deepseek_v4_1/nvidia/model.py:114-150`; `nvidia/flashinfer_sparse.py:96-114`
  (kernel block sizes), `:542-547` (SM120 class, fp8_ds_mla layout).
* SWA spec/backends: `v1/attention/backends/mla/sparse_swa.py:72-127`, `:132-160`.
* Spec classes: `v1/kv_cache_interface.py:151-195` (num_states), `:386-443` (page bytes), `:446-476`
  (FullAttention max mem), `:543-549` (576 alignment), `:553-617` (MLA spec/merge), `:708-757`
  (sliding-window admission), `:798-862` (SlidingWindowMLASpec), `:1036-1120` (UniformTypeKVCacheSpecs:
  `page_size_bytes` = Σ pages :1056-1058, `is_uniform_type` block-size check :1083-1092), `:326-345`
  (kernel-block split needs dense pages).
* Grouping/allocation: `v1/core/kv_cache_utils.py:2188-2276` (`get_kv_cache_groups`), `:1936-2071`
  (`_get_packed_kv_cache_groups`; gcd :1985-2000; state buckets :2017-2036), `:1901-1935` (`_approximate_gcd`),
  `:1559-1576` (bytes/block = widest group), `:1612-1766` (`get_kv_cache_config_from_groups`; num_blocks :1703-1705),
  `:1049-1071` (max concurrency), `:2300-2327` (token count log line 2321), `:2527-2682`
  (`get_kv_cache_configs`; extra_retained :2570-2582), `:2073-2137` (eagle group annotation).
* In-flight tokens: `config/vllm.py:577-598`. DSpark index width: `v1/attention/backends/mla/compressor_utils.py:19-28`.
* Memory pin: `v1/worker/gpu_worker.py:534-556`; `multimodal/gpu_ipc_memory.py:155-215`.
* Layout: `v1/kv_cache_layout.py:15-58`; `v1/attention/backends/utils.py:240-285`; indexer backend layouts
  `v1/attention/backends/mla/indexer.py:286-296`; MXFP4 indexer gate `:54-66`.
* Drafter: `models/deepseek_v4_1/nvidia/dspark.py:68-120`, `:299-341`.
* Launch: `dsv41_tp4_launch.sh:14` (`--block-size 128`), `:44-50` (patch mounts), `:65` (dspark k=5),
  `:81` (`--kv-cache-memory-bytes`, `--max-model-len`, `--max-num-batched-tokens`); `ops/accepted-A.env`
  (`KV_BYTES=12884901888`, `MAXLEN=1000000`, `MAX_BATCHED=16384`, `ASYNC=1`).
