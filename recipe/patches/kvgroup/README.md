# kvgroup — finer KV cache group packing for DeepSeek-V4.1-Flash TP4 (vLLM 0.28.1rc1, image `vllm-dsv41:overlay8`)

Background and full accounting: `../../ops/KV_ACCOUNTING_REPORT.md`. Status 2026-09-11: **simulated only, not booted**.
Hold per `feedback-hold-upstream-patches-until-confirmed`: nothing goes upstream until a canary boot confirms.

## What it changes

`kv_cache_utils.py` is the image's `vllm/v1/core/kv_cache_utils.py` (md5 `ba85578d…`) with one gated change in
`_get_packed_kv_cache_groups` (patched file lines 1990-2014, upstream lines 1985-2000):

```python
_fine_grouping = os.environ.get("DSV41_KV_GROUPING", "") == "fine"
repeats_per_group = _approximate_gcd([...], lower_bound=1 if _fine_grouping else min_repeats_per_group)
```

Upstream forces every KV cache group to hold at least `min_repeats_per_group` = 3 repeats of the widest
mixed-page pattern, so the shared pool block is sized by the 3 ratio-2 kv-source layers
(3 × (37,440 compressed + 8,640 indexer) = **138,240 B**) while the ratio-1 layer-20 group (64-token blocks, cannot
share a group with 128-token blocks: `kv_cache_interface.py:1083-1092`) fills only 46,080 B of each block yet takes
`cdiv(L, 64)` blocks per request (57 % of a 1 M request). With `DSV41_KV_GROUPING=fine` the repeat floor is 1, the
pool block becomes **46,080 B**, and every group fills 81-100 % of its block. Env unset (or any other value) →
upstream behaviour, byte-for-byte identical code path.

Groups: 17 → **47** (43 SWA groups of 1 layer, 3 groups of one ratio-2 compressed+indexer pair, 1 group for layer 20;
the DSpark draft SWA group stays separate and eagle-flagged, as today). Per-layer page format, block sizes, layout
(BLHNC) and hash/scheduler block sizes (gcd 64 / lcm 128) are unchanged.

## Mount

Add to the boot patch manifest (`~/patches/dsv41-boot3/mounts.txt` format, consumed by `dsv41_tp4_launch.sh:44-50`):

```
kv_cache_utils.py v1/core/kv_cache_utils.py
```

and enable per boot with `-e DSV41_KV_GROUPING=fine` (e.g. via `EXTRA_ENV="-e DSV41_KV_GROUPING=fine"` in the
launcher env). Without the env the mounted file behaves exactly like the image's. Verify in the boot log:
`DSV41_KV_GROUPING=fine: packing KV cache groups at one pattern repeat per group (upstream floor was 3 repeats).`
and `GPU KV cache size` matching the table below.

## Expected gains (12 GiB/rank pin, simulated with the real grouping code inside the image, patch mounted)

`sim_kvgroup.py` run 2026-09-11 on sp4 (`docker run --rm -i -v …/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro --entrypoint python3 vllm-dsv41:overlay8 - < sim_kvgroup.py`, CPU only).
Default mode reproduces every logged value (2,561,073 / 3,411,140 / 2,990,822) exactly.

| max_model_len | max_num_batched_tokens | async | in-flight T | default tokens (17 groups, 138,240 B/block, 93,206 blocks, util) | fine tokens (47 groups, 46,080 B/block, 279,620 blocks, util) | gain |
|---|---|---|---|---|---|---|
| 300,000 | 8,192 | off | 8,192 | 3,107,557 (60.4 %) | 4,833,256 (93.9 %) | +56 % |
| 300,000 | 8,192 | on | 16,384 | 2,561,073 (63.4 %) | 3,669,553 (90.9 %) | +43 % |
| 300,000 | 16,384 | off | 16,384 | **2,561,073** (logged) (63.4 %) | 3,669,553 (90.9 %) | +43 % |
| 300,000 | 16,384 | on | 32,768 | 1,894,687 (67.1 %) | 2,476,851 (87.7 %) | +31 % |
| 1,000,000 | 8,192 | off | 8,192 | 3,668,949 (57.3 %) | 6,255,480 (97.6 %) | +70 % |
| 1,000,000 | 8,192 | on | 16,384 | 3,411,140 (58.7 %) | 5,569,675 (95.8 %) | +63 % |
| 1,000,000 | 16,384 | off | 16,384 | **3,411,140** (logged) (58.7 %) | 5,569,675 (95.8 %) | +63 % |
| 1,000,000 | 16,384 | on | 32,768 | **2,990,822** (logged, MAXB16K-A) (61.0 %) | 4,568,058 (93.2 %) | +53 % |

"util" = bytes written into the charged pool blocks per max-length request / bytes charged. The token count is
`int(max_concurrency × max_model_len)`; the in-flight term T = (2 if async else 1) × max_num_batched_tokens is a
sliding-window admission reservation (43 layers × `cdiv(127+T, 64)+1` blocks), which is why async with 16 K
batched costs 12-26 % of the metric. Note the current `ops/accepted-A.env` (`MAX_BATCHED=16384`, `ASYNC=1`) is the
last row.

## Risks and assumptions

1. **Scheduler CPU: 47 block tables instead of 17.** `KVCacheManager`/`HybridKVCacheCoordinator` allocate, free and
   prefix-match per group (`v1/core/kv_cache_coordinator.py`, `find_longest_cache_hit` iterates groups); the model
   runner builds per-group block-table tensors and slot mappings each step. Per-layer slot mappings already exist for
   all 48 cache tensors, so the growth is in group-level bookkeeping (~2.8×), not in kernel work. Measure decode
   step time / TTFT at max_num_seqs 8 before adopting; if it hurts, an intermediate variant (d=2: 25 groups, pool block
   92,160 B, +11 % @1M) is a fallback (set `lower_bound=2`). The 47-group hybrid path is not exercised anywhere else
   in our fleet — canary boot first.
2. **Kernel assumptions about group layout.** Each layer's page keeps its exact format (64-state pages: 37,440 B
   compressed/SWA, 8,640 B indexer, 576-B aligned) and stays one contiguous chunk inside a block; only the per-block
   stride of the pool changes (138,240 → 46,080 B). The SM120 FlashInfer sparse path already runs with a non-dense
   block stride today and reads none itself (`_packed_block_span`, which requires block_stride % token_stride == 0, is
   only used by the SM100 class: live `flashinfer_sparse.py:433-434` inside `DeepseekV4FlashInferMLAAttention`).
   Any kernel that hard-codes the 138,240-B stride or assumes block_stride == n × page would break; none found, but
   this is the assumption to validate with the correctness probes (needle / long-ctx / tool-call) after the canary boot.
   `validate_kv_cache_layout` still passes (mixed page sizes need a block-compact, block-outermost layout: BLHNC).
3. **Interaction with the `sm12x-pages` patch (Kai/Tech2Wild).** This patch only pays off *because* of
   `sm12x-pages`: it is the 64-token block of the ratio-1 layer that splits the MLA bucket. With the unpatched image
   geometry (all MLA caches at 128-token blocks, SWA 32) there is no balanced mixed-page bucket,
   `min_repeats_per_group` is 0 and `DSV41_KV_GROUPING=fine` is a no-op. Mount order does not matter (different
   files). If `sm12x-pages` later moves layer 20 back to 128-token blocks (kernel-side fix, option 2 in the report),
   this patch becomes a no-op again — safe to leave mounted.
4. **DSpark draft group.** The 3 draft SWA layers keep their own group (window > 128) and the
   `is_eagle_group` flag (`kv_cache_utils.py:2084-2137` positional fallback: group holding the last registered layer);
   the eagle last-block drop therefore still targets only that group. Per-request cost unchanged (260 blocks).
   `use_multi_module_mtp()` is false for dspark, so `extra_retained_tokens` stays 0.
5. **Memory pin unchanged.** `--kv-cache-memory-bytes` is honoured as before; `num_blocks = 12 GiB // 46,080 =
   279,620`. Nothing else in `get_kv_cache_configs` (null-block reservation, enough-memory check, min-blocks across
   ranks) is affected. No effect on weights/activations.
6. **Prefix caching.** hash_block_size stays gcd(64,128) = 64 and scheduler block size lcm = 128
   (`resolve_kv_cache_block_sizes`), unchanged; more groups means more per-group `find_longest_cache_hit` work.
7. The gated code path is also what `sim_kvgroup.py` exercises; if the image is rebuilt, re-derive the file from
   the new `kv_cache_utils.py` (the hunk is 1 line + logging) and re-run the sim — default mode must keep
   reproducing the logged values.

## Files

| file | purpose |
|---|---|
| `kv_cache_utils.py` | image file + gated change (md5 `c86d8833…`); mount over `v1/core/kv_cache_utils.py` |
| `sim_kvgroup.py` | CPU-only reproduction; exits 0 only if default mode reproduces the three logged values |
| `README.md` | this file |

Scratch copy used for the simulation mount: `user@RANK_LAN_IP:~/kvgroup-sim/kv_cache_utils.py` (same md5; can be removed).
