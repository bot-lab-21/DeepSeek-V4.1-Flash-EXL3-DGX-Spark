# woproj: DeepSeek-V4.1 `_o_proj` through the native b12x MXFP8 WO kernel

Status 2026-09-11: **drafted + unit-tested in the image on sp4 (PASS), not yet
served.** Pure-Python mount patch for `vllm-dsv41:overlay8`
(vLLM 0.28.1rc1.dev388, dsv41-feat tree). Default OFF; behaviour is byte-for-byte
the original unless `DSV41_WOPROJ_B12X=1`.

## Problem

`wo_a` is a `ColumnParallelLinear` with `is_bmm=True`. For bmm layers
`init_mxfp8_linear_kernel` only tries `[DeepGemmMxfp8BmmLinearKernel,
EmulationMxfp8LinearKernel]`; DeepGEMM is unsupported on SM121, so the
emulation kernel wins and **dequantizes wo_a to bf16 at load**. `o_proj.py`
then takes the bf16 `torch.bmm` branch for every token (16 MiB bf16 weight read
per layer per step) and `wo_b` goes through the FlashInfer-CUTLASS MXFP8 linear.
b12x ships a fused kernel for exactly this shape: inverse-RoPE + MXFP8 quant ->
grouped wo_a GEMM -> MXFP8 quant -> wo_b GEMM (`b12x.gemm.wo_projection`).

## Files

| file | purpose |
|---|---|
| `emulation.py` | copy of `vllm/model_executor/kernels/linear/mxfp8/emulation.py`; for `is_bmm` layers stashes the ORIGINAL fp8 weight / uint8 scale as `layer.mxfp8_weight_fp8` / `layer.mxfp8_weight_scale_u8` before the bf16 dequant (dequant kept, so the fallback path is unchanged) |
| `flashinfer_sparse_woproj.py` | boot3's `flashinfer_sparse.py` + `_B12xWoProjMixin` on both attention classes (`DeepseekV4FlashInferMLAAttention`, `DeepseekV4FlashInferSM120Attention`); original `_o_proj` kept as `_o_proj_reference` |
| `apply_woproj_patch.py` | regenerates `flashinfer_sparse_woproj.py` from any base `flashinfer_sparse.py` (`python3 apply_woproj_patch.py <base> <out>`); re-run when boot3's copy changes |
| `test_woproj.py` | in-image GPU unit test (fake TP4-rank layer, real vLLM kernels for the reference path) |
| `test_run_sp4_20260911.log` | the passing run |
| `orig/` | pristine image sources + boot3's `flashinfer_sparse.py`/`attention.py` used for the diff/rebase (reference only, not mounted) |

## Mount

`mounts.txt` lines (launcher format `<file> <path under $SITE>`, `$SITE=.../dist-packages/vllm`).
`flashinfer_sparse_woproj.py` REPLACES boot3's `flashinfer_sparse.py` line (same
target path; it already contains the Tech2Wild/Kai SM120 64-page block-size
changes from boot3). `emulation.py` is a new line:

```
emulation.py model_executor/kernels/linear/mxfp8/emulation.py
flashinfer_sparse_woproj.py models/deepseek_v4_1/nvidia/flashinfer_sparse.py
```

Env switches (container `-e`):

| var | default | meaning |
|---|---|---|
| `DSV41_WOPROJ_B12X` | `0` | `1` = use b12x WO path (per layer, after weights load); anything else = original path |
| `DSV41_WOPROJ_WARMUP` | `1` | `0` skips the load-time JIT warmup (do NOT skip on a served row: compiles would land inside CUDA-graph capture) |
| `DSV41_WOPROJ_BIND` | `0` | `1` = per-call `plan.bind_inv_rope()` + `binding.run()` instead of the direct `run_inv_rope()` call; A/B only, results identical |

## How it works

1. `__init__` (both attention classes): after the einsum-recipe block,
   `self._woproj_setup(vllm_config)` wraps the per-layer
   `quant_method.process_weights_after_loading` of `wo_a` and `wo_b`
   (ModelOpt builds one linear method per layer via `build_linear_method`, so
   the wrap is instance-local; the hook chains for any other layer).
2. Load time: the hook stashes the checkpoint MXFP8 payload (`[N,K]` e4m3 +
   `[N,K/32]` UE8M0 uint8) BEFORE the kernel touches it, then calls the
   original. `emulation.py` does the same for `wo_a` as belt-and-braces (both
   are idempotent; if the hook already stashed, emulation does nothing).
   After `wo_b` is processed, `_woproj_try_init` packs and warms up.
3. Packing: `wo_projection.pack_weights` expects DSV4-style 128x128 block scales
   and re-expands them; our ModelOpt checkpoint already carries native per-32
   UE8M0 scales, so `_build_woproj_state` builds `MXFP8Rows` directly:
   `wo_a.values = fp8.view(G,R,D).permute(1,2,0)` (dense-GEMM `[M,K,L]` view over
   physical `[L,M,K]`), `scale_rows = u8.view(G,R,D/32).view(e8m0)`,
   `scale_mma = pack_mxfp8_scales_for_dense_gemm(...)`; same for `wo_b`
   (`[hidden, G*R]`, one group); `sfb_k_replicated=False`. Verified: b12x's own
   `dequantize_mxfp8_rows_torch` of these rows == the checkpoint dequant, bit-exact.
4. Plan: `wo.plan(Caps(max_tokens=self.max_num_batched_tokens, ...))` (the
   attention layer already stores `scheduler_config.max_num_batched_tokens`).
   NOTE: for the inverse-RoPE variant the scratch/binding is *unused* by b12x:
   `wo_projection_inv_rope_mxfp8` runs one opaque fused custom op
   (`torch.ops.b12x.wo_projection_inv_rope_mxfp8_fused`) that allocates
   internally (capture-safe via the caching allocator). So the default path calls
   `run_inv_rope(o, positions, cos_sin_cache, weights, heads_per_group=8,
   nope_dim=448, rope_dim=64, expected_m=o.shape[0])` directly; no per-call bind.
5. Warmup (once per process per shape): runs token counts
   `{1..9,12,16,17,24,32,48,64,96,...,8192} ∪ cudagraph_capture_sizes ∪ {max_tokens}`
   (all <= max_tokens) so every dense-GEMM plan variant is JIT-compiled before
   the profile run / capture. 5.8 s cold, 0.4 s with `B12X_COMPILE_CACHE_DIR`
   populated (test dims, max_tokens=1024).
6. Forward: `out = state.run(o, positions, self.rotary_emb.cos_sin_cache)`
   -> `[tokens, hidden]` bf16, then exactly what `RowParallelLinear.forward`
   does today: `if self.wo_b.reduce_results and self.wo_b.tp_size > 1:
   out = tensor_model_parallel_all_reduce(out)`. No bias (`bias=False`).
   The sequence-parallel path (`model.py:201` sets `wo_b.reduce_results=False`)
   is honoured automatically. `o` may be the strided `o_padded[:, :n_local_heads]`
   view (tested).
7. Any init failure -> `logger.warning("DSV41 woproj: b12x path disabled for
   <prefix>: <reason>")` and that layer stays on the original path. Runtime
   errors are NOT swallowed (loud, not silent regressions).

Inverse-RoPE semantics match vLLM's `fused_inv_rope_fp8_quant` exactly: cache
`[max_pos, rope_dim]` = `cat(cos, sin)` fp32, interleaved pairs
(`even = x*cos + partner*sin`, `odd = x*cos - partner*sin`), `positions[token]`.

## What was verified (sp4, GB10, image `vllm-dsv41:overlay8`, live TP4 container sharing the GPU)

Fake TP4-rank layer: groups=2, group_width=4096 (8 heads x 512), rank=1024,
hidden=5120, wo_b K-slice 2048, random ModelOpt-style MXFP8 weights, random
`o`/positions, cos_sin_cache built like vLLM's rotary embedding. Reference
"today" = the real `deep_gemm_fp8_o_proj` bf16 branch on the emulation-dequantized
`wo_a` + `wo_b` via the real `FlashInferCutlassMxfp8LinearKernel` (what the
served TP4 uses). "exact" = fp32 torch, no activation quant.

Relative Frobenius error vs exact (max-abs-rel in the log is the same order):

| M | new (b12x) | today | new vs today |
|---|---|---|---|
| 1 | 3.79e-2 | 2.77e-2 | 4.00e-2 |
| 6 | 3.75e-2 | 2.66e-2 | 4.05e-2 |
| 12 | 3.76e-2 | 2.71e-2 | 4.04e-2 |
| 64 | 3.77e-2 | 2.67e-2 | 4.10e-2 |
| 512 | 3.77e-2 | 2.68e-2 | 4.10e-2 |

The extra ~1e-2 is the one additional e4m3 activation quantization the b12x path
performs (wo_a's input is MXFP8 instead of bf16); wo_b's input is MXFP8 in both.
Same order as today's path; not a free lunch - a served-row quality gate
(ppl / HE+ on the usual set) is still required.

Timing, us per call, 200 iters, CUDA-event timed (`graph` = `torch.cuda.graph`
replay, which is what decode runs under; eager includes ~80 us of b12x Python
validation that disappears under graphs):

| M | new eager | today eager | new graph | today graph | graph speedup |
|---|---|---|---|---|---|
| 1 | 113 | 261 | 31-43 | 260 | 6.1-8.5x |
| 6 | 113 | 189 | 29-42 | 246 | 5.9-8.9x |
| 12 | 162 | 231 | 44-49 | 234 | 4.8-5.5x |
| 48 | 162 | 215 | 46-50 | 217 | 4.4-4.7x |
| 64 | 161 | 153 | 45-63 | 147 | 2.3-3.1x |
| 128 | 163 | 165 | 78 | 156 | 2.0x |
| 512 | 247 | 385 | 238 | 375 | 1.6x |
| 1024 | 440 | 733 | 470 | 722 | 1.5x |

(Two runs; ranges = run-to-run jitter with the live container on the same GPU.)
Per layer per decode step the WO projection drops from ~250 us to ~30-45 us at
the MTP-k5 batch sizes (M=6/12/48). Eager M=64 is the one regime where the new
path is slower (161 vs 153 us) - Python overhead; irrelevant under graphs,
relevant only if `_o_proj` ever runs eager at mid M.

CUDA graph: captured at M=6 under `torch.cuda.graph`, replay == eager bit-exact,
in-place input mutation + replay tracks the buffers (rel err 3.8e-2 vs exact on
the new data). No allocation/JIT inside capture after warmup.
`bind_inv_rope` + `binding.run()` == direct call bit-exact (scratch spec 24.6 MiB
at max_tokens=1024, unused).

## Open risks before a served row

1. **Quality gate not run.** The extra activation quant on wo_a's input is a
   real numeric change; run the standard ppl/HE+ gate on a TP4 line with
   `DSV41_WOPROJ_B12X=1` vs the same line without.
2. **JIT compile at load.** Warmup covers the dense-GEMM plan variants for our
   dims up to `max_num_batched_tokens` (8192 on the current lines; more sizes
   than the test's 1024). Mount a persistent `B12X_COMPILE_CACHE_DIR` in the
   launch recipe or expect a one-off boot delay; watch the boot log for
   `DSV41 woproj: b12x MXFP8 wo_projection ACTIVE` (one line per process) and
   for any `b12x path disabled for ...` warnings.
3. **TP all-reduce** is mirrored, not shared with `RowParallelLinear` (no bias,
   `reduce_results` honoured). Not exercised at tp>1 in the unit test (tp_size=1
   fake); the first TP4 boot must compare outputs (e.g. a short greedy decode A/B).
4. **Memory.** Per attention layer per rank: wo_a fp8 stash 8 MiB + wo_b fp8
   10 MiB + scales <1 MiB + `scale_mma` copies <1 MiB. With the default CUTLASS
   wo_b kernel `layer.weight` IS the same fp8 storage (no extra bytes for wo_b);
   under the `B12XLIN` lever b12x replaces it, so the stash becomes the only
   copy (+10 MiB). The bf16 dequant copy of wo_a (16 MiB) is kept for the
   fallback path - reclaiming it once b12x is active is a follow-up. Warmup
   transiently allocates `o` for `max_tokens` (8192 x 16 x 512 bf16 = 128 MiB)
   plus kernel internals.
5. **torch.compile.** `_o_proj` runs after the eager attention break; the b12x
   entry is a registered custom op with a fake impl and the mixin's branch is a
   constant guard, so tracing should be fine, but it is untested under
   `VLLM_USE_V2_MODEL_RUNNER`/piecewise compile in a real boot.
6. **Idle-time stash on non-MXFP8 checkpoints**: hooks are no-ops (dtype
   checks) and the layer logs one warning and stays on the original path.
7. `cos_sin_cache` dtype: b12x's Triton quantizer promotes whatever it loads;
   vLLM's path asserts fp32 and that is what the DSV4 rope builds today.

## Credits

- Kernel: **Luke Alonso / b12x** - `b12x.gemm.wo_projection` (`_shared/wo_mxfp8.py`,
  fused inverse-RoPE MXFP8 quantizer + CuTe dense GEMMs), commit 6627d342 as
  shipped in the image.
- Idea: **MiaAI-Lab** - routing the DeepSeek-V4 WO projection through the b12x
  MXFP8 path under SGLang. Ideas only; their code is AGPL and none of it was
  read into or copied into this patch.
- vLLM upstream: `deep_gemm_fp8_o_proj` / `fused_inv_rope_fp8_quant`
  (Apache-2.0) are the reference path and were left untouched.
