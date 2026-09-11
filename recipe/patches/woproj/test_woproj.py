#!/usr/bin/env python3
"""In-image GPU unit test for the b12x MXFP8 WO-projection patch.

Builds a fake DeepSeek-V4.1 attention layer (one TP4 rank: 2 local o-groups of
8 heads x 512, o_lora_rank 1024, hidden 5120, wo_b K-slice 2048) with random
MXFP8 weights, runs the "loader" (stash hooks + the real vLLM MXFP8 kernels'
process_weights_after_loading) and compares

  new   = patched _o_proj -> b12x wo_projection (inverse-RoPE fused, MXFP8 A/B)
  today = original path: fused_inv_rope (bf16) -> bf16 bmm(wo_a dequant) ->
          wo_b via FlashInferCutlassMxfp8LinearKernel (what the served TP4 uses)
  exact = fp32 torch reference (no activation quant)

for M in {1, 6, 12, 64, 512}, then times new/today at M=1/6/64 and checks
CUDA-graph capture + replay of the new path at M=6.

Usage (inside vllm-dsv41:overlay8, GPU):
  python3 test_woproj.py --patch-dir /patch
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import types

import torch
import torch.nn as nn

os.environ.setdefault("DSV41_WOPROJ_B12X", "1")


def load_as(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mxfp8_quantize(x: torch.Tensor):
    """Realistic ModelOpt-style MXFP8: fp8 e4m3 values + per-32 UE8M0 u8 scale."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )

    vals, scales = _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout=False)
    return vals.contiguous(), scales.contiguous()


def dequant_fp32(w_fp8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    return (
        w_fp8.float().view(n, k // 32, 32) * torch.exp2(s_u8.float() - 127.0)[..., None]
    ).view(n, k)


def build_cos_sin_cache(max_pos: int, rope_dim: int, theta: float, device):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()  # [max_pos, rope_dim]


def inv_rope_fp32(o, positions, cos_sin, nope_dim, rope_dim):
    x = o.float()
    cs = cos_sin[positions]  # [M, rope_dim]
    half = rope_dim // 2
    cos = cs[:, :half][:, None, :]
    sin = cs[:, half:][:, None, :]
    r = x[..., nope_dim:]
    r_even, r_odd = r[..., 0::2], r[..., 1::2]
    out = x.clone()
    out[..., nope_dim::2] = r_even * cos + r_odd * sin
    out[..., nope_dim + 1 :: 2] = r_odd * cos - r_even * sin
    return out


def exact_ref(o, positions, cos_sin, wa_deq, wb_deq, groups, nope_dim, rope_dim):
    m = o.shape[0]
    x = inv_rope_fp32(o, positions, cos_sin, nope_dim, rope_dim).view(m, groups, -1)
    z = torch.einsum("tgd,grd->tgr", x, wa_deq)  # [M, G, R]
    return z.reshape(m, -1) @ wb_deq.t()  # [M, H]


def rel_fro(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def max_abs_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def cuda_time(fn, iters=200, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0  # us


class FakeWoA(nn.Module):
    def __init__(self, w_fp8, s_u8, groups):
        super().__init__()
        self.weight = nn.Parameter(w_fp8, requires_grad=False)
        self.weight_scale = nn.Parameter(s_u8, requires_grad=False)
        self.is_bmm = True
        self.bmm_batch_size = groups
        self.weight_block_size = [1, 32]


class FakeWoB(nn.Module):
    """RowParallelLinear stand-in (tp_size=1): kernel.apply_weights, no bias."""

    def __init__(self, w_fp8, s_u8):
        super().__init__()
        self.weight = nn.Parameter(w_fp8, requires_grad=False)
        self.weight_scale = nn.Parameter(s_u8, requires_grad=False)
        self.reduce_results = True
        self.tp_size = 1
        self.kernel = None
        self.fallback_deq = None  # torch emulation if cutlass unavailable

    def forward(self, x):
        if self.kernel is not None:
            return self.kernel.apply_weights(self, x)
        # torch emulation of a dynamic MXFP8 A x static MXFP8 B GEMM
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            _mxfp8_e4m3_quantize_torch,
        )

        xq, xs = _mxfp8_e4m3_quantize_torch(x, False)
        return (dequant_fp32(xq, xs) @ self.fallback_deq.t()).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024, help="plan/warmup cap for the test")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    print(f"device: {torch.cuda.get_device_name(dev)} torch {torch.__version__}")

    # --- mount-equivalent: patched emulation replaces the installed module ---
    # Import the parent package first (its __init__ imports emulation), then
    # swap the patched file in under the same module name.
    import vllm.model_executor.kernels.linear  # noqa: F401
    emu = load_as(
        "vllm.model_executor.kernels.linear.mxfp8.emulation",
        os.path.join(args.patch_dir, "emulation.py"),
    )
    patch = load_as("dsv41_woproj_patch", os.path.join(args.patch_dir, "flashinfer_sparse_woproj.py"))
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        compute_fp8_einsum_recipe,
        deep_gemm_fp8_o_proj,
    )

    import b12x.gemm.wo_projection as wo

    print("b12x wo_projection.is_supported():", wo.is_supported())

    # --- dims: one TP4 rank of DeepSeek-V4.1-Flash ---
    G, HPG, NOPE, ROPE = 2, 8, 448, 64
    HEAD = NOPE + ROPE
    D = HPG * HEAD  # 4096 group_width
    R = 1024  # o_lora_rank
    H = 5120  # hidden
    KB = G * R  # wo_b local K slice = 2048
    HEADS = G * HPG  # 16 local heads
    MAX_POS = 8192

    wa_bf16 = torch.randn(G * R, D, device=dev, dtype=torch.bfloat16) * 0.02
    wb_bf16 = torch.randn(H, KB, device=dev, dtype=torch.bfloat16) * 0.02
    wa_fp8, wa_s = mxfp8_quantize(wa_bf16)
    wb_fp8, wb_s = mxfp8_quantize(wb_bf16)
    assert wa_fp8.dtype == torch.float8_e4m3fn and wa_s.dtype == torch.uint8
    assert wa_s.shape == (G * R, D // 32) and wb_s.shape == (H, KB // 32)
    wa_deq = dequant_fp32(wa_fp8, wa_s).view(G, R, D)
    wb_deq = dequant_fp32(wb_fp8, wb_s)
    cos_sin = build_cos_sin_cache(MAX_POS, ROPE, 10000.0, dev)

    # --- fake attention layer using the patch mixin ---
    wo_a = FakeWoA(wa_fp8.clone(), wa_s.clone(), G).to(dev)
    wo_b = FakeWoB(wb_fp8.clone(), wb_s.clone()).to(dev)
    emu_kernel = emu.EmulationMxfp8LinearKernel(Mxfp8LinearLayerConfig(bmm_batch_size=G))
    wo_a.quant_method = types.SimpleNamespace(
        process_weights_after_loading=emu_kernel.process_weights_after_loading
    )
    wob_kernel_name = "FlashInferCutlassMxfp8LinearKernel"
    try:
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
            FlashInferCutlassMxfp8LinearKernel,
        )

        ok, why = FlashInferCutlassMxfp8LinearKernel.is_supported()
        if not ok:
            raise RuntimeError(why)
        wo_b.kernel = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())
        wo_b.quant_method = types.SimpleNamespace(
            process_weights_after_loading=wo_b.kernel.process_weights_after_loading
        )
    except Exception as exc:  # noqa: BLE001
        wob_kernel_name = f"torch-emulation (cutlass unavailable: {exc})"
        wo_b.fallback_deq = wb_deq

        def _noop(layer):
            return None

        wo_b.quant_method = types.SimpleNamespace(process_weights_after_loading=_noop)
    print("wo_b 'today' kernel:", wob_kernel_name)

    class FakeAttn(patch._B12xWoProjMixin):
        pass

    attn = FakeAttn()
    attn.prefix = "model.layers.0.attn"
    attn.wo_a, attn.wo_b = wo_a, wo_b
    attn.n_local_groups, attn.o_lora_rank, attn.n_local_heads = G, R, HEADS
    attn.nope_head_dim, attn.rope_head_dim, attn.hidden_size = NOPE, ROPE, H
    attn.max_num_batched_tokens = args.max_tokens
    attn.rotary_emb = types.SimpleNamespace(cos_sin_cache=cos_sin)
    attn._einsum_recipe, attn._tma_aligned_scales = compute_fp8_einsum_recipe(128)

    def _o_proj_reference(o, positions):
        return deep_gemm_fp8_o_proj(
            o, positions, attn.rotary_emb.cos_sin_cache, attn.wo_a, attn.wo_b,
            n_groups=G, heads_per_group=HPG, nope_dim=NOPE, rope_dim=ROPE, o_lora_rank=R,
            einsum_recipe=attn._einsum_recipe, tma_aligned_scales=attn._tma_aligned_scales,
        )

    attn._o_proj_reference = _o_proj_reference
    fake_cfg = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64]),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=args.max_tokens),
    )
    attn._woproj_setup(fake_cfg)

    # --- "loader": process_weights_after_loading in module order (wo_a, wo_b) ---
    t0 = time.time()
    wo_a.quant_method.process_weights_after_loading(wo_a)
    assert wo_a.weight.dtype == torch.bfloat16, "emulation should dequant wo_a to bf16"
    assert getattr(wo_a, "mxfp8_weight_fp8", None) is not None, "wo_a stash missing"
    assert wo_a.mxfp8_weight_fp8.dtype == torch.float8_e4m3fn
    assert torch.equal(wo_a.mxfp8_weight_fp8, wa_fp8) and torch.equal(wo_a.mxfp8_weight_scale_u8, wa_s)
    wo_b.quant_method.process_weights_after_loading(wo_b)  # -> stash + kernel + pack + warmup
    torch.cuda.synchronize()
    print(f"load-time hooks + pack + warmup: {time.time() - t0:.1f}s")
    assert getattr(wo_b, "mxfp8_weight_fp8", None) is not None, "wo_b stash missing"
    assert torch.equal(wo_b.mxfp8_weight_fp8, wb_fp8) and torch.equal(wo_b.mxfp8_weight_scale_u8, wb_s)
    if attn._woproj_state is None:
        print("FAIL: b12x path not active:", attn._woproj_disabled_reason)
        sys.exit(1)
    state = attn._woproj_state
    print(f"b12x state: groups={state.groups} width={state.group_width} rank={state.rank} "
          f"hidden={state.hidden} max_tokens={state.max_tokens} bind={state.use_bind}")

    # --- layout sanity: b12x's own dequant of our MXFP8Rows == checkpoint dequant ---
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

    wa_b12x = dequantize_mxfp8_rows_torch(state.weights.wo_a.values, state.weights.wo_a.scale_rows)
    # [rank, group_width, groups] -> [groups, rank, group_width]
    assert torch.equal(wa_b12x.permute(2, 0, 1), wa_deq), "wo_a b12x layout mismatch"
    wb_b12x = dequantize_mxfp8_rows_torch(state.weights.wo_b.values, state.weights.wo_b.scale_rows)
    assert torch.equal(wb_b12x, wb_deq), "wo_b b12x layout mismatch"
    print("layout sanity: b12x dequant(wo_a/wo_b) == checkpoint dequant  OK")

    # --- correctness ---
    print("\nM     | new vs exact (fro, maxabs) | today vs exact (fro, maxabs) | new vs today (fro) | strided-o new vs exact")
    worst = 0.0
    for M in (1, 6, 12, 64, 512):
        o = torch.randn(M, HEADS, HEAD, device=dev, dtype=torch.bfloat16)
        pos = torch.randint(0, MAX_POS, (M,), device=dev, dtype=torch.int64)
        exact = exact_ref(o, pos, cos_sin, wa_deq, wb_deq, G, NOPE, ROPE)
        new = attn._o_proj(o, pos)
        today = attn._o_proj_reference(o, pos)
        assert new.shape == (M, H) and new.dtype == torch.bfloat16, (new.shape, new.dtype)
        assert torch.isfinite(new).all()
        # real forward passes o = o_padded[:, :n_local_heads, :] (strided over heads)
        o_pad = torch.zeros(M, 32, HEAD, device=dev, dtype=torch.bfloat16)
        o_pad[:, :HEADS] = o
        new_strided = attn._o_proj(o_pad[:, :HEADS, :], pos)
        e_new, e_today = rel_fro(new, exact), rel_fro(today, exact)
        worst = max(worst, e_new)
        print(f"{M:<5d} | {e_new:.3e}  {max_abs_rel(new, exact):.3e}      | "
              f"{e_today:.3e}  {max_abs_rel(today, exact):.3e}        | "
              f"{rel_fro(new, today):.3e}          | {rel_fro(new_strided, exact):.3e}")
    # both paths quantize activations to e4m3 (per-32 MXFP8 / bf16+MXFP8): expect ~1e-2
    assert worst < 5e-2, f"new path error too large: {worst}"

    # --- bind path (plan.bind_inv_rope + binding.run) == direct call ---
    M = 6
    o6 = torch.randn(M, HEADS, HEAD, device=dev, dtype=torch.bfloat16)
    p6 = torch.randint(0, MAX_POS, (M,), device=dev, dtype=torch.int64)
    direct = state.run(o6, p6, cos_sin)
    spec = state.plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = state.plan.bind_inv_rope(
        scratch=scratch, o=o6, positions=p6, cos_sin_cache=cos_sin, weights=state.weights,
        heads_per_group=HPG, nope_dim=NOPE, rope_dim=ROPE, expected_m=M,
    )
    bound = wo.run_inv_rope(binding=binding)
    print(f"\nbind path vs direct (M=6): rel_fro={rel_fro(bound, direct):.3e} "
          f"(scratch spec {spec.shape[0] / 2**20:.1f} MiB, unused by run_inv_rope)")

    # --- CUDA graph capture of the new path at M=6 ---
    exact6 = exact_ref(o6, p6, cos_sin, wa_deq, wb_deq, G, NOPE, ROPE)
    eager6 = attn._o_proj(o6, p6)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            attn._o_proj(o6, p6)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = attn._o_proj(o6, p6)
    g.replay()
    torch.cuda.synchronize()
    print(f"cuda-graph M=6: replay vs eager rel_fro={rel_fro(out_g, eager6):.3e}, vs exact={rel_fro(out_g, exact6):.3e}")
    # mutate inputs in place -> replay must follow the buffers
    o6.copy_(torch.randn_like(o6))
    p6.copy_(torch.randint(0, MAX_POS, (M,), device=dev, dtype=torch.int64))
    g.replay()
    torch.cuda.synchronize()
    exact6b = exact_ref(o6, p6, cos_sin, wa_deq, wb_deq, G, NOPE, ROPE)
    e_replay = rel_fro(out_g, exact6b)
    print(f"cuda-graph M=6 after in-place input change: replay vs exact rel_fro={e_replay:.3e}")
    assert e_replay < 5e-2, "graph replay did not track input buffers"

    # --- timing ---
    def graph_of(fn):
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):
            fn()
        return gg

    print(f"\ntiming ({args.iters} iters, us/call, CUDA-event timed; graph = torch.cuda.graph replay)")
    print("M     | new eager | today eager | new graph | today graph | graph speedup (today/new)")
    for M in (1, 6, 12, 24, 48, 64, 128, 512, 1024):
        o = torch.randn(M, HEADS, HEAD, device=dev, dtype=torch.bfloat16)
        pos = torch.randint(0, MAX_POS, (M,), device=dev, dtype=torch.int64)
        t_new = cuda_time(lambda: attn._o_proj(o, pos), args.iters)
        t_today = cuda_time(lambda: attn._o_proj_reference(o, pos), args.iters)
        g_new = graph_of(lambda: attn._o_proj(o, pos))
        g_today = graph_of(lambda: attn._o_proj_reference(o, pos))
        tg_new = cuda_time(g_new.replay, args.iters)
        tg_today = cuda_time(g_today.replay, args.iters)
        print(f"{M:<5d} | {t_new:9.1f} | {t_today:11.1f} | {tg_new:9.1f} | {tg_today:11.1f} | {tg_today / tg_new:5.2f}x")

    print("\nPASS")


if __name__ == "__main__":
    main()
