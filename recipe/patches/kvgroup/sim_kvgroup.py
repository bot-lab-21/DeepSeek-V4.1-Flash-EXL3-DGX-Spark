#!/usr/bin/env python3
"""CPU-only reproduction of vLLM's KV cache grouping/capacity for DeepSeek-V4.1-Flash TP4
(GB10 / SM12x, sm12x-pages patch geometry) and of the DSV41_KV_GROUPING=fine patch.

Run inside the serving image (no GPU needed), with the patched kv_cache_utils.py mounted:

  ssh user@RANK_LAN_IP 'docker run --rm -i \
     -v ~/kvgroup/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro \
     --entrypoint python3 vllm-dsv41:overlay8 -' < sim_kvgroup.py

It builds exactly the per-layer KVCacheSpec dict the model registers (4 compressed MLA caches on the
kv-source layers 2/8/14/20, their 4 indexer K caches, 43 SWA caches incl. the 3 DSpark draft layers) and
runs the real get_kv_cache_groups / get_kv_cache_config_from_groups / get_kv_cache_capacity.
Default mode must reproduce the logged "GPU KV cache size" values:
  300K, batched 16384, async off -> 2,561,073 ; 1M, 16384, off -> 3,411,140 ; 1M, 16384, on -> 2,990,822
"""
import os, sys, types
import torch
from vllm.v1.kv_cache_interface import (MLAAttentionSpec, SlidingWindowMLASpec,
                                        UniformTypeKVCacheSpecs, get_kv_quant_mode)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.core import kv_cache_utils as U
from vllm.utils.math_utils import cdiv

MEM = 12884901888  # KV_BYTES pin, 12 GiB per rank
CR = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
KV_SRC = [2, 8, 14, 20]
LOGGED = {(300000, 16384, False): 2561073, (1000000, 16384, False): 3411140, (1000000, 16384, True): 2990822}


def specs(swa_bs=64, n_draft=3, draft_win=192):
    """Live geometry (sm12x-pages patch): SWA 64-token pages, compressed/indexer 64*ratio tokens (64 states)."""
    d, q = {}, get_kv_quant_mode("fp8_ds_mla")
    for i in range(40 + n_draft):
        cr, p = CR[i], f"model.layers.{i}.self_attn"
        if i < 40 and i in KV_SRC:
            d[p] = MLAAttentionSpec(block_size=64 * cr, num_kv_heads=1, head_size=512, dtype=torch.uint8,
                                    tokens_per_state=cr, cache_dtype_str="fp8_ds_mla", alignment=576,
                                    model_version="deepseek_v4", kv_quant_mode=q, state_content_bytes=584)
            d[p + ".indexer.k_cache"] = MLAAttentionSpec(block_size=64 * cr, num_kv_heads=1, head_size=132,
                                                         dtype=torch.uint8, tokens_per_state=cr, alignment=576)
        d[p + ".swa_cache"] = SlidingWindowMLASpec(block_size=swa_bs, num_kv_heads=1, head_size=512,
                                                   dtype=torch.uint8, sliding_window=(draft_win if i >= 40 else 128),
                                                   cache_dtype_str="fp8_ds_mla", state_content_bytes=584,
                                                   alignment=576, model_version="deepseek_v4", kv_quant_mode=q)
    return d


def cfg(L, mnbt, async_on):
    ns = types.SimpleNamespace
    T = (2 if async_on else 1) * mnbt  # VllmConfig.max_in_flight_tokens (config/vllm.py:577-598)
    return ns(scheduler_config=ns(disable_hybrid_kv_cache_manager=False, async_scheduling=async_on,
                                  max_num_batched_tokens=mnbt),
              model_config=ns(max_model_len=L, original_max_model_len=L, hf_config=ns(model_type="deepseek_v41")),
              parallel_config=ns(decode_context_parallel_size=1, pipeline_parallel_size=1),
              max_in_flight_tokens=T,
              cache_config=ns(block_size=128, num_gpu_blocks_override=None, prefix_cache_retention_interval=None,
                              cache_dtype="fp8_ds_mla", get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC),
              speculative_config=ns(method="dspark", num_speculative_tokens=5, use_eagle=lambda: True,
                                    use_eagle_block_drop=lambda: True, use_multi_module_mtp=lambda: False))


def run(mode, L, mnbt, async_on):
    os.environ["DSV41_KV_GROUPING"] = "fine" if mode == "fine" else ""
    c = cfg(L, mnbt, async_on)
    groups = U.get_kv_cache_groups(c, specs())
    bpb = U._get_kv_cache_bytes_per_block(groups)
    kvc = U.get_kv_cache_config_from_groups(c, groups, MEM)
    tokens, conc = U.get_kv_cache_capacity(c, kvc)
    per = [cdiv(g.kv_cache_spec.max_memory_usage_bytes(c), g.kv_cache_spec.page_size_bytes) for g in groups]
    used = sum(p * g.kv_cache_spec.page_size_bytes for p, g in zip(per, groups))
    charged = sum(per) * bpb
    exp = LOGGED.get((L, mnbt, async_on)) if mode == "default" else None
    tag = "" if exp is None else ("  == logged OK" if exp == tokens else f"  !! logged {exp:,} MISMATCH")
    print(f"{mode:7s} L={L:>9,} batched={mnbt:5d} async={'on ' if async_on else 'off'} T={c.max_in_flight_tokens:5d} | "
          f"groups={len(groups):2d} bytes/block={bpb:6d} blocks={kvc.num_blocks:6d} per_req_blocks={sum(per):6d} "
          f"conc={conc:7.3f} -> tokens {tokens:>9,} | pool util {used / charged * 100:5.1f}% "
          f"({charged / 2**20:6.0f} MiB charged / {used / 2**20:6.0f} MiB used per req){tag}")
    return tokens, exp


def main():
    ok = True
    res = {}
    for mode in ("default", "fine"):
        for L in (300000, 1000000):
            for mnbt in (8192, 16384):
                for async_on in (False, True):
                    t, exp = run(mode, L, mnbt, async_on)
                    res[(mode, L, mnbt, async_on)] = t
                    if exp is not None and exp != t:
                        ok = False
        print()
    print("gain fine/default:")
    for L in (300000, 1000000):
        for mnbt in (8192, 16384):
            for a in (False, True):
                d, f = res[("default", L, mnbt, a)], res[("fine", L, mnbt, a)]
                print(f"  L={L:>9,} batched={mnbt:5d} async={'on ' if a else 'off'}: {d:>9,} -> {f:>9,}  ({(f / d - 1) * 100:+.0f}%)")
    print("\nDEFAULT-MODE LOGGED VALUES:", "ALL REPRODUCED" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
