"""In-image GPU smoke for P7 (uneven 128-aligned TP split of the EXL3 expert intermediate dim).

For tp=4 and I_total=2304 (DeepSeek-V4.1-Flash): expect widths [512, 640, 640, 512] and offsets [0, 512, 1152, 1792].
Checks: (1) the split math; (2) _place slices the full-width checkpoint tensors at the rank offset (values are recognizable);
(3) the four rank slices tile the full dim exactly; (4) exl3_moe_gemm accepts [I, I] shards of 640 and 512 (random trellis data,
values not checked -- the kernel's TORCH_CHECK on shard width is what failed in production).
"""
import torch
from types import SimpleNamespace
from cuda_exl3 import moe as M

TILE = M.TILE
E, H, I_TOTAL, TP = 2, 512, 2304, 4
BITS = 3


def make_method(rank):
    info = SimpleNamespace(bits=BITS, cb=2)
    moe = SimpleNamespace(tp_size=TP, tp_rank=rank)
    m = M.Exl3MoEMethod.__new__(M.Exl3MoEMethod)
    m.moe = moe; m.quant_config = None; m.prefix = f"test.rank{rank}"; m.swiglu_limit = 10.0
    m.w13_bits = BITS; m.w2_bits = BITS; m.cb = 2; m.cb_name = "mul1"
    return m


widths, offs = [], []
for r in range(TP):
    m = make_method(r)
    I, off = m._tp_split(I_TOTAL // TP)
    widths.append(I); offs.append(off)
print("widths", widths, "offsets", offs)
assert widths == [512, 640, 640, 512], widths
assert offs == [0, 512, 1152, 1792], offs
assert sum(widths) == I_TOTAL

# even case must be untouched (GLM-5.3: 1536/4)
m0 = make_method(2)
assert m0._tp_split(384) == (384, 768)

# full-width checkpoint tensors with recognizable values: column index encoded
gate_trellis = torch.arange(I_TOTAL // TILE, dtype=torch.int16).view(1, -1, 1).expand(H // TILE, I_TOTAL // TILE, TILE * BITS).contiguous()
down_trellis = torch.arange(I_TOTAL // TILE, dtype=torch.int16).view(-1, 1, 1).expand(I_TOTAL // TILE, H // TILE, TILE * BITS).contiguous()
gate_svh = torch.arange(I_TOTAL, dtype=torch.half)
down_suh = torch.arange(I_TOTAL, dtype=torch.half)

for r in range(TP):
    m = make_method(r)
    layer = torch.nn.Module()
    m.create_weights(layer, E, H, I_TOTAL // TP, torch.half)
    I, off = layer.exl3_inter, layer.exl3_inter_off
    assert layer.w13_trellis.shape == (E, H // TILE, 2 * I // TILE, TILE * BITS), layer.w13_trellis.shape
    assert layer.w2_trellis.shape == (E, I // TILE, H // TILE, TILE * BITS)
    m._place(layer, 0, "gate_proj", "trellis", gate_trellis.cuda())
    m._place(layer, 0, "up_proj", "trellis", gate_trellis.cuda())
    m._place(layer, 0, "down_proj", "trellis", down_trellis.cuda())
    m._place(layer, 0, "gate_proj", "svh", gate_svh.cuda())
    m._place(layer, 0, "up_proj", "svh", gate_svh.cuda())
    m._place(layer, 0, "down_proj", "suh", down_suh.cuda())
    g = layer.w13_trellis.data[0][0, : I // TILE, 0].cpu()          # gate half, tile column ids
    u = layer.w13_trellis.data[0][0, I // TILE :, 0].cpu()          # up half
    d = layer.w2_trellis.data[0][:, 0, 0].cpu()
    exp = torch.arange(off // TILE, off // TILE + I // TILE, dtype=torch.int16)
    assert torch.equal(g, exp) and torch.equal(u, exp) and torch.equal(d, exp), (r, g[:3], exp[:3])
    assert torch.equal(layer.w13_svh.data[0][:I].cpu(), torch.arange(off, off + I, dtype=torch.half))
    assert torch.equal(layer.w2_suh.data[0][0].cpu(), torch.arange(off, off + I, dtype=torch.half))
    print(f"rank {r}: I={I} off={off} slices OK")

# kernel accepts 640- and 512-wide shards (random data; shape/TORCH_CHECK smoke only)
ops = torch.ops.cuda_exl3_C
for I in (640, 512):
    rows, T, Mtok = 64, 6, 16
    block_m = 16
    trellis = torch.randint(-32768, 32767, (E, H // TILE, 2 * I // TILE, TILE * BITS), dtype=torch.int16, device="cuda")
    suh = torch.randn(E, 2, H, dtype=torch.half, device="cuda")
    svh = torch.randn(E, 2 * I, dtype=torch.half, device="cuda")
    a13 = torch.randn(2, rows, H, dtype=torch.half, device="cuda")
    expert_ids = torch.randint(0, E, (rows // block_m,), dtype=torch.int32, device="cuda")
    n_rows = torch.tensor([rows], dtype=torch.int32, device="cuda")
    out = ops.exl3_moe_gemm(a13, trellis, suh, svh, expert_ids, n_rows, [I, I], 2, block_m, torch.half, None, None, Mtok, T)
    torch.cuda.synchronize()
    assert out.shape[-1] == 2 * I, out.shape
    print(f"exl3_moe_gemm accepted shards [{I}, {I}] -> {tuple(out.shape)}")
print("P7-UNEVEN-SPLIT-OK")
