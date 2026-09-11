#!/usr/bin/env python3
"""fill_card.py <exp> [--apply] — fill the served TBD cells of the HF card and the GitHub README from a gate row's bench json.

Reads results/<exp>/bench-*.json (headline C1/C4/C6, prefill) and the "Available KV cache memory" line of that boot in boot_tp4.log.
Prints the substitutions; writes the files only with --apply. Quality cells (ppl probe, HumanEval+/MBPP+) stay TBD — fill by hand
from the quality battery output. KV tokens are scaled from the shipped baseline (6.99 GiB/rank = 1.16 M tokens, fp8 DS-MLA).
"""
import glob, json, re, sys, os

K = os.path.expanduser("~/glm53-v4-quant/dsv41")
exp = sys.argv[1]
apply = "--apply" in sys.argv
f = glob.glob(f"{K}/results/{exp}/bench-*.json")
if not f:
    sys.exit(f"no bench json for {exp}")
j = json.load(open(f[0]))
h = {x["level"]: x for x in j["headline"]}
pre = " / ".join(f"{p['prefill_tok_s']:.0f}" for p in j["prefill"])

kv = None
log = open(f"{K}/boot_tp4.log").read()
sec = log.split(f"== TP4 boot {exp} ")
if len(sec) > 1:
    m = re.findall(r"Available KV cache memory: ([0-9.]+) GiB", sec[-1])
    if m:
        kv = float(m[-1])
    tok = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", sec[-1])
    pin = re.findall(r"reserved ([0-9.]+) GiB memory for KV Cache", sec[-1])
kv_cell = "TBD"
if tok:
    ntok = int(tok[-1].replace(",", "")) / 1e6
    kv_cell = (f"{float(pin[-1]):.0f} GiB/rank pinned = {ntok:.2f} M tokens" if pin
               else (f"{kv:.2f} GiB/rank = {ntok:.2f} M tokens" if kv else f"{ntok:.2f} M tokens"))
elif kv:
    kv_cell = f"{kv:.2f} GiB/rank = {kv * 1.16 / 6.99:.2f} M tokens"

rows = {
    "| single stream, aggregate / per-stream tok/s |": f"{h['C1']['agg_tok_s']:.1f} / {h['C1']['per_stream_tok_s']:.1f}",
    "| 4 streams aggregate tok/s |": f"{h['C4']['agg_tok_s']:.1f}",
    "| 6 streams aggregate tok/s |": f"{h['C6']['agg_tok_s']:.1f}",
    "| cold prefill 3K / 12K / 47K / 93K tok/s |": pre,
    "| KV capacity at gmu 0.80 |": kv_cell,
}
for path in [f"{K}/release/HF_MODEL_CARD_dsv41_exl3.md", f"{K}/release/GITHUB_README.md"]:
    s = open(path).read(); out = []
    for line in s.split("\n"):
        for key, val in rows.items():
            if line.startswith(key) and line.rstrip().endswith("| TBD |"):
                line = line.rstrip()[: -len("TBD |")] + f"{val} |"
                print(f"{os.path.basename(path)}: {key} -> {val}")
        out.append(line)
    if apply:
        open(path, "w").write("\n".join(out))
print("applied" if apply else "dry run (add --apply to write)")
