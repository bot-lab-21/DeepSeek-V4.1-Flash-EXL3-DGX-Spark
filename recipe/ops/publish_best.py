#!/usr/bin/env python3
"""publish_best.py — rewrite the 'Best serving configuration so far' block in the HF card and the GitHub README from the ladders.

Best = the bench row with the highest 6-stream aggregate among rows that are either a measured base (G0) or an ACCEPTED rung
(ops/ladder-<LINE>.md), across lines A and B. Its configuration = that line's accepted base file + the row's lever.
Also emits the per-rung table (both lines). Idempotent; the block sits between BEST-SERVING markers. Prints a one-line summary.
"""
import glob, json, os, re, datetime

K = os.path.expanduser("~/glm53-v4-quant/dsv41")
PIN = {"KV_BYTES": "KV cache pinned", "GMU": "gpu-memory-utilization", "MEM_MIN_GIB": "launcher MemFree floor (GiB)",
       "NCCL_CH": "NCCL channels", "ROCE": "b12x RoCE one-shot all-reduce", "SPEC_K": "DSpark k", "ASYNC": "async scheduling",
       "HOT_DIR": "resident Engram hot rows dir", "HOT_ROWS": "resident Engram hot rows", "ENGRAM_THREADS": "Engram disk threads",
       "MAXLEN": "max context", "SEQS": "max concurrent seqs"}


def bench(label):
    f = glob.glob(f"{K}/results/{label}/bench-*.json")
    if not f:
        return None
    j = json.load(open(f[0])); h = {x["level"]: x for x in j["headline"]}
    return {"c1": h["C1"]["agg_tok_s"], "c1s": h["C1"]["per_stream_tok_s"], "c4": h["C4"]["agg_tok_s"], "c6": h["C6"]["agg_tok_s"],
            "ttft": h["C1"]["ttft_mean_s"], "pre": [p["prefill_tok_s"] for p in j["prefill"]], "when": j.get("finished", "")}


def accepted(line):
    p = f"{K}/ops/accepted-{line}.env"
    return [l.strip() for l in open(p)] if os.path.exists(p) else []


rows = []
for line in "AB":
    # measured base(s): the newest G0 row for the line
    for d in sorted(glob.glob(f"{K}/results/exl3-G0-{line}*")):
        b = bench(os.path.basename(d))
        if b: rows.append({"line": line, "label": os.path.basename(d), "lever": "base", "verdict": "base", **b})
    md = f"{K}/ops/ladder-{line}.md"
    if os.path.exists(md):
        for l in open(md):
            m = re.match(r"\| (\w+) \| ([^|]*) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| (ACCEPT[^|]*) \|", l)
            if m:
                lab = f"exl3-{m.group(1)}-{line}"; b = bench(lab)
                if b: rows.append({"line": line, "label": lab, "lever": m.group(2).strip(), "verdict": "ACCEPT", **b})
rows = [r for r in rows if not r["label"].endswith("kvfree")]  # the overcommitted first boot is not a servable config
best = max(rows, key=lambda r: r["c6"]) if rows else None

def env_of(r):
    acc = [x for x in accepted(r["line"]) if x and not x.startswith("#")]
    lever = [] if r["lever"] == "base" else r["lever"].split()
    seen = {}
    for kv in acc + lever:
        k, _, v = kv.partition("="); seen[k] = v
    return seen

def fmt_env(e):
    return ", ".join(f"{PIN.get(k, k)} = {v}" for k, v in e.items() if v)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
if best:
    e = env_of(best)
    block = [f"<!-- BEST-SERVING-START -->",
             f"## Best serving configuration so far (auto-updated {stamp} Pacific)",
             f"Line {best['line']}, row `{best['label']}` ({'measured base' if best['verdict']=='base' else 'accepted rung: ' + best['lever']}). Settings on top of the recipe defaults: {fmt_env(e)}.",
             "",
             "| | best so far |", "|---|---|",
             f"| single stream, aggregate / per-stream tok/s | {best['c1']:.1f} / {best['c1s']:.1f} |",
             f"| 4 streams aggregate tok/s | {best['c4']:.1f} |",
             f"| 6 streams aggregate tok/s | {best['c6']:.1f} |",
             f"| mean TTFT at C1 | {best['ttft']:.2f} s |",
             f"| cold prefill 3K / 12K / 47K / 93K tok/s | {' / '.join(f'{p:.0f}' for p in best['pre'])} |",
             "", "Tuning ladder (accept = +3 % single-stream or 6-stream aggregate with ≤5 % prefill loss at 47K and smokes passing; levers stack per line, winners cross-applied):", "",
             "| line | row | lever | C1/stream | C6 agg | prefill@47K | verdict |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["line"], r["when"])):
        block.append(f"| {r['line']} | {r['label']} | {r['lever']} | {r['c1s']:.1f} | {r['c6']:.1f} | {r['pre'][2]:.0f} | {r['verdict']} |")
    # rejected rungs too (from the ladder tables), for the record
    for line in "AB":
        md = f"{K}/ops/ladder-{line}.md"
        if os.path.exists(md):
            for l in open(md):
                m = re.match(r"\| (\w+) \| ([^|]*) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| (reject[^|]*) \|", l)
                if m: block.append(f"| {line} | exl3-{m.group(1)}-{line} | {m.group(2).strip()} | {m.group(3)} | {m.group(4)} | {m.group(5)} | reject |")
    block.append("<!-- BEST-SERVING-END -->")
    text = "\n".join(block)
    for p in [f"{K}/release/HF_MODEL_CARD_dsv41_exl3.md", f"{K}/release/GITHUB_README.md"]:
        s = open(p).read()
        if "<!-- BEST-SERVING-START -->" in s:
            s = re.sub(r"<!-- BEST-SERVING-START -->.*?<!-- BEST-SERVING-END -->", lambda _: text, s, flags=re.S)
        else:
            anchor = "## Measured" if "## Measured" in s else "## Results"
            s = s.replace(anchor, text + "\n\n" + anchor, 1)
        open(p, "w").write(s)
    print(f"BEST {best['line']} {best['label']} C1/stream {best['c1s']:.1f} C6 {best['c6']:.1f} prefill47K {best['pre'][2]:.0f} | rows {len(rows)}")
else:
    print("no rows")
