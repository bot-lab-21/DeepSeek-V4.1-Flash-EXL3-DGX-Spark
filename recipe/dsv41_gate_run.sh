#!/bin/bash
# dsv41_gate_run.sh <profile> <row>... — run gate rows sequentially on the TP4 line: each row = boot (dsv41_boot_tp4.sh with the row's env)
# → Tony bench + smokes (inside the boot script) → teardown → next. Rows: G0 (base), ROCE (ROCE=1), ET64 (ENGRAM_THREADS=64), CH8 (NCCL_CH=8),
# ASYNC (--async-scheduling), or KEY=VAL[,KEY=VAL] literals. Results: results/<profile>-<row>/ ; summary table appended to gate_summary.md
# (aggregate C1/C4/C6, per-stream C1, prefill, KV tokens) with results/boot3 as the reference row. Log gate_run.log.
set -u; K=~/glm53-v4-quant/dsv41; LOG=$K/gate_run.log; log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
PROFILE=${1:?profile exl3|shipped}; shift; ROWS=${*:-"G0 ROCE"}
LINE=${LINE:-A}; CNAME=vllm_dsv41; declare -A N=( [0]=user@RANK_LAN_IP [1]=user@RANK_LAN_IP [2]=user@RANK_LAN_IP [3]=user@RANK_LAN_IP ); [ "$LINE" = B ] && { declare -A N=( [0]=user@RANK_LAN_IP [1]=user@RANK_LAN_IP [2]=user@RANK_LAN_IP [3]=user@RANK_LAN_IP ); CNAME=vllm_dsv41b; }
teardown(){ local d=$K/results/${exp:-teardown}; mkdir -p $d
  for r in 0 1 2 3; do timeout 120 ssh -o BatchMode=yes ${N[$r]} "docker logs --tail 400 $CNAME 2>&1 | grep -vE 'Warning|warn|Capturing'" > $d/rank$r.log 2>/dev/null; timeout 60 ssh -o BatchMode=yes ${N[$r]} "docker rm -f $CNAME >/dev/null 2>&1; pkill -f cache_flusher >/dev/null 2>&1; true"; done; }
row_env(){ case $1 in G0) echo "";; ROCE) echo "ROCE=1";; ET64) echo "ENGRAM_THREADS=64";; CH8) echo "NCCL_CH=8";; K7) echo "SPEC_K=7";; K4) echo "SPEC_K=4";; SERVE1M) echo "MAXLEN=1000000 SEQS=8";; HOT) echo "HOT_DIR=/mnt/glm52/dsv41engram/hotset HOT_ROWS=${HOT_ROWS:-20000000}";; HOTROCE) echo "ROCE=1 HOT_DIR=/mnt/glm52/dsv41engram/hotset HOT_ROWS=${HOT_ROWS:-20000000}";; ASYNC) echo "VLLM_EXTRA=--block-size_128_PLACEHOLDER";; *) echo "$1" | tr ',' ' ';; esac; }
summ(){ # $1 = results label → markdown row from results/<label>/bench-*.json (headline: level/agg_tok_s/per_stream_tok_s; prefill: prefill_tok_s)
  python3 - "$1" <<'PY'
import json,sys,os,glob
lab=sys.argv[1]; f=glob.glob(os.path.expanduser(f"~/glm53-v4-quant/dsv41/results/{lab}/bench-*.json"))
if not f: print(f"| {lab} | (no bench json) | | | | |"); sys.exit()
j=json.load(open(f[0])); h={x["level"]:x for x in j.get("headline",[])}
g=lambda c,k: f"{h[c][k]:.1f}" if c in h else "?"
pre=" / ".join(f"{p.get('prefill_tok_s',0):.0f}" for p in j.get("prefill",[]))
print(f"| {lab} | {g('C1','agg_tok_s')} | {g('C1','per_stream_tok_s')} | {g('C4','agg_tok_s')} | {g('C6','agg_tok_s')} | {pre} |")
PY
}
log "== gate run profile=$PROFILE rows=[$ROWS]"
[ -f $K/gate_summary.md ] || printf "| row | C1 agg | C1/stream | C4 agg | C6 agg | prefill 3K/12K/47K/93K |\n|---|---|---|---|---|---|\n" > $K/gate_summary.md
grep -q "| boot3 |" $K/gate_summary.md || summ boot3 >> $K/gate_summary.md
for row in $ROWS; do exp="${PROFILE}-${LABEL:-$row}${LINE:+-$LINE}"; env_kv=$(row_env $row)
  case $row in ASYNC) env_kv='VLLM_EXTRA=--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1 --async-scheduling';; esac
  teardown; log "-- row $row: $env_kv"
  ( cd $K && env LINE=$LINE PROFILE=$PROFILE EXP_NAME=$exp $env_kv bash dsv41_boot_tp4.sh ) 2>&1 | grep -E "DSV41-TP4|smoke:|warmup|toolcall|KV:|ABORT|NODE-PREP-(ALL|SOME)" | tee -a "$LOG"
  grep -q "DSV41-TP4-BENCH-DONE → .*$exp" $K/boot_tp4.log && { summ $exp >> $K/gate_summary.md; log "row $row done"; } || log "row $row FAILED (see boot_tp4.log)"
  case " ${QUALITY_ROWS:-G0} " in *" $row "*) if grep -q "DSV41-TP4-BENCH-DONE → .*$exp" $K/boot_tp4.log; then log "-- quality battery on row $row (line $LINE) while it is up"; base=http://LAN_IP:8000; [ "$LINE" = B ] && base=http://LAN_IP:8001; DSV41_BASE=$base timeout 5400 bash $K/dsv41_quality_gate.sh "$exp" 2>&1 | grep -E "=== |PASS|FAIL|pass@1|ppl|needle|tool|DONE|ABORT" | cut -c1-160 | tee -a "$LOG"; fi;; esac
  kv=$(grep -oE "Available KV cache memory: [0-9.]+ GiB" $K/boot_tp4.log | tail -n 1); log "row $row KV: ${kv:-?}"
  teardown
done
log "GATE-RUN-DONE"; cat $K/gate_summary.md | tee -a "$LOG"; ~/bin/tg-send "DSV41 TP4 gate rows done ($PROFILE: $ROWS). Table in gate_summary.md" >/dev/null 2>&1
