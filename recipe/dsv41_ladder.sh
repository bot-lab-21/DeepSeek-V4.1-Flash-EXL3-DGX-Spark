#!/bin/bash
# dsv41_ladder.sh <LINE> <lever>... — cumulative tuning ladder on one TP4 line. Each lever = one gate row on top of the ACCEPTED base
# (starts from G0). Accept rule (vs the current base, same line, Tony bench): decode wins when C1 per-stream OR C6 aggregate improves by
# >= ACCEPT_PCT (3 %) AND cold prefill @47K does not regress by more than PREFILL_TOL (5 %) AND smoke + tool-call smoke passed.
# Profile levers (SERVE1M) are accepted on health + smoke alone (they trade KV for context, not speed). Rejected levers are dropped;
# accepted ones are carried in the env for every later rung. Ladder state: ops/ladder-<LINE>.md (table) and ops/accepted-<LINE>.env.
# Levers (env deltas): ROCE=1 · ET64 (Engram disk threads 64) · CH8 (NCCL channels 8) · K7 / K4 (DSpark k) · ASYNC (--async-scheduling)
#   · HOT (resident Engram hot set) · SERVE1M (1 M ctx, 8 seqs). Our 0.29-port priors: RoCE one-shot all-reduce +34 % C1 on GLM; channels
#   8/16 −5 % prefill on a switched 100G port; async scheduling in our recipe; k=3 −40 % (tenaiaiai), k=7 untested.
set -u; K=~/glm53-v4-quant/dsv41; LINE=${1:?LINE}; shift; LEVERS=${*:-"ROCE K7 ASYNC SERVE1M"}; ACCEPT_PCT=${ACCEPT_PCT:-3}; PREFILL_TOL=${PREFILL_TOL:-5}
LOG=$K/ops/ladder-$LINE.log; TBL=$K/ops/ladder-$LINE.md; ACC=$K/ops/accepted-$LINE.env; mkdir -p $K/ops; log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
lever_env(){ case $1 in G0) echo "";; ROCE) echo "ROCE=1";; ET64) echo "ENGRAM_THREADS=64";; CH8) echo "NCCL_CH=8";; K7) echo "SPEC_K=7";; K4) echo "SPEC_K=4";; ASYNC) echo "ASYNC=1";; HOT) echo "HOT_DIR=/mnt/glm52/dsv41engram/hot90 HOT_ROWS=${HOT_ROWS:-100000000}";; SERVE1M) echo "MAXLEN=1000000 SEQS=8";; *) echo "$1";; esac; }
metrics(){ python3 - "$1" <<'PY'
import json,sys,os,glob
lab=sys.argv[1]; f=glob.glob(os.path.expanduser(f"~/glm53-v4-quant/dsv41/results/{lab}/bench-*.json"))
if not f: print("nan nan nan"); sys.exit()
j=json.load(open(f[0])); h={x["level"]:x for x in j.get("headline",[])}; pre={p.get("target"):p.get("prefill_tok_s",0) for p in j.get("prefill",[])}
print(h.get("C1",{}).get("per_stream_tok_s","nan"), h.get("C6",{}).get("agg_tok_s","nan"), pre.get(32000, pre.get("32000","nan")))
PY
}
run_row(){ # $1 label, $2 env string → runs one gate row with the accepted env + this lever's env; returns 0 if bench done
  local label=$1 envs="$2"; local acc=""; [ -f $ACC ] && acc=$(grep -vE '^#|^$' $ACC | tr '\n' ' ')
  log "== rung $label: base [${acc:-none}] + lever [$envs]"
  ( cd $K && env LINE=$LINE PROFILE=exl3 LABEL=$label QUALITY_ROWS="${QUALITY_ROWS:-G0}" $acc $envs bash dsv41_gate_run.sh exl3 "$( [ -n "$envs" ] && echo "$envs" | tr ' ' ',' || echo G0 )" ) 2>&1 | grep -E "row |DSV41-TP4|smoke:|toolcall|KV:|ABORT|NODE-PREP-(ALL|SOME)|pass@1|ppl|needle" | cut -c1-180 | tee -a "$LOG"
  grep -q "DSV41-TP4-BENCH-DONE → .*exl3-$label-$LINE" $K/boot_tp4.log
}
[ -f $TBL ] || printf "| rung | env | C1/stream | C6 agg | prefill@47K | verdict |\n|---|---|---|---|---|---|\n" > $TBL
if ! [ -f $K/results/exl3-G0-$LINE/bench-exl3-G0-$LINE.json ]; then run_row G0 "" || { log "G0 failed — ladder stopped"; exit 1; }; fi
read b1 b6 bp <<<"$(metrics exl3-G0-$LINE)"; log "base G0: C1/stream $b1 C6 $b6 prefill47K $bp"; grep -q "| G0 " $TBL || echo "| G0 | (Tony config) | $b1 | $b6 | $bp | base |" >> $TBL
for lv in $LEVERS; do e=$(lever_env $lv); label=$lv; ok=0
  if [ "$lv" = HOT ]; then hn=user@RANK_LAN_IP; [ "$LINE" = B ] && hn=user@RANK_LAN_IP; timeout 20 ssh -o BatchMode=yes $hn 'test -s /mnt/glm52/dsv41engram/hot90/engram_hot_L01.safetensors' || { log "rung HOT skipped: hot-set files not on the line's head yet"; echo "| HOT | $e | skipped (no hot-set files yet) | | | skip |" >> $TBL; continue; }; fi
  if run_row $label "$e"; then read c1 c6 cp <<<"$(metrics exl3-$label-$LINE)"
    smoke=$(grep -A3 "exl3-$label-$LINE" $K/boot_tp4.log | grep -cE "smoke: 1, 2, 3|toolcall: \['get_weather"); 
    if [ "$lv" = SERVE1M ]; then [ "$smoke" -ge 1 ] && ok=1
    else ok=$(python3 -c "b1,b6,bp,c1,c6,cp=map(float,'$b1 $b6 $bp $c1 $c6 $cp'.split()); dec=(c1>=b1*(1+$ACCEPT_PCT/100)) or (c6>=b6*(1+$ACCEPT_PCT/100)); pre=cp>=bp*(1-$PREFILL_TOL/100); print(int(dec and pre and $smoke>=1))" 2>/dev/null || echo 0); fi
    verdict=$([ "$ok" = 1 ] && echo ACCEPT || echo reject); echo "| $label | $e | $c1 | $c6 | $cp | $verdict |" >> $TBL; log "rung $label: C1/stream $c1 (base $b1) C6 $c6 (base $b6) prefill $cp (base $bp) → $verdict"
    if [ "$ok" = 1 ]; then for kv in $e; do echo "$kv" >> $ACC; done; b1=$c1; b6=$c6; bp=$cp; fi
  else echo "| $label | $e | boot/bench failed | | | reject |" >> $TBL; log "rung $label: FAILED → reject"; fi
done
log "LADDER-DONE line $LINE: accepted [$(grep -vE '^#|^$' $ACC 2>/dev/null | tr '\n' ' ')]"; cat $TBL | tee -a "$LOG"; ~/bin/tg-send "DSV41 line $LINE ladder done. Accepted: $(grep -vE '^#|^$' $ACC 2>/dev/null | tr '\n' ' '). Table: ops/ladder-$LINE.md" >/dev/null 2>&1
