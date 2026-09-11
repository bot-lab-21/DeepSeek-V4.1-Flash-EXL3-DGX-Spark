#!/bin/bash
# ops/stack_next.sh <LINE> <WAIT_LEVER|LADDER-DONE> [+KEY=VAL ...] -- <levers...>
# Keeps a line stacking wins: waits for the verdict of WAIT_LEVER (or LADDER-DONE) in ops/ladder-<LINE>.log, re-derives that
# verdict from its logged numbers (the old ladder's smoke check rejected everything), appends the lever env to the line's accepted
# base if it passes, appends any +KEY=VAL cross-applied wins from the other line, retires the old G0 row so the fixed ladder re-runs
# a clean base with the full accepted set, then relaunches `dsv41_ladder2.sh <LINE> <levers>` via ops/ladder_relaunch.sh.
# Lives in a file on purpose (pattern kills in bash -c self-match). Env passes through (HOT_ROWS, QUALITY_ROWS).
set -u; K=~/glm53-v4-quant/dsv41; LINE=${1:?LINE}; WAITLV=${2:?lever|LADDER-DONE}; shift 2; XAPPLY=(); while [ "${1:-}" != "--" ] && [ $# -gt 0 ]; do XAPPLY+=("${1#+}"); shift; done; [ "${1:-}" = "--" ] && shift
LOG=$K/ops/ladder-$LINE.log; ACC=$K/ops/accepted-$LINE.env
log(){ echo "[$(date '+%F %T')] stack-$LINE: $*" | tee -a $K/PLAN.md; }
pat="rung $WAITLV: .*→"; [ "$WAITLV" = LADDER-DONE ] && pat="LADDER-DONE"
n0=$(wc -l < $LOG)
until tail -n +$((n0+1)) $LOG | grep -qE "$pat"; do pgrep -f "dsv41_ladder[23]?.sh $LINE( |$)" >/dev/null || { log "ladder gone before '$pat'"; break; }; sleep 30; done
line=$(tail -n +$((n0+1)) $LOG | grep -E "$pat" | tail -n 1); log "seen: ${line:0:160}"
if [ "$WAITLV" != LADDER-DONE ]; then
  lv_env=$(grep -oE "$WAITLV\) echo \"[^\"]+\"" $K/dsv41_ladder3.sh | sed -E 's/.*echo "([^"]+)"/\1/')
  ok=$(python3 - "$line" <<'PY'
import re,sys
m=re.search(r"C1/stream ([0-9.]+) \(base ([0-9.]+)\) C6 ([0-9.]+) \(base ([0-9.]+)\) prefill ([0-9.]+) \(base ([0-9.]+)\)", sys.argv[1])
if not m: print(0); sys.exit()
c1,b1,c6,b6,cp,bp=map(float,m.groups()); print(int(((c1>=b1*1.03) or (c6>=b6*1.03)) and cp>=bp*0.95))
PY
)
  if [ "$ok" = 1 ]; then for kv in $lv_env; do grep -qx "$kv" $ACC || echo "$kv" >> $ACC; done; log "$WAITLV re-derived → ACCEPT, appended [$lv_env] to accepted base"; else log "$WAITLV re-derived → reject (numbers below rule)"; fi
fi
for kv in "${XAPPLY[@]}"; do grep -qx "$kv" $ACC || { echo "$kv" >> $ACC; log "cross-applied $kv"; }; done
[ -d $K/results/exl3-G0-$LINE ] && { mv $K/results/exl3-G0-$LINE $K/results/exl3-G0-$LINE-base$(date +%H%M); log "retired old G0 row → fresh base will be measured with accepted [$(grep -vE '^#|^$' $ACC | tr '\n' ' ')]"; }
exec bash $K/ops/ladder_relaunch.sh $LINE -- "$@"
