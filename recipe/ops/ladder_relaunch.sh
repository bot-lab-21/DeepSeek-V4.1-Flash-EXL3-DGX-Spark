#!/bin/bash
# ops/ladder_relaunch.sh <LINE> [WAIT_PATTERN] -- <levers...>
# Waits (optional) until WAIT_PATTERN appears as a NEW line in ops/ladder-<LINE>.log (e.g. "rung ROCE: .*→" = that rung's verdict,
# or "LADDER-DONE"), then stops that line's ladder chain BY PID TREE (ladder → gate_run → boot_tp4 → node_prep), removes the line's
# containers, and relaunches `dsv41_ladder.sh <LINE> <levers>` with the current environment (e.g. HOT_ROWS=20000000 QUALITY_ROWS=NONE).
# Lives in a file on purpose: a bash -c that contains both a pkill pattern and the relaunch text kills itself (09-10, twice).
set -u; K=~/glm53-v4-quant/dsv41; LINE=${1:?LINE}; shift; WAIT=""; if [ "${1:-}" != "--" ]; then WAIT="$1"; shift; fi; [ "${1:-}" = "--" ] && shift
LOG=$K/ops/ladder-$LINE.log; CN=vllm_dsv41; NODES="user@RANK_LAN_IP user@RANK_LAN_IP user@RANK_LAN_IP user@RANK_LAN_IP"
[ "$LINE" = B ] && { CN=vllm_dsv41b; NODES="user@RANK_LAN_IP user@RANK_LAN_IP user@RANK_LAN_IP user@RANK_LAN_IP"; }
log(){ echo "[$(date '+%F %T')] relaunch-$LINE: $*" | tee -a $K/PLAN.md; }
if [ -n "$WAIT" ]; then n0=$(wc -l < $LOG); until tail -n +$((n0+1)) $LOG | grep -qE "$WAIT"; do pgrep -f "dsv41_ladder[23]?.sh $LINE( |$)" >/dev/null || { log "ladder gone before '$WAIT'"; break; }; sleep 20; done
  tail -n +$((n0+1)) $LOG | grep -E "$WAIT" | tail -n 1 | cut -c1-200; fi
# stop the chain by PID tree
lp=$(pgrep -f "dsv41_ladder[23]?.sh $LINE( |$)"); kids(){ for p in "$@"; do echo $p; kids $(pgrep -P $p); done; }
all=$(kids $lp 2>/dev/null | sort -u); [ -n "$all" ] && { kill -TERM $all 2>/dev/null; sleep 3; kill -KILL $all 2>/dev/null; }
log "stopped ladder chain pids: $(echo $all | tr '\n' ' ')"
for h in $NODES; do timeout 60 ssh -o BatchMode=yes $h "docker rm -f $CN >/dev/null 2>&1; pkill -f 'bash /tmp/cache_flusher.sh' >/dev/null 2>&1; true"; done
log "containers removed on line $LINE; relaunching: $* (env HOT_ROWS=${HOT_ROWS:-default} QUALITY_ROWS=${QUALITY_ROWS:-default})"
cd $K && setsid nohup bash dsv41_ladder3.sh $LINE "$@" > /tmp/dsv41_ladder_$LINE.out 2>&1 < /dev/null &
sleep 3; pgrep -f "dsv41_ladder[23]?.sh $LINE( |$)" >/dev/null && log "ladder $LINE running: $*" || log "ladder $LINE FAILED to start"
