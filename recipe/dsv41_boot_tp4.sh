#!/bin/bash
# dsv41_boot_tp4.sh — boot the DeepSeek-V4.1-Flash TP4 baseline (shipped MXFP4/FP8, Tony boot-10 config) on sp4 (head) + sp6 + sp7 + sp10,
# worker-first like Tony's boot_dsv41.sh, wait for /health on the head, then run Tony's bench protocol (v41bench.py, levels 1-6 + prefill sweep).
# Preconditions (checked): no dsv41cap capture container on the 4 nodes, checkpoint signature identical on all 4, image present, patches staged,
# MemAvailable >= 100 GiB. Log ~/glm53-v4-quant/dsv41/boot_tp4.log; markers DSV41-TP4-HEALTHY / DSV41-TP4-BOOT-FAILED / DSV41-TP4-BENCH-DONE.
set -u; K=~/glm53-v4-quant/dsv41; LOG=$K/boot_tp4.log; log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
S(){ timeout ${T:-120} ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 "$@"; }
LINE=${LINE:-A}
case "$LINE" in
  A) declare -A N=( [0]=user@RANK_LAN_IP [1]=user@RANK_LAN_IP [2]=user@RANK_LAN_IP [3]=user@RANK_LAN_IP ); HEAD_API=http://LAN_IP:8000; CNAME=vllm_dsv41; PREP_NODES="sp4 sp6 sp7 sp10" ;;
  B) declare -A N=( [0]=user@RANK_LAN_IP [1]=user@RANK_LAN_IP [2]=user@RANK_LAN_IP [3]=user@RANK_LAN_IP ); HEAD_API=http://LAN_IP:8001; CNAME=vllm_dsv41b; PREP_NODES="sp3 sp5 sp8 sp9" ;;
  *) echo "LINE must be A or B"; exit 1 ;;
esac
EXP=${EXP_NAME:-boot1}
PROFILE=${PROFILE:-shipped}   # shipped = /mnt/glm52/hub/DeepSeek-V4.1-Flash everywhere; exl3 = EXL3 dir on sp4/sp6, in-place converted dir on sp7/sp10
model_host(){ case "$PROFILE:$1" in partial:*) echo /mnt/glm52/hub/DeepSeek-V4.1-Flash-EXL3-partial3;; exl3:0|exl3:1|exl3:2) echo /mnt/glm52/hub/DeepSeek-V4.1-Flash-EXL3-3p5;; exl3:3) [ "$LINE" = B ] && echo /mnt/glm52/hub/DeepSeek-V4.1-Flash-EXL3-3p5 || echo /mnt/glm52/hub/DeepSeek-V4.1-Flash;; *) echo /mnt/glm52/hub/DeepSeek-V4.1-Flash;; esac; }
declare -A SP=( [0]=sp4 [1]=sp6 [2]=sp7 [3]=sp10 ); [ "$LINE" = B ] && declare -A SP=( [0]=sp3 [1]=sp5 [2]=sp8 [3]=sp9 )
export EXP_NAME=$EXP IMAGE=${IMAGE:-vllm-dsv41:overlay8} GMU=${GMU:-0.80} MAXLEN=${MAXLEN:-300000} SEQS=${SEQS:-8} EAGER=${EAGER:-0} SPEC=${SPEC:-dspark} SPEC_K=${SPEC_K:-5} ENGRAM_DISK=1 TEXT_ONLY=${TEXT_ONLY:-0} THINKING=${THINKING:-false} PARSERS=${PARSERS:-1}
log "== TP4 boot $EXP (line $LINE): image $IMAGE roce ${ROCE:-0} async ${ASYNC:-0} k ${SPEC_K:-5} maxlen ${MAXLEN:-300000} nccl_ch ${NCCL_CH:-default} engram_threads ${ENGRAM_THREADS:-32} gmu $GMU maxlen $MAXLEN spec $SPEC k=$SPEC_K text_only $TEXT_ONLY parsers $PARSERS"
sig=""; for r in 0 1 2 3; do h=${N[$r]}
  S $h 'docker ps --format "{{.Names}}" | grep -qE "^dsv41cap$"' && { log "ABORT: capture container still running on $h"; exit 1; }
  md=$(model_host $r); if [ "$PROFILE" = exl3 ] || [ "$PROFILE" = partial ]; then if [ "${VERIFY_MD5:-0}" = 1 ]; then s=$(T=3600 S $h "cd $md && md5sum -c --quiet MD5SUMS.body >/dev/null 2>&1 && md5sum MD5SUMS.body | cut -c1-12 || echo BODY-MD5-FAIL"); else s=$(S $h "cd $md && md5sum MD5SUMS.body | cut -c1-12 && ls model-*.safetensors | wc -l" | tr "\n" " "); fi; else s=$(S $h "cd $md && echo \$(find . -type f | wc -l) \$(du -sb --apparent-size . | cut -f1)"); fi
  [ -z "$sig" ] && sig=$s; [ "$s" = "$sig" ] || { log "ABORT: checkpoint signature differs on $h ($s vs $sig)"; exit 1; }; grep -q FAIL <<<"$s" && { log "ABORT: body MD5 failed on $h"; exit 1; }
  S $h "docker image inspect $IMAGE >/dev/null 2>&1 && test -f ~/patches/dsv41-boot3/mounts.txt" || { log "ABORT: image or patches missing on $h"; exit 1; }
  avail=$(S $h "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo"); [ "${avail:-0}" -ge 100 ] || { log "ABORT: $h MemAvailable ${avail} GiB < 100"; exit 1; }
done
log "preflight OK (signature $sig)"; [ "${SKIP_PREP:-0}" = 1 ] || { PROFILE=$PROFILE IMAGE=$IMAGE bash $K/dsv41_node_prep.sh $PREP_NODES 2>&1 | grep -E "NODE-PREP|FAIL|CUDA free|GPU state" | tee -a "$LOG"; grep -q "NODE-PREP-ALL-OK" $K/node_prep.log || { log "ABORT: node prep failed"; exit 1; }; }; scp -q $K/dsv41_tp4_launch.sh ${N[0]}:~/ ; for r in 1 2 3; do scp -q $K/dsv41_tp4_launch.sh ${N[$r]}:~/; done
ENVS="LINE=$LINE ASYNC=${ASYNC:-0} EXP_NAME=$EXP_NAME IMAGE=$IMAGE KV_BYTES=${KV_BYTES:-} SERVED_ALIAS=${SERVED_ALIAS:-} HOT_DIR=${HOT_DIR:-} HOT_ROWS=${HOT_ROWS:-} ROCE=${ROCE:-} NCCL_CH=${NCCL_CH:-} ENGRAM_THREADS=${ENGRAM_THREADS:-32} GMU=$GMU MAXLEN=$MAXLEN SEQS=$SEQS EAGER=$EAGER SPEC=$SPEC SPEC_K=$SPEC_K ENGRAM_DISK=1 TEXT_ONLY=$TEXT_ONLY THINKING=$THINKING PARSERS=$PARSERS"
launch_rank(){ # $1 rank → runs the per-node launcher (which carries the per-rank MEM-GATE); any failure aborts the boot and removes the ranks already launched
  local r=$1 out; out=$(T=300 S ${N[$r]} "$ENVS MODEL_HOST=$(model_host $r) bash ~/dsv41_tp4_launch.sh $r" 2>&1 || true)
  grep -E "MEM-GATE|launched on" <<<"$out" | tee -a "$LOG"
  grep -q "launched on" <<<"$out" || { log "DSV41-TP4-BOOT-FAILED: rank $r launcher failed on ${N[$r]}: $(grep -vE 'MEM-GATE OK' <<<"$out" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; for rr in 0 1 2 3; do S ${N[$rr]} "docker rm -f $CNAME >/dev/null 2>&1; true"; done; exit 1; }; }
for r in 3 2 1; do launch_rank $r; done; sleep 5; launch_rank 0
log "== waiting for head /health (up to 45 min; Tony: weights ~5-12 min local, graphs ~1 min)"
for i in $(seq 1 90); do curl -sf -m 5 -o /dev/null $HEAD_API/health && break; sleep 30
  [ $((i % 10)) = 0 ] && S ${N[0]} "CNAME=$CNAME; "'docker logs $CNAME 2>&1 | grep -E "Loading weights took|KV cache|Capturing|graph|Engram|ERROR|Error" | tail -n 2 | cut -c1-160' | tee -a "$LOG"
  S ${N[0]} "CNAME=$CNAME; "'docker inspect -f "{{.State.Running}}" $CNAME 2>/dev/null' | grep -q true || { log "DSV41-TP4-BOOT-FAILED: head container exited"; for rr in 1 2 3; do S ${N[$rr]} "CNAME=$CNAME; "'docker logs --tail 300 $CNAME 2>&1 | grep -vE "Warning|warn|Capturing" | grep -iE "error|raise|Traceback|exl3|Exl3|assert|KeyError|ValueError|RuntimeError" | tail -n 6 | cut -c1-220' | sed "s/^/rank$rr: /" | tee -a "$LOG"; done; S ${N[0]} "CNAME=$CNAME; "'docker logs $CNAME 2>&1 | grep -vE "Warning" | tail -n 25 | cut -c1-200' | tee -a "$LOG"; exit 1; }
done
curl -sf -m 5 -o /dev/null $HEAD_API/health || { log "DSV41-TP4-BOOT-FAILED: no health after 45 min"; exit 1; }
for r in 0 1 2 3; do S ${N[$r]} 'pkill -f cache_flusher >/dev/null 2>&1; true'; done   # flusher only during load (memory tricks rule 7)
kvline=$(S ${N[0]} "CNAME=$CNAME; "'docker logs $CNAME 2>&1 | grep -oE "Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | tail -n 2 | tr "\n" ";"'); log "KV: $kvline (pin with KV_BYTES on the next boot)"
log "DSV41-TP4-HEALTHY $(S ${N[0]} "CNAME=$CNAME; "'docker logs $CNAME 2>&1 | grep -E "Loading weights took|GPU KV cache size|Available KV cache" | tail -n 3 | cut -c1-160' | tr '\n' ' | ')"
curl -s -m 60 $HEAD_API/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Count from 1 to 20, comma separated, nothing else."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"thinking":false}}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("smoke:", d["choices"][0]["message"]["content"][:120].replace("\n"," "), "| usage", d.get("usage"))' | tee -a "$LOG"
# warm-up (pays lazy compile / MTP warm-up once) + tool-call + vision smokes
curl -s -m 120 $HEAD_API/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Write a 120-word paragraph about tensor parallelism."}],"max_tokens":200,"temperature":0}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("warmup:", d["usage"]["completion_tokens"], "tokens")' | tee -a "$LOG"
curl -s -m 120 $HEAD_API/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":128,"temperature":0}' | python3 -c 'import sys,json; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("toolcall:", [t["function"]["name"]+" "+t["function"]["arguments"] for t in (m.get("tool_calls") or [])] or m.get("content","")[:80])' | tee -a "$LOG"
# Tony's bench protocol (his script, his prompt set), plus his cold-prefill sweep
mkdir -p $K/results/$EXP; cp ~/port/dsv41-tony/bench/v41bench.py ~/port/dsv41-tony/bench/prompts-v1.json $K/results/$EXP/ 2>/dev/null
cd $K/results/$EXP && timeout 3600 python3 v41bench.py --base $HEAD_API/v1 --model deepseek-v4.1-flash --label $EXP --out . --levels 1,2,3,4,5,6 --prefill 2000,8000,32000,64000 --notes "bot-lab-21 TP4 sp4/6/7/10, shipped MXFP4/FP8, Tony boot-10 config, patches ca662ac" 2>&1 | tail -n 12 | tee -a "$LOG"
log "DSV41-TP4-BENCH-DONE → $K/results/$EXP"; ~/bin/tg-send "DSV41 TP4 baseline ($EXP) healthy on sp4/6/7/10 and benched (Tony protocol). Results in ~/glm53-v4-quant/dsv41/results/$EXP" >/dev/null 2>&1
