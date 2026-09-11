#!/bin/bash
# dsv41_node_prep.sh <sp...> — the full GB10 pre-boot ritual on each TP4 node (user: "use all our memory tricks and tools on every recipe"):
#  1 mem_playbook.sh (sysctls min_free=1G / watermark 200 / vfs_cache_pressure / swappiness / dirty; compact; drop_caches; glm-reclaim
#    when no vllm runs) + install/start the cache_flusher sidecar (drop caches when Cached > 40 GiB, every 5 s — mandatory during load)
#  2 compaction_proactiveness=0, earlyoom inactive, glm-fabric-mtu-guard active, enp1s0f1np1 MTU 9000 + jumbo ping to the other TP4 fabric IPs
#  3 GPU hidden fast/slow state probe (Tony gpuflip; only on an idle GPU) — slow-state node = do not boot, report
#  4 CUDA-visible free memory gate (torch.cuda.mem_get_info in the vehicle) >= CUDA_MIN_GIB (100) — the vLLM guard reads THIS, not MemAvailable
#  5 model dir gate: config.json + 48 shards + quant_method (exl3 for PROFILE=exl3) + (VERIFY_MD5=1) MD5SUMS.body; image + patches present;
#    config parity: md5 of patch files + launcher across the nodes must be identical (rank mismatch = silent hang)
# env: PROFILE=exl3|shipped, IMAGE (vllm-dsv41:overlay8), CUDA_MIN_GIB, VERIFY_MD5. Log node_prep.log; marker NODE-PREP-OK|FAIL per node.
set -u; K=~/glm53-v4-quant/dsv41; LOG=$K/node_prep.log; log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
S(){ timeout ${T:-120} ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; }
declare -A N=( [sp3]=user@RANK_LAN_IP [sp4]=user@RANK_LAN_IP [sp5]=user@RANK_LAN_IP [sp6]=user@RANK_LAN_IP [sp7]=user@RANK_LAN_IP [sp8]=user@RANK_LAN_IP [sp9]=user@RANK_LAN_IP [sp10]=user@RANK_LAN_IP )
declare -A FAB=( [sp3]=FABRIC_IP [sp4]=FABRIC_IP [sp5]=FABRIC_IP [sp6]=FABRIC_IP [sp7]=FABRIC_IP [sp8]=FABRIC_IP [sp9]=FABRIC_IP [sp10]=FABRIC_IP )
PROFILE=${PROFILE:-exl3}; IMAGE=${IMAGE:-vllm-dsv41:overlay8}; CUDA_MIN_GIB=${CUDA_MIN_GIB:-100}; NODES=${*:?nodes}
model_dir(){ case "$PROFILE:$1" in partial:*) echo /mnt/glm52/hub/DeepSeek-V4.1-Flash-EXL3-partial3;; exl3:sp4|exl3:sp6|exl3:sp7|exl3:sp3|exl3:sp5|exl3:sp8|exl3:sp9) echo /mnt/glm52/hub/DeepSeek-V4.1-Flash-EXL3-3p5;; *) echo /mnt/glm52/hub/DeepSeek-V4.1-Flash;; esac; }
FLUSHER='#!/bin/bash
# cache_flusher sidecar (GB10): drop the page cache whenever Cached > 40 GiB; every 5 s. Weights live in GPU after load, so a big cache = leak.
while true; do c=$(awk "/^Cached:/{print int(\$2/1048576)}" /proc/meminfo); [ "${c:-0}" -gt 40 ] && { sudo -n /usr/local/bin/glm-drop-caches >/dev/null 2>&1 || { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; }; echo "$(date +%T) dropped at ${c}G"; }; sleep 5; done'
declare -A SIG; fail=0
for n in $NODES; do h=${N[$n]}; ok=1; log "== $n prep (profile $PROFILE)"
  S $h "cat > /tmp/cache_flusher.sh <<'F'
$FLUSHER
F
chmod +x /tmp/cache_flusher.sh"
  $HOME/mem_playbook.sh $h ${h%%@*} 2>&1 | tail -n 2 | tee -a "$LOG"
  r=$(S $h 'printf "proact=%s earlyoom=%s mtuguard=%s mtu=%s flusher=%s\n" $(cat /proc/sys/vm/compaction_proactiveness) $(systemctl is-active earlyoom 2>/dev/null; true) $(systemctl is-active glm-fabric-mtu-guard.timer 2>/dev/null; true) $(cat /sys/class/net/enp1s0f1np1/mtu) $(pgrep -fc cache_flusher)'); log "$n $r"
  grep -q "proact=0 " <<<"$r" || log "$n WARN compaction_proactiveness != 0"; grep -qE "earlyoom=(inactive|unknown|failed)" <<<"$r" || { log "$n FAIL earlyoom active"; ok=0; }; grep -q "mtu=9000" <<<"$r" || { log "$n FAIL fabric MTU != 9000"; ok=0; }
  peers=""; for p in $NODES; do [ $p = $n ] || peers="$peers ${FAB[$p]}"; done
  jp=$(S $h "for ip in $peers; do ping -M do -s 8972 -c 1 -W 2 \$ip >/dev/null 2>&1 && printf '%s:ok ' \$ip || printf '%s:FAIL ' \$ip; done"); log "$n jumbo path: $jp"; grep -q FAIL <<<"$jp" && ok=0
  busy=$(S $h 'docker ps --format "{{.Names}}" | grep -cE "^(vllm|dsv41cook|dsv41cap|dsv41fq)"'); if [ "${busy:-1}" = 0 ]; then
    cu=$(T=300 S $h "docker run --rm --gpus all --entrypoint python3 $IMAGE -c 'import torch; f,t=torch.cuda.mem_get_info(); print(int(f/2**30), int(t/2**30))' 2>/dev/null | tail -n 1"); log "$n CUDA free/total GiB: $cu"; [ "${cu%% *}" -ge $CUDA_MIN_GIB ] 2>/dev/null || { log "$n FAIL CUDA free < $CUDA_MIN_GIB GiB → clean (docker rm -f, glm-reclaim, drop_caches), never lower gmu"; ok=0; }
    gp=$(T=400 S $h "docker run --rm --gpus all --user \$(id -u):\$(id -g) -e HOME=/tmp -v /mnt/glm52:/mnt/glm52 --entrypoint python3 $IMAGE /mnt/glm52/dsv41kit/gputools/gpuflip.py 2>&1 | grep -E 'gemv_cont|mm_duty' | grep -oE 'p50= *[0-9.]+' | tr -d ' ' | tr '\n' ' '"); log "$n GPU state (gemv p50 GB/s, mm p50 TFLOPS): $gp"
    gemv=$(grep -oE "p50=[0-9]+" <<<"$gp" | head -1 | cut -d= -f2); [ "${gemv:-0}" -ge 150 ] 2>/dev/null || { log "$n FAIL GPU slow state (gemv p50 ${gemv:-?} GB/s < 150) — do not boot on it"; ok=0; }
  else log "$n GPU busy ($busy container) — CUDA-free gate + GPU probe skipped"; fi
  md=$(model_dir $n); if [ "${SKIP_MODEL:-0}" = 1 ]; then m="skip skip skip"; log "$n model check skipped (SKIP_MODEL=1)"; else m=$(S $h "cd $md 2>/dev/null && printf '%s %s %s ' \$(ls model-*.safetensors | wc -l) \$(python3 -c \"import json;print(json.load(open('config.json')).get('quantization_config',{}).get('quant_method','none'))\") \$(test -f MD5SUMS.body && echo md5file || echo nomd5); [ '${VERIFY_MD5:-0}' = 1 ] && test -f MD5SUMS.body && (md5sum -c --quiet MD5SUMS.body >/dev/null && echo md5-ok || echo md5-FAIL)"); log "$n model $md: shards/quant/md5 = $m"
  fi
  set -- $m; if [ "${SKIP_MODEL:-0}" != 1 ]; then [ "${1:-0}" = 48 ] || { log "$n FAIL shard count $1"; ok=0; }; { [ "$PROFILE" = exl3 ] || [ "$PROFILE" = partial ]; } && [ "${2:-}" != exl3 ] && { log "$n FAIL quant_method $2 (want exl3)"; ok=0; }; grep -q "md5-FAIL" <<<"$m" && ok=0; fi
  im=$(S $h "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null | cut -c8-19; test -f ~/patches/dsv41-boot3/mounts.txt && (cd ~/patches/dsv41-boot3 && cat mounts.txt \$(awk '{print \$1}' mounts.txt) | md5sum | cut -c1-12) || echo nopatches; md5sum ~/dsv41_tp4_launch.sh 2>/dev/null | cut -c1-12"); SIG[$n]=$(tr '\n' ' ' <<<"$im"); log "$n image/patches/launcher: ${SIG[$n]}"
  grep -qE "^ *$|nopatches" <<<"$im" && { log "$n FAIL image or patches missing"; ok=0; }
  [ $ok = 1 ] && log "NODE-PREP-OK $n" || { log "NODE-PREP-FAIL $n"; fail=1; }
done
first=""; for n in $NODES; do [ -z "$first" ] && first=${SIG[$n]}; [ "${SIG[$n]}" = "$first" ] || { log "PARITY FAIL: $n differs from the first node ($first vs ${SIG[$n]})"; fail=1; }; done
[ $fail = 0 ] && log "NODE-PREP-ALL-OK [$NODES]" || log "NODE-PREP-SOME-FAILED — fix before booting"
exit $fail
