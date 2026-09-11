#!/bin/bash
# roce_ar_2node.sh [nodeA nodeB] — run the RoCE one-shot all-reduce unit test on two idle Sparks (default sp7 + sp10) in vllm-dsv41:overlay7.
# Both nodes need the image + /mnt/glm52/dsv41kit/roce_port. Rank 1 detached on B, rank 0 foreground on A (master = A's fabric IP). Log roce_ar_test.log.
set -u; K=~/glm53-v4-quant/dsv41; LOG=$K/roce_port/roce_ar_test.log
declare -A H=( [sp4]=user@RANK_LAN_IP [sp6]=user@RANK_LAN_IP [sp7]=user@RANK_LAN_IP [sp10]=user@RANK_LAN_IP [sp3]=user@RANK_LAN_IP [sp5]=user@RANK_LAN_IP [sp8]=user@RANK_LAN_IP [sp9]=user@RANK_LAN_IP )
declare -A FAB=( [sp3]=FABRIC_IP [sp4]=FABRIC_IP [sp5]=FABRIC_IP [sp6]=FABRIC_IP [sp7]=FABRIC_IP [sp8]=FABRIC_IP [sp9]=FABRIC_IP [sp10]=FABRIC_IP )
A=${1:-sp7}; B=${2:-sp10}; S(){ timeout ${T:-120} ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; }
for n in $A $B; do S ${H[$n]} 'docker ps --format "{{.Names}}" | grep -qE "^(dsv41cook|dsv41cap|dsv41fq|vllm)"' && { echo "$n busy — not testing"; exit 1; }; done
S ${H[$B]} 'docker image inspect vllm-dsv41:overlay7 >/dev/null 2>&1' || { echo "syncing overlay6 image + kit sp7 → $B"; T=2400 S ${H[sp7]} 'docker save vllm-dsv41:overlay7' | T=2400 S ${H[$B]} 'docker load -q' >/dev/null; }
rsync -a -q $K/roce_port/roce_ar_test.py $K/roce_port/roce_ar_test.sh ${H[$B]}:/mnt/glm52/dsv41kit/roce_port/ 2>/dev/null || S ${H[$B]} 'mkdir -p /mnt/glm52/dsv41kit/roce_port' && rsync -a -q $K/roce_port/roce_ar_test.py $K/roce_port/roce_ar_test.sh ${H[$B]}:/mnt/glm52/dsv41kit/roce_port/
echo "[$(date '+%F %T')] == RoCE all-reduce test: rank0 $A (${FAB[$A]}) rank1 $B" | tee -a $LOG
S ${H[$B]} "docker rm -f roce_ar_test >/dev/null 2>&1; (setsid nohup bash /mnt/glm52/dsv41kit/roce_port/roce_ar_test.sh 1 2 ${FAB[$A]} > /tmp/roce_ar_rank1.log 2>&1 < /dev/null &)"; sleep 3
T=900 S ${H[$A]} "docker rm -f roce_ar_test >/dev/null 2>&1; bash /mnt/glm52/dsv41kit/roce_port/roce_ar_test.sh 0 2 ${FAB[$A]}" 2>&1 | tee -a $LOG | tail -n 14
S ${H[$B]} 'tail -n 3 /tmp/roce_ar_rank1.log; docker rm -f roce_ar_test >/dev/null 2>&1' | sed "s/^/[$B] /" | tee -a $LOG
