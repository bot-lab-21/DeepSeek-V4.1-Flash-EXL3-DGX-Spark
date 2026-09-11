#!/bin/bash
# dsv41_tp4_launch.sh <rank> — DeepSeek-V4.1-Flash TP4 on OUR Sparks, derived from tonyd2wild/Kai launch/dsv41-tp4.sh (boot 9 config).
# Rank map (worker-first launch: 3, 2, 1, then 0):  0 sp4 FABRIC_IP (head, API LAN_IP:8000)   1 sp6 FABRIC_IP   2 sp7 FABRIC_IP   3 sp10 FABRIC_IP
# Every node has a LOCAL copy of the checkpoint at /mnt/glm52/hub/DeepSeek-V4.1-Flash (no NFS); patches in ~/patches/dsv41-boot3.
# Fabric: enp1s0f1np1 / rocep1s0f1 / GID 3 / TC 106 (our GLM-5.3 recipes), RoCE range FABRIC_IP/24.
# Knobs (export before running, SAME on all four): IMAGE (vllm-dsv41:overlay5) EXP_NAME GMU (0.80) MAXLEN (300000) SEQS (8)
#   EXTRA_ENV (extra `-e K=V` docker env args, e.g. "-e CUDA_EXL3_MOE_BLOCK_M=16") MAX_BATCHED (8192) EAGER (0) CUDAGRAPH_MODE (FULL_AND_PIECEWISE) CG_SIZES SPEC (dspark) SPEC_K (5) SPEC_ADAPT (false)
#   ENGRAM_DISK (1) ENGRAM_THREADS (32) ENGRAM_CHUNK (16) TEXT_ONLY (0) THINKING (false) PARSERS (1) RUST_FE (0) VLLM_EXTRA NCCL_EXTRA
set -euo pipefail
NODE_RANK="${1:?usage: dsv41_tp4_launch.sh <0|1|2|3>}"
IMAGE="${IMAGE:-vllm-dsv41:overlay5}"; EXP_NAME="${EXP_NAME:-boot1}"; GMU="${GMU:-0.80}"; MAXLEN="${MAXLEN:-300000}"; SEQS="${SEQS:-8}"
MAX_BATCHED="${MAX_BATCHED:-8192}"; EAGER="${EAGER:-0}"; CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"; CG_SIZES="${CG_SIZES:-}"
SPEC="${SPEC:-dspark}"; SPEC_K="${SPEC_K:-5}"; ENGRAM_DISK="${ENGRAM_DISK:-1}"; TEXT_ONLY="${TEXT_ONLY:-0}"; THINKING="${THINKING:-false}"
PARSERS="${PARSERS:-1}"; RUST_FE="${RUST_FE:-0}"; VLLM_EXTRA_DEFAULT='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'; VLLM_EXTRA="${VLLM_EXTRA:-$VLLM_EXTRA_DEFAULT}"; [ "${ASYNC:-0}" = 1 ] && VLLM_EXTRA="$VLLM_EXTRA --async-scheduling"   # ladder lever ASYNC=1 (our 0.29 recipe)   # NB: a literal } inside ${VAR:-...} closes the expansion (boot1 failure)
NCCL_EXTRA="${NCCL_EXTRA:--e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton}"
NAME="vllm_dsv41"; MODEL_DIR="DeepSeek-V4.1-Flash"; MODEL_HOST="${MODEL_HOST:-/mnt/glm52/hub/$MODEL_DIR}"; CACHE_HOST_PATH="/mnt/glm52/dsv41-vllm-cache${LINE:+-$LINE}"
# MODEL_HOST may differ per node (EXL3 dir on sp4/sp6/sp7, in-place on sp10); the container path /models/$MODEL_DIR is the same on every rank
SITE="/usr/local/lib/python3.12/dist-packages/vllm"; IF=enp1s0f1np1; LINE="${LINE:-A}"
# LINE A = sp4(head) sp6 sp7 sp10 on :8000 / master 29541; LINE B = sp3(head) sp5 sp8 sp9 on :8001 / master 29542 (two TP4 lines side by side)
case "$LINE" in
  A) HEAD_IP="FABRIC_IP"; MPORT="29541"; PORT="8000"; RANK_IPS=(FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP) ;;
  B) HEAD_IP="FABRIC_IP"; MPORT="29542"; PORT="8001"; RANK_IPS=(FABRIC_IP FABRIC_IP FABRIC_IP FABRIC_IP); NAME="vllm_dsv41b" ;;
  *) echo "LINE must be A or B" >&2; exit 3 ;;
esac
HOST_IP=${RANK_IPS[$NODE_RANK]}; [ "$NODE_RANK" = 0 ] && HEADLESS="" || HEADLESS="--headless"
case "$NODE_RANK" in
  0|1|2|3) ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST" >&2; exit 3; }
test -f "$MODEL_HOST/model-00048-of-00048.safetensors" || { echo "MODEL INCOMPLETE at $MODEL_HOST (shard 48 missing)" >&2; exit 3; }
# --- per-rank MEMORY GATE (user 2026-09-10: "they should all have a mem gate in launch recipe"). vLLM's worker refuses to start when
# CUDA-free < GMU*total at its init snapshot (sp7 died at 96.08 < 97.35 GiB minutes after prep had seen 110). CUDA-free on GB10 ≈ MemFree
# (probe 114.1 vs MemFree 114.6), so gate on MemFree >= GMU*total + MEM_MARGIN_GIB right before docker run; drop caches and retry; never lower GMU.
MEM_MARGIN_GIB="${MEM_MARGIN_GIB:-4}"; MEM_MIN_GIB="${MEM_MIN_GIB:-0}"   # MEM_MIN_GIB: absolute MemFree floor (GiB) when KV is pinned and GMU is only vLLM's startup guard
pgrep -f "bash /tmp/cache_flusher.sh" >/dev/null 2>&1 || { [ -x /tmp/cache_flusher.sh ] && (setsid nohup bash /tmp/cache_flusher.sh >> /tmp/cache_flusher.log 2>&1 < /dev/null &) || true; }
mem_gate(){ awk -v g="$GMU" -v m="$MEM_MARGIN_GIB" -v a="$MEM_MIN_GIB" '/^MemFree:/{f=$2/1048576} /^MemTotal:/{t=$2/1048576} END{need=g*t+m; if (a+0>need) need=a+0; printf "%.1f %.1f %.1f %d\n", f, t, need, (f>=need)}' /proc/meminfo; }
for attempt in 1 2 3 4; do read -r MF MT MNEED MOK <<<"$(mem_gate)"; [ "$MOK" = 1 ] && break
  echo "MEM-GATE rank $NODE_RANK attempt $attempt: MemFree $MF GiB < need $MNEED (gmu $GMU*$MT + margin $MEM_MARGIN_GIB) → dropping caches"
  sync; sudo -n /usr/local/bin/glm-drop-caches >/dev/null 2>&1 || echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; sleep 3; done
read -r MF MT MNEED MOK <<<"$(mem_gate)"
[ "$MOK" = 1 ] || { echo "MEM-GATE FAIL rank $NODE_RANK on $(hostname): MemFree $MF/$MT GiB < need $MNEED GiB (gmu $GMU + margin $MEM_MARGIN_GIB) after cache drops — clean the node (stray processes / page cache), do not lower gmu" >&2; exit 4; }
echo "MEM-GATE OK rank $NODE_RANK on $(hostname): MemFree $MF/$MT GiB (need >= $MNEED)"
PATCH_DIR="${PATCH_DIR:-$HOME/patches/dsv41-boot3}"; PATCH_MOUNTS=""
[ -f "$PATCH_DIR/mounts.txt" ] || { echo "no mounts.txt in $PATCH_DIR" >&2; exit 3; }
while read -r f rel; do
  [ -z "$f" ] && continue
  if [ "$ENGRAM_DISK" != "1" ] && { [ "$f" = "engram.py" ] || [ "$f" = "weight_utils.py" ] || [ "$f" = "model_state.py" ]; }; then continue; fi
  test -f "$PATCH_DIR/$f" || { echo "PATCH FILE MISSING: $PATCH_DIR/$f" >&2; exit 3; }
  PATCH_MOUNTS="$PATCH_MOUNTS -v $PATCH_DIR/$f:$(realpath -m "$SITE/$rel"):ro"
done < "$PATCH_DIR/mounts.txt"
if [ "$ENGRAM_DISK" = "1" ]; then ENGRAM_ENV="-e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=${ENGRAM_THREADS:-32} -e DSV41_ENGRAM_DISK_CHUNK=${ENGRAM_CHUNK:-16} ${HOT_DIR:+-e DSV41_ENGRAM_HOT_DIR=/hot -e DSV41_ENGRAM_HOT_ROWS=${HOT_ROWS:-20000000}}"; else ENGRAM_ENV="-e DSV41_ENGRAM_DISK=0"; fi
mkdir -p "$CACHE_HOST_PATH"; docker rm -f "$NAME" 2>/dev/null || true
sudo -n /usr/local/bin/glm-drop-caches >/dev/null 2>&1 || true
AVAIL_GB=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 )); [ "$AVAIL_GB" -ge 100 ] || { echo "MemAvailable ${AVAIL_GB} GiB < 100 GiB, refusing to boot" >&2; exit 4; }
GRAPH_ENV=""
if [ "$EAGER" = "1" ]; then GRAPH_ARGS=(--enforce-eager); else
  grep -q '^model_state.py ' "$PATCH_DIR/mounts.txt" || [ "$ENGRAM_DISK" != "1" ] || { echo "EAGER=0 + ENGRAM_DISK=1 needs model_state.py in mounts.txt" >&2; exit 3; }
  if [ -z "$CG_SIZES" ]; then
    if [ "$SPEC" = "dspark" ]; then K="$SPEC_K"; CG_SIZES=$( { seq "$K" "$K" $((K * SEQS)); seq $((K + 1)) $((K + 1)) $(((K + 1) * SEQS)); } | sort -n -u | paste -sd, - ); else CG_SIZES=$(seq 1 "$SEQS" | paste -sd, -); fi
  fi
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":[$CG_SIZES]}"); GRAPH_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1"
fi
if [ "$EAGER" = "1" ]; then SPEC_ADAPT=false; else SPEC_ADAPT="${SPEC_ADAPT:-false}"; fi
if [ "$SPEC" = "dspark" ]; then SPEC_ARGS="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$SPEC_ADAPT}"; else SPEC_ARGS=""; fi
if [ "$TEXT_ONLY" = "1" ]; then TEXT_ARGS="--language-model-only"; else TEXT_ARGS=""; fi
if [ "$PARSERS" = "1" ]; then PARSER_ARGS="--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41"; else PARSER_ARGS=""; fi
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband --oom-score-adj 500 \
  -v "$MODEL_HOST:/models/$MODEL_DIR:ro" -v "$CACHE_HOST_PATH:/cache" ${HOT_DIR:+-v $HOT_DIR:/hot:ro} $PATCH_MOUNTS \
  -e VLLM_HOST_IP=$HOST_IP -e CUDA_EXL3_MODEL_PATH=/models/$MODEL_DIR -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_CACHE_ROOT="/cache/vllm-$EXP_NAME" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_USE_RUST_FRONTEND=$RUST_FE -e VLLM_HAS_FLASHINFER_CUBIN=1 \
  $ENGRAM_ENV $GRAPH_ENV -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=106 -e NCCL_IB_TIMEOUT=22 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=FABRIC_IP/24 \
  -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF -e MN_IF_NAME=$IF \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  ${NCCL_CH:+-e NCCL_MIN_NCHANNELS=$NCCL_CH -e NCCL_MAX_NCHANNELS=$NCCL_CH} ${EXTRA_ENV:-} ${MOE_BLOCK_M:+-e CUDA_EXL3_MOE_BLOCK_M=$MOE_BLOCK_M} ${NCCL_ALGO:+-e NCCL_ALGO=$NCCL_ALGO} ${NCCL_PROTO:+-e NCCL_PROTO=$NCCL_PROTO} ${NCCL_NTHREADS:+-e NCCL_NTHREADS=$NCCL_NTHREADS} ${ROCE:+-e VLLM_ENABLE_ROCE_ALLREDUCE=$ROCE -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=${ROCE_MAX:-2MB}} $NCCL_EXTRA \
  --entrypoint vllm "$IMAGE" serve "/models/$MODEL_DIR" --served-model-name deepseek-v4.1-flash ${SERVED_ALIAS:-} --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 4 --distributed-executor-backend mp --nnodes 4 --node-rank $NODE_RANK --master-addr $HEAD_IP --master-port $MPORT $HEADLESS \
  --gpu-memory-utilization "$GMU" ${KV_BYTES:+--kv-cache-memory-bytes $KV_BYTES} --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --engram-config '{"cpu_offload": false}' "${GRAPH_ARGS[@]}" $SPEC_ARGS $TEXT_ARGS $PARSER_ARGS \
  --default-chat-template-kwargs "{\"thinking\": $THINKING}" $VLLM_EXTRA
echo "rank $NODE_RANK launched on $HOST_IP ($IMAGE, exp $EXP_NAME, roce ${ROCE:-0}, gmu $GMU, maxlen $MAXLEN, spec $SPEC k=$SPEC_K adapt=$SPEC_ADAPT, engram_disk $ENGRAM_DISK, text_only $TEXT_ONLY)"
