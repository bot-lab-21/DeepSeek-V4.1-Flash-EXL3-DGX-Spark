#!/bin/bash
# roce_ar_test.sh <rank> <world> <master fabric ip> — run roce_ar_test.py on THIS node in vllm-dsv41:overlay6 with the TP4 launcher's NCCL/IB env.
set -u; R=$1; W=$2; M=$3; IF=enp1s0f1np1; HOST_IP=$(ip -4 -o addr show $IF | grep -oE "10\.0\.0\.[0-9]+" | head -1)
docker run --rm --name roce_ar_test --gpus all --network host --ipc host --shm-size 8g --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  -v /mnt/glm52/dsv41kit/roce_port:/rp:ro -e RANK=$R -e WORLD_SIZE=$W -e MASTER_ADDR=$M -e MASTER_PORT=29777 -e VLLM_HOST_IP=$HOST_IP \
  -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=106 -e NCCL_IB_TIMEOUT=22 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=FABRIC_IP/24 -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN --entrypoint python3 vllm-dsv41:overlay7 /rp/roce_ar_test.py 2>&1 | grep -vE "Warning|warn"
