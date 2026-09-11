#!/bin/bash
# ops/boot_prod.sh <LINE> [EXP_NAME] — boot a line on its ACCEPTED base (ops/accepted-<LINE>.env) via the v2 boot chain (prep → launch →
# health → smokes → bench). Use after a ladder ends (ladders tear the line down at LADDER-DONE and do not relaunch production).
set -u; K=~/glm53-v4-quant/dsv41; LINE=${1:?LINE}; EXP=${2:-prod-$LINE-$(date +%m%d%H%M)}
F="$K/ops/accepted-$LINE.env"; [ -f "$F" ] || { echo "no $F"; exit 1; }
set -a; while IFS= read -r l; do case "$l" in ''|'#'*) ;; *) eval "export $l";; esac; done < "$F"; set +a
echo "[$(date '+%F %T')] boot_prod $LINE exp=$EXP base: $(grep -vE '^#|^$' "$F" | tr '\n' ' ')" | tee -a $K/PLAN.md
cd $K && exec env LINE=$LINE PROFILE=exl3 EXP_NAME=$EXP bash dsv41_boot_tp4_v2.sh
