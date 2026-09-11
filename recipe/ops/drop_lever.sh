#!/bin/bash
# ops/drop_lever.sh <LINE> <KEY_PREFIX>... — remove lever lines whose KEY starts with a given prefix from the line's accepted base file.
set -u; K=~/glm53-v4-quant/dsv41; LINE=${1:?LINE}; shift; F="$K/ops/accepted-$LINE.env"
for pre in "$@"; do grep -vE "^${pre}[A-Z0-9_]*=" "$F" > "$F.tmp" && mv "$F.tmp" "$F"; echo "accepted-$LINE -= ${pre}*"; done
echo "accepted-$LINE now: $(grep -vE '^#|^$' "$F" | tr '\n' ' ')"
