#!/bin/bash
# ops/accept_lever.sh <LINE> <KEY=VAL>... — append lever settings to the line's accepted base file if not already present. Prints nothing secret.
set -u; K=~/glm53-v4-quant/dsv41; LINE=${1:?LINE}; shift; F="$K/ops/accepted-$LINE.env"
touch "$F"; for kv in "$@"; do grep -qx -- "$kv" "$F" || { echo "$kv" >> "$F"; echo "accepted-$LINE += $kv"; }; done
echo "accepted-$LINE now: $(grep -vE '^#|^$' "$F" | tr '\n' ' ')"
