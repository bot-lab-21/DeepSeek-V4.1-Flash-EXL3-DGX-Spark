#!/bin/bash
# dsv41_quality_gate.sh <label> [needle_tokens=300000] — the "Q" row for a running DSV41 TP4 endpoint (head sp4 :8000): the GLM quality
# battery re-pointed (run-time copies of our probes with endpoint/model substituted; originals untouched):
#   tool-call integrity (12) · correctness probe · ppl probe (24 fixed held-out texts) · golden refs (diffable vs a previous label) ·
#   needle 300K (two passphrases at 30 %/80 % depth) · image probe · evalplus HumanEval+/MBPP+ (greedy, openai backend).
# Output ~/quality_bench_dsv41_<label>/quality.log. Compare labels: golden_ref.py --diff <a> <b>. The DSV41 endpoint has no auth;
# the probes' bearer header (from the brain token file) is sent and ignored.
set -u; LABEL=${1:?label}; NEEDLE=${2:-300000}; BASE=${DSV41_BASE:-http://LAN_IP:8000}; MODEL=${DSV41_MODEL:-deepseek-v4.1-flash}
OUT=$HOME/quality_bench_dsv41_$LABEL; Q=$OUT/probes; mkdir -p "$Q"; LOG="$OUT/quality.log"
for f in ~/glm-steal-apply/tool_call_probe.py ~/glm-steal-apply/needle_probe.py ~/glm-steal-apply/correctness_probe.py ~/glm-steal-apply/image_probe.py ~/glm53-mtp-ft/ppl_probe.py ~/glm53-mtp-ft/golden_ref.py; do
  sed -e "s#http://LAN_IP:5001#$BASE#g" -e "s#LAN_IP:5001#${BASE#http://}#g" -e "s#GLM-5.3-Int4-Int8#$MODEL#g" -e "s#GLM-5.2-Int4-Int8#$MODEL#g" "$f" > "$Q/$(basename $f)"; done
HOST=$(echo ${BASE#http://} | cut -d: -f1); PORT=$(echo ${BASE#http://} | cut -d: -f2)
export OPENAI_BASE_URL="$BASE/v1" GLM_MODEL="$MODEL" BENCH_URL="$BASE/v1/chat/completions" BENCH_MODEL="$MODEL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dsv41-no-auth}"
{
echo "##### DSV41 QUALITY BATTERY [$LABEL] START $(date '+%F %T') base $BASE model $MODEL #####"
curl -s -m 30 "$BASE/v1/models" | grep -q "$MODEL" || { echo "ABORT-no-model at $BASE"; exit 1; }
cd "$Q"
echo "=== tool-call integrity"; python3 tool_call_probe.py 12 2>&1 | tail -n 2
echo "=== correctness probe"; python3 correctness_probe.py "$HOST" "$PORT" "dsv41-$LABEL" 2>&1 | tail -n 4
echo "=== ppl probe (24 held-out)"; python3 ppl_probe.py 24 2>&1 | tee "$OUT/ppl.txt" | tail -n 3
echo "=== golden"; python3 golden_ref.py "dsv41-$LABEL" 2>&1 | tail -n 2
echo "=== needle $NEEDLE"; python3 needle_probe.py "$NEEDLE" 2>&1 | tail -n 3
echo "=== image probe"; python3 image_probe.py 2>&1 | tail -n 3
if source $HOME/evalplus_venv/bin/activate 2>/dev/null; then for DS in humaneval mbpp; do
  echo "=== $DS codegen $(date '+%H:%M')"; ROOT="$OUT/$DS"; rm -rf "$ROOT"; mkdir -p "$ROOT"
  evalplus.codegen --model "$MODEL" --dataset "$DS" --backend openai --base_url "$BASE/v1" --greedy --root "$ROOT" 2>&1 | tail -n 2
  GEN=$(find "$ROOT" -name "*_temp_0.0.jsonl" ! -name "*.raw.jsonl" | head -1); evalplus.sanitize --samples "$GEN" >/dev/null 2>&1
  SAN=$(find "$ROOT" -name "*sanitized*.jsonl" | head -1); evalplus.evaluate --dataset "$DS" --samples "${SAN:-$GEN}" 2>&1 | tee "$OUT/${DS}_eval.txt" | grep -iE "pass@1"
done; else echo "evalplus venv missing — HE+/MBPP+ skipped"; fi
echo "##### DSV41 QUALITY BATTERY [$LABEL] DONE $(date '+%F %T') #####"
} 2>&1 | tee -a "$LOG"
