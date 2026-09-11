#!/bin/bash
# ops/publish_best_loop.sh — every 3 min: if a ladder produced a new verdict/base line, rebuild the "best serving" block and push the
# card to HF (README.md) and GitHub. User 2026-09-11: "always update HF with the best serving". Runs until killed by PID.
set -u; K=~/glm53-v4-quant/dsv41; cd $K; export SSH_ASKPASS=/tmp/qnap_askpass.sh SSH_ASKPASS_REQUIRE=force
S=ARCHIVE_ROOT/huggingface/DeepSeek-V4.1-Flash-EXL3-3p5
sig(){ cat ops/ladder-A.md ops/ladder-B.md 2>/dev/null | md5sum | cut -c1-12; ls results/*/bench-*.json 2>/dev/null | md5sum | cut -c1-12; }
last=""
while :; do
  now=$(sig | tr '\n' ' ')
  if [ "$now" != "$last" ]; then
    out=$(python3 ops/publish_best.py 2>&1 | tail -n 1); echo "[$(date '+%F %T')] publish_best: $out"
    if grep -q "^BEST" <<<"$out"; then
      timeout 60 scp -q -o StrictHostKeyChecking=accept-new release/HF_MODEL_CARD_dsv41_exl3.md user@ARCHIVE_HOST:$S/README_card.md && \
      vault-get HF_TOKEN | timeout 300 ssh user@ARCHIVE_HOST "bash -lc 'docker exec -i hf-uploader sh /hfroot/hf_tools/push_readme.sh \"card: best serving so far — $out\"'" 2>&1 | grep -viE 'token:|secret|password|bearer' | tail -n 1
      cp release/GITHUB_README.md ~/dsv41-release-repo/README.md; cp release/HF_MODEL_CARD_dsv41_exl3.md ~/dsv41-release-repo/docs/HF_MODEL_CARD.md
      (cd ~/dsv41-release-repo && git add -A && git -c user.name=bot-lab-21 -c user.email=maintainer@example.org commit -q -m "results: best serving so far — $out

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" >/dev/null 2>&1 && git push -q origin main && echo "github pushed")
      echo "[$(date '+%F %T')] HF card + GitHub updated: $out" >> $K/PLAN.md
    fi
    last=$now
  fi
  sleep 180
done
