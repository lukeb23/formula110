#!/bin/zsh
set -eu
cd /Users/lukebriden/formula110
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
resume_args=()
if [[ -f artifacts/evolution-bc-challenger-20260908/experiment.json ]]; then
  resume_args=(--resume)
fi
caffeinate -is uv run racing-evolve \
  "${resume_args[@]}" \
  --initial-population artifacts/evolution-bc-challenger-20260908/generation-000 \
  --generations 25 --elites 5 --mutation-std 0.002 \
  --mutation-stds 0.01 0.01 0.005 0.002 0.015 0.04 \
  --mutation-scopes steer_output throttle_output output all all all \
  --mutation-weights 0.20 0.25 0.25 0.25 0.05 0.0 \
  --training-seed 110 --simulator-seeds 110 111 112 271 997 \
  --round-seconds 30 --worst-seed-weight 0.25 \
  --distance-weight 1.5 --lap-completion-bonus 100 --fast-lap-bonus 300 \
  --health-exponent 1 --damage-penalty 0 \
  --wall-contact-penalty-per-s 20 --low-progress-penalty-per-s 10
