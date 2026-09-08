#!/bin/zsh
set -eu
cd /Users/lukebriden/formula110
# Run after stopping the old evolution process. Uses an isolated copy of generation 124.
uv run racing-evolve \
  --new-phase-from artifacts/evolution-development-20260908/generation-124 \
  --population-size 50 --generations 25 --elites 5 \
  --mutation-std 0.002 \
  --mutation-stds 0.01 0.01 0.005 0.002 0.015 0.04 \
  --mutation-scopes steer_output throttle_output output all all all \
  --mutation-weights 0.20 0.25 0.25 0.25 0.05 0.0 \
  --training-seed 110 --simulator-seeds 110 111 112 271 997 \
  --round-seconds 30 --worst-seed-weight 0.25 \
  --distance-weight 1.5 --lap-completion-bonus 100 --fast-lap-bonus 300 \
  --health-exponent 1 --damage-penalty 0 \
  --wall-contact-penalty-per-s 20 --low-progress-penalty-per-s 10
