#!/bin/bash
# Judge one dev matchup with the repo's evaluate_pairwise.py (DeepSeek, pairwise-v0.2).
# Usage: run_judge.sh <work-dir> <matchup> <run_id>
# Reads <work-dir>/eval/pairs/<matchup>.jsonl; writes results to <work-dir>/eval/pairwise/<run_id>/ and the call
# journal to <work-dir>/eval/pairwise_calls/<run_id>/. Re-running the same run_id resumes and retries failed orders.
set -u
W="$(cd "$1" && pwd)"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_deepseek.toml --pairs "$W/eval/pairs/$2.jsonl" --no-human-annotations \
  --output-dir "$W/eval/pairwise/$3" --calls-dir "$W/eval/pairwise_calls/$3" --run-id "$3"
