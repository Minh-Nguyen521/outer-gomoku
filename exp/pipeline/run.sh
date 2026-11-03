#!/bin/bash
# Full chained pipeline: distil -> ExIt -> Gumbel AZ.
#   ./run.sh              # all three stages (skips completed ones)
#   ./run.sh 3            # gumbel only, seeded by whatever exists
#   ./run.sh 12           # distil + ExIt only
set -e
cd "$(dirname "$0")/../.."
python3 exp/pipeline/run.py --stage "${1:-123}" "${@:2}"
