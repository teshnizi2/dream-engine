#!/bin/bash
# Submit N chained resumable training jobs: bash alice/submit_chain.sh <RUN> <N> [extra sbatch --export vars...]
# Each job starts only when the previous one has ended (afterany). Jobs after the run completes exit immediately.
set -euo pipefail
cd "$HOME/dream-engine"
mkdir -p logs
RUN=${1:?run name}; N=${2:?number of jobs}; shift 2
EXPORTS="ALL,RUN=$RUN"
for kv in "$@"; do EXPORTS="$EXPORTS,$kv"; done
prev=""
for i in $(seq 1 "$N"); do
  if [ -z "$prev" ]; then
    jid=$(sbatch --parsable --export="$EXPORTS" alice/train.slurm)
  else
    jid=$(sbatch --parsable --dependency=afterany:"$prev" --export="$EXPORTS" alice/train.slurm)
  fi
  echo "submitted job $i/$N: $jid"
  prev=$jid
done
