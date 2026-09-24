#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=${1:-qire_day}
HOURS=${2:-12}
CMD="source .venv/bin/activate && python qire_ai_console.py --preset day_research --profile oracle_free --time-budget-hours ${HOURS} --ram-mode throttle --export-every 50"
tmux new-session -d -s "$SESSION" "$CMD"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
