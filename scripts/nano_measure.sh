#!/usr/bin/env bash
# One measurement session on the Nano: pause the shared services, run the given
# command, then always restart the services, even if the command fails.
# Runs detached so a dropped SSH connection cannot leave the services down:
#   nohup bash scripts/nano_measure.sh "uv run memrag models probe" > /tmp/rag_measure.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "== $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg)"
bash scripts/nano_services.sh stop
trap 'echo "== $(date -Is) restarting services"; bash scripts/nano_services.sh start' EXIT
trap 'exit 130' INT TERM   # make the EXIT trap run when the session is killed

for cmd in "$@"; do
  echo "== $(date -Is) run: $cmd"
  bash -c "$cmd"
  echo "== exit $?"
done
