#!/usr/bin/env bash
# One summarization session on the Nano: pause the shared services, start our own
# llama-server (model, port and context from configs/pipeline.yaml), run the given
# commands, then always stop our server and restart the services.
# Runs detached so a dropped SSH connection cannot leave the services down:
#   nohup bash scripts/nano_llm_session.sh "uv run memrag corpus build" > /tmp/rag_llm.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/stt-test/services/llama_cpp/llama.cpp/build/bin/llama-server}"
cfg() { awk -v k="$1" '$1 == k":" {print $2; exit}' configs/pipeline.yaml; }
FILE=$(cfg file)
CTX=$(cfg context)
PORT=$(cfg base_url | sed 's/.*://')
MODEL=$(ls .hf_cache/hub/models--*/snapshots/"$(cfg revision)"/"$FILE" 2>/dev/null | head -1)
[[ -f "$MODEL" ]] || { echo "model $FILE not found under .hf_cache"; exit 1; }

echo "== $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg)"
bash scripts/nano_services.sh stop
LLM_PID=""
cleanup() {
  echo "== $(date -Is) stopping llama-server :$PORT"
  [[ -n "$LLM_PID" ]] && kill "$LLM_PID" 2>/dev/null && wait "$LLM_PID" 2>/dev/null
  echo "== $(date -Is) restarting services"
  bash scripts/nano_services.sh start
}
trap cleanup EXIT
trap 'exit 130' INT TERM   # make the EXIT trap run when the session is killed

mkdir -p results/service_logs
"$LLAMA_SERVER" -m "$MODEL" -ngl 99 -c "$CTX" -np 1 --reasoning off --host 127.0.0.1 --port "$PORT" \
  > results/service_logs/rag_llm_server.log 2>&1 &
LLM_PID=$!
for _ in $(seq 180); do curl -s "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 1; done
curl -s "http://127.0.0.1:$PORT/health" | grep -q ok || { echo "llama-server did not start"; exit 1; }
echo "== $(date -Is) llama-server ready: $FILE ctx=$CTX; MemAvailable $(awk '/MemAvailable/ {printf "%.2f GiB", $2/1048576}' /proc/meminfo)"

for cmd in "$@"; do
  echo "== $(date -Is) run: $cmd"
  bash -c "$cmd"
  echo "== exit $?"
done
