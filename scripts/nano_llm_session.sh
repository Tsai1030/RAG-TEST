#!/usr/bin/env bash
# One summarization session on the Nano: pause the shared services, start our own
# llama-server (model, port and context from configs/pipeline.yaml), run the given
# commands, then always stop our server and restart the services.
# Runs detached so a dropped SSH connection cannot leave the services down:
#   nohup bash scripts/nano_llm_session.sh "uv run memrag corpus build" > /tmp/rag_llm.log 2>&1 &
# Answer sessions need less context than summarizing (smaller KV cache):
#   LLM_CTX=4096 nohup bash scripts/nano_llm_session.sh "uv run memrag eval answers --device cpu" > ... &
# Memory guard: the Nano's 8 GB is shared by CPU and GPU and can freeze for hours before the
# OOM killer acts, so a command is stopped (and the session ends) when MemAvailable drops below MIN_MEM_MB.
# --cache-ram 0: this llama-server keeps past prompts' KV state in RAM (default up to 8192 MiB,
# more than the Nano has); memrag's own response cache already covers repeated prompts.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/stt-test/services/llama_cpp/llama.cpp/build/bin/llama-server}"
MIN_MEM_MB="${MIN_MEM_MB:-500}"
cfg() { awk -v k="$1" '$1 == k":" {print $2; exit}' configs/pipeline.yaml; }
mem_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
FILE=$(cfg file)
CTX="${LLM_CTX:-$(cfg context)}"
PORT=$(cfg base_url | sed 's/.*://')
MODEL=$(ls .hf_cache/hub/models--*/snapshots/"$(cfg revision)"/"$FILE" 2>/dev/null | head -1)
[[ -f "$MODEL" ]] || { echo "model $FILE not found under .hf_cache"; exit 1; }

echo "== $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg)"
bash scripts/nano_services.sh stop
LLM_PID=""
CMD_PID=""
cleanup() {
  [[ -n "$CMD_PID" ]] && kill -KILL -- -"$CMD_PID" 2>/dev/null
  echo "== $(date -Is) stopping llama-server :$PORT"
  [[ -n "$LLM_PID" ]] && kill "$LLM_PID" 2>/dev/null && wait "$LLM_PID" 2>/dev/null
  echo "== $(date -Is) restarting services"
  bash scripts/nano_services.sh start
}
trap cleanup EXIT
trap 'exit 130' INT TERM   # make the EXIT trap run when the session is killed

mkdir -p results/service_logs
MEM_BEFORE=$(mem_mb)
START=$(date +%s.%N)
"$LLAMA_SERVER" -m "$MODEL" -ngl 99 -c "$CTX" -np 1 --cache-ram 0 --reasoning off --host 127.0.0.1 --port "$PORT" \
  > results/service_logs/rag_llm_server.log 2>&1 &
LLM_PID=$!
for _ in $(seq 900); do curl -s "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 0.2; done
curl -s "http://127.0.0.1:$PORT/health" | grep -q ok || { echo "llama-server did not start"; exit 1; }
# Load time and memory for `memrag profile pipeline` (load time has 0.2 s resolution)
export RAG_LLM_LOAD_S=$(awk -v s="$START" -v e="$(date +%s.%N)" 'BEGIN {printf "%.1f", e - s}')
export RAG_LLM_MEM_BEFORE_MIB=$MEM_BEFORE RAG_LLM_MEM_AFTER_MIB=$(mem_mb) RAG_LLM_CTX=$CTX
echo "== $(date -Is) llama-server ready in ${RAG_LLM_LOAD_S} s: $FILE ctx=$CTX; MemAvailable ${MEM_BEFORE} -> ${RAG_LLM_MEM_AFTER_MIB} MiB"

for cmd in "$@"; do
  echo "== $(date -Is) run: $cmd"
  setsid bash -c "$cmd" &   # own process group, so the guard can stop all of it
  CMD_PID=$!
  low_mem=""
  min_mem=$(mem_mb)
  while kill -0 "$CMD_PID" 2>/dev/null; do
    mem=$(mem_mb)
    (( mem < min_mem )) && min_mem=$mem
    if (( mem < MIN_MEM_MB )); then
      echo "== $(date -Is) MemAvailable ${mem} MiB < ${MIN_MEM_MB} MiB, stopping the command"
      kill -TERM -- -"$CMD_PID" 2>/dev/null
      sleep 5
      kill -KILL -- -"$CMD_PID" 2>/dev/null
      low_mem=1
      break
    fi
    sleep 1
  done
  wait "$CMD_PID"
  echo "== exit $? (MemAvailable min ${min_mem} MiB)"
  CMD_PID=""
  [[ -n "$low_mem" ]] && { echo "== session ended: low memory"; exit 3; }
done
