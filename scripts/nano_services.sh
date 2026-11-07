#!/usr/bin/env bash
# Pause and resume the shared services on the Nano for measurements (plan D8).
# Runs on the Nano:  bash scripts/nano_services.sh stop|start|status
#
# stop  saves each running service's command line and working folder, then ends it
#       (SIGSTOP would not free memory).
# start restarts exactly the saved command from the same folder and waits until
#       the service answers. Only services this script stopped are started.
set -uo pipefail

STATE_DIR="$HOME/50019/.service_state"
LOG_DIR="$HOME/50019/rag_test/results/service_logs"
SERVICES=(llama-server pocket_tts_server)

find_pid() {
  case "$1" in
    llama-server)
      pgrep -f "llama.cpp/build/bin/llama-server .*--port 8090" | head -1 ;;
    pocket_tts_server)
      for p in $(pgrep -f "python server.py"); do
        [[ "$(readlink "/proc/$p/cwd")" == "$HOME/stt-test/services/pocket_tts_server" ]] && echo "$p" && break
      done ;;
  esac
}

is_ready() {
  case "$1" in
    llama-server) [[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/health)" == "200" ]] ;;
    pocket_tts_server) ss -ltn 2>/dev/null | grep -q '127.0.0.1:8092 ' ;;
  esac
}

mem_available() { awk '/MemAvailable/ {printf "%.2f GiB", $2/1048576}' /proc/meminfo; }

status() {
  for s in "${SERVICES[@]}"; do
    pid=$(find_pid "$s")
    if [[ -n "$pid" ]]; then
      echo "$s: running pid=$pid rss=$(( $(awk '/VmRSS/ {print $2}' "/proc/$pid/status") / 1024 )) MiB"
    elif [[ -f "$STATE_DIR/$s.cmdline" ]]; then
      echo "$s: stopped by this script"
    else
      echo "$s: not running"
    fi
  done
  echo "MemAvailable: $(mem_available)"
}

stop() {
  mkdir -p "$STATE_DIR"
  for s in "${SERVICES[@]}"; do
    pid=$(find_pid "$s")
    [[ -z "$pid" ]] && { echo "$s: not running, skipped"; continue; }
    cp "/proc/$pid/cmdline" "$STATE_DIR/$s.cmdline"
    readlink "/proc/$pid/cwd" > "$STATE_DIR/$s.cwd"
    readlink "/proc/$pid/fd/1" > "$STATE_DIR/$s.stdout"
    kill -TERM "$pid"
    for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
    echo "$s: stopped (pid $pid)"
  done
  echo "MemAvailable: $(mem_available)"
}

start() {
  mkdir -p "$LOG_DIR"
  local failed=0
  for s in "${SERVICES[@]}"; do
    [[ -f "$STATE_DIR/$s.cmdline" ]] || { echo "$s: not stopped by this script, skipped"; continue; }
    if [[ -n "$(find_pid "$s")" ]]; then echo "$s: already running"; rm -f "$STATE_DIR/$s".*; continue; fi
    mapfile -d '' -t cmd < "$STATE_DIR/$s.cmdline"
    out=$(cat "$STATE_DIR/$s.stdout")
    [[ -f "$out" ]] || out="$LOG_DIR/$s.log"
    (cd "$(cat "$STATE_DIR/$s.cwd")" && setsid nohup "${cmd[@]}" >> "$out" 2>&1 < /dev/null &)
    for _ in $(seq 180); do is_ready "$s" && break; sleep 1; done
    if is_ready "$s"; then
      echo "$s: ready (log $out)"
      rm -f "$STATE_DIR/$s".*
    else
      echo "$s: NOT ready after 180 s, see $out"; failed=1
    fi
  done
  echo "MemAvailable: $(mem_available)"
  return $failed
}

case "${1:-status}" in
  stop) stop ;;
  start) start ;;
  status) status ;;
  *) echo "usage: $0 stop|start|status"; exit 2 ;;
esac
