#!/usr/bin/env bash
# Wall-clock benchmark for asrctl. Runs a fixed sample set, reports timings.
#
# Usage:
#   bench/bench.sh                           # in-process only
#   bench/bench.sh --server http://...:8080  # also run server path
#
# Each sample is run twice: cold (fresh process) + warm (re-run, kernel cache
# may help). Both runs are full processes so "warm" still pays Metal kernel
# JIT cost — informative not optimal.

set -euo pipefail

ASRCTL=${ASRCTL:-./zig-out/bin/asrctl}
SAMPLES_DIR=${SAMPLES_DIR:-samples}
SERVER_URL=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --server)
      SERVER_URL=$2
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x $ASRCTL ]]; then
  echo "binary $ASRCTL not found, run 'zig build' first" >&2
  exit 1
fi

# millis() — wall time in ms via /usr/bin/python3 (always available on macOS).
millis() { /usr/bin/python3 -c 'import time; print(int(time.time()*1000))'; }

# duration_ms <wav> — return clip duration in ms via afinfo.
duration_ms() {
  local f=$1
  local secs
  secs=$(afinfo "$f" 2>/dev/null | awk '/estimated duration/ {print $3}')
  /usr/bin/python3 -c "print(int(${secs}*1000))"
}

run_once() {
  local label=$1
  shift
  local t0 t1
  t0=$(millis)
  local out
  out=$("$@" 2>/dev/null)
  t1=$(millis)
  local elapsed=$((t1 - t0))
  printf '%-22s  %5d ms   %s\n' "$label" "$elapsed" "$out"
}

run_sample() {
  local wav=$1
  local label=$2
  local dur
  dur=$(duration_ms "$wav")
  echo
  echo "$label  ($wav, ${dur} ms audio)"
  run_once "  qwen3   cold"  "$ASRCTL" --backend qwen3   "$wav"
  run_once "  qwen3   warm"  "$ASRCTL" --backend qwen3   "$wav"
  run_once "  whisper cold"  "$ASRCTL" --backend whisper "$wav"
  run_once "  whisper warm"  "$ASRCTL" --backend whisper "$wav"
  if [[ -n $SERVER_URL ]]; then
    run_once "  qwen3   server cold" "$ASRCTL" --server-url "$SERVER_URL" "$wav"
    run_once "  qwen3   server warm" "$ASRCTL" --server-url "$SERVER_URL" "$wav"
  fi
}

echo "asrctl bench  ($(date))"
echo "binary: $ASRCTL"
[[ -n $SERVER_URL ]] && echo "server: $SERVER_URL"

run_sample "$SAMPLES_DIR/en_short.wav" "EN short"
run_sample "$SAMPLES_DIR/en_long.wav"  "EN long"
run_sample "$SAMPLES_DIR/zh_short.wav" "ZH short"
run_sample "$SAMPLES_DIR/zh_long.wav"  "ZH long"
