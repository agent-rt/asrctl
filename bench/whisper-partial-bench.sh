#!/usr/bin/env bash
# Compare full-context vs audio_ctx-scaled whisper inference on the same wav.
# We can't easily call transcribePCMQuick from a CLI flag (it's a partial-only
# code path), but we can approximate by running whisper backend on the same
# wav with whisper.cpp's `whisper-cli --audio-ctx` if available, OR by
# inserting a debug branch. For now this script just measures the overall
# baseline so the v0.5/v0.6 reports have something to compare against.

set -euo pipefail
ASRCTL=${ASRCTL:-./zig-out/bin/asrctl}

# Force a "live preview" by truncating long sample to 3s (typical partial buffer length).
# Use ffmpeg to clip the long EN sample.
mkdir -p bench/clips
ffmpeg -y -loglevel error -i samples/en_long.wav -t 3 -ar 16000 -ac 1 bench/clips/en_3s.wav

millis() { /usr/bin/python3 -c 'import time; print(int(time.time()*1000))'; }

run() {
    local label=$1
    shift
    local total=0 best=999999
    for i in 1 2 3; do
        local t0=$(millis)
        "$@" >/dev/null 2>&1
        local elapsed=$(($(millis) - t0))
        total=$((total + elapsed))
        if [[ $elapsed -lt $best ]]; then best=$elapsed; fi
    done
    local avg=$((total / 3))
    printf "%-30s avg=%d ms  best=%d ms\n" "$label" "$avg" "$best"
}

# These are macroscopic process measurements (model load + Metal JIT + actual
# inference + teardown). They give an upper bound on partial-call latency from
# the user's perspective, even though our in-process partial loop avoids
# the model-load / JIT cost (paid once at listen start).
run "qwen3 / 3s wav"   "$ASRCTL" --backend qwen3   bench/clips/en_3s.wav
run "whisper / 3s wav" "$ASRCTL" --backend whisper bench/clips/en_3s.wav
run "qwen3 / 17s wav"   "$ASRCTL" --backend qwen3   samples/en_long.wav
run "whisper / 17s wav" "$ASRCTL" --backend whisper samples/en_long.wav
