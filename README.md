# asrctl

A small Zig CLI that turns audio into text on macOS Apple Silicon, with
**two state-of-the-art ASR backends in a single 6 MB static binary**:

- **Qwen3-ASR-0.6B** (Alibaba 2026, multilingual SOTA, especially Chinese)
  via [llama.cpp](https://github.com/ggml-org/llama.cpp) + `mtmd`
- **Whisper-large-v3-turbo Q5_0** (OpenAI multilingual SOTA)
  via vendored [whisper.cpp](https://github.com/ggml-org/whisper.cpp)

Same CLI for both backends. Real-time microphone (`asrctl listen`), neural
silero VAD, llama-server fallback. No Python.

```sh
asrctl audio.wav                          # → stdout, default qwen3 backend
asrctl --backend whisper audio.wav        # use whisper-large-v3-turbo
asrctl audio.wav -o out.txt               # → file
asrctl --server-url http://127.0.0.1:8080 audio.wav   # qwen3 via llama-server
asrctl listen                             # mic → text stream (Ctrl-C to stop)
asrctl listen --backend whisper --vad silero  # full SOTA stack
```

EN: 5 s wav → ~1.2 s wall time. ZH: same. See [`bench/`](bench/).

## Status

- ✅ v0.1: wav transcription, in-process + llama-server fallback.
- ✅ v0.2: real-time microphone streaming via `asrctl listen`.
- ✅ v0.3: silero neural VAD via vendored whisper.cpp (`--vad silero`).
- ✅ v0.4: dual ASR backends — Qwen3-ASR + whisper-large-v3-turbo
  (`--backend qwen3|whisper`).
- See [`docs/REQ.md`](docs/REQ.md) for the full requirements & milestones.

## Backend comparison

| | Qwen3-ASR (default) | Whisper-large-v3-turbo |
| --- | --- | --- |
| Repo | `ggml-org/Qwen3-ASR-0.6B-GGUF` | `ggerganov/whisper.cpp` |
| Disk | ~1.5 GB (model + mmproj) | ~547 MB (Q5_0) |
| 5s wav (warm) | ~1.2 s | ~2.2 s |
| 17s wav (warm) | ~1.7 s | ~2.2 s |
| Strengths | Faster; Chinese; full-width punctuation | LM context recovery; less hallucination on phonetics |
| Server fallback | ✅ via `--server-url` | ❌ (no llama-server equivalent) |

## Limitations

- macOS Apple Silicon only. No Linux / Intel Mac / Windows.
- `.wav` only (`.mp3` / `.m4a` planned for v0.3+).
- `<asr_text>` output protocol is a Qwen3-ASR specific quirk; mtmd marks
  audio support as "experimental" upstream.

## Install

### Pre-built (recommended)

```sh
brew install agent-rt/tap/asrctl
```

This is a fully self-contained binary — only depends on macOS system
frameworks (Metal / Foundation / Accelerate). No `brew install llama.cpp`
required at runtime.

### Build from source

```sh
brew install zig cmake          # cmake compiles the bundled llama.cpp
git clone https://github.com/agent-rt/asrctl
cd asrctl
zig build -Doptimize=ReleaseFast
./zig-out/bin/asrctl --help
```

- **Zig** 0.16.0 (new `std.Io` API).
- **CMake** to compile the vendored llama.cpp into static libraries.
- **macOS Apple Silicon** (M1/M2/M3/M4) with Xcode Command Line Tools.
- `curl` for HuggingFace download (ships with macOS).

The first `zig build` takes ~5 minutes to compile llama.cpp + ggml. Subsequent
builds reuse the cmake cache.

### Build

```sh
git clone https://github.com/agent-rt/asrctl
cd asrctl
# v0.3 silero VAD requires whisper.cpp source vendored locally.
# (build.zig.zon would zig-fetch it, but whisper-vad's API surface is in the
#  monolithic whisper.cpp file, so we cmake-build the whole project.)
git clone --depth=1 https://github.com/ggml-org/whisper.cpp _vendor/whisper.cpp
zig build -Doptimize=ReleaseFast
./zig-out/bin/asrctl --help
```

### Pre-pull the model (optional)

```sh
asrctl model pull   # downloads Qwen3-ASR-0.6B-Q8_0.gguf + mmproj (~1.5 GB)
asrctl model path   # show resolved cache paths
```

By default the cache lives at `$HF_HOME` / `$XDG_CACHE_HOME/huggingface` /
`~/.cache/huggingface`, matching `huggingface_hub` conventions. If a Python
toolchain (or `llama-mtmd-cli -hf`) already populated that cache, asrctl
reuses the existing files — no re-download.

## Usage

```
asrctl <wav-file> [options]            transcribe a wav file
asrctl model path                      print resolved model path
asrctl model pull                      download model + mmproj from HF
asrctl version                         print version
asrctl help                            show this help

Transcribe options:
  -o, --output PATH    write text to file instead of stdout
      --model PATH     override model gguf path
      --server-url URL forward to a running llama-server instead of
                       loading the model in-process
      --threads N      CPU threads (default 4)
  -v, --verbose        print timing/diagnostic info to stderr

Environment:
  HF_HOME              HuggingFace cache root (default ~/.cache/huggingface)
  HF_ENDPOINT          HF mirror, e.g. https://hf-mirror.com
```

Exit codes: `0` ok / `1` user error / `2` internal / `3` inference / `4` server.

### Live microphone (`listen`)

```sh
asrctl listen                     # speak; Ctrl-C to stop
asrctl listen --vad silero        # neural VAD, better noise robustness
asrctl listen -v                  # see VAD + per-utterance timing on stderr
asrctl listen -o transcript.log   # append each utterance as a new line
asrctl listen --threshold 0.5 --silence-ms 500   # tune VAD for your env
```

VAD backends:
- **`energy`** (default): RMS threshold. Fast, zero deps. Quiet rooms only.
- **`silero`**: neural VAD via vendored whisper.cpp. Robust to fans / ambient
  noise. Auto-downloads the 885 KB silero-v5 model on first use. CPU-only;
  ~1 ms per 32 ms frame in ReleaseFast (negligible overhead).

Pipeline: 16 kHz mono PCM from CoreAudio's AudioQueue → energy-based VAD
(RMS threshold + silence-duration cut) → for each detected utterance, the
loaded model runs `mtmd_bitmap_init_from_audio` + `mtmd_helper_eval_chunks`
+ greedy sampling → text printed.

The model loads once on `listen` start; per-utterance latency is just the
ASR encode/decode (~0.3-0.6 s for short phrases). First run prompts macOS
for microphone permission.

Caveat: Qwen3-ASR is not natively streaming — partial words are not emitted
during speech. The listen mode segments by silence and transcribes the
finished utterance.

### Server fallback

If you are processing many files and want to amortize the model load + Metal
kernel JIT cost, run llama-server once and point asrctl at it:

```sh
llama-server -hf ggml-org/Qwen3-ASR-0.6B-GGUF --port 8765 &

for f in *.wav; do
  asrctl --server-url http://127.0.0.1:8765 "$f"
done
```

The server path is 3–5× faster per request because each fresh in-process run
re-pays the ~1 s fixed startup cost.

## Architecture

Single Zig binary, ~5 modules under [`src/`](src/):

- `main.zig` — entry, subcommand dispatch, stdout/stderr routing
- `cli.zig` — argument parser
- `hf.zig` — HuggingFace cache resolver + `curl` downloader
- `asr.zig` — in-process pipeline: load → mtmd → eval → sample → parse
- `server.zig` — HTTP fallback via `llama-server` `/v1/audio/transcriptions`
- `errors.zig` — wrap internal errors into human-readable diagnostics

`build.zig` shells out to `cmake` to compile the vendored llama.cpp + ggml +
mtmd into static `.a` files (one-time, cached), then links them statically
into asrctl. `GGML_METAL_EMBED_LIBRARY=ON` puts the Metal shader source
inline so we don't need to ship a separate `default.metallib`.

The transcription pipeline (in `asr.zig`) follows the same shape as
`llama-mtmd-cli`:

```
ggml_backend_load_all_from_path  →  llama_backend_init
  →  llama_model_load_from_file (n_gpu_layers=99)
  →  llama_init_from_model
  →  mtmd_init_from_file(mmproj, model)
  →  mtmd_helper_bitmap_init_from_file(wav)   # miniaudio handles wav decode
  →  prompt: "<|im_start|>user\nTranscribe.<__media__><|im_end|>..."
  →  mtmd_tokenize → mtmd_helper_eval_chunks
  →  greedy sampling loop until EOG
  →  parse "language X<asr_text>Y" output
```

## Benchmark

Apple M2 Pro, see [`bench/results-2026-05-02.txt`](bench/results-2026-05-02.txt).

| Sample | Audio | In-process | Via server (warm) |
| --- | --- | --- | --- |
| EN short | 5.4 s | ~1.2 s | ~0.3 s |
| EN long | 17.0 s | ~1.7 s | ~0.6 s |
| ZH short | 3.7 s | ~1.2 s | ~0.3 s |
| ZH long | 17.2 s | ~1.7 s | ~0.6 s |

Reproduce:

```sh
bench/bench.sh                                          # in-process
llama-server -hf ggml-org/Qwen3-ASR-0.6B-GGUF --port 8765 &
bench/bench.sh --server http://127.0.0.1:8765
```

## Project history

The `docs/` directory has the full design trail:

- [`REQ.md`](docs/REQ.md) — requirements + milestones
- [`M0-spike-report.md`](docs/M0-spike-report.md) — initial feasibility spike
- [`M2-report.md`](docs/M2-report.md) — first end-to-end transcription
- [`M3-report.md`](docs/M3-report.md) — CLI completion + HF cache
- [`M4-report.md`](docs/M4-report.md) — server fallback
- [`llama.cpp.zig-research.md`](docs/llama.cpp.zig-research.md) — reference repo notes
- [`MLX.zig-research.md`](docs/MLX.zig-research.md) — alternative path that wasn't taken

## License

MIT (matching upstream llama.cpp).
