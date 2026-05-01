# asrctl

A small Zig CLI that turns `.wav` audio into text on Apple Silicon, using
[Qwen3-ASR-0.6B](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF) via
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `mtmd` (multimodal) module
on the Metal backend.

```sh
asrctl audio.wav            # → stdout
asrctl audio.wav -o out.txt # → file
asrctl --server-url http://127.0.0.1:8080 audio.wav   # forward to llama-server
```

EN: 5 s wav → ~1.2 s wall time. ZH: same. See [`bench/`](bench/).

## Status

- ✅ MVP (v0.1): wav transcription, in-process + llama-server fallback.
- 🚧 v0.2: real-time microphone streaming (planned).
- See [`docs/REQ.md`](docs/REQ.md) for the full requirements & milestones.

## Limitations

- macOS Apple Silicon only. No Linux / Intel Mac / Windows.
- `.wav` only (`.mp3` / `.m4a` planned for v0.3+).
- Currently dynamically linked against `brew install llama.cpp` and `ggml`.
  M5.5 (static vendoring) is in progress; binary is **not** yet self-contained.
- `<asr_text>` output protocol is a Qwen3-ASR specific quirk; mtmd marks
  audio support as "experimental" upstream.

## Install

### Prerequisites

```sh
brew install zig llama.cpp ggml
```

- Zig **0.16.0** or newer (uses the new `std.Io` API).
- llama.cpp **8990** or newer (need PR #19441 for Qwen3 ASR `qwen3a` projector).
- macOS Apple Silicon (M1/M2/M3/M4).
- `curl` for HuggingFace download (ships with macOS).

### Build

```sh
git clone <this repo>
cd asrctl
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
- `backend.zig` — discover the active brew `libexec/` for ggml backend dylibs
- `errors.zig` — wrap internal errors into human-readable diagnostics

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
