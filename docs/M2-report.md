# M2 报告

> 进程内推理：wav → mtmd → llama → 文本，端到端跑通固定文件。
> 状态：✅ 通过（2026-05-01）。

## 输出验证

```
$ ./zig-out/bin/asrctl /tmp/asrctl-spike/test_en.wav
--- text ---
The quick brown fox jumps over the lazy dog. Hello, world. This is a transcription test.

$ ./zig-out/bin/asrctl /tmp/asrctl-spike/test_zh.wav
--- text ---
今天天气真不错，我们去公园散步吧。
```

EN/ZH 输出与 M0 spike（用 upstream `llama-mtmd-cli`）逐字一致。

## 性能

| 测试 | 音频长度 | wall time |
| --- | --- | --- |
| EN | ~6 s | 1.48 s |
| ZH | ~4 s | 1.09 s |

模型 load + Metal kernel JIT 预热占大头（首跑尤其明显，M2 暂未做 warmup 缓存）。REQ.md §5 目标 30 s 音频 ≤ 5 s 远超达标。

## 实现要点

调用顺序（`src/main.zig`）：

1. `ggml_backend_load_all_from_path("/opt/homebrew/Cellar/ggml/0.10.1/libexec")` — **必须**，否则 `llama_model_load_from_file` 报 `no backends are loaded`。新 ggml 把 backend 拆成动态加载的 `.so`。
2. `llama_backend_init` → `llama_model_load_from_file` (`n_gpu_layers=99`) → `llama_init_from_model` (`n_ctx=4096`)。
3. `mtmd_init_from_file(mmproj_path, model, params)`，参数 `use_gpu=true`、`warmup=false`、`n_threads=4`。`mtmd_support_audio` 校验。
4. `mtmd_helper_bitmap_init_from_file(mctx, wav_path)` — **不需要自己解 wav**，mtmd 通过 miniaudio 直接吃 wav/mp3/flac。
5. Prompt：`<|im_start|>user\nTranscribe the audio.<__media__><|im_end|>\n<|im_start|>assistant\n`，`<__media__>` 是 `mtmd_default_marker()` 返回的占位符。
6. `mtmd_tokenize` 把 prompt + bitmap 拆成 chunks（text-before / audio / text-after）。
7. `mtmd_helper_eval_chunks(..., logits_last=true, ...)` 自动跑 `llama_decode`(text) + `mtmd_encode`+`llama_decode`(audio)，更新 `n_past`。
8. 采样循环：`llama_sampler_init_greedy`（ASR 要确定性），逐 token `llama_sampler_sample` → `llama_decode` 直到 `llama_vocab_is_eog`。
9. Output protocol parse：从生成文本里抓 `<asr_text>` 标签后的部分。

## 偏离 REQ.md 的设计决定

| 原计划 | 实际 | 理由 |
| --- | --- | --- |
| 自己写 RIFF/WAVE 解析（PCM16 + float32 + mono 下混 + 线性重采样） | 用 `mtmd_helper_bitmap_init_from_file` | mtmd 已内置 miniaudio，单函数搞定，省几百行代码。CLI 层加 `.wav` 后缀校验保住「MVP 仅支持 wav」语义。代价：用户把 mp3 改名 .wav 也能跑（不是真校验，是 UX 提示）。 |
| 通过 `--threads N` 透传 | M2 写死 `n_threads=4` | M3 接 CLI flag 时再连。 |

## 已知遗留问题（M3 处理）

1. **模型路径硬编码**到 HF cache 的具体 sha snapshot — M3 做 HF 路径解析。
2. **错误退出有 Zig stack trace** — M3 改 `std.process.exit(code)` 吃掉 trace。
3. **无 stdout 控制** — 所有输出走 `std.debug.print` 进 stderr，M3 改成纯文本到 stdout、诊断信息到 stderr，并支持 `-o`。
4. **ggml backend dir 硬编码**到 `0.10.1` — M3 用 `glob` 或 brew prefix 探测，或 M5 vendor 源码彻底解决。
5. **Prompt 写死英文** "Transcribe the audio." — Qwen3-ASR 自动检测语言，工作正常；但 `--language zh` 之类的 hint 暂未实现。
6. **没解析 `language XYZ` 前缀** — 当前 raw 输出是 `language English<asr_text>...`，可能想把 language 也作为元数据返回。

## 工程指标

```
$ ls -la zig-out/bin/asrctl
.rwxr-xr-x 2.0M  asrctl     # debug build
$ otool -L zig-out/bin/asrctl
        /opt/homebrew/opt/llama.cpp/lib/libllama.0.dylib
        /opt/homebrew/opt/llama.cpp/lib/libmtmd.0.dylib
        /opt/homebrew/opt/ggml/lib/libggml.0.dylib
        /opt/homebrew/opt/ggml/lib/libggml-base.0.dylib
        /usr/lib/libSystem.B.dylib
```

仍是 dylib 链接（M5 解决静态化）。binary 本身 2 MB，远低于 REQ.md ≤ 50 MB 目标，不过那是 ReleaseFast 静态后才有意义的指标。

## 决定

**进 M3**：CLI 完整化（参数解析、错误处理、HF 路径解析、stdout 输出协议）。
