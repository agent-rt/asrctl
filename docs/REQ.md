# asrctl 需求文档

> 一个 macOS 单二进制 ASR CLI 工具，把音频转成文字。
> 调研：[llama.cpp.zig-research.md](./llama.cpp.zig-research.md) / [MLX.zig-research.md](./MLX.zig-research.md)

## 版本规划总览

| 阶段 | 范围 | 状态 |
| --- | --- | --- |
| **MVP (v0.1)** | 本地 `.wav` 文件 → 文本，进程内 + llama-server 备用 | 当前文档 |
| **v0.2** | 实时音频（麦克风流式 ASR） | ✅ 已通过（2026-05-02），见 [v0.2-report.md](./v0.2-report.md) |
| **v0.3+** | mp3/m4a/flac 等额外格式、字幕输出、Linux 支持 | 未承诺 |

## 1. 目标（MVP）

提供一条命令把 wav 文件转录成文本，无 Python、无运行时依赖、离线可用：

```sh
asrctl transcribe input.wav            # → stdout 输出文本
asrctl transcribe input.wav -o out.txt # → 写文件
```

## 2. 范围（MVP）

### In scope

- 平台：**macOS Apple Silicon (aarch64)**，Metal 后端。
- 输入：本地 `.wav` 文件（PCM 16-bit / 32-bit float，mono 优先；多声道下混到 mono）。
- 输出：纯文本到 stdout 或 `-o` 指定文件。
- 模型：`ggml-org/Qwen3-ASR-0.6B-GGUF`，本地缓存。
- 单二进制分发（macOS 必须随发 `default.metallib`，作为已知约束）。
- 进程内推理为主路径；`llama-server` 远程模式为备用路径。

### Out of scope（MVP 不做，分别归属未来版本）

- **实时麦克风流式 / WebSocket** → **v0.2 第二期**。
- mp3 / m4a / flac / ogg / opus 等额外音频格式 → 未来版本。
- Linux / Windows / Intel mac → 未来版本。
- 说话人分离 / 时间戳对齐 / 字幕格式（srt/vtt）→ 未承诺。
- 多并发 server 化（直接复用 upstream `llama-server`）。
- GUI。

## 3. 技术方案

### 主路径：嵌入 llama.cpp

- 用 Zig `build.zig` 把 upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) 作为依赖编译进来，参考 [diogok/llama.cpp.zig](https://github.com/diogok/llama.cpp.zig) 的构建脚本，但**不直接依赖该仓库**（其只有 5 stars 单作者，自己 vendor 一份 build 脚本更稳）。
- 链接 `llama_cpp` + `mtmd`（multimodal 模块，处理音频输入）+ Metal/Foundation/QuartzCore/Accelerate framework。
- 通过 `zig translate-c` 把 `llama.h` 和 `mtmd.h` 暴露为 Zig 模块直接调用。
- 构建命令：`zig build -Dbackend=metal -Doptimize=ReleaseFast`。

### 备用路径：llama-server HTTP 客户端

- CLI 提供 `--server-url=http://127.0.0.1:8080` flag。
- 命中时改走 `POST /v1/audio/transcriptions`（OpenAI 兼容），不在进程内做推理。
- 用户可用同一份 GGUF 启动 upstream `llama-server`，asrctl 退化为 HTTP 客户端。
- 切换由 CLI flag 控制，无运行时自动 fallback（明确显式优于隐式）。

### 音频解码

- **MVP 只解 wav**：直接读 RIFF/WAVE 头，支持 PCM 16-bit 和 32-bit float，多声道下混到 mono。代码量很小，不引入 `dr_wav`，避免无谓依赖。
- 解码后输出 mono 16kHz f32 PCM 给 `mtmd`（具体格式以 mtmd API 实际要求为准，spike 阶段确认）。
- 输入采样率不是 16kHz 时：MVP 用最简线性重采样保证可用；性能不达标再换 `Accelerate.framework` 的 vDSP。
- mp3 / m4a 等格式延后到未来版本（届时引入 `minimp3` 等单头库）。

### 模型管理

- **从 HuggingFace 自动下载**：默认仓库 `ggml-org/Qwen3-ASR-0.6B-GGUF`，按量化档位选具体文件（spike 后定，候选 `Q4_K_M` / `Q8_0`）。
- 缓存路径：复用 HuggingFace 标准约定 `~/.cache/huggingface/hub/`，避免与已有 Python 工具链重复下载；asrctl 自身不再维护独立 cache 目录。
  - 解析顺序：`HF_HOME` 环境变量 > `XDG_CACHE_HOME/huggingface` > `~/.cache/huggingface`，与 huggingface_hub 行为对齐。
- 下载实现：MVP **直接用 HTTPS GET** 拉 `https://huggingface.co/{repo}/resolve/main/{file}`，断点续传可选（`Range` 头），不依赖 Python `huggingface_hub`。
  - mtmd 模型可能需要两个文件（主权重 + mmproj 视觉/音频 projector），下载逻辑要支持多文件。
  - 失败处理：网络错误显式报错并提示重试 / 用 `--model PATH` 指向已下载文件 / 设置 `HF_ENDPOINT` 走镜像。
- 镜像支持：尊重 `HF_ENDPOINT` 环境变量（如国内用户可设 `https://hf-mirror.com`）。
- 私有/受限模型：MVP **不支持**，因为 Qwen3-ASR-GGUF 是公开仓库；如需要再接 `HF_TOKEN`。
- 覆盖：`--model PATH` 直接指向本地 GGUF 文件，跳过下载/缓存解析。
- 子命令辅助：
  - `asrctl model path` — 打印当前解析路径（已下载或将下载到的位置）。
  - `asrctl model pull` — 仅下载不推理，方便 CI / 离线分发场景预热缓存。

## 4. CLI 接口（MVP 草案）

```
asrctl transcribe <wav-file> [options]
  -o, --output PATH       输出到文件（默认 stdout）
      --model PATH        模型 GGUF 路径
      --server-url URL    走 llama-server，不在进程内推理
      --language LANG     提示语言（zh/en/...）；不传则自动
      --threads N         CPU 线程数（默认自动）
  -v, --verbose           打印推理统计
  -h, --help

asrctl version            打印版本 + llama.cpp commit
asrctl model path         打印当前模型解析路径
asrctl model pull         从 HF 下载默认模型到缓存
```

环境变量：

- `HF_HOME` / `XDG_CACHE_HOME` — 模型缓存根目录，遵循 huggingface_hub 约定。
- `HF_ENDPOINT` — 自定义 HF 镜像（如 `https://hf-mirror.com`）。
- `HF_TOKEN` — 预留给未来私有模型，MVP 不使用。

输入文件后缀必须为 `.wav`，否则报错并提示「MVP 仅支持 wav，其他格式见 v0.3+」。

退出码：`0` 成功 / `1` 用户输入错误 / `2` 模型缺失 / `3` 推理失败 / `4` 网络/server 错误。

## 5. 非功能需求

| 项 | 目标 |
| --- | --- |
| 冷启动到首字延迟 | 30 秒音频 ≤ 3 秒（M1 及以上） |
| 二进制大小 | ≤ 50 MB（不含模型，不含 metallib） |
| 内存峰值 | ≤ 2 GB（0.6B Q4/Q8 量化档） |
| 离线可用 | 主路径完全离线，仅备用路径需要本地 server |
| 错误信息 | 失败时打印人类可读原因，不打印 C 栈 |

## 6. 已知风险与前置验证

**spike 0（开工前必须验证）**：upstream llama.cpp 当前是否能用 mtmd 吃 `Qwen3-ASR-0.6B-GGUF` 跑出文字？

- 验证方式：本机装 upstream llama.cpp，用其 multimodal CLI 跑一段 mp3 → 看是否输出合理文本。
- 失败处理：如果 upstream 还没接入，**整个方案要重估**——选项是等 upstream / 切回 MLX 路线 / 用 whisper.cpp + Qwen3-ASR 不可得时退到 whisper-large-v3-turbo。

**其他风险**：

- llama.cpp upstream API 不稳定 → 通过 `build.zig.zon` pin commit 缓解。
- Metal toolchain 必须本地存在（`xcodebuild -downloadComponent MetalToolchain`），README 里明确写出。
- `default.metallib` 必须随 binary 分发，不是真"单文件"——MVP 接受，未来版本可考虑 `@embedFile` 内嵌。
- mtmd 的音频输入格式（采样率 / 通道 / 量化）以 spike 实测为准，本文档规格留待 spike 后回填。

## 7. 验收标准（MVP）

- [ ] `asrctl transcribe samples/zh.wav` 输出可读中文，与 Python 参考实现差异在 token 级 ≤ 5%。
- [ ] `asrctl transcribe samples/en.wav` 同上，英文。
- [ ] 输入非 wav 文件时给出明确报错，不 crash。
- [ ] 模型不存在时自动从 HF 下载并缓存；无网络时给出明确报错。
- [ ] `asrctl model pull` 可在无 Python 环境下成功下载并复用 huggingface 标准缓存。
- [ ] 设置 `HF_ENDPOINT` 时走镜像下载。
- [ ] `--server-url` 能切到 `llama-server` 并得到一致输出。
- [ ] `zig build -Dbackend=metal` 在干净的 mac 上一次成功（前提：Zig + Xcode CLT + Metal toolchain）。
- [ ] 30 秒音频从启动到出全部文字 ≤ 5 秒（M1）。

## 8. 里程碑（MVP）

1. **M0 — Spike**：✅ 已通过（2026-05-01），见 [M0-spike-report.md](./M0-spike-report.md)。结论：llama.cpp ≥ 8990 + `-hf ggml-org/Qwen3-ASR-0.6B-GGUF` + `--audio` 能直接出文字，6s 英文音频端到端 1.4s，逐字一致。
2. **M1 — 构建骨架**：✅ 已通过（2026-05-01）。Zig 0.16.0 + `build.zig` 链 brew dylib (`libllama` / `libmtmd` / `libggml*`)，translate-c 暴露 `llama.h` + `mtmd.h`，hello world 调通 `llama_backend_init` / `mtmd_context_params_default`。**注意**：当前是动态链 brew，不是真单二进制；M5 再决定 vendor 源码做静态。Metal toolchain 缺失也不影响（brew 已编好 metallib）。
3. **M2 — 进程内推理**：✅ 已通过（2026-05-01）。`asrctl <wav>` 跑通 EN/ZH，输出与 M0 逐字一致，wall time 1.1-1.5s。**注意**：原计划自己写 RIFF/WAVE 解析，实际改用 `mtmd_helper_bitmap_init_from_file`（mtmd 内置 miniaudio），CLI 层做 `.wav` 后缀校验保住 REQ 语义。需要在 `llama_backend_init` 前调 `ggml_backend_load_all_from_path("/opt/homebrew/Cellar/ggml/<ver>/libexec")` 加载 metal/cpu/blas 后端 dylib（M5 vendor 源码时改成自动）。
4. **M3 — CLI 完整**：✅ 已通过（2026-05-01）。参数解析（`-o`/`--model`/`--threads`/`-v`）、subcmd（`transcribe`/`model path`/`model pull`/`version`/`help`）、HF cache 解析（双 `--` 分隔、复用 huggingface_hub 标准 layout）、curl 子进程下载、stdout/stderr 分离（文本到 stdout、log/诊断到 stderr）、`std.process.exit` 干净退出。代码拆 5 文件：`main.zig` / `cli.zig` / `hf.zig` / `asr.zig` / `backend.zig`。
5. **M4 — 备用路径**：✅ 已通过（2026-05-02）。`--server-url URL` 命中时短路本地推理，POST wav 到 `<URL>/v1/audio/transcriptions`（OpenAI 兼容），用 curl 子进程，写临时文件读 JSON `text` 字段，过同一份 `asr.parseOutput` → `<asr_text>` 标签解析，输出与主路径逐字一致。错误退出码 4。
6. **M5 — 打磨**：✅ 通过（2026-05-02）。
   - M5.1 友好错误信息：✅ `errors.zig` 集中映射，所有路径过 `errors.print`
   - M5.2 性能 bench：✅ `bench/bench.sh` + `bench/results-2026-05-02.txt`，server 路径 3-5x 快于进程内
   - M5.3 README：✅
   - M5.4 lint：✅ zig fmt 通过
   - M5.5 vendor 静态化：✅ 通过（2026-05-02），见 [M5.5-report.md](./M5.5-report.md)。换思路 cmake-shellout 取代手写 build.zig，1 小时完成（spike 估 3-4 天）。`otool -L` 只剩系统 framework，二进制 9.3 MB，性能与 dylib 版完全一致。`GGML_METAL_EMBED_LIBRARY=ON` 让 shader 源码内嵌运行时编译，无需 metallib 文件。

## 9. 待决事项

- [x] 量化档位：**Q8_0**（M0 spike 实测精度逐字一致，速度也充裕，不必上 Q4）。
- [ ] mtmd 的 Zig API 是直接 translate-c 调，还是包一层 idiomatic Zig wrapper？倾向先直接调，跑通再说。
- [ ] 是否需要 `--format json` 输出（含置信度 / 时间戳）？MVP 默认不做，看 mtmd 是否原生暴露。
- [ ] 重采样质量：MVP 的线性重采样在 8kHz/44.1kHz/48kHz 输入下精度是否够？不够就要把 vDSP 升到 MVP 范围。

## 10. v0.2 — 实时音频（占位，进入第二期再细化）

第二期目标：支持麦克风实时流式 ASR，边说边出文字。

预期形态：

```sh
asrctl listen                          # 麦克风 → 文本流到 stdout
asrctl listen --device <name>          # 指定输入设备
asrctl listen --vad                    # 静音检测自动断句
```

需要在第二期解决（占位，不在本 MVP 文档承诺细节）：

- 音频采集：macOS `AVAudioEngine` / `AudioToolbox` / `CoreAudio`，封装为 Zig 接口。
- VAD（语音活动检测）：决定何时开始 / 结束一段推理；候选 webrtcvad 或 silero。
- 流式分段策略：Qwen3-ASR 是 encoder-decoder 一次性架构，**不天然支持流式**。需要决定是「滑动窗口 + 重叠拼接」还是「VAD 切句后整段送 mtmd」，前者实现复杂、后者延迟更高。spike 验证后定方案。
- 部分结果输出协议：行刷新 / JSONL / 终端 ANSI 重写。
- 与 MVP 的复用：wav 解码层换成实时 PCM 缓冲、mtmd 调用层尽量复用、CLI 子命令独立避免污染 `transcribe`。

> v0.2 启动前需要重新评估：是否还坚持 Qwen3-ASR（不利于流式）vs 切到 whisper.cpp（流式更成熟）。这个判断留到第二期 spike 阶段做，不在 MVP 阶段提前承诺。
