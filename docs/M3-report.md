# M3 报告

> CLI 完整化：参数解析、HF 路径解析、子命令、stdout/stderr 分离、清爽错误退出。
> 状态：✅ 通过（2026-05-01）。

## 目标对照

| 需求 | 状态 |
| --- | --- |
| 参数解析（-o / --model / --threads / -v / -h） | ✅ |
| 错误处理 | ✅ `std.process.exit(code)`，无 Zig stack trace |
| 非 wav 拒绝 | ✅ M2 已做，M3 沿用 |
| HF 自动下载 + 缓存解析 | ✅ 复用 huggingface_hub layout，curl 子进程下载 |
| `asrctl model path` | ✅ 输出到 stdout 可管道 |
| `asrctl model pull` | ✅ 命中缓存幂等，下载错误信息走 stderr |

## 文件结构

```
src/
├── c.h           # 聚合 llama.h + mtmd.h + mtmd-helper.h + ggml-backend.h
├── main.zig      # 入口 + 子命令分发 + stdout/stderr 路由
├── cli.zig       # 手写 arg parser，无外部依赖
├── hf.zig        # HF cache 解析 + curl 下载
├── asr.zig       # transcribe pipeline（Options/Result 结构化接口）
└── backend.zig   # ggml libexec 路径解析
```

## 关键设计

### HF 缓存解析（`hf.zig`）

1. cacheRoot：`HF_HOME` > `XDG_CACHE_HOME/huggingface` > `~/.cache/huggingface`。
2. 优先扫 huggingface_hub 标准 layout：`<root>/hub/models--{org}--{repo}/snapshots/*/{filename}`，**双 `--` 分隔 org/repo**（这是个容易踩的坑：第一次实现用了单 `-`，结果 M0 已下载的 1.5 GB 文件认不到要重下）。
3. 不命中则用 asrctl-owned layout：`<root>/asrctl/{org}--{repo}/{filename}`，写入只走这条。
4. `predictPath` 给 `model path` 用，不下载；`ensureFile` 给 transcribe / pull 用，不命中就 curl 下。

代价：写入不模拟 huggingface_hub 的 blob+symlink 结构，所以 asrctl 自己下载的文件不会被 huggingface_hub 识别为缓存命中（反向不成立）。MVP 可接受。

### Curl 子进程下载

- `std.process.spawn(io, .{ .argv = &.{"curl", "-L", "--fail", "--progress-bar", "-o", dest, url}, ... })`
- `-L` 跟随 30x（HF 重定向到 CDN 必须）
- `--fail` 让 4xx/5xx 退出非零
- `--progress-bar` 让用户看到进度
- HF_ENDPOINT 透传到 URL，国内镜像直接生效

M5 vendor 源码时再考虑替换为 `std.http.Client`（Zig 0.16 std.http 还在 io 接口大改造中）。

### Backend 路径

最初实现遍历 `Cellar/ggml/*` 取第一个找到的版本——结果命中了未删的 `0.9.11`，与当前 `0.10.1` 的 libllama ABI 不匹配，**model load 段错误**。改成直接用 brew 维护的 `/opt/homebrew/opt/ggml/libexec` symlink，永远指向当前版本。

### stdout / stderr 分离

- 内容输出（转录文本 / `model path`/`pull` 路径 / version / help）走 **stdout**，可管道。
- 诊断（`-v` 信息、错误、ggml/mtmd log）走 **stderr**。
- 实现：写了 `writeStdout` / `printStdout` helper 用 `std.Io.File.stdout().writer(io, &buf)`，因为 0.16 的 std.Io 写法和 0.13 完全不同。
- `std.debug.print` 默认走 stderr，所以错误信息直接用它就行。

### 错误退出

`pub fn main(init: std.process.Init) void` 不返回错误，`run` 返回 `!u8`，main 里 `catch std.process.exit(2)`。退出码：
- `0` 成功
- `1` 用户错误（参数 / 非 wav / 缺命令）
- `2` 内部错误（缓存目录创建失败等）
- `3` 推理失败（model load / mtmd / decode）

退出码语义已对齐 REQ.md §4。

## 端到端验证

```
$ asrctl /tmp/asrctl-spike/test_en.wav
The quick brown fox jumps over the lazy dog. Hello, world. This is a transcription test.

$ asrctl /tmp/asrctl-spike/test_zh.wav
今天天气真不错，我们去公园散步吧。

$ asrctl -v test_en.wav -o /tmp/out.txt
# stderr: model: ...  mmproj: ...  backend: ...  audio: ...  language: English  + ggml log
# /tmp/out.txt: transcription text + \n

$ asrctl model path
/Users/.../snapshots/.../Qwen3-ASR-0.6B-Q8_0.gguf
/Users/.../snapshots/.../mmproj-Qwen3-ASR-0.6B-Q8_0.gguf

$ asrctl version
asrctl 0.0.1

$ asrctl --bogus foo  # exit=1, error to stderr, stdout 干净

$ HF_ENDPOINT=https://hf-mirror.com asrctl model path
# 仍命中本地 HF cache（HF_ENDPOINT 只在下载阶段生效）
```

## 踩到的 Zig 0.16 API 坑（备忘）

| 旧 API | 0.16 替换 |
| --- | --- |
| `std.fs.cwd()` | `std.Io.Dir.cwd()` |
| `dir.openDir(path, opts)` | `dir.openDir(io, path, opts)` 显式传 io |
| `dir.iterate(io)` | `dir.iterate()` 不传 io（next 才传） |
| `dir.makePath` | `dir.createDirPath(io, path)` |
| `Term.Exited` | `Term.exited`（小写化） |
| `std.time.milliTimestamp()` | 没了，用 `std.Io.Timestamp.now(io, clock)` |
| `std.mem.trimLeft` | `std.mem.trimStart` |
| `std.posix.write` | 没了，用 File writer |
| `std.process.argsAlloc` | 没了，用 `init.minimal.args.vector` |
| `addLibraryPath` on Compile | 在 Module 上 |
| `linkSystemLibrary("x")` | `linkSystemLibrary("x", .{})` |

整体感受：0.16 的 Io 接口 + Module/Compile 拆分还在演进中，写起来比 0.13 时代麻烦不少，但接口更显式，长期更可控。

## 遗留给后续阶段

- **M4**：`--server-url` 走 `llama-server` HTTP，复用同一份 GGUF。
- **M5**：vendor llama.cpp 源码静态编译，干掉 brew 依赖；用 `std.http` 替代 curl；`@embedFile` 内嵌 metallib 做真单二进制；探测改成 `glob` 第二备份方案。
- 性能 / warmup 缓存优化：当前每次冷启动都要 JIT Metal kernel，~1s 开销。可以做后台 daemon 或预热。
- 子命令 `model rm` / `model list` / `model use --quant Q4` 等 model 管理。
- 友好失败信息：当前 LoadModelFailed 等只打 enum 名，可以根据 errno 给更细的提示（"模型未下载？"、"模型损坏？"）。

## 决定

**进 M4**：备用路径 `--server-url` 走 HTTP，输出对齐主路径。
