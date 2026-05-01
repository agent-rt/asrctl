# M4 报告

> 备用路径：`--server-url` 走 HTTP POST 到 `llama-server`，输出与主路径对齐。
> 状态：✅ 通过（2026-05-02）。

## 端点验证

```
$ llama-server -hf ggml-org/Qwen3-ASR-0.6B-GGUF --port 8765
$ curl -X POST http://127.0.0.1:8765/v1/audio/transcriptions -F file=@test.wav
{"type":"transcript.text.done",
 "text":"language English<asr_text>The quick brown fox jumps over the lazy dog. Hello, world. ...",
 "usage":{...}}
```

注意：response.text 仍是 `language X<asr_text>...` 协议，**和本地 mtmd 调用产出的 raw 完全一样**。这意味着 client 不需要做协议转换——把 JSON 里的 `text` 字段当 raw 喂给现有的 `asr.parseOutput` 就行。这是 server fallback 之所以"输出对齐主路径"的关键。

## 实现要点

1. **CLI flag**：`--server-url URL` 在 `cli.TranscribeArgs` 加字段、`parseTranscribe` 加分支、help 文案加说明。
2. **新模块 `server.zig`**：
   - `transcribe(allocator, io, url, wav_path) → asr.Result`，签名与 `asr.transcribe` 形态对称（虽然返回 Result 一致但不能完全替换，因为 asr.transcribe 接 `Options`）。
   - URL 处理：如果 URL 已包含 `/v1/`，原样；否则追加 `/v1/audio/transcriptions`。容易理解，也允许网关/反代路径。
   - HTTP 实现：`curl -sS --fail-with-body -X POST -o <tmp> <url> -F file=@<wav>`。
   - 临时文件 `/tmp/asrctl-server-response.json`：避免在 Zig 0.16 流式 reader 接口上耗时间。单进程工具不会并发，固定文件名 + 用完删足够。
   - JSON parse：`std.json.parseFromSlice(std.json.Value, ...)` 拿 `text` 字段，dupe 出来喂 `asr.parseOutput`。
3. **复用 `asr.parseOutput`**：从 `asr.transcribe` 内部抽出来，导出为公共函数。两条路径走同一个解析，"输出对齐"自然。
4. **main.zig 分发**：`if (args.server_url) |url|` 短路在 quiet log + libexec 解析 + 模型 load 之前，server 模式根本不碰 ggml/llama.cpp（除了 link 进 binary 但不调）。
5. **`writeText` helper**：`-o` 写文件 vs stdout 的逻辑两路径共享。

## 端到端验证

```
$ asrctl --server-url http://127.0.0.1:8765 test_en.wav
The quick brown fox jumps over the lazy dog. Hello, world. This is a transcription test.

$ asrctl --server-url http://127.0.0.1:8765 test_zh.wav
今天天气真不错，我们去公园散步吧。

$ asrctl -v --server-url http://127.0.0.1:8765 test_en.wav
# stderr: server: http://127.0.0.1:8765 / language: English
# stdout: 转录文本

$ asrctl --server-url http://127.0.0.1:9999 test_en.wav
# stderr: curl: (7) Failed to connect to 127.0.0.1 port 9999 ...
# stderr: error: server transcribe failed: HttpFailed
# exit=4
```

输出与主路径逐字一致，错误信息走 stderr，退出码 4 = 网络/server（对齐 REQ.md §4）。

## 遗留 / 后续

- `--server-url` + 自定义 endpoint 路径（如 `/api/transcribe`）已支持。
- 当前不支持设置 timeout / retries。若需要，curl 加 `--max-time` flag。
- 不支持 HTTP auth / API key（llama-server 公开 endpoint 无需）。需要时透传 `Authorization` header。
- 临时文件路径写死 `/tmp/asrctl-server-response.json`：单进程足够，但 root 进程或多用户机器要 racy。M5 vendor 时换 `std.http`，response 留内存。

## 决定

**进 M5 — 打磨**：性能测量、错误信息友好化、README、release。
