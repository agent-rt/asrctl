# M0 Spike 报告

> 验收 REQ.md §6 的 spike 0：upstream llama.cpp + mtmd 能否跑 `ggml-org/Qwen3-ASR-0.6B-GGUF`。
> 结果：✅ **通过**，可以进 M1。

## 环境

- macOS aarch64，Apple M2 Pro，统一内存 32 GB（`recommendedMaxWorkingSetSize` 26.8 GB）。
- llama.cpp `8990`（brew，2026-04-13 release）。**8680（4 月 15 日装的版本）不支持**，需要 ≥ commit `21a4933042`（PR #19441，2026-04-12 合并的「mtmd: qwen3 audio support」）。
- 模型：HF `ggml-org/Qwen3-ASR-0.6B-GGUF`，文件 `Qwen3-ASR-0.6B-Q8_0.gguf`（749 MB）+ `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf`（720 MB），总缓存 ~1.5 GB。

## 命令

```sh
llama-mtmd-cli \
  -hf ggml-org/Qwen3-ASR-0.6B-GGUF \
  --audio test.wav \
  -p "Transcribe the audio." \
  -n 128 --no-warmup
```

`-hf` 自动从 HuggingFace 下载主模型 + mmproj projector，缓存在 `~/.cache/huggingface/hub/`，复用 huggingface_hub 标准约定，符合 REQ.md §3 的模型管理设计。

## 测试结果

### 英文

```
输入: "The quick brown fox jumps over the lazy dog. Hello world, this is a transcription test."
输出: language English<asr_text>The quick brown fox jumps over the lazy dog. Hello, world. This is a transcription test.
```

逐字基本一致，仅标点差异（`Hello world` → `Hello, world.`）。

### 中文

```
输入: 今天天气真不错，我们去公园散步吧。
输出: language Chinese<asr_text>今天天气真不错，我们去公园散步吧。
```

逐字完全一致。

## 性能

英文 ~6 秒 16kHz mono PCM16 wav：

| 阶段 | 耗时 |
| --- | --- |
| 模型 load | 1.24 s |
| Audio encode | 0.30 s |
| Audio decode batch | 3 ms |
| Token eval（23 tokens） | 0.17 s |
| **端到端 total** | **1.42 s** |

吞吐：prompt eval 974 tok/s，generation 133 tok/s。

REQ.md §5 的目标是 30 秒音频 ≤ 5 秒，按当前 audio encode 0.05 s/s 音频外推完全达标。

## 关键发现

1. **mtmd-cli `--audio FILE` 直接吃 wav**，输出 PCM 转换由 mtmd 内部完成。asrctl 在 Zig 侧只需解析 RIFF/WAVE 头给出 mono 16kHz f32 PCM（具体格式 spike 后可在 M2 进一步缩窄，目前看 `mtmd_helper_eval_audio` 的 C API 直接接收 PCM 数组）。
2. **输出格式是 `language X<asr_text>...`**：自动检测语言，用 `<asr_text>` 标签包裹文本。**asrctl 需要 parse 这个标签**取出纯文本（M3 任务）。
3. **mmproj 是必须的**——`init_audio` 阶段 mtmd 会要求 audio encoder。`-hf` 自动下载，但 asrctl 自己实现 HF 下载逻辑时也要拉这个文件（REQ.md 已埋了一笔，确认）。
4. **HF cache 路径解析**：实际命中 `~/.cache/huggingface/hub/models--ggml-org--Qwen3-ASR-0.6B-GGUF/snapshots/<sha>/`，文件名是仓库里的原始 `rfilename`。Zig 实现要按 huggingface_hub 的目录约定生成 symlink + blob，或者直接平铺到 snapshot 目录都行（mtmd 不挑）。
5. **量化档位**：用 `Q8_0`，单个 `.gguf` 749 MB；同仓库还有 `bf16` 双倍大小。MVP 直接默认 `Q8_0`，REQ.md §9 的待决事项可以收敛——精度上 Q8_0 已经做到逐字一致，没必要再上 Q4。
6. **mtmd 仍标注 experimental**：`init_audio: audio input is in experimental stage and may have reduced quality`，asrctl 的 README 里要照抄一句免责。
7. **Flash Attention 自动开**（`auto` → `on`），Metal 上 KV cache fused Gated Delta Net 也开了。这些都是默认行为，asrctl 透传 `llama_*` 默认参数即可。

## 对 REQ.md 的修正/收敛

- §3「量化档位 spike 后定」**收敛为 Q8_0**。
- §3 模型管理新增：mmproj 文件名为 `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf`，与主模型同 snapshot 目录。
- §9 待决「量化档位 Q4 vs Q8」可以删除。
- §9 新增确认事项：输出 parse 协议是 `language <X><asr_text><TEXT>`（M3 实现）。

## 风险更新

| 原风险 | 新认识 |
| --- | --- |
| upstream 是否支持 Qwen3-ASR | ✅ 已支持，PR #19441 合并 |
| mtmd 音频输入格式 | ✅ 内部处理，Zig 侧只需 wav → PCM f32 |
| 性能是否达标 | ✅ 大幅冗余，30s 音频外推 ~3.5s 总耗时 |
| 输出格式未知 | ⚠️ 需要 parse `<asr_text>` 标签 |
| `qwen3-asr wrong output` issue (#22343) | 需要复盘，目前 Q8_0 表现良好，暂不阻塞 |

## 决定

**进 M1**。pin upstream llama.cpp commit 用 8990 对应的源码 commit（接下来在 M1 拉源码时定）。
