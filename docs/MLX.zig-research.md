# 调研：jaco-bro/MLX.zig

- 仓库：https://github.com/jaco-bro/MLX.zig
- 调研日期：2026-05-01
- License：Apache-2.0
- 主语言：Zig（README 要求 Zig **0.13.0**）
- 仓库体量：~5.4 MB（仓库内打包了 tokenizer model、tiktoken 词表、whisper 预计算权重和示例 `alive.mp3`）
- Stars 34，Forks 5；最近活跃日期 2025-04-14（项目处于早期 PoC 阶段，已停更约一年）
- Topics：`llama` `llm` `mlx` `pcre2` `regex` `tiktoken` `zig` `zig-package`
- 描述：**Phi-4, Llama 3.2, and Whisper in Zig**

## 一句话定位

把 Apple 的 [MLX](https://github.com/ml-explore/mlx)（Apple Silicon 上的数组/ML 框架，类似 PyTorch + Metal）
通过 [`mlx-c`](https://github.com/ml-explore/mlx-c) 暴露成 Zig 绑定，并在上面**用纯 Zig 重新实现**了若干
Transformer 模型（Llama、Phi、Qwen）和 Whisper 的前向 + 生成循环。**只跑 Apple Silicon。**

和 `llama.cpp.zig` 完全是不同思路：
- `llama.cpp.zig`：Zig 只负责构建系统，模型/推理仍在 llama.cpp（C++）里。
- `MLX.zig`：Zig 既做绑定又做模型实现，MLX 提供张量/算子/Metal 内核。

## 它解决的问题

1. 让 Zig 工程能直接调 MLX，不用走 Python/Swift。
2. 提供「`zig build run-llm`」一条命令就能跑起 Llama-3.2 / Phi-4 / Qwen-2.5 / QwQ / R1-Distill 等 mlx-community 模型。
3. 提供「`zig build run-whisper alive.mp3`」一条命令跑 Whisper-Turbo-Large-v3 ASR。

## 仓库结构

```
build.zig              依赖装配 + 两个 exe (llm, whisper)
build.zig.zon          仅声明 pcre2 依赖（mlx-c 是 build.zig 里 curl 拉的）
src/
  mlx.zig       (52K)  MLX C API 的 Zig 封装层（算子、Array、Stream、Map…）
  llm.zig       (12K)  LLM 入口：参数解析 + ChatConfig + TransformerUnion 分发
  llama.zig     (13K)  Llama Transformer 实现
  phi.zig        (7K)  Phi Transformer 实现
  qwen.zig       (9K)  Qwen Transformer 实现
  whisper.zig   (24K)  Whisper 编解码 + STFT + 生成
  tokenizer.zig (20K)  tokenizer + chat template
  regex.zig      (2K)  pcre2 包装（给 tokenizer 的 BPE pre-tokenize 用）
  utils.zig     (13K)  HF 模型下载、safetensors 加载等
  main.zig            test 聚合入口
multilingual.tiktoken  Whisper 用的 tiktoken 词表 (816 KB, 内嵌入仓库)
tokenizer.model        SentencePiece 模型 (2.1 MB)
whisper_precomputed.safetensors  Whisper 预计算的 mel filter 等 (104 KB)
alive.mp3              测试音频 (3.6 MB)
```

把权重/词表直接 commit 进仓库是这个项目的特点，clone 体积大，但开箱即跑。

## 构建链路（关键点）

1. `build.zig` 在 `setupDependencies` 里**手动 curl** 下载 `mlx-c v0.1.2`：
   ```
   curl -L https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.1.2.tar.gz | tar xz
   cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
   ```
   产物 `libmlxc.a` 和 `_deps/mlx-build/libmlx.a` 被 `addObjectFile` 静态链入 exe。
2. 链接 macOS 框架：`Metal` / `Foundation` / `QuartzCore` / `Accelerate`，外加 `linkLibCpp`。
3. 把 `mlx.metallib`（Metal 内核库）拷到 `zig-out/lib/metal/`。
4. `pcre2-10.45` 是真正用 `zig fetch` 拉的依赖，给 `regex.zig` 用。
5. 所以**先决条件**：Apple Silicon Mac、Zig 0.13.0、CMake、网络（首次构建会拉 mlx-c）。

`mlx-c` 被钉死在 v0.1.2（commit message：「stick to mlx-c v0.1.2 for now」），意味着不会自动跟随 MLX 上游升级。

## 模型/CLI 用法

```fish
# LLM（默认 qwen）
zig build run-llm -- "Write a python function to check if a number is prime"
zig-out/bin/llm --config=phi --max=100 "..."

# 支持的 config 预设：llama, phi, qwen, olympic
# 模型从 mlx-community/<model_name> 自动下载（utils.zig:downloadModel）

# Whisper ASR
zig build run-whisper -- audio_file.mp3
zig-out/bin/whisper alive.mp3
```

`llm.zig` 主流程：
```
ChatConfig.initFromBuild → downloadModel(mlx-community/...)
  → Tokenizer.init → encodeChat
  → TransformerUnion(.llama / .phi / .qwen).init
  → transformer.generate(input_ids, num_tokens)
  → tokenizer.decode → 打印
```
模型分发是一个 `union(enum) { llama, phi, qwen }` —— 想加新模型类型需要改源码扩 union，
不是插件式架构。

## `mlx.zig` 绑定层

通过 `@cImport(@cInclude("mlx/c/mlx.h"))` 直接拿到 mlx-c 全部 C API，然后用 `defineBinaryOp` /
`defineUnaryOp` 这种 comptime 元编程把算子（add/sub/mul/matmul/sigmoid/exp/...）批量包成 Zig 函数。
导出 `Array` / `Stream` / `VectorArray` / `MapStrArr` 等核心类型。
风格上属于「薄封装」：保留 C API 形态，错误用 `MLXError` 集合包一层。

## 评估

**优点**

- 完整跑通了 LLM + ASR 两条端到端链路（不是只贴一个 hello world）。
- MLX 在 M 系芯片上跑量化模型性能很好（统一内存、Metal 内核），这条路线本身有价值。
- 兼容 mlx-community HuggingFace 仓库 → 模型生态可复用。
- tokenizer / chat template / safetensors 加载都自己写，不依赖 Python，完全离线。

**风险 / 局限**

- **平台锁死 macOS + Apple Silicon**，无 fallback。
- **久未维护**：最后一次提交 2025-04-14，距今约 13 个月；MLX 上游、Zig 标准库都已大幅演进。
- **Zig 0.13.0 锁版本**：当前 Zig 主线已远超 0.13，直接 `zig build` 大概率不通过，需自己回退 toolchain 或 port 代码。
- `mlx-c` 钉在 v0.1.2，无法享受新算子/bugfix。
- 仓库把模型词表 / 音频样本 commit 进 git，clone 体积大，且更新成本高。
- 单作者 PoC，注释里多处 `wip` / `inconsistent poc`，API 不稳定。
- 模型类型用 `union(enum)` 硬编码，扩展性弱。
- 没有 CI badge、没有 release tag（version 是 `0.0.0`）。

## 适用场景判断

- ✅ 想在 Apple Silicon 上做**纯 Zig 的 LLM/ASR 推理实验**、研究 MLX 内部用法。
- ✅ 想要一份「MLX 模型怎么用纯系统语言搭起来」的参考实现。
- ❌ 生产部署：维护停滞 + Zig/mlx-c 版本锁死，需要先做一轮 fork + 升级。
- ❌ 跨平台需求（Linux / Windows / NVIDIA）：MLX 路线根本不覆盖，应该选 llama.cpp 系。

## 与 `llama.cpp.zig` 的对比

| 维度 | llama.cpp.zig | MLX.zig |
| --- | --- | --- |
| 思路 | 给 llama.cpp 写 `build.zig` | 给 MLX 写 Zig 绑定 + 重写模型 |
| 推理后端 | llama.cpp (CPU/Vulkan/Metal) | MLX (Metal-only) |
| 平台 | Linux/Win/macOS × x86_64/aarch64 | macOS Apple Silicon only |
| 模型实现 | upstream C++（自带几乎所有架构） | 手写 Zig（仅 Llama/Phi/Qwen/Whisper） |
| 多模态 | mtmd vision/audio | Whisper（语音） |
| 维护状态 | 几乎每日同步 upstream（活跃） | 2025-04 后停更 |
| Zig 版本 | 0.16.0+ | 0.13.0 |
| Stars | 1 | 34 |
| 依赖体积 | upstream 通过 zon 拉 | mlx-c 通过 curl 拉，权重直接放仓库 |
| 用途 | 把 llama.cpp 嵌进 Zig 生态 | Apple Silicon 上做 MLX 实验 / 学习参考 |

简单结论：**要跑模型选 `llama.cpp.zig`，要研究 MLX/Apple Silicon 优化或拿来当 PoC 学习材料选 `MLX.zig`（且需准备升级 Zig 0.13 适配的工作量）。**

## 参考链接

- 本仓库：https://github.com/jaco-bro/MLX.zig
- MLX：https://github.com/ml-explore/mlx
- mlx-c：https://github.com/ml-explore/mlx-c
- 灵感来源：https://github.com/ErikKaum/zig-build-mlx
