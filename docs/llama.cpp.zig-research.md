# 调研：diogok/llama.cpp.zig

- 仓库：https://github.com/diogok/llama.cpp.zig
- 调研日期：2026-05-01
- License：MIT
- 主语言：Zig（最低 Zig 版本 0.16.0）
- 仓库体量：~890 KB；Stars 1，Forks 2（项目非常新）
- 创建时间：2025-10-15，最近一次提交：2026-05-01（持续活跃）
- Topics：`ggml` `llama` `llamacpp` `llm`

## 一句话定位

为 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 提供一个**纯 Zig `build.zig`**，
让 llama.cpp 可以在 Zig 工程里作为依赖被 `zig fetch` 拉取并编译，并支持跨平台/跨架构交叉编译，
同时把 `llama.h` 通过 `zig translate-c` 暴露成 Zig 模块直接调用。

它**不是** llama.cpp 的 Zig 重写，也不是高层封装，而是「构建系统 + C 头绑定」适配层。

## 它解决的问题

upstream llama.cpp 用 CMake 构建，对 Zig 工程不友好。该仓库把整个编译图改写成 `build.zig`：

1. 用 `b.dependency("llama_cpp", ...)` 把 upstream llama.cpp 作为 zon 依赖（pin 到具体 commit）拉下来。
2. 扫描 llama.cpp `src/` 下所有 `.c` / `.cpp` 文件，挂到 Zig 的 module 上，自带 `cflags` / `cppflags`。
3. 生成 `build-info.cpp`（注入 build 元信息），自动 `link_libc` + `link_libcpp`。
4. 通过 `b.addTranslateC` 把 `include/llama.h` 转成 Zig 模块，用 key `"c"` 暴露给下游。
5. 同时编译出几个 upstream 的可执行入口，下游可直接拿来用。

## 支持矩阵

| Target              | CPU | Vulkan | Metal |
| ------------------- | --- | ------ | ----- |
| Linux x86_64        | ✅   | ✅      | -     |
| Linux aarch64       | ✅   | ⚠️ (RPi5 内存不足) | - |
| Windows x86_64      | ✅   | ✅      | -     |
| Windows aarch64     | ✅   | ⚠️ (SQ2 缺特性)   | - |
| macOS aarch64 (M*)  | ✅   | -      | ✅    |
| Termux aarch64      | ✅   | ⚠️     | -     |

通过 `-Dbackend={cpu,vulkan,metal}` 选择后端；`-Dtarget=...` 走标准 Zig 交叉编译。

> Metal 后端需要先 `xcodebuild -downloadComponent MetalToolchain`，产物里会多一个
> `default.metallib`，必须随二进制一起分发。

## 产物（`zig build install` → `zig-out/bin/`）

- `llama-run`：单次推理 CLI（即 upstream 的 `llama-cli`）。
- `llama-bench`：性能基准测试。
- `llama-server`：HTTP / OpenAI 兼容 server，内嵌 upstream Web UI（`/`）。
- `demo`：链接到 library 的最小 Zig 示例。

## 仓库结构

```
build.zig          (~23 KB) 构建图全在这里
build.zig.zon      声明依赖：llama_cpp (pin commit) + 本地 ggml 目录
src/
  demo.zig         调用 c 模块跑 TinyStories-656K-Q8_0.gguf 的最小例子
  test.zig
  xxd.c            把文件嵌成 C 数组的工具（用于内嵌 web UI）
ggml/              本地路径依赖（GGML 适配壳）
models/            (示例模型放置位)
.github/           CI on PRs
```

`build.zig.zon` 关键片段：

```zig
.dependencies = .{
    .llama_cpp = .{
        .url  = "git+https://github.com/ggml-org/llama.cpp#aab68217b7bd...",
        .hash = "N-V-__8AAGnF6ghOYbxmqALUDqDT_UtVAWUVZGKUx34T86im",
    },
    .ggml = .{ .path = "ggml" },
},
```

最近的提交节奏基本是「每天 bump 一次 upstream llama.cpp commit」，作者用脚本/CI 跟着上游走。

## 怎么在自己的 Zig 项目里用

```sh
zig fetch --save git+https://github.com/diogok/llama.cpp.zig
```

```zig
const llama_cpp_dep = b.dependency("llama_cpp_zig", .{
    .target = target,
    .optimize = optimize,
    .backend = .metal, // 或 .cpu / .vulkan
});
const llama_cpp_lib = llama_cpp_dep.artifact("llama_cpp");
your_module.linkLibrary(llama_cpp_lib);

// 多模态（vision / audio）需要额外链接 mtmd
const mtmd_lib = llama_cpp_dep.artifact("mtmd");
your_module.linkLibrary(mtmd_lib);

// 直接拿 llama.h 的 Zig 绑定
const c_mod = llama_cpp_dep.module("c");
your_module.addImport("c", c_mod);
```

`src/demo.zig` 是最小可用调用示例：`llama_backend_init` → `llama_model_load_from_file`
→ 构造 sampler chain（top_k / top_p / min_p / temp / dist）→ `llama_tokenize`
→ `llama_decode` 循环采样 → `llama_token_to_piece` 输出。整个流程纯 Zig，但调的是 translate-c
出来的 C 符号，没有再包一层 Zig 风格 API。

## 评估

**优点**

- 真的「`zig fetch --save` 就能用」，对 Zig 用户友好，省掉手写 CMake 集成。
- 跨编译矩阵开箱即用（Linux/Win/macOS × x86_64/aarch64 × cpu/vulkan/metal）。
- 跟 upstream 节奏紧（几乎每日 bump），不会很快烂尾。
- MIT，依赖透明。

**风险 / 局限**

- 项目极新、单作者、Stars 仅 1，没有形成社区，长期维护性存疑。
- 没有 Zig 风格的高层封装，下游仍然在写 C-style 代码（裸指针、`null` 检查、手动 `defer free`）。
- ARM + Vulkan 组合实际跑不动（README 自陈）。
- `optimize` 被强制 override 成 `ReleaseFast`，调试构建受限。
- 暂未列出 ROCm / CUDA / SYCL / OpenCL 后端。
- 锁死 `minimum_zig_version = 0.16.0`，跟随 Zig 主干，对稳定 release 用户不友好。

## 适用场景判断

- ✅ 想在 Zig 工程里嵌入本地 LLM 推理、又不想引入 CMake / 不想手搓 FFI 包装。
- ✅ 需要把 llama.cpp 跨编译到多个 target 分发（Zig 的强项）。
- ✅ 想要一个 Metal / Vulkan 都能切的本地 server 二进制做小工具。
- ❌ 生产环境核心依赖：太新、社区太小，建议自己 fork 钉住 commit。
- ❌ 期待 idiomatic Zig API：本仓库不提供，需要自己再封一层。

## 参考链接

- 本仓库：https://github.com/diogok/llama.cpp.zig
- 上游：https://github.com/ggml-org/llama.cpp
- 示例代码：`src/demo.zig`
- 构建脚本：`build.zig`（约 600 行，所有平台/后端分支都在里面）
