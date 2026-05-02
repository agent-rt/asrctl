const std = @import("std");

// M5.5 vendor static build:
//   1. `b.dependency("llama_cpp")` gives us upstream source via build.zig.zon.
//   2. We shell out to cmake to compile upstream into static libraries — much
//      less code than hand-porting the cmake tree (ggml + cpu + metal + blas
//      multi-arch detection is ~600 lines of cmake-script logic).
//   3. We link the resulting .a files + system frameworks statically into
//      asrctl. Result: `otool -L` shows only system dylibs, no brew deps.
//
// `-DGGML_METAL_EMBED_LIBRARY=ON` makes ggml-metal compile its .metal shaders
// inline as a C string, so we don't ship a separate `default.metallib`.
//
// Trade-off: cmake is now a build prereq alongside Zig. README documents it.

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const llama_dep = b.dependency("llama_cpp", .{});
    const llama_root = llama_dep.path("");

    const whisper_dep = b.dependency("whisper_cpp", .{});
    const whisper_root = whisper_dep.path("");

    // Run upstream's cmake build to produce static .a files. Outputs land in
    // a private build dir under the zig cache so incremental rebuilds work.
    const cmake_build = b.addSystemCommand(&.{
        "bash", "-eu", "-c",
        \\set -eu
        \\SRC="$1"; OUT="$2"
        \\mkdir -p "$OUT"
        \\if [ ! -f "$OUT/.configured" ]; then
        \\  cmake "$SRC" -B "$OUT" \
        \\    -DCMAKE_BUILD_TYPE=Release \
        \\    -DBUILD_SHARED_LIBS=OFF \
        \\    -DGGML_BACKEND_DL=OFF \
        \\    -DGGML_NATIVE=OFF \
        \\    -DGGML_METAL=ON \
        \\    -DGGML_METAL_EMBED_LIBRARY=ON \
        \\    -DGGML_BLAS=ON \
        \\    -DGGML_OPENMP=OFF \
        \\    -DLLAMA_BUILD_TESTS=OFF \
        \\    -DLLAMA_BUILD_EXAMPLES=OFF \
        \\    -DLLAMA_BUILD_TOOLS=ON \
        \\    -DLLAMA_BUILD_SERVER=OFF \
        \\    -DLLAMA_CURL=OFF >/dev/null
        \\  touch "$OUT/.configured"
        \\fi
        \\cmake --build "$OUT" --target mtmd --target llama --target ggml -- -j$(sysctl -n hw.ncpu) >/dev/null
        \\for lib in src/libllama.a tools/mtmd/libmtmd.a \
        \\           ggml/src/libggml.a ggml/src/libggml-base.a \
        \\           ggml/src/libggml-cpu.a ggml/src/ggml-metal/libggml-metal.a \
        \\           ggml/src/ggml-blas/libggml-blas.a; do
        \\  test -f "$OUT/$lib" || { echo "missing $OUT/$lib" >&2; exit 1; }
        \\done
        ,
        "--",
    });
    cmake_build.addDirectoryArg(llama_root);
    const cmake_out = cmake_build.addOutputDirectoryArg("upstream-build");

    // whisper.cpp is fetched via build.zig.zon (same pattern as llama.cpp).
    // Used for the silero VAD implementation in `asrctl listen`. We only link
    // `libwhisper.a` and reuse llama.cpp's ggml; both projects pin ggml at the
    // same major.minor version (0.10.x).
    const whisper_build = b.addSystemCommand(&.{
        "bash", "-eu", "-c",
        \\set -eu
        \\SRC="$1"; OUT="$2"
        \\mkdir -p "$OUT"
        \\if [ ! -f "$OUT/.configured" ]; then
        \\  cmake "$SRC" -B "$OUT" \
        \\    -DCMAKE_BUILD_TYPE=Release \
        \\    -DBUILD_SHARED_LIBS=OFF \
        \\    -DGGML_BACKEND_DL=OFF \
        \\    -DGGML_NATIVE=OFF \
        \\    -DGGML_METAL=ON \
        \\    -DGGML_METAL_EMBED_LIBRARY=ON \
        \\    -DGGML_BLAS=ON \
        \\    -DGGML_OPENMP=OFF \
        \\    -DGGML_CCACHE=OFF \
        \\    -DCMAKE_C_COMPILER_LAUNCHER= \
        \\    -DCMAKE_CXX_COMPILER_LAUNCHER= \
        \\    -DWHISPER_BUILD_TESTS=OFF \
        \\    -DWHISPER_BUILD_EXAMPLES=OFF \
        \\    -DWHISPER_BUILD_SERVER=OFF >/dev/null
        \\  touch "$OUT/.configured"
        \\fi
        \\cmake --build "$OUT" --target whisper -- -j$(sysctl -n hw.ncpu) >/dev/null
        \\test -f "$OUT/src/libwhisper.a" || { echo "missing libwhisper.a" >&2; exit 1; }
        ,
        "--",
    });
    whisper_build.addDirectoryArg(whisper_root);
    const whisper_out = whisper_build.addOutputDirectoryArg("whisper-build");

    // Module setup.
    const c_translate = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
    });
    c_translate.addIncludePath(llama_root.path(b, "include"));
    c_translate.addIncludePath(llama_root.path(b, "ggml/include"));
    c_translate.addIncludePath(llama_root.path(b, "tools/mtmd"));
    c_translate.addIncludePath(whisper_root.path(b, "include"));
    const c_mod = c_translate.createModule();

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("c", c_mod);

    const exe = b.addExecutable(.{
        .name = "asrctl",
        .root_module = exe_mod,
    });

    // Link the static libs produced by cmake. Order matters: consumers first,
    // dependencies later. mtmd → llama → ggml → ggml-base, plus the backend
    // libs which contribute their `register_backend` constructors.
    exe_mod.addObjectFile(cmake_out.path(b, "tools/mtmd/libmtmd.a"));
    exe_mod.addObjectFile(cmake_out.path(b, "src/libllama.a"));
    exe_mod.addObjectFile(whisper_out.path(b, "src/libwhisper.a"));
    exe_mod.addObjectFile(cmake_out.path(b, "ggml/src/libggml.a"));
    exe_mod.addObjectFile(cmake_out.path(b, "ggml/src/ggml-blas/libggml-blas.a"));
    exe_mod.addObjectFile(cmake_out.path(b, "ggml/src/ggml-metal/libggml-metal.a"));
    exe_mod.addObjectFile(cmake_out.path(b, "ggml/src/libggml-cpu.a"));
    exe_mod.addObjectFile(cmake_out.path(b, "ggml/src/libggml-base.a"));

    exe_mod.linkFramework("Metal", .{});
    exe_mod.linkFramework("MetalKit", .{});
    exe_mod.linkFramework("Foundation", .{});
    exe_mod.linkFramework("Accelerate", .{});
    exe_mod.linkFramework("CoreFoundation", .{});
    // AudioToolbox.AudioQueue: microphone capture for `asrctl listen` (v0.2).
    exe_mod.linkFramework("AudioToolbox", .{});
    exe_mod.linkFramework("CoreAudio", .{});
    exe_mod.link_libc = true;
    exe_mod.link_libcpp = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run asrctl");
    run_step.dependOn(&run_cmd.step);
}
