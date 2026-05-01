const std = @import("std");

// M1 strategy: link against brew-installed llama.cpp dylibs instead of building
// upstream from source. Single-binary static linking is deferred to M5.
const brew_prefix_default = "/opt/homebrew";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const brew_prefix = b.option(
        []const u8,
        "brew-prefix",
        "Homebrew prefix (default: /opt/homebrew)",
    ) orelse brew_prefix_default;

    const include_path = b.pathJoin(&.{ brew_prefix, "include" });
    const lib_path = b.pathJoin(&.{ brew_prefix, "lib" });

    // translate-c the upstream C headers we need.
    const c_translate = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
    });
    c_translate.addIncludePath(.{ .cwd_relative = include_path });
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

    // Link against system llama.cpp / mtmd / ggml. In Zig 0.16 these helpers
    // live on the module, not the Compile step.
    exe_mod.addLibraryPath(.{ .cwd_relative = lib_path });
    exe_mod.linkSystemLibrary("llama", .{});
    exe_mod.linkSystemLibrary("mtmd", .{});
    exe_mod.linkSystemLibrary("ggml", .{});
    exe_mod.linkSystemLibrary("ggml-base", .{});
    exe_mod.link_libc = true;

    // Embed an rpath so the produced binary can find the brew dylibs at runtime
    // without DYLD_LIBRARY_PATH. M5 will replace this with static linking.
    exe_mod.addRPathSpecial(lib_path);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run asrctl");
    run_step.dependOn(&run_cmd.step);
}
