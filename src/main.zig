const std = @import("std");
const c = @import("c");
const cli = @import("cli.zig");
const hf = @import("hf.zig");
const asr = @import("asr.zig");
const server = @import("server.zig");
const errors = @import("errors.zig");

const version = "0.0.1";

const repo = "ggml-org/Qwen3-ASR-0.6B-GGUF";
const model_filename = "Qwen3-ASR-0.6B-Q8_0.gguf";
const mmproj_filename = "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf";

/// Write to stdout. Use this for content the user is meant to capture / pipe
/// (transcribed text, paths, versions). Diagnostics and errors go through
/// `std.debug.print`, which writes to stderr.
fn writeStdout(io: std.Io, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.writeAll(bytes);
    try w.interface.flush();
}

fn printStdout(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.print(fmt, args);
    try w.interface.flush();
}

pub fn main(init: std.process.Init) void {
    const exit_code = run(init) catch |err| {
        printError(err);
        std.process.exit(2);
    };
    std.process.exit(exit_code);
}

fn run(init: std.process.Init) !u8 {
    const allocator = init.gpa;
    const io = init.io;
    var env = init.environ_map.*;

    const cmd = cli.parse(init.minimal.args.vector) catch |err| {
        switch (err) {
            error.InvalidArgs, error.MissingValue, error.UnknownFlag => {
                std.debug.print("error: {s}\n\n{s}", .{ @errorName(err), cli.usage_text });
                return 1;
            },
            else => return err,
        }
    };

    switch (cmd) {
        .help => {
            try writeStdout(io, cli.usage_text);
            return 0;
        },
        .version => {
            try printStdout(io, "asrctl {s}\n", .{version});
            return 0;
        },
        .model_path => return modelPath(allocator, io, &env),
        .model_pull => return modelPull(allocator, io, &env),
        .transcribe => |args| return transcribe(allocator, io, &env, args),
    }
}

fn modelPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
) !u8 {
    const m = try hf.predictPath(allocator, io, env, .{ .repo = repo, .filename = model_filename });
    defer allocator.free(m);
    const p = try hf.predictPath(allocator, io, env, .{ .repo = repo, .filename = mmproj_filename });
    defer allocator.free(p);
    try printStdout(io, "{s}\n{s}\n", .{ m, p });
    return 0;
}

fn modelPull(allocator: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map) !u8 {
    const m = try hf.ensureFile(allocator, io, env, .{ .repo = repo, .filename = model_filename });
    defer allocator.free(m);
    const p = try hf.ensureFile(allocator, io, env, .{ .repo = repo, .filename = mmproj_filename });
    defer allocator.free(p);
    try printStdout(io, "{s}\n{s}\n", .{ m, p });
    return 0;
}

fn transcribe(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    args: cli.TranscribeArgs,
) !u8 {
    if (!std.mem.endsWith(u8, args.audio_path, ".wav")) {
        std.debug.print("error: MVP only supports .wav (got '{s}')\n", .{args.audio_path});
        return 1;
    }

    // Server fallback: short-circuit before loading any local model state.
    // Same Result shape, same output formatting → fully fungible with the
    // in-process path from the user's perspective.
    if (args.server_url) |url| {
        if (args.verbose) std.debug.print("server:  {s}\n", .{url});
        const result = server.transcribe(allocator, io, url, args.audio_path) catch |err| {
            errors.print("error", err);
            return 4;
        };
        defer result.deinit(allocator);
        if (args.verbose) std.debug.print("language: {s}\n", .{result.language});
        try writeText(io, args.output_path, result.text);
        return 0;
    }

    // Quiet llama/mtmd unless --verbose.
    if (!args.verbose) {
        c.llama_log_set(quietLogCb, null);
        c.mtmd_helper_log_set(quietLogCb, null);
    }

    // Resolve paths.
    const model_path = if (args.model_path) |p|
        try allocator.dupeZ(u8, p)
    else blk: {
        const p = try hf.ensureFile(allocator, io, env, .{ .repo = repo, .filename = model_filename });
        defer allocator.free(p);
        break :blk try allocator.dupeZ(u8, p);
    };
    defer allocator.free(model_path);

    const mmproj_path = blk: {
        // mmproj follows main model: live next to it, or be downloaded.
        const p = try hf.ensureFile(allocator, io, env, .{ .repo = repo, .filename = mmproj_filename });
        defer allocator.free(p);
        break :blk try allocator.dupeZ(u8, p);
    };
    defer allocator.free(mmproj_path);

    const audio_z = try allocator.dupeZ(u8, args.audio_path);
    defer allocator.free(audio_z);

    if (args.verbose) {
        std.debug.print("model:   {s}\n", .{model_path});
        std.debug.print("mmproj:  {s}\n", .{mmproj_path});
        std.debug.print("audio:   {s}\n", .{audio_z});
    }

    const result = asr.transcribe(allocator, .{
        .model_path = model_path,
        .mmproj_path = mmproj_path,
        .audio_path = audio_z,
        .n_threads = args.threads orelse 4,
    }) catch |err| {
        errors.print("error", err);
        return 3;
    };
    defer result.deinit(allocator);

    if (args.verbose) {
        std.debug.print("language: {s}\n", .{result.language});
    }

    try writeText(io, args.output_path, result.text);
    return 0;
}

fn writeText(io: std.Io, output_path: ?[]const u8, text: []const u8) !void {
    var buf: [4096]u8 = undefined;
    if (output_path) |path| {
        const cwd = std.Io.Dir.cwd();
        var file = try cwd.createFile(io, path, .{});
        defer file.close(io);
        var w = file.writer(io, &buf);
        try w.interface.writeAll(text);
        try w.interface.writeAll("\n");
        try w.interface.flush();
    } else {
        var w = std.Io.File.stdout().writer(io, &buf);
        try w.interface.writeAll(text);
        try w.interface.writeAll("\n");
        try w.interface.flush();
    }
}

fn printError(err: anyerror) void {
    errors.print("fatal", err);
}

fn quietLogCb(level: c.ggml_log_level, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    _ = level;
}
