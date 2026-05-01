const std = @import("std");
const c = @import("c");
const cli = @import("cli.zig");
const hf = @import("hf.zig");
const asr = @import("asr.zig");
const server = @import("server.zig");
const errors = @import("errors.zig");
const audio = @import("audio.zig");
const vad = @import("vad.zig");

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
        .listen => |args| return listen(allocator, io, &env, args),
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

    // v0.2.0 spike: route through Session.transcribePCM (raw f32 PCM) to
    // verify that our wav decoder + mtmd_bitmap_init_from_audio path produces
    // the same output as mtmd_helper_bitmap_init_from_file. If it does, the
    // listen subcommand can reuse this exact code path with mic samples.
    // v0.2.0: route through Session + decoded PCM. Same path the listen
    // subcommand will use with mic samples; one less code path to maintain.
    var session = asr.Session.open(allocator, .{
        .model_path = model_path,
        .mmproj_path = mmproj_path,
        .n_threads = args.threads orelse 4,
    }) catch |err| {
        errors.print("error", err);
        return 3;
    };
    defer session.close();

    const decoded = asr.Wav.decodeFile(allocator, io, args.audio_path) catch |err| {
        errors.print("error", err);
        return 1;
    };
    defer decoded.deinit(allocator);
    if (args.verbose) std.debug.print(
        "wav: {d} samples @ {d} Hz ({d:.2}s)\n",
        .{
            decoded.samples.len,
            decoded.sample_rate,
            @as(f64, @floatFromInt(decoded.samples.len)) / @as(f64, @floatFromInt(decoded.sample_rate)),
        },
    );
    const result = session.transcribePCM(decoded.samples) catch |err| {
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

// ---------- listen subcommand (v0.2 real-time mic) ----------

var g_running: std.atomic.Value(bool) = .init(true);

extern "c" fn usleep(usec: u32) c_int;

fn sigintHandler(_: std.posix.SIG) callconv(.c) void {
    g_running.store(false, .release);
}

fn listen(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    args: cli.ListenArgs,
) !u8 {
    if (!args.verbose) {
        c.llama_log_set(quietLogCb, null);
        c.mtmd_helper_log_set(quietLogCb, null);
    }

    // Resolve model paths (same flow as transcribe).
    const model_path = if (args.model_path) |p|
        try allocator.dupeZ(u8, p)
    else blk: {
        const p = try hf.ensureFile(allocator, io, env, .{ .repo = repo, .filename = model_filename });
        defer allocator.free(p);
        break :blk try allocator.dupeZ(u8, p);
    };
    defer allocator.free(model_path);

    const mmproj_path = blk: {
        const p = try hf.ensureFile(allocator, io, env, .{ .repo = repo, .filename = mmproj_filename });
        defer allocator.free(p);
        break :blk try allocator.dupeZ(u8, p);
    };
    defer allocator.free(mmproj_path);

    var session = asr.Session.open(allocator, .{
        .model_path = model_path,
        .mmproj_path = mmproj_path,
        .n_threads = args.threads orelse 4,
    }) catch |err| {
        errors.print("error", err);
        return 3;
    };
    defer session.close();

    var detector = vad.Detector.init(.{
        .threshold = args.threshold orelse 0.012,
        .silence_ms = args.silence_ms orelse 600,
    });
    defer detector.deinit(allocator);

    const capture = audio.Capture.start(allocator, 16_000) catch |err| {
        errors.print("error", err);
        return 3;
    };
    defer capture.stop(allocator);

    // Wire Ctrl-C → stop loop. Restore in `defer` so a re-run gets fresh state.
    var sa: std.posix.Sigaction = .{
        .handler = .{ .handler = sigintHandler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &sa, null);
    defer {
        var dfl: std.posix.Sigaction = .{
            .handler = .{ .handler = std.posix.SIG.DFL },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.INT, &dfl, null);
    }

    if (args.verbose) std.debug.print("listening (Ctrl-C to stop)…\n", .{});

    const Ctx = struct {
        session: *asr.Session,
        allocator: std.mem.Allocator,
        io: std.Io,
        output_path: ?[]const u8,
        verbose: bool,
        utterance_id: u32 = 0,
    };
    var ctx: Ctx = .{
        .session = &session,
        .allocator = allocator,
        .io = io,
        .output_path = args.output_path,
        .verbose = args.verbose,
    };

    const onSegment = struct {
        fn cb(c_ctx: *Ctx, samples: []const f32) anyerror!void {
            c_ctx.utterance_id += 1;
            const dur_ms = samples.len * 1000 / 16_000;
            if (c_ctx.verbose) std.debug.print(
                "[utt {d}] {d}ms ({d} samples) → ASR…\n",
                .{ c_ctx.utterance_id, dur_ms, samples.len },
            );
            const result = try c_ctx.session.transcribePCM(samples);
            defer result.deinit(c_ctx.allocator);
            if (c_ctx.verbose) std.debug.print(
                "[utt {d}] language={s}\n",
                .{ c_ctx.utterance_id, result.language },
            );
            if (c_ctx.output_path) |path| {
                try appendLine(c_ctx.io, path, result.text);
            } else {
                try writeStdout(c_ctx.io, result.text);
                try writeStdout(c_ctx.io, "\n");
            }
        }
    }.cb;

    var pcm_buf: std.ArrayList(f32) = .empty;
    defer pcm_buf.deinit(allocator);

    while (g_running.load(.acquire)) {
        // 50 ms pacing: matches our AudioQueue buffer cadence so we drain ~1
        // callback's worth at a time. Direct usleep since std.posix.nanosleep
        // and std.Thread.sleep both moved/disappeared in Zig 0.16.
        _ = usleep(50_000);

        pcm_buf.clearRetainingCapacity();
        _ = capture.drain(allocator, &pcm_buf) catch continue;
        if (pcm_buf.items.len == 0) continue;

        detector.feed(allocator, pcm_buf.items, &ctx, onSegment) catch |err| {
            errors.print("warn", err);
        };
    }

    detector.flush(&ctx, onSegment) catch |err| errors.print("warn", err);
    if (args.verbose) std.debug.print("\nstopped after {d} utterance(s)\n", .{ctx.utterance_id});
    return 0;
}

fn appendLine(io: std.Io, path: []const u8, line: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, path, .{ .mode = .write_only }) catch
        try cwd.createFile(io, path, .{});
    defer file.close(io);
    const offset = (try file.stat(io)).size;
    try file.writePositionalAll(io, line, offset);
    try file.writePositionalAll(io, "\n", offset + line.len);
}

// ---------- helpers ----------

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
