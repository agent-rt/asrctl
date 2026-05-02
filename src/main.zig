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

// Default Qwen3-ASR model (backend=qwen3).
const qwen3_repo = "ggml-org/Qwen3-ASR-0.6B-GGUF";
const qwen3_model_filename = "Qwen3-ASR-0.6B-Q8_0.gguf";
const qwen3_mmproj_filename = "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf";

// Default whisper model (backend=whisper). large-v3-turbo Q5_0: 547 MB,
// multilingual SOTA, ~30x realtime on Apple Silicon. Override via --model.
const whisper_repo = "ggerganov/whisper.cpp";
const whisper_model_filename = "ggml-large-v3-turbo-q5_0.bin";

// Silero VAD model on HF — ggml-format binaries maintained alongside whisper.cpp.
const vad_repo = "ggml-org/whisper-vad";
const vad_filename = "ggml-silero-v5.1.2.bin";

fn parseBackend(s: ?[]const u8) ?asr.Backend {
    if (s == null) return null;
    if (std.mem.eql(u8, s.?, "qwen3")) return .qwen3;
    if (std.mem.eql(u8, s.?, "whisper")) return .whisper;
    return null;
}

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
    // Show both backends so the user can see what's resolved either way.
    const q_m = try hf.predictPath(allocator, io, env, .{ .repo = qwen3_repo, .filename = qwen3_model_filename });
    defer allocator.free(q_m);
    const q_p = try hf.predictPath(allocator, io, env, .{ .repo = qwen3_repo, .filename = qwen3_mmproj_filename });
    defer allocator.free(q_p);
    const w = try hf.predictPath(allocator, io, env, .{ .repo = whisper_repo, .filename = whisper_model_filename });
    defer allocator.free(w);
    try printStdout(io, "qwen3 model:   {s}\nqwen3 mmproj:  {s}\nwhisper model: {s}\n", .{ q_m, q_p, w });
    return 0;
}

fn modelPull(allocator: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map) !u8 {
    // Pull both default backends. Cheaper to do at once if the user wants to
    // pre-warm everything.
    const q_m = try hf.ensureFile(allocator, io, env, .{ .repo = qwen3_repo, .filename = qwen3_model_filename });
    defer allocator.free(q_m);
    const q_p = try hf.ensureFile(allocator, io, env, .{ .repo = qwen3_repo, .filename = qwen3_mmproj_filename });
    defer allocator.free(q_p);
    const w = try hf.ensureFile(allocator, io, env, .{ .repo = whisper_repo, .filename = whisper_model_filename });
    defer allocator.free(w);
    try printStdout(io, "qwen3 model:   {s}\nqwen3 mmproj:  {s}\nwhisper model: {s}\n", .{ q_m, q_p, w });
    return 0;
}

const ResolvedPaths = struct {
    backend: asr.Backend,
    model: [:0]u8,
    mmproj: ?[:0]u8, // null for whisper

    fn deinit(self: ResolvedPaths, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        if (self.mmproj) |p| allocator.free(p);
    }
};

/// Resolves the model files for the chosen backend, downloading from HF on
/// cache miss. `--model PATH` skips download for the main model file.
fn resolvePaths(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    backend: asr.Backend,
    override_model: ?[]const u8,
) !ResolvedPaths {
    switch (backend) {
        .qwen3 => {
            const model = if (override_model) |p|
                try allocator.dupeZ(u8, p)
            else blk: {
                const p = try hf.ensureFile(allocator, io, env, .{ .repo = qwen3_repo, .filename = qwen3_model_filename });
                defer allocator.free(p);
                break :blk try allocator.dupeZ(u8, p);
            };
            const mmproj = blk: {
                const p = try hf.ensureFile(allocator, io, env, .{ .repo = qwen3_repo, .filename = qwen3_mmproj_filename });
                defer allocator.free(p);
                break :blk try allocator.dupeZ(u8, p);
            };
            return .{ .backend = .qwen3, .model = model, .mmproj = mmproj };
        },
        .whisper => {
            const model = if (override_model) |p|
                try allocator.dupeZ(u8, p)
            else blk: {
                const p = try hf.ensureFile(allocator, io, env, .{ .repo = whisper_repo, .filename = whisper_model_filename });
                defer allocator.free(p);
                break :blk try allocator.dupeZ(u8, p);
            };
            return .{ .backend = .whisper, .model = model, .mmproj = null };
        },
    }
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

    const backend = parseBackend(args.backend) orelse .qwen3;
    if (args.backend != null and parseBackend(args.backend) == null) {
        std.debug.print("error: unknown --backend '{s}' (qwen3|whisper)\n", .{args.backend.?});
        return 1;
    }

    // Server fallback (qwen3 only — llama-server doesn't host whisper).
    if (args.server_url) |url| {
        if (backend != .qwen3) {
            std.debug.print("error: --server-url only works with --backend qwen3\n", .{});
            return 1;
        }
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

    if (!args.verbose) {
        c.llama_log_set(quietLogCb, null);
        c.mtmd_helper_log_set(quietLogCb, null);
    }

    const paths = try resolvePaths(allocator, io, env, backend, args.model_path);
    defer paths.deinit(allocator);

    const language_z: ?[:0]const u8 = if (args.language) |l|
        try allocator.dupeZ(u8, l)
    else
        null;
    defer if (language_z) |l| allocator.free(l);

    const audio_z = try allocator.dupeZ(u8, args.audio_path);
    defer allocator.free(audio_z);

    if (args.verbose) {
        std.debug.print("backend: {s}\n", .{@tagName(backend)});
        std.debug.print("model:   {s}\n", .{paths.model});
        if (paths.mmproj) |p| std.debug.print("mmproj:  {s}\n", .{p});
        std.debug.print("audio:   {s}\n", .{audio_z});
    }

    var session = asr.Session.open(allocator, .{
        .backend = backend,
        .model_path = paths.model,
        .mmproj_path = paths.mmproj,
        .language = language_z,
        .n_threads = args.threads orelse 4,
    }) catch |err| {
        errors.print("error", err);
        return 3;
    };
    defer session.close();

    const result = session.transcribeFile(audio_z) catch |err| {
        errors.print("error", err);
        return 3;
    };
    defer result.deinit(allocator);

    if (args.verbose) std.debug.print("language: {s}\n", .{result.language});
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
    const backend = parseBackend(args.backend) orelse .qwen3;
    if (args.backend != null and parseBackend(args.backend) == null) {
        std.debug.print("error: unknown --backend '{s}' (qwen3|whisper)\n", .{args.backend.?});
        return 1;
    }

    if (!args.verbose) {
        c.llama_log_set(quietLogCb, null);
        c.mtmd_helper_log_set(quietLogCb, null);
    }

    const paths = try resolvePaths(allocator, io, env, backend, args.model_path);
    defer paths.deinit(allocator);

    const language_z: ?[:0]const u8 = if (args.language) |l|
        try allocator.dupeZ(u8, l)
    else
        null;
    defer if (language_z) |l| allocator.free(l);

    var session = asr.Session.open(allocator, .{
        .backend = backend,
        .model_path = paths.model,
        .mmproj_path = paths.mmproj,
        .language = language_z,
        .n_threads = args.threads orelse 4,
    }) catch |err| {
        errors.print("error", err);
        return 3;
    };
    defer session.close();

    const vad_backend: vad.Backend = blk: {
        if (args.vad) |s| {
            if (std.mem.eql(u8, s, "energy")) break :blk .energy;
            if (std.mem.eql(u8, s, "silero")) break :blk .silero;
            std.debug.print("error: unknown --vad backend '{s}' (energy|silero)\n", .{s});
            return 1;
        }
        break :blk .energy;
    };

    // Silero needs its model file; resolve and pass through.
    var silero_path_owned: ?[:0]u8 = null;
    defer if (silero_path_owned) |p| allocator.free(p);
    const silero_model_path: ?[:0]const u8 = if (vad_backend == .silero) blk: {
        const p = hf.ensureFile(allocator, io, env, .{
            .repo = vad_repo,
            .filename = vad_filename,
        }) catch |err| {
            errors.print("error", err);
            return 2;
        };
        defer allocator.free(p);
        const z = try allocator.dupeZ(u8, p);
        silero_path_owned = z;
        break :blk z;
    } else null;

    var detector = vad.Detector.init(allocator, .{
        .backend = vad_backend,
        .energy_threshold = args.threshold orelse 0.012,
        .silero_threshold = args.threshold orelse 0.5,
        .silero_model_path = silero_model_path,
        .silence_ms = args.silence_ms orelse 600,
    }) catch |err| {
        errors.print("error", err);
        return 3;
    };
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
