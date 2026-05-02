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

// Whisper models. We map a friendly name → HF filename. Default (final) is
// large-v3-turbo Q5_0: multilingual SOTA, ~30× realtime on M2 Pro.
//
// `--whisper-partial-model` defaults to `tiny` when --partial is used with a
// `medium` or larger main model — gives ~50 ms partial latency vs ~500 ms for
// running large twice. Smaller mains use the same model for both.
const whisper_repo = "ggerganov/whisper.cpp";

const WhisperModelSize = enum {
    tiny,
    base,
    small,
    medium,
    large_v3_turbo,

    fn parse(s: []const u8) ?WhisperModelSize {
        if (std.mem.eql(u8, s, "tiny")) return .tiny;
        if (std.mem.eql(u8, s, "base")) return .base;
        if (std.mem.eql(u8, s, "small")) return .small;
        if (std.mem.eql(u8, s, "medium")) return .medium;
        if (std.mem.eql(u8, s, "large-v3-turbo") or std.mem.eql(u8, s, "large")) return .large_v3_turbo;
        return null;
    }

    fn filename(self: WhisperModelSize) []const u8 {
        return switch (self) {
            .tiny => "ggml-tiny-q5_1.bin", // 31 MB
            .base => "ggml-base-q5_1.bin", // 57 MB
            .small => "ggml-small-q5_1.bin", // 181 MB
            .medium => "ggml-medium-q5_0.bin", // 514 MB
            .large_v3_turbo => "ggml-large-v3-turbo-q5_0.bin", // 547 MB
        };
    }

    /// Used to decide whether `--partial` should auto-pair with tiny. We
    /// only auto-switch when the main model is "noticeably bigger" than tiny,
    /// otherwise the saving is marginal and the cost of a 2nd model load
    /// outweighs the partial speedup.
    fn isLargeForPartial(self: WhisperModelSize) bool {
        return switch (self) {
            .tiny, .base => false,
            .small, .medium, .large_v3_turbo => true,
        };
    }
};

const whisper_default_main: WhisperModelSize = .large_v3_turbo;
const whisper_default_partial: WhisperModelSize = .tiny;

// Silero VAD model on HF — ggml-format binaries maintained alongside whisper.cpp.
const vad_repo = "ggml-org/whisper-vad";
const vad_filename = "ggml-silero-v5.1.2.bin";

/// Tri-state result of parsing the `--backend` flag. `auto` is its own value
/// — it triggers a language probe and then picks one of the real backends.
const BackendChoice = union(enum) {
    real: asr.Backend,
    auto,
};

fn parseBackend(s: ?[]const u8) ?BackendChoice {
    if (s == null) return null;
    if (std.mem.eql(u8, s.?, "qwen3")) return .{ .real = .qwen3 };
    if (std.mem.eql(u8, s.?, "whisper")) return .{ .real = .whisper };
    if (std.mem.eql(u8, s.?, "auto")) return .auto;
    return null;
}

/// Heuristic: which backend handles a given language best?
///   - zh / yue / wuu (Chinese family): qwen3 (Alibaba 2026 SOTA on Chinese)
///   - everything else: whisper-large-v3-turbo (OpenAI multilingual SOTA)
fn backendForLanguage(lang: []const u8) asr.Backend {
    if (std.mem.eql(u8, lang, "zh") or
        std.mem.eql(u8, lang, "yue") or
        std.mem.eql(u8, lang, "wuu")) return .qwen3;
    return .whisper;
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
    const w = try hf.predictPath(allocator, io, env, .{ .repo = whisper_repo, .filename = whisper_default_main.filename() });
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
    const w = try hf.ensureFile(allocator, io, env, .{ .repo = whisper_repo, .filename = whisper_default_main.filename() });
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
    whisper_size: WhisperModelSize,
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
                const p = try hf.ensureFile(allocator, io, env, .{ .repo = whisper_repo, .filename = whisper_size.filename() });
                defer allocator.free(p);
                break :blk try allocator.dupeZ(u8, p);
            };
            return .{ .backend = .whisper, .model = model, .mmproj = null };
        },
    }
}

fn resolveWhisperSizePath(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    size: WhisperModelSize,
) ![:0]u8 {
    const p = try hf.ensureFile(allocator, io, env, .{ .repo = whisper_repo, .filename = size.filename() });
    defer allocator.free(p);
    return try allocator.dupeZ(u8, p);
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

    const choice: BackendChoice = blk: {
        if (args.backend) |s| {
            if (parseBackend(s)) |bc| break :blk bc;
            std.debug.print("error: unknown --backend '{s}' (qwen3|whisper|auto)\n", .{s});
            return 1;
        }
        break :blk .{ .real = .qwen3 };
    };

    // Auto: probe the audio with tiny whisper to identify the language, then
    // route to the real backend. Adds ~300-500 ms one-time latency.
    var auto_decoded: ?asr.Wav.Decoded = null;
    defer if (auto_decoded) |d| d.deinit(allocator);
    const backend: asr.Backend = switch (choice) {
        .real => |b| b,
        .auto => blk: {
            if (args.server_url != null) {
                std.debug.print("error: --server-url is incompatible with --backend auto\n", .{});
                return 1;
            }
            if (!args.verbose) {
                c.llama_log_set(quietLogCb, null);
                c.mtmd_helper_log_set(quietLogCb, null);
            }
            const tiny_path = try resolveWhisperSizePath(allocator, io, env, .tiny);
            defer allocator.free(tiny_path);

            // Decode wav once; stash for the real transcribe below to reuse.
            const decoded = asr.Wav.decodeFile(allocator, io, args.audio_path) catch |err| {
                errors.print("error", err);
                return 1;
            };
            auto_decoded = decoded;

            const lang = asr.detectLanguage(allocator, tiny_path, decoded.samples) catch |err| {
                errors.print("error", err);
                return 3;
            };
            defer allocator.free(lang);

            const picked = backendForLanguage(lang);
            if (args.verbose) std.debug.print(
                "auto: detected language='{s}' → backend={s}\n",
                .{ lang, @tagName(picked) },
            );
            break :blk picked;
        },
    };

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

    const whisper_size = blk: {
        if (args.whisper_model) |s| {
            if (WhisperModelSize.parse(s)) |sz| break :blk sz;
            std.debug.print("error: unknown --whisper-model '{s}' (tiny|base|small|medium|large-v3-turbo)\n", .{s});
            return 1;
        }
        break :blk whisper_default_main;
    };

    const paths = try resolvePaths(allocator, io, env, backend, args.model_path, whisper_size);
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

    // If --backend auto already decoded the wav for the language probe, reuse
    // those samples instead of re-reading the file. Same for --quick path.
    const result = blk: {
        if (auto_decoded) |d| {
            break :blk session.transcribePCM(d.samples) catch |err| {
                errors.print("error", err);
                return 3;
            };
        }
        if (args.quick) {
            if (backend != .whisper) {
                std.debug.print("note: --quick is whisper-specific; ignored for qwen3\n", .{});
            }
            const decoded = asr.Wav.decodeFile(allocator, io, args.audio_path) catch |err| {
                errors.print("error", err);
                return 3;
            };
            defer decoded.deinit(allocator);
            break :blk session.transcribePCMQuick(decoded.samples) catch |err| {
                errors.print("error", err);
                return 3;
            };
        }
        break :blk session.transcribeFile(audio_z) catch |err| {
            errors.print("error", err);
            return 3;
        };
    };
    defer result.deinit(allocator);

    if (args.verbose) std.debug.print("language: {s}\n", .{result.language});
    try writeText(io, args.output_path, result.text);
    return 0;
}

// ---------- listen subcommand (v0.2 real-time mic) ----------

var g_running: std.atomic.Value(bool) = .init(true);

extern "c" fn usleep(usec: u32) c_int;
extern "c" fn isatty(fd: c_int) c_int;

fn sigintHandler(_: std.posix.SIG) callconv(.c) void {
    g_running.store(false, .release);
}

fn listen(
    allocator: std.mem.Allocator,
    io: std.Io,
    env: *std.process.Environ.Map,
    args: cli.ListenArgs,
) !u8 {
    const choice: BackendChoice = blk: {
        if (args.backend) |s| {
            if (parseBackend(s)) |c2| break :blk c2;
            std.debug.print("error: unknown --backend '{s}' (qwen3|whisper|auto)\n", .{s});
            return 1;
        }
        break :blk .{ .real = .qwen3 };
    };
    const backend: asr.Backend = switch (choice) {
        .real => |b| b,
        .auto => {
            std.debug.print("error: --backend auto is not yet supported in listen mode\n" ++
                "       (per-utterance language probe would add 50-100 ms latency).\n" ++
                "       Workaround: pick qwen3 for Chinese-heavy use, whisper otherwise.\n", .{});
            return 1;
        },
    };

    if (!args.verbose) {
        c.llama_log_set(quietLogCb, null);
        c.mtmd_helper_log_set(quietLogCb, null);
    }

    const whisper_size = blk: {
        if (args.whisper_model) |s| {
            if (WhisperModelSize.parse(s)) |sz| break :blk sz;
            std.debug.print("error: unknown --whisper-model '{s}'\n", .{s});
            return 1;
        }
        break :blk whisper_default_main;
    };

    // Smart default: when --partial is on with a medium+ main, transparently
    // load `tiny` alongside it so partial preview is ~10× faster. User can
    // override with --whisper-partial-model NAME or disable by passing
    // --whisper-partial-model <same-as-main>.
    const partial_size: ?WhisperModelSize = blk: {
        if (backend != .whisper or !args.partial) break :blk null;
        if (args.whisper_partial_model) |s| {
            if (WhisperModelSize.parse(s)) |sz| {
                if (sz == whisper_size) break :blk null; // no separate model needed
                break :blk sz;
            }
            std.debug.print("error: unknown --whisper-partial-model '{s}'\n", .{s});
            return 1;
        }
        if (whisper_size.isLargeForPartial()) break :blk whisper_default_partial;
        break :blk null;
    };

    const paths = try resolvePaths(allocator, io, env, backend, args.model_path, whisper_size);
    defer paths.deinit(allocator);

    const partial_path: ?[:0]u8 = if (partial_size) |sz|
        try resolveWhisperSizePath(allocator, io, env, sz)
    else
        null;
    defer if (partial_path) |p| allocator.free(p);

    if (args.verbose) {
        std.debug.print("whisper main:    {s}\n", .{paths.model});
        if (partial_path) |p| std.debug.print("whisper partial: {s}\n", .{p});
    }

    const language_z: ?[:0]const u8 = if (args.language) |l|
        try allocator.dupeZ(u8, l)
    else
        null;
    defer if (language_z) |l| allocator.free(l);

    var session = asr.Session.open(allocator, .{
        .backend = backend,
        .model_path = paths.model,
        .mmproj_path = paths.mmproj,
        .whisper_partial_model_path = partial_path,
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

    // --partial only makes sense on whisper (qwen3 is one-shot enc-dec) and
    // only renders correctly on a TTY (the redraw uses CR + ANSI erase).
    const stdout_tty = isatty(1) != 0;
    const partial_enabled = args.partial and backend == .whisper and
        stdout_tty and args.output_path == null;
    if (args.partial and backend != .whisper) {
        std.debug.print("error: --partial requires --backend whisper\n", .{});
        return 1;
    }
    if (args.partial and !stdout_tty and args.verbose)
        std.debug.print("note: stdout is not a TTY, partials disabled\n", .{});

    const partial_ms = args.partial_ms orelse 500;
    const drain_ms: u32 = 50;
    const partial_target_iters: u32 = if (partial_ms < drain_ms) 1 else partial_ms / drain_ms;
    // Don't bother running whisper on <1s of audio; quality is unstable.
    const min_partial_samples: usize = 16_000;

    if (args.verbose) std.debug.print(
        "listening (Ctrl-C to stop, partial={s})…\n",
        .{if (partial_enabled) "on" else "off"},
    );

    const Ctx = struct {
        session: *asr.Session,
        allocator: std.mem.Allocator,
        io: std.Io,
        output_path: ?[]const u8,
        verbose: bool,
        partial_drawn: bool, // true if a partial line is currently on stdout
        utterance_id: u32 = 0,
    };
    var ctx: Ctx = .{
        .session = &session,
        .allocator = allocator,
        .io = io,
        .output_path = args.output_path,
        .verbose = args.verbose,
        .partial_drawn = false,
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
                if (c_ctx.partial_drawn) {
                    // Erase the in-flight dim partial, then commit final.
                    try writeStdout(c_ctx.io, "\r\x1b[K");
                    c_ctx.partial_drawn = false;
                }
                try writeStdout(c_ctx.io, result.text);
                try writeStdout(c_ctx.io, "\n");
            }
        }
    }.cb;

    var pcm_buf: std.ArrayList(f32) = .empty;
    defer pcm_buf.deinit(allocator);
    var partial_iters: u32 = 0;

    while (g_running.load(.acquire)) {
        // 50 ms pacing: matches our AudioQueue buffer cadence so we drain ~1
        // callback's worth at a time. Direct usleep since std.posix.nanosleep
        // and std.Thread.sleep both moved/disappeared in Zig 0.16.
        _ = usleep(drain_ms * 1000);

        pcm_buf.clearRetainingCapacity();
        _ = capture.drain(allocator, &pcm_buf) catch continue;
        if (pcm_buf.items.len == 0) continue;

        detector.feed(allocator, pcm_buf.items, &ctx, onSegment) catch |err| {
            errors.print("warn", err);
        };

        // Streaming partial preview: while we're collecting an utterance,
        // periodically run whisper on the in-flight buffer and redraw the
        // line in dim text. No-op for energy/qwen3 / file output / non-TTY.
        if (partial_enabled and detector.state == .active) {
            partial_iters += 1;
            if (partial_iters >= partial_target_iters and
                detector.segment.items.len >= min_partial_samples)
            {
                partial_iters = 0;
                if (session.transcribePCMQuick(detector.segment.items)) |result| {
                    defer result.deinit(allocator);
                    // \r move-to-col-0, \x1b[K erase-to-end-of-line, dim ANSI.
                    printStdout(io, "\r\x1b[K\x1b[2m{s}\x1b[0m", .{result.text}) catch {};
                    ctx.partial_drawn = true;
                } else |_| {}
            }
        } else {
            partial_iters = 0;
        }
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
