//! Minimal hand-rolled argument parser. Subcommands:
//!   transcribe <file.wav>  — default if first non-flag arg ends with .wav
//!   model path             — print resolved model path
//!   model pull             — download model + mmproj
//!   version                — print version
//!   help / -h / --help

const std = @import("std");

pub const Subcommand = union(enum) {
    transcribe: TranscribeArgs,
    listen: ListenArgs,
    model_path,
    model_pull,
    version,
    help,
};

pub const ListenArgs = struct {
    output_path: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    backend: ?[]const u8 = null, // "qwen3" | "whisper"
    language: ?[]const u8 = null,
    vad: ?[]const u8 = null, // "energy" | "silero"
    threads: ?i32 = null,
    threshold: ?f32 = null,
    silence_ms: ?u32 = null,
    /// Stream partial transcriptions while user is still speaking. Whisper
    /// backend only; needs a TTY stdout to render the in-place redraw.
    partial: bool = false,
    partial_ms: ?u32 = null,
    verbose: bool = false,
};

pub const TranscribeArgs = struct {
    audio_path: []const u8,
    output_path: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    server_url: ?[]const u8 = null,
    backend: ?[]const u8 = null, // "qwen3" | "whisper"
    language: ?[]const u8 = null,
    threads: ?i32 = null,
    /// Whisper only: scale audio_ctx to actual audio length so the encoder
    /// skips the silence-padded portion of its 30s window. Faster for short
    /// clips, marginal accuracy loss on edge cases.
    quick: bool = false,
    verbose: bool = false,
};

pub const ParseError = error{
    InvalidArgs,
    UnknownFlag,
    MissingValue,
} || std.mem.Allocator.Error || std.fmt.ParseIntError;

pub const usage_text =
    \\asrctl — macOS Apple Silicon ASR CLI
    \\
    \\Backends (--backend):
    \\  qwen3    Qwen3-ASR-0.6B via llama.cpp + mtmd. Strong multilingual,
    \\           best Chinese accuracy. ~1.5 GB model + mmproj. (default)
    \\  whisper  whisper.cpp + ggml-large-v3-turbo (Q5_0). OpenAI SOTA.
    \\           ~547 MB. Best for English / Latin languages.
    \\
    \\Usage:
    \\  asrctl <wav-file> [options]            transcribe a wav file
    \\  asrctl listen [options]                live mic → text (Ctrl-C to stop)
    \\  asrctl model path                      print resolved model paths
    \\  asrctl model pull                      download default backend model
    \\  asrctl version                         print version
    \\  asrctl help                            show this help
    \\
    \\Transcribe options:
    \\  -o, --output PATH    write text to file instead of stdout
    \\      --backend NAME   ASR backend: qwen3 (default) | whisper
    \\      --model PATH     override model gguf/bin path
    \\      --language CODE  hint language for whisper (en/zh/auto/...). Qwen3 auto-detects.
    \\      --server-url URL forward to llama-server (qwen3 only) instead of
    \\                       loading the model in-process
    \\      --quick          whisper: scale audio_ctx to actual length (fast,
    \\                       slight accuracy trade-off on edge cases)
    \\      --threads N      CPU threads (default 4)
    \\  -v, --verbose        print timing/diagnostic info to stderr
    \\
    \\Listen options (asrctl listen):
    \\  -o, --output PATH    append each utterance to file instead of stdout
    \\      --backend NAME   ASR backend: qwen3 (default) | whisper
    \\      --model PATH     override model path
    \\      --language CODE  hint language for whisper
    \\      --vad BACKEND    VAD backend: energy (default) | silero
    \\      --threshold F    VAD threshold (energy: RMS 0..1, silero: P 0..1)
    \\      --silence-ms N   silence duration that ends an utterance (default 600)
    \\      --partial        stream partial transcriptions during speech (whisper only)
    \\      --partial-ms N   partial cadence in ms (default 500)
    \\      --threads N      CPU threads (default 4)
    \\  -v, --verbose        print VAD/timing info to stderr
    \\
    \\Environment:
    \\  HF_HOME              HuggingFace cache root (default ~/.cache/huggingface)
    \\  HF_ENDPOINT          HF mirror, e.g. https://hf-mirror.com
    \\
;

/// argv contains the program name at [0]; parses argv[1..].
pub fn parse(argv: []const [*:0]const u8) ParseError!Subcommand {
    if (argv.len < 2) return .help;

    const a1 = std.mem.span(argv[1]);

    // Top-level keywords.
    if (std.mem.eql(u8, a1, "help") or std.mem.eql(u8, a1, "-h") or std.mem.eql(u8, a1, "--help")) {
        return .help;
    }
    if (std.mem.eql(u8, a1, "version") or std.mem.eql(u8, a1, "--version")) {
        return .version;
    }
    if (std.mem.eql(u8, a1, "model")) {
        if (argv.len < 3) return error.InvalidArgs;
        const sub = std.mem.span(argv[2]);
        if (std.mem.eql(u8, sub, "path")) return .model_path;
        if (std.mem.eql(u8, sub, "pull")) return .model_pull;
        return error.InvalidArgs;
    }
    if (std.mem.eql(u8, a1, "transcribe")) {
        return parseTranscribe(argv[2..]);
    }
    if (std.mem.eql(u8, a1, "listen")) {
        return parseListen(argv[2..]);
    }
    // Default: treat as `transcribe <a1> ...` when a1 looks like a file.
    return parseTranscribe(argv[1..]);
}

fn parseListen(rest: []const [*:0]const u8) ParseError!Subcommand {
    var args: ListenArgs = .{};
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const a = std.mem.span(rest[i]);
        if (std.mem.eql(u8, a, "-o") or std.mem.eql(u8, a, "--output")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.output_path = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--model")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.model_path = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--threads")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.threads = try std.fmt.parseInt(i32, std.mem.span(rest[i]), 10);
        } else if (std.mem.eql(u8, a, "--backend")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.backend = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--language")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.language = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--vad")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.vad = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--threshold")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.threshold = std.fmt.parseFloat(f32, std.mem.span(rest[i])) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, a, "--silence-ms")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.silence_ms = try std.fmt.parseInt(u32, std.mem.span(rest[i]), 10);
        } else if (std.mem.eql(u8, a, "--partial")) {
            args.partial = true;
        } else if (std.mem.eql(u8, a, "--partial-ms")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.partial_ms = try std.fmt.parseInt(u32, std.mem.span(rest[i]), 10);
        } else if (std.mem.eql(u8, a, "-v") or std.mem.eql(u8, a, "--verbose")) {
            args.verbose = true;
        } else {
            return error.UnknownFlag;
        }
    }
    return .{ .listen = args };
}

fn parseTranscribe(rest: []const [*:0]const u8) ParseError!Subcommand {
    var args: TranscribeArgs = .{ .audio_path = "" };
    var positional_seen = false;

    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const a = std.mem.span(rest[i]);
        if (std.mem.eql(u8, a, "-o") or std.mem.eql(u8, a, "--output")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.output_path = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--model")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.model_path = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--server-url")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.server_url = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--backend")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.backend = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--language")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.language = std.mem.span(rest[i]);
        } else if (std.mem.eql(u8, a, "--quick")) {
            args.quick = true;
        } else if (std.mem.eql(u8, a, "--threads")) {
            i += 1;
            if (i >= rest.len) return error.MissingValue;
            args.threads = try std.fmt.parseInt(i32, std.mem.span(rest[i]), 10);
        } else if (std.mem.eql(u8, a, "-v") or std.mem.eql(u8, a, "--verbose")) {
            args.verbose = true;
        } else if (std.mem.startsWith(u8, a, "-")) {
            return error.UnknownFlag;
        } else {
            if (positional_seen) return error.InvalidArgs;
            args.audio_path = a;
            positional_seen = true;
        }
    }
    if (!positional_seen) return error.InvalidArgs;
    return .{ .transcribe = args };
}
