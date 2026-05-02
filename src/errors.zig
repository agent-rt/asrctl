//! Map internal errors to human-readable diagnostic strings.
//!
//! All asrctl error returns funnel through `friendly` so the user never sees a
//! bare `LoadModelFailed` enum name; they get a sentence + a hint.

const std = @import("std");

pub const Diagnostic = struct {
    summary: []const u8,
    hint: ?[]const u8 = null,
};

pub fn friendly(err: anyerror) Diagnostic {
    return switch (err) {
        // asr.zig
        error.BackendLoadFailed => .{
            .summary = "could not load ggml backend dylibs",
            .hint = "is `brew install ggml` installed and up to date?",
        },
        error.LoadModelFailed => .{
            .summary = "failed to load the model file",
            .hint = "model file may be corrupted; try `asrctl model pull` to redownload",
        },
        error.InitContextFailed => .{
            .summary = "failed to initialize the llama context",
            .hint = "out of memory? try lowering --threads or closing other apps",
        },
        error.InitMtmdFailed => .{
            .summary = "failed to initialize the multimodal context",
            .hint = "the mmproj file may be corrupted or for a different model",
        },
        error.LoadAudioFailed => .{
            .summary = "could not decode the audio file",
            .hint = "is it a valid 16-bit PCM or float wav?",
        },
        error.TokenizeFailed => .{ .summary = "tokenizer rejected the prompt" },
        error.EvalChunksFailed => .{ .summary = "model failed to evaluate audio chunks" },
        error.DecodeFailed => .{ .summary = "model failed to decode tokens" },

        // hf.zig
        error.HomeNotFound => .{
            .summary = "could not determine HuggingFace cache root",
            .hint = "set HF_HOME, XDG_CACHE_HOME, or HOME",
        },
        error.DownloadFailed => .{
            .summary = "model download failed",
            .hint = "check network, or set HF_ENDPOINT to a mirror like https://hf-mirror.com",
        },

        // server.zig
        error.SpawnFailed => .{
            .summary = "could not spawn curl",
            .hint = "is curl installed? (it ships with macOS by default)",
        },
        error.HttpFailed => .{
            .summary = "transcription request to llama-server failed",
            .hint = "is the server running? try `curl <url>/health`",
        },
        error.InvalidResponse => .{
            .summary = "llama-server returned a response we can't parse",
            .hint = "is the server actually serving Qwen3-ASR?",
        },

        // vad.zig
        error.SileroLoadFailed => .{
            .summary = "could not load silero VAD model",
            .hint = "try `asrctl model pull` to redownload, or omit --vad silero",
        },
        error.SileroInferenceFailed => .{ .summary = "silero VAD inference failed" },
        error.SileroNotImplemented => .{
            .summary = "silero VAD backend is not yet implemented",
            .hint = "use --vad energy (default) for now; silero lands in v0.3.1",
        },

        // generic
        error.OutOfMemory => .{ .summary = "out of memory" },
        else => .{ .summary = @errorName(err) },
    };
}

pub fn print(prefix: []const u8, err: anyerror) void {
    const d = friendly(err);
    if (d.hint) |hint| {
        std.debug.print("{s}: {s}\n  hint: {s}\n", .{ prefix, d.summary, hint });
    } else {
        std.debug.print("{s}: {s}\n", .{ prefix, d.summary });
    }
}
