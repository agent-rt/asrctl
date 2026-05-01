//! Transcription pipeline.
//!
//! Two entry points:
//!   - `Session` — load model + mtmd + sampler once, transcribe many segments.
//!     Used by `asrctl listen` (v0.2 real-time).
//!   - `transcribe()` — one-shot helper for `asrctl transcribe <wav>`. Internally
//!     just wraps Session.
//!
//! Audio input is always raw f32 PCM at the model's expected sample rate
//! (typically 16 kHz). For wav files, the caller decodes via wav.zig first;
//! for the real-time path, the audio capture layer feeds PCM directly.

const std = @import("std");
const c = @import("c");
const wav = @import("wav.zig");

pub const Error = error{
    LoadModelFailed,
    InitContextFailed,
    InitMtmdFailed,
    LoadAudioFailed,
    TokenizeFailed,
    EvalChunksFailed,
    DecodeFailed,
    OutOfMemory,
    UnsupportedFormat,
    UnsupportedChannelCount,
    NotRiffWave,
    Truncated,
};

pub const Options = struct {
    model_path: [:0]const u8,
    mmproj_path: [:0]const u8,
    n_threads: i32 = 4,
    n_ctx: u32 = 4096,
    max_tokens: usize = 256,
};

pub const Result = struct {
    /// Parsed transcription text (after `<asr_text>` marker).
    text: []u8,
    /// Detected language (between "language " and "<asr_text>"), may be empty.
    language: []u8,
    /// Raw model output for debugging.
    raw: []u8,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.language);
        allocator.free(self.raw);
    }
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    model: *c.llama_model,
    lctx: *c.llama_context,
    mctx: *c.mtmd_context,
    smpl: *c.llama_sampler,
    vocab: *const c.llama_vocab,
    n_batch: i32,
    max_tokens: usize,

    pub fn open(allocator: std.mem.Allocator, opts: Options) Error!Session {
        c.llama_backend_init();
        errdefer c.llama_backend_free();

        var mparams = c.llama_model_default_params();
        mparams.n_gpu_layers = 99;
        const model = c.llama_model_load_from_file(opts.model_path.ptr, mparams) orelse
            return error.LoadModelFailed;
        errdefer c.llama_model_free(model);

        var cparams = c.llama_context_default_params();
        cparams.n_ctx = opts.n_ctx;
        cparams.n_batch = 2048;
        cparams.n_ubatch = 512;
        cparams.no_perf = true;
        const lctx = c.llama_init_from_model(model, cparams) orelse
            return error.InitContextFailed;
        errdefer c.llama_free(lctx);

        var mtmd_params = c.mtmd_context_params_default();
        mtmd_params.use_gpu = true;
        mtmd_params.print_timings = false;
        mtmd_params.n_threads = opts.n_threads;
        mtmd_params.warmup = false;
        const mctx = c.mtmd_init_from_file(opts.mmproj_path.ptr, model, mtmd_params) orelse
            return error.InitMtmdFailed;
        errdefer c.mtmd_free(mctx);

        if (!c.mtmd_support_audio(mctx)) return error.InitMtmdFailed;

        const sparams = c.llama_sampler_chain_default_params();
        const smpl = c.llama_sampler_chain_init(sparams) orelse return error.InitContextFailed;
        errdefer c.llama_sampler_free(smpl);
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_greedy());

        return .{
            .allocator = allocator,
            .model = model,
            .lctx = lctx,
            .mctx = mctx,
            .smpl = smpl,
            .vocab = c.llama_model_get_vocab(model).?,
            .n_batch = @intCast(cparams.n_batch),
            .max_tokens = opts.max_tokens,
        };
    }

    pub fn close(self: *Session) void {
        c.llama_sampler_free(self.smpl);
        c.mtmd_free(self.mctx);
        c.llama_free(self.lctx);
        c.llama_model_free(self.model);
        c.llama_backend_free();
    }

    /// Transcribe a buffer of mono f32 PCM samples. Resets the KV cache + the
    /// sampler so each call is independent.
    pub fn transcribePCM(self: *Session, samples: []const f32) Error!Result {
        const bitmap = c.mtmd_bitmap_init_from_audio(samples.len, samples.ptr) orelse
            return error.LoadAudioFailed;
        defer c.mtmd_bitmap_free(bitmap);
        return self.transcribeBitmap(bitmap);
    }

    /// File path entry point used by the one-shot `asrctl transcribe` command.
    /// Lets mtmd's bundled miniaudio handle wav/mp3/flac decoding.
    pub fn transcribeFile(self: *Session, path: [:0]const u8) Error!Result {
        const bitmap = c.mtmd_helper_bitmap_init_from_file(self.mctx, path.ptr) orelse
            return error.LoadAudioFailed;
        defer c.mtmd_bitmap_free(bitmap);
        return self.transcribeBitmap(bitmap);
    }

    fn transcribeBitmap(self: *Session, bitmap: *c.mtmd_bitmap) Error!Result {
        // Fresh KV cache for each segment so previous utterances don't bleed in.
        c.llama_memory_clear(c.llama_get_memory(self.lctx), true);
        c.llama_sampler_reset(self.smpl);

        const media_marker = std.mem.span(c.mtmd_default_marker());
        const prompt = try std.fmt.allocPrintSentinel(
            self.allocator,
            "<|im_start|>user\nTranscribe the audio.{s}<|im_end|>\n<|im_start|>assistant\n",
            .{media_marker},
            0,
        );
        defer self.allocator.free(prompt);

        const chunks = c.mtmd_input_chunks_init() orelse return error.TokenizeFailed;
        defer c.mtmd_input_chunks_free(chunks);

        var input_text: c.mtmd_input_text = .{
            .text = prompt.ptr,
            .add_special = true,
            .parse_special = true,
        };
        var bitmaps = [_]?*const c.mtmd_bitmap{bitmap};
        if (c.mtmd_tokenize(self.mctx, chunks, &input_text, &bitmaps, bitmaps.len) != 0)
            return error.TokenizeFailed;

        var n_past: c.llama_pos = 0;
        var new_n_past: c.llama_pos = 0;
        if (c.mtmd_helper_eval_chunks(
            self.mctx,
            self.lctx,
            chunks,
            n_past,
            0,
            self.n_batch,
            true,
            &new_n_past,
        ) != 0) return error.EvalChunksFailed;
        n_past = new_n_past;

        var raw: std.ArrayList(u8) = .empty;
        errdefer raw.deinit(self.allocator);

        var i: usize = 0;
        while (i < self.max_tokens) : (i += 1) {
            const token_id = c.llama_sampler_sample(self.smpl, self.lctx, -1);
            c.llama_sampler_accept(self.smpl, token_id);
            if (c.llama_vocab_is_eog(self.vocab, token_id)) break;

            var piece: [256]u8 = undefined;
            const n = c.llama_token_to_piece(self.vocab, token_id, &piece, piece.len, 0, false);
            if (n < 0) return error.DecodeFailed;
            try raw.appendSlice(self.allocator, piece[0..@intCast(n)]);

            var token_mut = token_id;
            const batch = c.llama_batch_get_one(&token_mut, 1);
            if (c.llama_decode(self.lctx, batch) != 0) return error.DecodeFailed;
            n_past += 1;
        }

        const raw_owned = try raw.toOwnedSlice(self.allocator);
        return parseOutput(self.allocator, raw_owned);
    }
};

/// One-shot helper: open a session, transcribe a file, close. Used by the
/// non-interactive `transcribe` subcommand.
pub fn transcribe(
    allocator: std.mem.Allocator,
    opts: Options,
    audio_path: [:0]const u8,
) Error!Result {
    var session = try Session.open(allocator, opts);
    defer session.close();
    return session.transcribeFile(audio_path);
}

/// Parses Qwen3-ASR's `language X<asr_text>Y` output into a Result. Takes
/// ownership of `raw`. Shared by both in-process and HTTP server paths.
pub fn parseOutput(allocator: std.mem.Allocator, raw: []u8) !Result {
    const tag = "<asr_text>";
    var lang_owned: []u8 = &.{};
    var text_owned: []u8 = &.{};
    if (std.mem.indexOf(u8, raw, tag)) |idx| {
        const lang_section = raw[0..idx];
        const trimmed_lang = std.mem.trim(
            u8,
            std.mem.trimStart(u8, lang_section, "language "),
            " \n",
        );
        lang_owned = try allocator.dupe(u8, trimmed_lang);
        const text_section = std.mem.trim(u8, raw[idx + tag.len ..], " \n");
        text_owned = try allocator.dupe(u8, text_section);
    } else {
        text_owned = try allocator.dupe(u8, std.mem.trim(u8, raw, " \n"));
    }
    return .{ .text = text_owned, .language = lang_owned, .raw = raw };
}

// Re-export for callers that want the wav decoder.
pub const Wav = wav;
