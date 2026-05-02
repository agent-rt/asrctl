//! Transcription pipeline with two backends:
//!   - `.qwen3`:   Qwen3-ASR-0.6B via llama.cpp + mtmd. Strong multilingual,
//!                 best Chinese performance, 1.5 GB total (model + mmproj).
//!   - `.whisper`: whisper.cpp + ggml-large-v3-turbo (Q5_0). OpenAI SOTA,
//!                 streaming-capable transcription, ~547 MB.
//!
//! Both expose the same `Session` shape so callers don't care which is loaded.
//! `transcribe()` is a one-shot wrapper for `asrctl <wav>`. `Session.open` /
//! `transcribePCM` is what `asrctl listen` drives across utterances.
//!
//! Audio in is always mono f32 PCM at 16 kHz.

const std = @import("std");
const c = @import("c");
const wav = @import("wav.zig");

pub const Backend = enum { qwen3, whisper };

pub const Error = error{
    LoadModelFailed,
    InitContextFailed,
    InitMtmdFailed,
    LoadAudioFailed,
    TokenizeFailed,
    EvalChunksFailed,
    DecodeFailed,
    WhisperFullFailed,
    OutOfMemory,
    UnsupportedFormat,
    UnsupportedChannelCount,
    NotRiffWave,
    Truncated,
};

pub const Options = struct {
    backend: Backend = .qwen3,
    /// Qwen3 main model.
    model_path: [:0]const u8,
    /// Qwen3 mmproj. Ignored for `.whisper`.
    mmproj_path: ?[:0]const u8 = null,
    n_threads: i32 = 4,
    n_ctx: u32 = 4096,
    max_tokens: usize = 256,
    /// Whisper language code ("auto" / "en" / "zh" / ...). Only used by .whisper.
    language: ?[:0]const u8 = null,
};

pub const Result = struct {
    /// Parsed transcription text.
    text: []u8,
    /// Detected language for Qwen3 (from "<asr_text>" header). Whisper sets
    /// this from `whisper_full_lang_id`-style call when available.
    language: []u8,
    /// Raw model output for debugging. For Qwen3 this is the unparsed
    /// "language X<asr_text>Y" string; for whisper it equals `text`.
    raw: []u8,

    pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.language);
        allocator.free(self.raw);
    }
};

pub const Session = union(Backend) {
    qwen3: Qwen3Session,
    whisper: WhisperSession,

    pub fn open(allocator: std.mem.Allocator, opts: Options) Error!Session {
        return switch (opts.backend) {
            .qwen3 => .{ .qwen3 = try Qwen3Session.open(allocator, opts) },
            .whisper => .{ .whisper = try WhisperSession.open(allocator, opts) },
        };
    }

    pub fn close(self: *Session) void {
        switch (self.*) {
            .qwen3 => |*s| s.close(),
            .whisper => |*s| s.close(),
        }
    }

    pub fn transcribePCM(self: *Session, samples: []const f32) Error!Result {
        return switch (self.*) {
            .qwen3 => |*s| s.transcribePCM(samples),
            .whisper => |*s| s.transcribePCM(samples),
        };
    }

    /// Streaming-tuned partial transcribe. Whisper takes the audio_ctx-scaled
    /// fast path; qwen3 has no equivalent so it falls back to the regular
    /// transcribe (qwen3 is one-shot anyway and `--partial` gates on whisper).
    pub fn transcribePCMQuick(self: *Session, samples: []const f32) Error!Result {
        return switch (self.*) {
            .qwen3 => |*s| s.transcribePCM(samples),
            .whisper => |*s| s.transcribePCMQuick(samples),
        };
    }

    pub fn transcribeFile(self: *Session, path: [:0]const u8) Error!Result {
        return switch (self.*) {
            .qwen3 => |*s| s.transcribeFile(path),
            .whisper => |*s| s.transcribeFile(path),
        };
    }
};

// ---------- Qwen3 backend (llama.cpp + mtmd) ----------

pub const Qwen3Session = struct {
    allocator: std.mem.Allocator,
    model: *c.llama_model,
    lctx: *c.llama_context,
    mctx: *c.mtmd_context,
    smpl: *c.llama_sampler,
    vocab: *const c.llama_vocab,
    n_batch: i32,
    max_tokens: usize,

    pub fn open(allocator: std.mem.Allocator, opts: Options) Error!Qwen3Session {
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

        const mmproj = opts.mmproj_path orelse return error.InitMtmdFailed;
        var mtmd_params = c.mtmd_context_params_default();
        mtmd_params.use_gpu = true;
        mtmd_params.print_timings = false;
        mtmd_params.n_threads = opts.n_threads;
        mtmd_params.warmup = false;
        const mctx = c.mtmd_init_from_file(mmproj.ptr, model, mtmd_params) orelse
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

    pub fn close(self: *Qwen3Session) void {
        c.llama_sampler_free(self.smpl);
        c.mtmd_free(self.mctx);
        c.llama_free(self.lctx);
        c.llama_model_free(self.model);
        c.llama_backend_free();
    }

    pub fn transcribePCM(self: *Qwen3Session, samples: []const f32) Error!Result {
        const bitmap = c.mtmd_bitmap_init_from_audio(samples.len, samples.ptr) orelse
            return error.LoadAudioFailed;
        defer c.mtmd_bitmap_free(bitmap);
        return self.transcribeBitmap(bitmap);
    }

    pub fn transcribeFile(self: *Qwen3Session, path: [:0]const u8) Error!Result {
        const bitmap = c.mtmd_helper_bitmap_init_from_file(self.mctx, path.ptr) orelse
            return error.LoadAudioFailed;
        defer c.mtmd_bitmap_free(bitmap);
        return self.transcribeBitmap(bitmap);
    }

    fn transcribeBitmap(self: *Qwen3Session, bitmap: *c.mtmd_bitmap) Error!Result {
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
        return parseQwen3Output(self.allocator, raw_owned);
    }
};

/// Parses Qwen3-ASR's `language X<asr_text>Y` output into a Result. Takes
/// ownership of `raw`. Shared with the HTTP server path.
pub fn parseQwen3Output(allocator: std.mem.Allocator, raw: []u8) !Result {
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

// Backwards-compat alias for server.zig.
pub const parseOutput = parseQwen3Output;

// ---------- Whisper backend (whisper.cpp) ----------

pub const WhisperSession = struct {
    allocator: std.mem.Allocator,
    ctx: *c.whisper_context,
    n_threads: i32,
    language: ?[:0]const u8,
    verbose_timings: bool = false,

    pub fn open(allocator: std.mem.Allocator, opts: Options) Error!WhisperSession {
        var cparams = c.whisper_context_default_params();
        cparams.use_gpu = true;
        cparams.flash_attn = true;
        const ctx = c.whisper_init_from_file_with_params(opts.model_path.ptr, cparams) orelse
            return error.LoadModelFailed;
        // ASRCTL_TIMINGS=1 prints whisper internal timings after each
        // inference. Used by bench/whisper-partial-bench.sh; not a CLI flag.
        const timings = std.c.getenv("ASRCTL_TIMINGS") != null;
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .n_threads = opts.n_threads,
            .language = opts.language,
            .verbose_timings = timings,
        };
    }

    pub fn close(self: *WhisperSession) void {
        c.whisper_free(self.ctx);
    }

    pub fn transcribePCM(self: *WhisperSession, samples: []const f32) Error!Result {
        return self.transcribeWithCtx(samples, 0); // 0 = full 30s encoder context
    }

    /// Streaming-tuned variant: scale `audio_ctx` to the actual audio length
    /// so the encoder skips the silence-padded portion of its 30s window.
    /// Used by `--partial` to cut per-call cost ~10x for short buffers.
    /// Final transcribe (after silence cut) still uses full context for max
    /// accuracy.
    pub fn transcribePCMQuick(self: *WhisperSession, samples: []const f32) Error!Result {
        // Scale audio_ctx so the encoder skips most of its 30s padding window.
        // Empirically, audio_ctx units = mel frames at 50 Hz; 30s = 1500.
        // Naive `seconds * 50` produces broken decoder output (the decoder
        // attends over too few audio frames and either skips text or loops).
        // A floor of 768 (≈15 s of context) is the sweet spot: encoder saves
        // ~30 % vs full 1500 while decoder still has enough context to behave.
        const sec_x50 = (samples.len * 50) / 16_000;
        const audio_ctx: c_int = @intCast(@min(@as(usize, 1500), @max(@as(usize, 768), sec_x50 + 256)));
        return self.transcribeWithCtx(samples, audio_ctx);
    }

    fn transcribeWithCtx(self: *WhisperSession, samples: []const f32, audio_ctx: c_int) Error!Result {
        var fparams = c.whisper_full_default_params(c.WHISPER_SAMPLING_GREEDY);
        fparams.print_progress = false;
        fparams.print_special = false;
        fparams.print_realtime = false;
        fparams.print_timestamps = false;
        fparams.translate = false;
        fparams.n_threads = self.n_threads;
        fparams.no_context = true; // independent utterances
        fparams.single_segment = false;
        fparams.suppress_blank = true;
        fparams.no_timestamps = true;
        fparams.suppress_nst = true;
        fparams.audio_ctx = audio_ctx;
        if (self.language) |lang| {
            fparams.language = lang.ptr;
        } else {
            fparams.language = "auto";
        }

        if (c.whisper_full(self.ctx, fparams, samples.ptr, @intCast(samples.len)) != 0)
            return error.WhisperFullFailed;

        if (self.verbose_timings) c.whisper_print_timings(self.ctx);

        // Concatenate all segment texts.
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.allocator);

        const n = c.whisper_full_n_segments(self.ctx);
        var i: c_int = 0;
        while (i < n) : (i += 1) {
            const seg = c.whisper_full_get_segment_text(self.ctx, i);
            if (seg) |s| {
                const slice = std.mem.span(s);
                // Whisper segments often start with a leading space.
                try out.appendSlice(self.allocator, slice);
            }
        }

        const text_owned = try self.allocator.dupe(u8, std.mem.trim(u8, out.items, " \n"));
        const raw_owned = try out.toOwnedSlice(self.allocator);

        // Detected language id → string.
        const lang_id = c.whisper_full_lang_id(self.ctx);
        const lang_str = c.whisper_lang_str(lang_id);
        const lang_owned = try self.allocator.dupe(u8, std.mem.span(lang_str));

        return .{ .text = text_owned, .language = lang_owned, .raw = raw_owned };
    }

    pub fn transcribeFile(self: *WhisperSession, path: [:0]const u8) Error!Result {
        const decoded = wav.decodeFileSimple(self.allocator, std.mem.span(path.ptr)) catch
            return error.LoadAudioFailed;
        defer decoded.deinit(self.allocator);
        return self.transcribePCM(decoded.samples);
    }
};

// ---------- one-shot helper ----------

pub fn transcribe(
    allocator: std.mem.Allocator,
    opts: Options,
    audio_path: [:0]const u8,
) Error!Result {
    var session = try Session.open(allocator, opts);
    defer session.close();
    return session.transcribeFile(audio_path);
}

pub const Wav = wav;
