//! Voice-activity detection.
//!
//! Plug-in voicer backends share the same FSM (idle/active state machine
//! with pre-roll, silence-cut, max-segment cap). Each backend just provides
//! a `frameSamples()` natural window size and an `isVoiced(frame)` predicate.
//!
//! Backends:
//!   - `.energy`:  RMS threshold. Zero deps. Adequate in quiet rooms.
//!   - `.silero`:  ggml-silero-v6 inference via vendored whisper.cpp VAD.
//!                 Better noise robustness; downloads ~1 MB model on first use.

const std = @import("std");
const c = @import("c");

pub const Backend = enum { energy, silero };

pub const Config = struct {
    backend: Backend = .energy,
    sample_rate: u32 = 16_000,
    /// Energy backend: RMS threshold (linear, 0..1). Ignored for silero.
    energy_threshold: f32 = 0.012,
    /// Silero backend: P(speech) threshold above which a frame is voiced.
    /// Ignored for energy.
    silero_threshold: f32 = 0.5,
    /// Path to the silero ggml model. Required when backend=.silero.
    silero_model_path: ?[:0]const u8 = null,
    /// How long an ambiguous silence run ends a segment. Used as the safety
    /// fallback when frames sit in the gray zone (energy backend always uses
    /// this; silero uses it for frames above `silero_strong_silence_p` but
    /// below `silero_threshold`).
    silence_ms: u32 = 600,
    /// Silero only: shorter silence cut when frames are confidently silent.
    /// Triggered by a streak of frames with P(speech) ≤ `silero_strong_silence_p`.
    /// Default 400 ms is the result of `bench-vad` on samples/{en,zh}_{short,long}:
    /// 250 ms cuts mid-speech in en_short (which contains a ~300 ms strong-
    /// silence stretch between syllables); 400 ms commits all 4 samples
    /// cleanly at 416-448 ms past silero's last-voiced frame, beating the
    /// pre-v0.11 safety-only path that fragmented every recording at the
    /// first 600 ms inner pause.
    silero_quick_silence_ms: u32 = 400,
    /// P(speech) at or below which a silero frame counts as "definitely
    /// silent" (drives the quick-silence cut).
    silero_strong_silence_p: f32 = 0.05,
    /// Audio kept before the first voiced frame so we don't clip onsets.
    preroll_ms: u32 = 200,
    /// Drop segments shorter than this (likely noise).
    min_segment_ms: u32 = 300,
    /// Force a segment cut after this much continuous voiced audio so a long
    /// monologue still gets transcribed periodically.
    max_segment_ms: u32 = 30_000,
};

pub const Error = error{
    SileroLoadFailed,
    SileroInferenceFailed,
    SileroNotImplemented, // v0.3.0: scaffolding only
} || std.mem.Allocator.Error;

/// Per-frame classification. `voiced` drives the FSM transitions; `strong_silence`
/// lets the Detector apply a quicker silence cut when the backend is highly
/// confident the room is silent (silero only — energy always reports false).
pub const FrameClass = struct {
    voiced: bool,
    strong_silence: bool,
};

/// Frame-level "is voiced" classifier.
pub const Voicer = union(Backend) {
    energy: EnergyVoicer,
    silero: SileroVoicer,

    pub fn frameSamples(self: Voicer) usize {
        return switch (self) {
            inline else => |v| v.frame_samples,
        };
    }

    pub fn classify(self: *Voicer, frame: []const f32) FrameClass {
        return switch (self.*) {
            .energy => |*v| v.classify(frame),
            .silero => |*v| v.classify(frame),
        };
    }

    pub fn deinit(self: *Voicer) void {
        switch (self.*) {
            .energy => {},
            .silero => |*v| v.deinit(),
        }
    }
};

pub fn voicerFromConfig(allocator: std.mem.Allocator, cfg: Config) Error!Voicer {
    switch (cfg.backend) {
        .energy => return .{
            .energy = .{
                .frame_samples = (cfg.sample_rate * 20) / 1000, // 20 ms
                .threshold = cfg.energy_threshold,
            },
        },
        .silero => {
            const path = cfg.silero_model_path orelse return error.SileroLoadFailed;
            return .{ .silero = try SileroVoicer.open(
                allocator,
                path,
                cfg.silero_threshold,
                cfg.silero_strong_silence_p,
            ) };
        },
    }
}

// ---------- Energy backend ----------

pub const EnergyVoicer = struct {
    frame_samples: usize,
    threshold: f32,

    pub fn classify(self: *EnergyVoicer, frame: []const f32) FrameClass {
        // Energy backend has no probability — we never claim strong silence
        // and let the detector fall through to its fixed silence_ms timeout.
        return .{ .voiced = rms(frame) > self.threshold, .strong_silence = false };
    }
};

fn rms(frame: []const f32) f32 {
    var sum: f64 = 0;
    for (frame) |s| sum += @as(f64, s) * s;
    return @floatCast(@sqrt(sum / @as(f64, @floatFromInt(frame.len))));
}

// ---------- Silero backend (scaffolding) ----------

/// Wraps a vendored whisper.cpp `whisper_vad_*` context. Frame size is fixed
/// at silero's expected window (512 samples = 32 ms at 16 kHz). Per-frame
/// inference uses `whisper_vad_detect_speech_no_reset` so LSTM state carries
/// over (correct streaming semantics; reset between utterances is
/// instantaneous and not currently needed).
pub const SileroVoicer = struct {
    frame_samples: usize,
    threshold: f32,
    strong_silence_p: f32,
    ctx: *c.whisper_vad_context,
    allocator: std.mem.Allocator,

    pub fn open(
        allocator: std.mem.Allocator,
        model_path: [:0]const u8,
        threshold: f32,
        strong_silence_p: f32,
    ) Error!SileroVoicer {
        var params = c.whisper_vad_default_context_params();
        // VAD is tiny; CPU is plenty and avoids Metal contention with the
        // main ASR context that's also using the GPU.
        params.use_gpu = false;
        params.n_threads = 1;
        const ctx = c.whisper_vad_init_from_file_with_params(model_path.ptr, params) orelse
            return error.SileroLoadFailed;
        return .{
            .frame_samples = 512,
            .threshold = threshold,
            .strong_silence_p = strong_silence_p,
            .ctx = ctx,
            .allocator = allocator,
        };
    }

    pub fn classify(self: *SileroVoicer, frame: []const f32) FrameClass {
        if (!c.whisper_vad_detect_speech_no_reset(self.ctx, frame.ptr, @intCast(frame.len)))
            return .{ .voiced = false, .strong_silence = true };
        const n = c.whisper_vad_n_probs(self.ctx);
        if (n <= 0) return .{ .voiced = false, .strong_silence = true };
        const probs = c.whisper_vad_probs(self.ctx);
        // For a 512-sample frame, n_probs is 1. If we ever feed larger
        // chunks (e.g. catch-up after lag), max across the window.
        var max_p: f32 = 0;
        var i: c_int = 0;
        while (i < n) : (i += 1) {
            const p = probs[@intCast(i)];
            if (p > max_p) max_p = p;
        }
        return .{
            .voiced = max_p >= self.threshold,
            .strong_silence = max_p <= self.strong_silence_p,
        };
    }

    pub fn deinit(self: *SileroVoicer) void {
        c.whisper_vad_free(self.ctx);
    }
};

// ---------- Detector (shared FSM) ----------

pub const Detector = struct {
    cfg: Config,
    voicer: Voicer,
    frame_samples: usize,
    silence_frames: u32,
    /// Frame count for the silero quick-cut path. 0 disables (energy backend
    /// or any non-positive computed value).
    quick_silence_frames: u32,
    preroll_samples: usize,
    min_samples: usize,
    max_samples: usize,

    state: enum { idle, active } = .idle,
    silence_run: u32 = 0,
    strong_silence_run: u32 = 0,
    /// Pre-roll buffer: kept rolling while idle, prepended to a segment when
    /// a voiced frame triggers transition.
    preroll: std.ArrayList(f32) = .empty,
    /// Active-segment buffer.
    segment: std.ArrayList(f32) = .empty,

    pub fn init(allocator: std.mem.Allocator, cfg: Config) Error!Detector {
        const voicer = try voicerFromConfig(allocator, cfg);
        const sr = cfg.sample_rate;
        const frame_ms = @as(u32, @intCast(voicer.frameSamples() * 1000 / sr));
        // Silence frames is at frame-cadence; ceil to avoid 0 when frame_ms
        // happens to exceed silence_ms (unusual but be defensive).
        const silence_frames = if (frame_ms == 0) 1 else @max(1, cfg.silence_ms / frame_ms);
        const quick_silence_frames: u32 = blk: {
            if (cfg.backend != .silero) break :blk 0;
            if (frame_ms == 0) break :blk 0;
            // Cap at the regular silence window — quick-cut should never run
            // longer than the safety fallback.
            break :blk @min(silence_frames, @max(1, cfg.silero_quick_silence_ms / frame_ms));
        };
        return .{
            .cfg = cfg,
            .voicer = voicer,
            .frame_samples = voicer.frameSamples(),
            .silence_frames = silence_frames,
            .quick_silence_frames = quick_silence_frames,
            .preroll_samples = (sr * cfg.preroll_ms) / 1000,
            .min_samples = (sr * cfg.min_segment_ms) / 1000,
            .max_samples = (sr * cfg.max_segment_ms) / 1000,
        };
    }

    pub fn deinit(self: *Detector, allocator: std.mem.Allocator) void {
        self.voicer.deinit();
        self.preroll.deinit(allocator);
        self.segment.deinit(allocator);
    }

    pub fn feed(
        self: *Detector,
        allocator: std.mem.Allocator,
        chunk: []const f32,
        ctx: anytype,
        on_segment: fn (@TypeOf(ctx), []const f32) anyerror!void,
    ) anyerror!void {
        var i: usize = 0;
        while (i + self.frame_samples <= chunk.len) : (i += self.frame_samples) {
            const frame = chunk[i .. i + self.frame_samples];
            const class = self.voicer.classify(frame);
            try self.processFrame(allocator, frame, class, ctx, on_segment);
        }
        if (i < chunk.len) {
            const tail = chunk[i..];
            switch (self.state) {
                .idle => try appendBounded(allocator, &self.preroll, tail, self.preroll_samples),
                .active => try self.segment.appendSlice(allocator, tail),
            }
        }
    }

    fn processFrame(
        self: *Detector,
        allocator: std.mem.Allocator,
        frame: []const f32,
        class: FrameClass,
        ctx: anytype,
        on_segment: fn (@TypeOf(ctx), []const f32) anyerror!void,
    ) anyerror!void {
        switch (self.state) {
            .idle => {
                try appendBounded(allocator, &self.preroll, frame, self.preroll_samples);
                if (class.voiced) {
                    try self.segment.appendSlice(allocator, self.preroll.items);
                    self.preroll.clearRetainingCapacity();
                    self.state = .active;
                    self.silence_run = 0;
                    self.strong_silence_run = 0;
                }
            },
            .active => {
                try self.segment.appendSlice(allocator, frame);
                if (class.voiced) {
                    self.silence_run = 0;
                    self.strong_silence_run = 0;
                } else {
                    self.silence_run += 1;
                    if (class.strong_silence) {
                        self.strong_silence_run += 1;
                    } else {
                        self.strong_silence_run = 0;
                    }
                }

                const safety_cut = self.silence_run >= self.silence_frames;
                const quick_cut = self.quick_silence_frames > 0 and
                    self.strong_silence_run >= self.quick_silence_frames;
                const too_long = self.segment.items.len >= self.max_samples;
                if (safety_cut or quick_cut or too_long) {
                    if (self.segment.items.len >= self.min_samples) {
                        try on_segment(ctx, self.segment.items);
                    }
                    self.segment.clearRetainingCapacity();
                    self.silence_run = 0;
                    self.strong_silence_run = 0;
                    self.state = .idle;
                }
            },
        }
    }

    pub fn flush(
        self: *Detector,
        ctx: anytype,
        on_segment: fn (@TypeOf(ctx), []const f32) anyerror!void,
    ) anyerror!void {
        if (self.state == .active and self.segment.items.len >= self.min_samples) {
            try on_segment(ctx, self.segment.items);
            self.segment.clearRetainingCapacity();
            self.state = .idle;
            self.silence_run = 0;
        }
    }
};

fn appendBounded(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(f32),
    src: []const f32,
    cap: usize,
) !void {
    try list.appendSlice(allocator, src);
    if (list.items.len > cap) {
        const drop = list.items.len - cap;
        std.mem.copyForwards(f32, list.items[0 .. list.items.len - drop], list.items[drop..]);
        list.shrinkRetainingCapacity(list.items.len - drop);
    }
}
