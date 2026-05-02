//! Voice-activity detection.
//!
//! Plug-in voicer backends share the same FSM (idle/active state machine
//! with pre-roll, silence-cut, max-segment cap). Each backend just provides
//! a `frameSamples()` natural window size and an `isVoiced(frame)` predicate.
//!
//! Backends:
//!   - `.energy`:  RMS threshold. Zero deps. Adequate in quiet rooms.
//!   - `.silero`:  ggml-silero-v5 inference via vendored whisper.cpp VAD.
//!                 Better noise robustness; downloads ~2 MB model on first use.
//!                 (NOTE: scaffolding only; real inference lands in v0.3.1
//!                 once we vendor the upstream whisper-vad source.)

const std = @import("std");

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
    /// How long a silence run ends a segment.
    silence_ms: u32 = 600,
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

/// Frame-level "is voiced" classifier.
pub const Voicer = union(Backend) {
    energy: EnergyVoicer,
    silero: SileroVoicer,

    pub fn frameSamples(self: Voicer) usize {
        return switch (self) {
            inline else => |v| v.frame_samples,
        };
    }

    pub fn isVoiced(self: *Voicer, frame: []const f32) bool {
        return switch (self.*) {
            .energy => |*v| v.isVoiced(frame),
            .silero => |*v| v.isVoiced(frame),
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
            return .{ .silero = try SileroVoicer.open(allocator, path, cfg.silero_threshold) };
        },
    }
}

// ---------- Energy backend ----------

pub const EnergyVoicer = struct {
    frame_samples: usize,
    threshold: f32,

    pub fn isVoiced(self: *EnergyVoicer, frame: []const f32) bool {
        return rms(frame) > self.threshold;
    }
};

fn rms(frame: []const f32) f32 {
    var sum: f64 = 0;
    for (frame) |s| sum += @as(f64, s) * s;
    return @floatCast(@sqrt(sum / @as(f64, @floatFromInt(frame.len))));
}

// ---------- Silero backend (scaffolding) ----------

/// Wraps a vendored whisper.cpp `whisper_vad_*` context. Frame size is fixed
/// at silero v5's expected window (512 samples = 32 ms at 16 kHz).
pub const SileroVoicer = struct {
    frame_samples: usize,
    threshold: f32,
    /// Opaque handle to the loaded VAD context. v0.3.0: nil until v0.3.1
    /// wires up the actual whisper-vad C API.
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,

    pub fn open(
        allocator: std.mem.Allocator,
        model_path: [:0]const u8,
        threshold: f32,
    ) Error!SileroVoicer {
        _ = model_path;
        // v0.3.0: not yet wired. The infrastructure (CLI flag, model fetch,
        // backend dispatch) is in place so v0.3.1 only needs to fill in the
        // ggml inference call.
        return .{
            .frame_samples = 512,
            .threshold = threshold,
            .ctx = null,
            .allocator = allocator,
        };
    }

    pub fn isVoiced(self: *SileroVoicer, frame: []const f32) bool {
        _ = self;
        _ = frame;
        // Returning true here would make every frame voiced → infinite
        // segment. Return false → no segments. Neither is right; bail out.
        @panic("silero VAD not yet implemented (v0.3.1)");
    }

    pub fn deinit(self: *SileroVoicer) void {
        _ = self;
    }
};

// ---------- Detector (shared FSM) ----------

pub const Detector = struct {
    cfg: Config,
    voicer: Voicer,
    frame_samples: usize,
    silence_frames: u32,
    preroll_samples: usize,
    min_samples: usize,
    max_samples: usize,

    state: enum { idle, active } = .idle,
    silence_run: u32 = 0,
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
        return .{
            .cfg = cfg,
            .voicer = voicer,
            .frame_samples = voicer.frameSamples(),
            .silence_frames = silence_frames,
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
            const voiced = self.voicer.isVoiced(frame);
            try self.processFrame(allocator, frame, voiced, ctx, on_segment);
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
        voiced: bool,
        ctx: anytype,
        on_segment: fn (@TypeOf(ctx), []const f32) anyerror!void,
    ) anyerror!void {
        switch (self.state) {
            .idle => {
                try appendBounded(allocator, &self.preroll, frame, self.preroll_samples);
                if (voiced) {
                    try self.segment.appendSlice(allocator, self.preroll.items);
                    self.preroll.clearRetainingCapacity();
                    self.state = .active;
                    self.silence_run = 0;
                }
            },
            .active => {
                try self.segment.appendSlice(allocator, frame);
                if (voiced) {
                    self.silence_run = 0;
                } else {
                    self.silence_run += 1;
                }

                const long_enough = self.silence_run >= self.silence_frames;
                const too_long = self.segment.items.len >= self.max_samples;
                if (long_enough or too_long) {
                    if (self.segment.items.len >= self.min_samples) {
                        try on_segment(ctx, self.segment.items);
                    }
                    self.segment.clearRetainingCapacity();
                    self.silence_run = 0;
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
