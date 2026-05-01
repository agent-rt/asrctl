//! Energy-based voice-activity detection.
//!
//! Splits a continuous PCM stream into utterances by RMS thresholding +
//! "this much silence ends a segment" rules. Not as good as silero/webrtc,
//! but adequate for v0.2 listen mode and ships with zero deps.
//!
//! Frame: a fixed-size window (default 20 ms). Each frame is "voiced" if its
//! RMS exceeds `threshold`. State machine:
//!   - idle:    waiting for first voiced frame; collected samples discarded
//!              except a small pre-roll so utterance onsets aren't clipped.
//!   - active:  collecting samples; if `silence_ms` of unvoiced frames seen
//!              → emit segment, return to idle.
//!
//! Output is via callback (push) rather than poll, so the caller can drive
//! transcription synchronously and back-pressure naturally.

const std = @import("std");

pub const Config = struct {
    sample_rate: u32 = 16_000,
    /// RMS threshold (linear, 0..1). 0.01 ≈ -40 dBFS, reasonable for a quiet
    /// room. Override per-environment via CLI later if needed.
    threshold: f32 = 0.012,
    /// Frame analysis window in ms.
    frame_ms: u32 = 20,
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

pub const Detector = struct {
    cfg: Config,
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

    pub fn init(cfg: Config) Detector {
        const sr = cfg.sample_rate;
        return .{
            .cfg = cfg,
            .frame_samples = (sr * cfg.frame_ms) / 1000,
            .silence_frames = cfg.silence_ms / cfg.frame_ms,
            .preroll_samples = (sr * cfg.preroll_ms) / 1000,
            .min_samples = (sr * cfg.min_segment_ms) / 1000,
            .max_samples = (sr * cfg.max_segment_ms) / 1000,
        };
    }

    pub fn deinit(self: *Detector, allocator: std.mem.Allocator) void {
        self.preroll.deinit(allocator);
        self.segment.deinit(allocator);
    }

    /// Feed a chunk of samples. For each completed segment, calls `on_segment`.
    /// Caller owns the slice handed to the callback for the duration of the
    /// callback only — copy if you need to retain.
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
            const voiced = rms(frame) > self.cfg.threshold;
            try self.processFrame(allocator, frame, voiced, ctx, on_segment);
        }
        // Tail: if there's a partial frame, append it to whatever buffer is
        // active. Not analyzed for VAD but not lost either.
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
                    // Promote pre-roll to segment.
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

    /// Force-emit any in-flight segment. Call on shutdown so the user's last
    /// utterance doesn't get lost.
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

fn rms(frame: []const f32) f32 {
    var sum: f64 = 0;
    for (frame) |s| sum += @as(f64, s) * s;
    return @floatCast(@sqrt(sum / @as(f64, @floatFromInt(frame.len))));
}

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
