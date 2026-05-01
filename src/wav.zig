//! Minimal RIFF/WAVE decoder.
//!
//! Supports PCM 16-bit, PCM 32-bit float, mono and stereo (downmixed to mono).
//! Returns a heap-allocated f32 PCM array. Used by the future `listen` command
//! and the in-process file path when we want raw samples (instead of letting
//! mtmd's miniaudio decode internally).

const std = @import("std");

pub const Error = error{
    NotRiffWave,
    UnsupportedFormat,
    Truncated,
    UnsupportedChannelCount,
} || std.mem.Allocator.Error;

pub const Decoded = struct {
    samples: []f32, // mono, native float
    sample_rate: u32,

    pub fn deinit(self: Decoded, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }
};

pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) Error!Decoded {
    if (bytes.len < 44) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF")) return error.NotRiffWave;
    if (!std.mem.eql(u8, bytes[8..12], "WAVE")) return error.NotRiffWave;

    // Walk chunks looking for fmt + data.
    var i: usize = 12;
    var fmt_audio_format: u16 = 0;
    var fmt_channels: u16 = 0;
    var fmt_sample_rate: u32 = 0;
    var fmt_bits_per_sample: u16 = 0;
    var data_offset: usize = 0;
    var data_size: usize = 0;

    while (i + 8 <= bytes.len) {
        const id = bytes[i .. i + 4];
        const sz = std.mem.readInt(u32, bytes[i + 4 .. i + 8][0..4], .little);
        const body = i + 8;
        if (body + sz > bytes.len) return error.Truncated;

        if (std.mem.eql(u8, id, "fmt ")) {
            if (sz < 16) return error.Truncated;
            fmt_audio_format = std.mem.readInt(u16, bytes[body .. body + 2][0..2], .little);
            fmt_channels = std.mem.readInt(u16, bytes[body + 2 .. body + 4][0..2], .little);
            fmt_sample_rate = std.mem.readInt(u32, bytes[body + 4 .. body + 8][0..4], .little);
            fmt_bits_per_sample = std.mem.readInt(u16, bytes[body + 14 .. body + 16][0..2], .little);
        } else if (std.mem.eql(u8, id, "data")) {
            data_offset = body;
            data_size = sz;
        }
        // Chunks pad to even sizes.
        i = body + sz + (sz & 1);
        if (data_offset != 0 and fmt_channels != 0) break;
    }

    if (data_offset == 0 or fmt_channels == 0) return error.Truncated;
    if (fmt_channels != 1 and fmt_channels != 2) return error.UnsupportedChannelCount;

    const data = bytes[data_offset .. data_offset + data_size];
    return switch (fmt_audio_format) {
        1 => switch (fmt_bits_per_sample) {
            16 => decodePcm16(allocator, data, fmt_channels, fmt_sample_rate),
            else => error.UnsupportedFormat,
        },
        3 => switch (fmt_bits_per_sample) {
            32 => decodeFloat32(allocator, data, fmt_channels, fmt_sample_rate),
            else => error.UnsupportedFormat,
        },
        else => error.UnsupportedFormat,
    };
}

pub fn decodeFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Decoded {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const buf = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(buf);

    var read_total: usize = 0;
    while (read_total < buf.len) {
        const slice = buf[read_total..];
        const bufs: []const []u8 = &.{slice};
        const n = file.readStreaming(io, bufs) catch |e| switch (e) {
            error.EndOfStream => break,
            else => return e,
        };
        if (n == 0) break;
        read_total += n;
    }
    return decode(allocator, buf[0..read_total]);
}

fn decodePcm16(allocator: std.mem.Allocator, data: []const u8, channels: u16, sample_rate: u32) Error!Decoded {
    const frames = data.len / (2 * channels);
    const out = try allocator.alloc(f32, frames);
    errdefer allocator.free(out);

    const inv = 1.0 / 32768.0;
    var f: usize = 0;
    while (f < frames) : (f += 1) {
        var sum: f32 = 0;
        var ch: usize = 0;
        while (ch < channels) : (ch += 1) {
            const off = (f * channels + ch) * 2;
            const s: i16 = std.mem.readInt(i16, data[off .. off + 2][0..2], .little);
            sum += @as(f32, @floatFromInt(s)) * inv;
        }
        out[f] = sum / @as(f32, @floatFromInt(channels));
    }
    return .{ .samples = out, .sample_rate = sample_rate };
}

fn decodeFloat32(allocator: std.mem.Allocator, data: []const u8, channels: u16, sample_rate: u32) Error!Decoded {
    const frames = data.len / (4 * channels);
    const out = try allocator.alloc(f32, frames);
    errdefer allocator.free(out);

    var f: usize = 0;
    while (f < frames) : (f += 1) {
        var sum: f32 = 0;
        var ch: usize = 0;
        while (ch < channels) : (ch += 1) {
            const off = (f * channels + ch) * 4;
            const bits = std.mem.readInt(u32, data[off .. off + 4][0..4], .little);
            sum += @bitCast(bits);
        }
        out[f] = sum / @as(f32, @floatFromInt(channels));
    }
    return .{ .samples = out, .sample_rate = sample_rate };
}
