//! HTTP client for the `--server-url` fallback path.
//!
//! POSTs the wav file to llama-server's OpenAI-compatible
//! `/v1/audio/transcriptions` endpoint and returns the parsed transcription.
//! Same output protocol as the in-process path, so we hand the JSON `text`
//! field back through `asr.parseOutput` for the `<asr_text>` split.
//!
//! Implementation: spawn curl, capture stdout, parse JSON. M5 may switch to
//! std.http once the Io interface settles.

const std = @import("std");
const asr = @import("asr.zig");

pub const Error = error{
    SpawnFailed,
    HttpFailed,
    InvalidResponse,
} || std.mem.Allocator.Error || std.Io.Reader.Error;

pub fn transcribe(
    allocator: std.mem.Allocator,
    io: std.Io,
    server_url: []const u8,
    wav_path: []const u8,
) !asr.Result {
    // Endpoint: <server_url>/v1/audio/transcriptions. We don't append if the
    // user already gave a full path (e.g. they're routing through nginx).
    const endpoint = if (std.mem.indexOf(u8, server_url, "/v1/") != null)
        try allocator.dupe(u8, server_url)
    else
        try std.fmt.allocPrint(
            allocator,
            "{s}/v1/audio/transcriptions",
            .{std.mem.trimEnd(u8, server_url, "/")},
        );
    defer allocator.free(endpoint);

    const file_arg = try std.fmt.allocPrint(allocator, "file=@{s}", .{wav_path});
    defer allocator.free(file_arg);

    // Avoid wrangling pipe streams (Zig 0.16 std.Io.File.reader is in flux):
    // have curl write the JSON response to a temp file, then read it.
    // Single-process tool: use a fixed temp filename. We delete it after read.
    const tmp_path = "/tmp/asrctl-server-response.json";

    var child = std.process.spawn(io, .{
        .argv = &.{
            "curl", "-sS",    "--fail-with-body", "-X", "POST",
            "-o",   tmp_path, endpoint,           "-F", file_arg,
        },
        .stdout = .inherit,
        .stderr = .inherit,
    }) catch return error.SpawnFailed;
    const term = child.wait(io) catch return error.HttpFailed;
    switch (term) {
        .exited => |code| if (code != 0) return error.HttpFailed,
        else => return error.HttpFailed,
    }

    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, tmp_path, .{}) catch return error.InvalidResponse;
    defer file.close(io);
    defer cwd.deleteFile(io, tmp_path) catch {};

    // Drain the file into a heap buffer. ASR responses are small (low kB).
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    var read_buf: [4096]u8 = undefined;
    while (true) {
        const bufs: []const []u8 = &.{&read_buf};
        const n = file.readStreaming(io, bufs) catch |e| switch (e) {
            error.EndOfStream => break,
            else => return e,
        };
        if (n == 0) break;
        try body.appendSlice(allocator, read_buf[0..n]);
    }

    return parseJsonText(allocator, body.items);
}

/// Extract the "text" field from llama-server's JSON response and feed it
/// through asr.parseOutput.
fn parseJsonText(allocator: std.mem.Allocator, body: []const u8) !asr.Result {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        body,
        .{},
    ) catch return error.InvalidResponse;
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.InvalidResponse,
    };
    const text_val = root.get("text") orelse return error.InvalidResponse;
    const text = switch (text_val) {
        .string => |s| s,
        else => return error.InvalidResponse,
    };

    // parseOutput takes ownership of the slice it gets, so dupe.
    const raw = try allocator.dupe(u8, text);
    return asr.parseOutput(allocator, raw);
}
