//! HTTP client for the `--server-url` fallback path.
//!
//! POSTs the wav file to llama-server's OpenAI-compatible
//! `/v1/audio/transcriptions` endpoint and returns the parsed transcription.
//! Same output protocol as the in-process path, so we hand the JSON `text`
//! field back through `asr.parseOutput` for the `<asr_text>` split.
//!
//! Implementation: hand-rolled HTTP/1.1 over a TCP stream. We can't use
//! `std.http.Client.fetch` here because Zig 0.16's name lookup on macOS does
//! not short-circuit IP literals or `localhost` correctly — see
//! docs/v0.10-std-http.md. Bypassing the lookup is straightforward:
//! parse the URL, treat the host as an IP literal (or rewrite `localhost` →
//! `127.0.0.1`), `IpAddress.connect`, write request line + headers + multipart
//! body, read response, parse JSON.

const std = @import("std");
const asr = @import("asr.zig");

pub const Error = error{
    InvalidServerUrl,
    HttpFailed,
    InvalidResponse,
} || std.mem.Allocator.Error;

pub fn transcribe(
    allocator: std.mem.Allocator,
    io: std.Io,
    server_url: []const u8,
    wav_path: []const u8,
) !asr.Result {
    const target = parseUrl(server_url) catch return error.InvalidServerUrl;
    if (!target.is_http) return error.InvalidServerUrl; // we don't do TLS here

    // Endpoint path: append /v1/audio/transcriptions unless the user already
    // gave a full path (e.g. routing through nginx with a custom mount).
    const path = if (std.mem.indexOf(u8, target.path, "/v1/") != null)
        try allocator.dupe(u8, target.path)
    else blk: {
        const base = std.mem.trimEnd(u8, target.path, "/");
        break :blk try std.fmt.allocPrint(
            allocator,
            "{s}/v1/audio/transcriptions",
            .{base},
        );
    };
    defer allocator.free(path);

    const addr = resolveLocal(target.host, target.port) orelse return error.InvalidServerUrl;

    // Read wav into memory. ASR clips are small (<60s @ 16k mono = <2 MB), so
    // a single allocation is fine and lets us emit a known Content-Length.
    const wav_bytes = try readWholeFile(allocator, io, wav_path);
    defer allocator.free(wav_bytes);

    const boundary = "asrctlBoundaryX9F3K2A1";
    const filename = std.fs.path.basename(wav_path);
    const head = try std.fmt.allocPrint(
        allocator,
        "--{s}\r\n" ++
            "Content-Disposition: form-data; name=\"file\"; filename=\"{s}\"\r\n" ++
            "Content-Type: audio/wav\r\n\r\n",
        .{ boundary, filename },
    );
    defer allocator.free(head);
    const tail = try std.fmt.allocPrint(allocator, "\r\n--{s}--\r\n", .{boundary});
    defer allocator.free(tail);
    const body_len = head.len + wav_bytes.len + tail.len;

    const stream = addr.connect(io, .{ .mode = .stream }) catch return error.HttpFailed;
    defer stream.close(io);

    var write_buf: [4096]u8 = undefined;
    var sw = stream.writer(io, &write_buf);
    const w = &sw.interface;

    w.print("POST {s} HTTP/1.1\r\n", .{path}) catch return error.HttpFailed;
    w.print("Host: {s}:{d}\r\n", .{ target.host, target.port }) catch return error.HttpFailed;
    w.print("Content-Type: multipart/form-data; boundary={s}\r\n", .{boundary}) catch return error.HttpFailed;
    w.print("Content-Length: {d}\r\n", .{body_len}) catch return error.HttpFailed;
    // Connection: close lets us drain the response body by reading until EOF
    // instead of parsing Content-Length / chunked from the response headers.
    w.writeAll("Connection: close\r\n\r\n") catch return error.HttpFailed;
    w.writeAll(head) catch return error.HttpFailed;
    w.writeAll(wav_bytes) catch return error.HttpFailed;
    w.writeAll(tail) catch return error.HttpFailed;
    w.flush() catch return error.HttpFailed;

    var read_buf: [16 * 1024]u8 = undefined;
    var sr = stream.reader(io, &read_buf);
    const r = &sr.interface;

    const status = readStatusCode(r) catch return error.HttpFailed;
    skipHeaders(r) catch return error.HttpFailed;

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    r.appendRemainingUnlimited(allocator, &body) catch return error.HttpFailed;

    if (status >= 400) return error.HttpFailed;

    return parseJsonText(allocator, body.items);
}

const Url = struct {
    is_http: bool,
    host: []const u8,
    port: u16,
    path: []const u8,
};

fn parseUrl(url: []const u8) !Url {
    const after_scheme = if (std.mem.startsWith(u8, url, "http://"))
        url["http://".len..]
    else if (std.mem.startsWith(u8, url, "https://"))
        return error.InvalidServerUrl // TLS not supported on this path
    else
        return error.InvalidServerUrl;

    const slash = std.mem.indexOfScalar(u8, after_scheme, '/');
    const authority = if (slash) |i| after_scheme[0..i] else after_scheme;
    const path = if (slash) |i| after_scheme[i..] else "/";

    var host: []const u8 = authority;
    var port: u16 = 80;
    if (authority.len > 0 and authority[0] == '[') {
        // IPv6 literal: [::1]:8080
        const close = std.mem.indexOfScalar(u8, authority, ']') orelse return error.InvalidServerUrl;
        host = authority[1..close];
        if (close + 1 < authority.len) {
            if (authority[close + 1] != ':') return error.InvalidServerUrl;
            port = try std.fmt.parseInt(u16, authority[close + 2 ..], 10);
        }
    } else if (std.mem.lastIndexOfScalar(u8, authority, ':')) |colon| {
        host = authority[0..colon];
        port = try std.fmt.parseInt(u16, authority[colon + 1 ..], 10);
    }

    return .{ .is_http = true, .host = host, .port = port, .path = path };
}

/// Resolve a hostname to an IpAddress without going through the OS resolver.
/// We only support IP literals and `localhost` because those are the cases
/// std.http.Client breaks on. Real DNS names would route the user back into
/// the broken stdlib path.
fn resolveLocal(host: []const u8, port: u16) ?std.Io.net.IpAddress {
    if (std.mem.eql(u8, host, "localhost")) {
        return .{ .ip4 = .loopback(port) };
    }
    if (std.Io.net.IpAddress.parseIp4(host, port)) |a| return a else |_| {}
    if (std.Io.net.IpAddress.parseIp6(host, port)) |a| return a else |_| {}
    return null;
}

fn readWholeFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var fr = file.reader(io, &buf);
    return fr.interface.allocRemaining(allocator, .unlimited) catch |e| switch (e) {
        error.StreamTooLong => unreachable, // .unlimited
        else => |x| return x,
    };
}

/// Reads one CRLF-terminated line from `r`, consuming the LF. Returned slice
/// points into the reader's buffer (valid until the next read) with `\r` (if
/// any) trimmed.
fn readLine(r: *std.Io.Reader) ![]const u8 {
    const line = try r.takeDelimiterInclusive('\n');
    return std.mem.trimEnd(u8, line[0 .. line.len - 1], "\r");
}

/// Reads the HTTP status line ("HTTP/1.1 NNN ...") and returns the numeric code.
fn readStatusCode(r: *std.Io.Reader) !u16 {
    const line = try readLine(r);
    var it = std.mem.splitScalar(u8, line, ' ');
    _ = it.next() orelse return error.InvalidResponse; // HTTP/1.1
    const code_s = it.next() orelse return error.InvalidResponse;
    return std.fmt.parseInt(u16, code_s, 10) catch error.InvalidResponse;
}

/// Reads header lines until the empty CRLF terminator.
fn skipHeaders(r: *std.Io.Reader) !void {
    while (true) {
        const line = try readLine(r);
        if (line.len == 0) return;
    }
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
