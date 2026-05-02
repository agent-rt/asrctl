//! HuggingFace model file resolver and downloader.
//!
//! Strategy:
//!   1. Compute cache root from $HF_HOME / $XDG_CACHE_HOME/huggingface /
//!      ~/.cache/huggingface, matching huggingface_hub conventions.
//!   2. When resolving a file, FIRST scan the standard HF layout
//!      `<root>/hub/models--{org}--{repo}/snapshots/*/` so we transparently
//!      reuse files already downloaded by huggingface_hub or
//!      `llama-mtmd-cli -hf`.
//!   3. If not found there, fall back to a simple asrctl-owned layout
//!      `<root>/asrctl/{org}--{repo}/{filename}`. This is what we write to;
//!      we don't try to recreate the symlink+blob structure used by
//!      huggingface_hub.
//!   4. Downloads use `std.http.Client.fetch` (HTTPS to public domains works
//!      fine on macOS — Bug 1 in docs/v0.10-std-http.md only bites
//!      IP literals / `localhost`, not domain names).

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

pub const Error = error{
    HomeNotFound,
    DownloadFailed,
} || Allocator.Error;

pub const FileSpec = struct {
    repo: []const u8,
    filename: []const u8,
};

pub fn cacheRoot(allocator: Allocator, env: *std.process.Environ.Map) ![]u8 {
    if (env.get("HF_HOME")) |v| return allocator.dupe(u8, v);
    if (env.get("XDG_CACHE_HOME")) |v| {
        return std.fmt.allocPrint(allocator, "{s}/huggingface", .{v});
    }
    if (env.get("HOME")) |v| {
        return std.fmt.allocPrint(allocator, "{s}/.cache/huggingface", .{v});
    }
    return error.HomeNotFound;
}

pub fn endpoint(allocator: Allocator, env: *std.process.Environ.Map) ![]u8 {
    if (env.get("HF_ENDPOINT")) |v| return allocator.dupe(u8, v);
    return allocator.dupe(u8, "https://huggingface.co");
}

pub fn ensureFile(
    allocator: Allocator,
    io: Io,
    env: *std.process.Environ.Map,
    spec: FileSpec,
) ![]u8 {
    const root = try cacheRoot(allocator, env);
    defer allocator.free(root);

    if (try findInHfHub(allocator, io, root, spec)) |p| return p;

    const owned = try ownedPath(allocator, root, spec);

    const cwd = std.Io.Dir.cwd();
    if (cwd.access(io, owned, .{})) {
        return owned;
    } else |_| {}

    try download(allocator, io, env, spec, owned);
    return owned;
}

pub fn predictPath(
    allocator: Allocator,
    io: Io,
    env: *std.process.Environ.Map,
    spec: FileSpec,
) ![]u8 {
    const root = try cacheRoot(allocator, env);
    defer allocator.free(root);
    if (try findInHfHub(allocator, io, root, spec)) |p| return p;
    return ownedPath(allocator, root, spec);
}

fn findInHfHub(allocator: Allocator, io: Io, root: []const u8, spec: FileSpec) !?[]u8 {
    const owner_repo = try expandSlash(allocator, spec.repo);
    defer allocator.free(owner_repo);

    const snapshots_dir = try std.fmt.allocPrint(
        allocator,
        "{s}/hub/models--{s}/snapshots",
        .{ root, owner_repo },
    );
    defer allocator.free(snapshots_dir);

    const cwd = std.Io.Dir.cwd();
    var dir = cwd.openDir(io, snapshots_dir, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .directory) continue;
        const candidate = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}/{s}",
            .{ snapshots_dir, entry.name, spec.filename },
        );
        if (cwd.access(io, candidate, .{})) |_| {
            return candidate;
        } else |_| {
            allocator.free(candidate);
        }
    }
    return null;
}

/// Replace each '/' with '--' to match huggingface_hub's models-- prefix scheme.
fn expandSlash(allocator: Allocator, s: []const u8) ![]u8 {
    const slashes = std.mem.count(u8, s, "/");
    const out = try allocator.alloc(u8, s.len + slashes);
    var i: usize = 0;
    for (s) |ch| {
        if (ch == '/') {
            out[i] = '-';
            out[i + 1] = '-';
            i += 2;
        } else {
            out[i] = ch;
            i += 1;
        }
    }
    return out;
}

fn ownedPath(allocator: Allocator, root: []const u8, spec: FileSpec) ![]u8 {
    const owner_repo = try expandSlash(allocator, spec.repo);
    defer allocator.free(owner_repo);

    return std.fmt.allocPrint(
        allocator,
        "{s}/asrctl/{s}/{s}",
        .{ root, owner_repo, spec.filename },
    );
}

fn download(
    allocator: Allocator,
    io: Io,
    env: *std.process.Environ.Map,
    spec: FileSpec,
    dest: []const u8,
) !void {
    const ep = try endpoint(allocator, env);
    defer allocator.free(ep);

    const url = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}/resolve/main/{s}",
        .{ ep, spec.repo, spec.filename },
    );
    defer allocator.free(url);

    const cwd = std.Io.Dir.cwd();
    if (std.fs.path.dirname(dest)) |parent| {
        cwd.createDirPath(io, parent) catch |e| switch (e) {
            error.PathAlreadyExists => {},
            else => return e,
        };
    }

    // Status to stderr.
    std.debug.print("downloading {s}\n   → {s}\n", .{ url, dest });

    const file = cwd.createFile(io, dest, .{}) catch return error.DownloadFailed;
    defer file.close(io);

    var write_buf: [64 * 1024]u8 = undefined;
    var fw = file.writer(io, &write_buf);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &fw.interface,
    }) catch |e| {
        std.debug.print("  http error: {s}\n", .{@errorName(e)});
        return error.DownloadFailed;
    };
    fw.flush() catch return error.DownloadFailed;

    if (@intFromEnum(result.status) >= 400) {
        std.debug.print("  http status: {d}\n", .{@intFromEnum(result.status)});
        return error.DownloadFailed;
    }
}
