//! Locate ggml's runtime backend dylibs.
//!
//! Recent ggml ships its CPU/Metal/BLAS backends as separate `.so` files that
//! must be loaded at runtime before `llama_model_load_from_file`, otherwise it
//! fails with "no backends are loaded".
//!
//! Brew maintains `/opt/homebrew/opt/ggml` as a symlink to the active version
//! (e.g. `Cellar/ggml/0.10.1`). Always preferring the symlink avoids picking
//! up a stale older version still sitting in `Cellar/`, which would otherwise
//! ABI-mismatch with the linked libggml-base.

const std = @import("std");

const candidates = [_][]const u8{
    "/opt/homebrew/opt/ggml/libexec",
    "/usr/local/opt/ggml/libexec",
};

pub const ResolveError = error{NoBackendsFound} || std.mem.Allocator.Error;

pub fn resolveLibexec(allocator: std.mem.Allocator, io: std.Io) ResolveError![]u8 {
    const cwd = std.Io.Dir.cwd();
    for (candidates) |path| {
        cwd.access(io, path, .{}) catch continue;
        return allocator.dupe(u8, path);
    }
    return error.NoBackendsFound;
}
