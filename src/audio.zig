//! macOS microphone capture via AudioToolbox AudioQueue.
//!
//! Avoids `translate-c <AudioToolbox/AudioQueue.h>` because Apple's SDK pulls
//! in mach_msg.h which translate-c can't size. Instead we declare the small
//! subset we need as raw externs. Stable C ABI; no Objective-C blocks.
//!
//! Captures mono f32 PCM at 16 kHz (matching what mtmd's audio encoder
//! expects). Two real-time-safety properties:
//!
//!   1. **Lock-free SPSC ring buffer** — fixed-size, preallocated. The audio
//!      callback (real-time thread) only does atomic head/tail loads + stores
//!      and a memcpy. Zero allocator calls on the hot path. Drop-newest on
//!      overflow with an atomic dropped-sample counter for diagnostics.
//!
//!   2. **Wakeup pipe** — the callback writes one non-blocking byte to a pipe
//!      so a `poll()`-blocked consumer wakes the moment audio arrives. Removes
//!      the 50 ms `usleep` polling floor that used to sit between mic capture
//!      and VAD reaction. write(2) ≤ PIPE_BUF is atomic and real-time-safe;
//!      EAGAIN (pipe already full) just means a wakeup is already pending —
//!      ignore it.

const std = @import("std");

pub const Error = error{
    NewQueueFailed,
    AllocBufferFailed,
    EnqueueFailed,
    StartFailed,
    PipeFailed,
    DeviceNotFound,
    DeviceLookupFailed,
} || std.mem.Allocator.Error;

// ---------- AudioQueue C ABI bindings ----------

const OSStatus = i32;
const Float64 = f64;
const UInt32 = u32;
const SInt64 = i64;

const AudioStreamBasicDescription = extern struct {
    mSampleRate: Float64,
    mFormatID: u32,
    mFormatFlags: u32,
    mBytesPerPacket: u32,
    mFramesPerPacket: u32,
    mBytesPerFrame: u32,
    mChannelsPerFrame: u32,
    mBitsPerChannel: u32,
    mReserved: u32,
};

const AudioQueueRef = ?*anyopaque;
const AudioQueueBufferRef = ?*AudioQueueBuffer;

const AudioQueueBuffer = extern struct {
    mAudioDataBytesCapacity: u32,
    mAudioData: ?*anyopaque,
    mAudioDataByteSize: u32,
    mUserData: ?*anyopaque,
    // The struct continues with mPacketDescriptionCapacity etc., but writing
    // to the prefix is enough for our PCM use case (no packet descriptions).
};

const AudioTimeStamp = extern struct {
    mSampleTime: Float64,
    mHostTime: u64,
    mRateScalar: Float64,
    mWordClockTime: u64,
    mSMPTETime: extern struct {
        mSubframes: i16,
        mSubframeDivisor: i16,
        mCounter: u32,
        mType: u32,
        mFlags: u32,
        mHours: i16,
        mMinutes: i16,
        mSeconds: i16,
        mFrames: i16,
    },
    mFlags: u32,
    mReserved: u32,
};

const AudioStreamPacketDescription = extern struct {
    mStartOffset: SInt64,
    mVariableFramesInPacket: UInt32,
    mDataByteSize: UInt32,
};

const InputCallback = *const fn (
    user_data: ?*anyopaque,
    queue: AudioQueueRef,
    buffer: AudioQueueBufferRef,
    start_time: ?*const AudioTimeStamp,
    n_packet_descriptions: u32,
    packet_descs: ?*const AudioStreamPacketDescription,
) callconv(.c) void;

extern "c" fn AudioQueueNewInput(
    inFormat: *const AudioStreamBasicDescription,
    inCallbackProc: InputCallback,
    inUserData: ?*anyopaque,
    inCallbackRunLoop: ?*anyopaque,
    inCallbackRunLoopMode: ?*anyopaque,
    inFlags: u32,
    outAQ: *AudioQueueRef,
) OSStatus;

extern "c" fn AudioQueueAllocateBuffer(
    inAQ: AudioQueueRef,
    inBufferByteSize: u32,
    outBuffer: *AudioQueueBufferRef,
) OSStatus;

extern "c" fn AudioQueueEnqueueBuffer(
    inAQ: AudioQueueRef,
    inBuffer: AudioQueueBufferRef,
    inNumPacketDescs: u32,
    inPacketDescs: ?*const AudioStreamPacketDescription,
) OSStatus;

extern "c" fn AudioQueueStart(inAQ: AudioQueueRef, inStartTime: ?*const AudioTimeStamp) OSStatus;
extern "c" fn AudioQueueStop(inAQ: AudioQueueRef, inImmediate: u8) OSStatus;
extern "c" fn AudioQueueDispose(inAQ: AudioQueueRef, inImmediate: u8) OSStatus;
extern "c" fn AudioQueueSetProperty(
    inAQ: AudioQueueRef,
    inID: u32,
    inData: *const anyopaque,
    inDataSize: u32,
) OSStatus;

const kLinearPCMFormatFlagIsFloat: u32 = 1 << 0;
const kLinearPCMFormatFlagIsPacked: u32 = 1 << 3;

// ---------- CoreAudio HAL + CoreFoundation bindings (device enumeration) ----------

const AudioObjectID = u32;

const AudioObjectPropertyAddress = extern struct {
    mSelector: u32,
    mScope: u32,
    mElement: u32,
};

const kAudioObjectSystemObject: AudioObjectID = 1;

extern "c" fn AudioObjectGetPropertyDataSize(
    inObjectID: AudioObjectID,
    inAddress: *const AudioObjectPropertyAddress,
    inQualifierDataSize: u32,
    inQualifierData: ?*const anyopaque,
    outDataSize: *u32,
) OSStatus;

extern "c" fn AudioObjectGetPropertyData(
    inObjectID: AudioObjectID,
    inAddress: *const AudioObjectPropertyAddress,
    inQualifierDataSize: u32,
    inQualifierData: ?*const anyopaque,
    ioDataSize: *u32,
    outData: *anyopaque,
) OSStatus;

const CFStringRef = ?*opaque {};
const kCFStringEncodingUTF8: u32 = 0x08000100;

extern "c" fn CFStringGetLength(s: CFStringRef) c_long;
extern "c" fn CFStringGetMaximumSizeForEncoding(len: c_long, enc: u32) c_long;
extern "c" fn CFStringGetCString(
    s: CFStringRef,
    buffer: [*]u8,
    bufferSize: c_long,
    encoding: u32,
) u8;
extern "c" fn CFStringCreateWithCString(
    alloc: ?*anyopaque,
    cstr: [*:0]const u8,
    encoding: u32,
) CFStringRef;
extern "c" fn CFRelease(o: ?*anyopaque) void;

pub const Device = struct {
    /// User-facing device name, e.g. "MacBook Air Microphone" or "BlackHole 2ch".
    name: []u8,
    /// Stable UID that AudioQueue accepts via kAudioQueueProperty_CurrentDevice.
    uid: []u8,

    pub fn deinit(self: Device, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.uid);
    }
};

/// Enumerate all input-capable devices on the system. Caller owns the slice
/// and each Device's name/uid; use `freeDevices` to release them all.
pub fn listInputDevices(allocator: std.mem.Allocator) ![]Device {
    const all_devices_addr = AudioObjectPropertyAddress{
        .mSelector = magic("dev#"), // kAudioHardwarePropertyDevices
        .mScope = magic("glob"), // kAudioObjectPropertyScopeGlobal
        .mElement = 0,
    };
    var size: u32 = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &all_devices_addr, 0, null, &size) != 0)
        return error.DeviceLookupFailed;
    const n = size / @sizeOf(AudioObjectID);
    if (n == 0) return allocator.alloc(Device, 0);

    const ids = try allocator.alloc(AudioObjectID, n);
    defer allocator.free(ids);
    if (AudioObjectGetPropertyData(
        kAudioObjectSystemObject,
        &all_devices_addr,
        0,
        null,
        &size,
        @ptrCast(ids.ptr),
    ) != 0)
        return error.DeviceLookupFailed;

    var list: std.ArrayList(Device) = .empty;
    errdefer {
        for (list.items) |d| d.deinit(allocator);
        list.deinit(allocator);
    }

    for (ids) |id| {
        if (!hasInputStream(id)) continue;
        const name = (cfStringProperty(allocator, id, magic("lnam")) catch continue) orelse continue;
        errdefer allocator.free(name);
        const uid = (cfStringProperty(allocator, id, magic("uid ")) catch null) orelse {
            allocator.free(name);
            continue;
        };
        try list.append(allocator, .{ .name = name, .uid = uid });
    }
    return list.toOwnedSlice(allocator);
}

pub fn freeDevices(allocator: std.mem.Allocator, devices: []Device) void {
    for (devices) |d| d.deinit(allocator);
    allocator.free(devices);
}

fn hasInputStream(id: AudioObjectID) bool {
    const addr = AudioObjectPropertyAddress{
        .mSelector = magic("stm#"), // kAudioDevicePropertyStreams
        .mScope = magic("inpt"), // kAudioObjectPropertyScopeInput
        .mElement = 0,
    };
    var size: u32 = 0;
    if (AudioObjectGetPropertyDataSize(id, &addr, 0, null, &size) != 0) return false;
    return size > 0;
}

/// Reads a CFStringRef-valued property and returns it as a UTF-8 owned slice.
fn cfStringProperty(
    allocator: std.mem.Allocator,
    id: AudioObjectID,
    selector: u32,
) !?[]u8 {
    const addr = AudioObjectPropertyAddress{
        .mSelector = selector,
        .mScope = magic("glob"),
        .mElement = 0,
    };
    var size: u32 = @sizeOf(CFStringRef);
    var ref: CFStringRef = null;
    if (AudioObjectGetPropertyData(id, &addr, 0, null, &size, @ptrCast(&ref)) != 0) return null;
    const r = ref orelse return null;
    defer CFRelease(@ptrCast(r));

    const len = CFStringGetLength(r);
    const max = CFStringGetMaximumSizeForEncoding(len, kCFStringEncodingUTF8) + 1;
    const buf = try allocator.alloc(u8, @intCast(max));
    errdefer allocator.free(buf);
    if (CFStringGetCString(r, buf.ptr, max, kCFStringEncodingUTF8) == 0) {
        allocator.free(buf);
        return null;
    }
    const actual = std.mem.indexOfScalar(u8, buf, 0) orelse buf.len;
    return try allocator.realloc(buf, actual);
}

/// Case-insensitive ASCII substring match. Used so users can pass partial
/// names like "blackhole" or "macbook" without remembering exact device names.
fn matchesDevice(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) break;
        }
        if (j == needle.len) return true;
    }
    return false;
}

// ---------- Public API ----------

pub const Capture = struct {
    allocator: std.mem.Allocator,
    queue: AudioQueueRef,
    buffers: [n_buffers]AudioQueueBufferRef,
    sample_rate: u32,

    /// Pre-allocated SPSC ring (producer = audio callback; consumer = listen
    /// loop). Indexed modulo `ring.len`. Sized to absorb a 30 s consumer
    /// stall, which is way beyond anything we'd ever see in practice (drains
    /// happen on every audio arrival, ~50 ms).
    ring: []f32,
    /// Producer cursor. Monotonically increasing — the producer is the only
    /// one to advance it, the consumer reads with `.acquire`.
    head: std.atomic.Value(usize),
    /// Consumer cursor. Monotonically increasing — symmetric to `head`.
    tail: std.atomic.Value(usize),
    /// Total samples dropped due to overflow. Diagnostic only; surfaces in
    /// `-v` when stopping.
    overflow: std.atomic.Value(u64),

    /// Wakeup pipe. `wakeup_w` is non-blocking; the callback writes 1 byte
    /// after every successful ring write. The consumer `poll()`s `wakeup_r`
    /// and drains it on each wakeup. EAGAIN on write means the pipe is
    /// already saturated with a pending wakeup — that's fine, the consumer
    /// will see the new data on its next drain anyway.
    wakeup_r: c_int,
    wakeup_w: c_int,

    const n_buffers = 3;

    /// Buffer size: ~50 ms of audio per AudioQueue buffer. Small enough that
    /// the consumer wakes promptly, large enough to keep callback rate low.
    const buffer_frames = 800; // 50 ms at 16 kHz

    /// Ring depth: 30 s. Overflow only happens if the consumer hangs for
    /// longer than that, which is a different kind of bug.
    const ring_seconds = 30;

    /// `device_name` is a case-insensitive substring match against the system
    /// device names; null means use the system default input. Pass e.g.
    /// "blackhole" to capture from a virtual loopback device.
    pub fn start(
        allocator: std.mem.Allocator,
        sample_rate: u32,
        device_name: ?[]const u8,
    ) Error!*Capture {
        var self = try allocator.create(Capture);
        errdefer allocator.destroy(self);

        const ring = try allocator.alloc(f32, ring_seconds * @as(usize, @intCast(sample_rate)));
        errdefer allocator.free(ring);

        var pipe_fds: [2]c_int = undefined;
        if (std.c.pipe(&pipe_fds) != 0) return error.PipeFailed;
        const wakeup_r = pipe_fds[0];
        const wakeup_w = pipe_fds[1];
        errdefer {
            _ = std.c.close(wakeup_r);
            _ = std.c.close(wakeup_w);
        }
        // O_NONBLOCK on both ends. Producer (callback): write must not block
        // the real-time thread on a full pipe. Consumer (drain): reads after
        // poll fires non-deterministic byte counts; non-block lets us loop
        // until EAGAIN without a separate "how much" query.
        try setNonBlocking(wakeup_r);
        try setNonBlocking(wakeup_w);

        self.* = .{
            .allocator = allocator,
            .queue = null,
            .buffers = .{null} ** n_buffers,
            .sample_rate = sample_rate,
            .ring = ring,
            .head = .init(0),
            .tail = .init(0),
            .overflow = .init(0),
            .wakeup_r = wakeup_r,
            .wakeup_w = wakeup_w,
        };

        const fmt = AudioStreamBasicDescription{
            .mSampleRate = @floatFromInt(sample_rate),
            .mFormatID = magic("lpcm"),
            .mFormatFlags = kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
            .mBytesPerPacket = 4,
            .mFramesPerPacket = 1,
            .mBytesPerFrame = 4,
            .mChannelsPerFrame = 1,
            .mBitsPerChannel = 32,
            .mReserved = 0,
        };

        if (AudioQueueNewInput(&fmt, callback, self, null, null, 0, &self.queue) != 0)
            return error.NewQueueFailed;
        errdefer _ = AudioQueueDispose(self.queue, 1);

        // Optional device override. AudioQueue accepts the device's UID as a
        // CFStringRef via kAudioQueueProperty_CurrentDevice. Sample-rate /
        // channel conversion (BlackHole's stereo 48 kHz → our mono 16 kHz)
        // is handled by AudioQueue's built-in converter.
        if (device_name) |name| try setQueueDevice(allocator, self.queue, name);

        for (&self.buffers) |*b| {
            if (AudioQueueAllocateBuffer(self.queue, buffer_frames * 4, b) != 0)
                return error.AllocBufferFailed;
            if (AudioQueueEnqueueBuffer(self.queue, b.*, 0, null) != 0)
                return error.EnqueueFailed;
        }

        if (AudioQueueStart(self.queue, null) != 0) return error.StartFailed;
        return self;
    }

    pub fn stop(self: *Capture, allocator: std.mem.Allocator) void {
        _ = AudioQueueStop(self.queue, 1);
        _ = AudioQueueDispose(self.queue, 1);
        _ = std.c.close(self.wakeup_r);
        _ = std.c.close(self.wakeup_w);
        allocator.free(self.ring);
        allocator.destroy(self);
    }

    /// Block until audio is available, or `timeout_ms` elapses. The consumer
    /// must call `drain` afterwards regardless of return value — `poll` can
    /// return spuriously, and a wakeup may have been written between the
    /// last drain and our `poll` entering the kernel.
    pub fn waitForData(self: *Capture, timeout_ms: i32) void {
        var fds = [_]std.posix.pollfd{
            .{ .fd = self.wakeup_r, .events = std.c.POLL.IN, .revents = 0 },
        };
        _ = std.posix.poll(&fds, timeout_ms) catch {};
        // Drain whatever bytes the callback wrote since last wake. Non-blocking,
        // so EAGAIN naturally exits the loop. We don't care about the bytes
        // themselves — they're just edge triggers.
        var dump: [64]u8 = undefined;
        while (true) {
            const n = std.c.read(self.wakeup_r, &dump, dump.len);
            if (n <= 0) break;
        }
    }

    /// Move all currently-buffered samples into `dest`. Returns the number
    /// pushed.
    pub fn drain(self: *Capture, allocator: std.mem.Allocator, dest: *std.ArrayList(f32)) !usize {
        const head = self.head.load(.acquire);
        const tail = self.tail.load(.monotonic);
        const n = head - tail;
        if (n == 0) return 0;

        try dest.ensureUnusedCapacity(allocator, n);
        const cap = self.ring.len;
        const start_idx = tail % cap;
        const first_chunk = @min(n, cap - start_idx);
        dest.appendSliceAssumeCapacity(self.ring[start_idx .. start_idx + first_chunk]);
        if (first_chunk < n) {
            dest.appendSliceAssumeCapacity(self.ring[0 .. n - first_chunk]);
        }
        self.tail.store(head, .release);
        return n;
    }

    /// Total samples dropped due to ring overflow since `start`. Useful in
    /// `-v` to surface a stuck consumer.
    pub fn overflowSamples(self: *const Capture) u64 {
        return self.overflow.load(.monotonic);
    }
};

/// Find an input device whose name contains `needle` (case-insensitive) and
/// bind the queue to it. On miss, prints the available devices to stderr so
/// the user can copy-paste the right substring.
fn setQueueDevice(
    allocator: std.mem.Allocator,
    queue: AudioQueueRef,
    needle: []const u8,
) Error!void {
    const devices = try listInputDevices(allocator);
    defer freeDevices(allocator, devices);

    const match: ?Device = blk: {
        for (devices) |d| if (matchesDevice(d.name, needle)) break :blk d;
        break :blk null;
    };
    const m = match orelse {
        std.debug.print("error: no input device matching '{s}'\n", .{needle});
        std.debug.print("available input devices:\n", .{});
        for (devices) |d| std.debug.print("  - {s}\n", .{d.name});
        return error.DeviceNotFound;
    };

    const uid_z = try allocator.dupeZ(u8, m.uid);
    defer allocator.free(uid_z);
    const uid_ref = CFStringCreateWithCString(null, uid_z.ptr, kCFStringEncodingUTF8) orelse
        return error.DeviceLookupFailed;
    defer CFRelease(@ptrCast(uid_ref));

    if (AudioQueueSetProperty(
        queue,
        magic("aqcd"), // kAudioQueueProperty_CurrentDevice
        @ptrCast(&uid_ref),
        @sizeOf(CFStringRef),
    ) != 0) return error.NewQueueFailed;
}

fn setNonBlocking(fd: c_int) !void {
    const cur = std.c.fcntl(fd, std.c.F.GETFL, @as(c_int, 0));
    if (cur < 0) return error.PipeFailed;
    const nb_bit: c_int = 1 << @bitOffsetOf(std.posix.O, "NONBLOCK");
    if (std.c.fcntl(fd, std.c.F.SETFL, cur | nb_bit) < 0) return error.PipeFailed;
}

fn callback(
    user_data: ?*anyopaque,
    queue: AudioQueueRef,
    buffer: AudioQueueBufferRef,
    start_time: ?*const AudioTimeStamp,
    n_packet_descriptions: u32,
    packet_descs: ?*const AudioStreamPacketDescription,
) callconv(.c) void {
    _ = start_time;
    _ = n_packet_descriptions;
    _ = packet_descs;
    const self: *Capture = @ptrCast(@alignCast(user_data orelse return));
    const buf = buffer orelse return;

    if (buf.mAudioData) |data_ptr| {
        const n_samples = buf.mAudioDataByteSize / 4;
        const samples = @as([*]const f32, @ptrCast(@alignCast(data_ptr)))[0..n_samples];

        // SPSC producer side. We are the only writer of `head`, so a
        // monotonic load suffices for our own state. We need acquire on
        // `tail` to see the consumer's progress and compute free space.
        const head = self.head.load(.monotonic);
        const tail = self.tail.load(.acquire);
        const cap = self.ring.len;
        const used = head - tail;
        const free = if (used >= cap) 0 else cap - used;
        const to_write = @min(n_samples, free);
        const dropped = n_samples - to_write;

        if (to_write > 0) {
            const start_idx = head % cap;
            const first_chunk = @min(to_write, cap - start_idx);
            @memcpy(self.ring[start_idx .. start_idx + first_chunk], samples[0..first_chunk]);
            if (first_chunk < to_write) {
                @memcpy(self.ring[0 .. to_write - first_chunk], samples[first_chunk..to_write]);
            }
            // Release: publish the writes above to the consumer.
            self.head.store(head + to_write, .release);

            // Wake the consumer. Non-blocking; EAGAIN means a wakeup is
            // already pending in the pipe — equally good.
            const byte: u8 = 0;
            _ = std.c.write(self.wakeup_w, @ptrCast(&byte), 1);
        }
        if (dropped > 0) {
            _ = self.overflow.fetchAdd(@intCast(dropped), .monotonic);
        }
    }
    _ = AudioQueueEnqueueBuffer(queue, buf, 0, null);
}

fn magic(s: *const [4]u8) u32 {
    // FourCC big-endian to host word: 'lpcm' → 0x6c70636d on big-endian, but
    // CoreAudio constants are typed as native-byte-order u32 of the ASCII
    // bytes, treated as a host-endian integer. The AudioToolbox headers
    // produce these via a macro that reads s[0]<<24|s[1]<<16|... — same here.
    return (@as(u32, s[0]) << 24) | (@as(u32, s[1]) << 16) |
        (@as(u32, s[2]) << 8) | @as(u32, s[3]);
}
