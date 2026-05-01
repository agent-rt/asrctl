//! macOS microphone capture via AudioToolbox AudioQueue.
//!
//! Avoids `translate-c <AudioToolbox/AudioQueue.h>` because Apple's SDK pulls
//! in mach_msg.h which translate-c can't size. Instead we declare the small
//! subset we need as raw externs. Stable C ABI; no Objective-C blocks.
//!
//! Captures mono f32 PCM at 16 kHz (matching what mtmd's audio encoder
//! expects). Pushes samples into a thread-safe ring buffer; consumer threads
//! call `drain()` to copy them out.

const std = @import("std");

pub const Error = error{
    NewQueueFailed,
    AllocBufferFailed,
    EnqueueFailed,
    StartFailed,
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

const kLinearPCMFormatFlagIsFloat: u32 = 1 << 0;
const kLinearPCMFormatFlagIsPacked: u32 = 1 << 3;

// ---------- Public API ----------

pub const Capture = struct {
    allocator: std.mem.Allocator,
    queue: AudioQueueRef,
    buffers: [n_buffers]AudioQueueBufferRef,
    ring: SampleRing,
    sample_rate: u32,

    const n_buffers = 3;

    /// Buffer size: ~50 ms of audio. Smaller → lower latency in the listen
    /// loop's drain rate; larger → fewer callbacks per second.
    const buffer_frames = 800; // 50 ms at 16 kHz

    pub fn start(allocator: std.mem.Allocator, sample_rate: u32) Error!*Capture {
        var self = try allocator.create(Capture);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .queue = null,
            .buffers = .{null} ** n_buffers,
            .ring = .{ .mutex = .{}, .data = .empty },
            .sample_rate = sample_rate,
        };
        errdefer self.ring.data.deinit(allocator);

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
        self.ring.data.deinit(allocator);
        allocator.destroy(self);
    }

    /// Move all currently-buffered samples into `dest`. Returns the number
    /// pushed. Caller drives the consumption rate.
    pub fn drain(self: *Capture, allocator: std.mem.Allocator, dest: *std.ArrayList(f32)) !usize {
        self.ring.mutex.lock();
        defer self.ring.mutex.unlock();
        try dest.appendSlice(allocator, self.ring.data.items);
        const n = self.ring.data.items.len;
        self.ring.data.clearRetainingCapacity();
        return n;
    }
};

const SampleRing = struct {
    mutex: SpinLock,
    data: std.ArrayList(f32),
};

/// Tiny atomic spinlock. The audio callback and the consumer thread each hold
/// it for ~microseconds, so contention is negligible. Avoids depending on
/// std.Io.Mutex (which would require threading an Io reference through the
/// CoreAudio callback path).
const SpinLock = struct {
    held: std.atomic.Value(bool) = .init(false),

    fn lock(self: *SpinLock) void {
        while (self.held.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }
    fn unlock(self: *SpinLock) void {
        self.held.store(false, .release);
    }
};

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
        const samples: []const f32 = @as([*]const f32, @ptrCast(@alignCast(data_ptr)))[0..n_samples];
        self.ring.mutex.lock();
        // Bound the buffer so a stalled consumer doesn't grow it unbounded.
        // 30 s of audio @ 16 kHz = 480 000 samples. Drop oldest beyond that.
        const max_samples: usize = 30 * @as(usize, @intCast(self.sample_rate));
        if (self.ring.data.items.len + n_samples > max_samples) {
            const overflow = self.ring.data.items.len + n_samples - max_samples;
            const remaining = self.ring.data.items.len - overflow;
            std.mem.copyForwards(
                f32,
                self.ring.data.items[0..remaining],
                self.ring.data.items[overflow..],
            );
            self.ring.data.shrinkRetainingCapacity(remaining);
        }
        self.ring.data.appendSlice(self.allocator, samples) catch {};
        self.ring.mutex.unlock();
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
