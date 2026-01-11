const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn toCString(allocator: Allocator, slice: []const u8) ![:0]const u8 {
    const c_str = try allocator.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}
