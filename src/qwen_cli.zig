const std = @import("std");
const chat = @import("chat.zig");
const config = @import("config.zig");

const fs = std.fs;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        if (gpa.deinit() == .leak) {
            std.debug.print("Warning: Memory leak detected!\n", .{});
        }
    }

    const allocator = gpa.allocator();

    var args = try config.parseArgs(allocator);
    defer config.freeArgs(allocator, &args);

    if (args.show_help) {
        const exe_path = try fs.selfExePathAlloc(allocator);
        defer allocator.free(exe_path);
        const exe_name = fs.path.basename(exe_path);
        config.printHelp(exe_name);
        return;
    }

    try chat.run(allocator, args);
}
