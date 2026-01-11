const std = @import("std");

pub const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in software development tasks.";
pub const DEFAULT_TEMPERATURE: f32 = 0.7;

pub const CliArgs = struct {
    model_path: []const u8,
    system_prompt: []const u8,
    temperature: f32,
    show_help: bool,

    model_path_owned: ?[]u8 = null,
    system_prompt_owned: ?[]u8 = null,

    pub fn deinit(self: *CliArgs, allocator: std.mem.Allocator) void {
        if (self.model_path_owned) |buf| allocator.free(buf);
        if (self.system_prompt_owned) |buf| allocator.free(buf);
        self.* = undefined;
    }
};

pub fn parseArgs(allocator: std.mem.Allocator) !CliArgs {
    var args_iter = try std.process.argsWithAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.next();

    var args = CliArgs{
        .model_path = "",
        .system_prompt = DEFAULT_SYSTEM_PROMPT,
        .temperature = DEFAULT_TEMPERATURE,
        .show_help = false,
    };

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            const value_arg = args_iter.next() orelse return error.MissingModelPath;
            const duped = try allocator.dupe(u8, value_arg);
            if (args.model_path_owned) |old| allocator.free(old);
            args.model_path_owned = duped;
            args.model_path = duped;
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
            const value_arg = args_iter.next() orelse return error.MissingSystemPrompt;
            const duped = try allocator.dupe(u8, value_arg);
            if (args.system_prompt_owned) |old| allocator.free(old);
            args.system_prompt_owned = duped;
            args.system_prompt = duped;
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temp")) {
            const temp_str = args_iter.next() orelse return error.MissingTemperature;
            args.temperature = try std.fmt.parseFloat(f32, temp_str);
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            args.show_help = true;
            break;
        }
    }

    return args;
}

pub fn printHelp(exe_name: []const u8) void {
    std.debug.print(
        \\Usage: {s} [options]
        \\
        \\A simple CLI chat interface for Qwen3 models using llama.cpp.
        \\
        \\Options:
        \\  -m, --model <path>    Path to the GGUF model file (required)
        \\  -s, --system <prompt> System prompt to use (default: \"{s}\")
        \\  -t, --temp <value>    Sampling temperature (default: {d:.1})
        \\  -h, --help            Show this help message
        \\
    , .{ exe_name, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE });
}
