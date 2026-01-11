const std = @import("std");
const process = std.process;
const print = std.debug.print;
const fmt = std.fmt;
const Allocator = std.mem.Allocator;

// --- Default Configuration ---
pub const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in software development tasks.";
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_MODEL_PATH = "";

// --- Command Line Arguments ---
pub const CliArgs = struct {
    model_path: []const u8 = DEFAULT_MODEL_PATH,
    system_prompt: []const u8 = DEFAULT_SYSTEM_PROMPT,
    temperature: f32 = DEFAULT_TEMPERATURE,
    show_help: bool = false,
};

pub fn parseArgs(allocator: Allocator) !CliArgs {
    var args_iter = try process.argsWithAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.next(); // Skip executable name

    var args = CliArgs{};

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            const value_arg = args_iter.next() orelse return error.MissingModelPath;
            args.model_path = try allocator.dupe(u8, value_arg);
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
            const value_arg = args_iter.next() orelse return error.MissingSystemPrompt;
            args.system_prompt = try allocator.dupe(u8, value_arg);
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temp")) {
            const temp_str = args_iter.next() orelse return error.MissingTemperature;
            args.temperature = try fmt.parseFloat(f32, temp_str);
        } else {
            print("Unknown argument: {s}\n", .{arg});
            args.show_help = true;
            break;
        }
    }

    return args;
}

pub fn printHelp(exe_name: []const u8) void {
    print(
        \\Usage: {s} [options]
        \\
        \\A simple CLI chat interface for Qwen3 models using llama.cpp.
        \\
        \\Options:
        \\  -m, --model <path>    Path to the GGUF model file (required)
        \\  -s, --system <prompt> System prompt to use (default: "{s}")
        \\  -t, --temp <value>    Sampling temperature (default: {d:.1})
        \\  -h, --help            Show this help message
        \\
    , .{ exe_name, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE });
}
