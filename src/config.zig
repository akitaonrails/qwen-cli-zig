const std = @import("std");

const process = std.process;
const fmt = std.fmt;
const fs = std.fs;
const print = std.debug.print;

pub const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in software development tasks.";
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_MODEL_PATH = "";

pub const CliArgs = struct {
    model_path: []const u8 = DEFAULT_MODEL_PATH,
    system_prompt: []const u8 = DEFAULT_SYSTEM_PROMPT,
    temperature: f32 = DEFAULT_TEMPERATURE,
    show_help: bool = false,
    owns_model_path: bool = false,
    owns_system_prompt: bool = false,
};

pub const ParseError = error{
    MissingModelPath,
    MissingSystemPrompt,
    MissingTemperature,
};

pub fn parseArgs(allocator: std.mem.Allocator) !CliArgs {
    var args_iter = try process.argsWithAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.next();

    var args = CliArgs{};
    errdefer freeArgs(allocator, &args);

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            const value_arg = args_iter.next() orelse return ParseError.MissingModelPath;
            const duplicated = try allocator.dupe(u8, value_arg);
            if (args.owns_model_path) allocator.free(@constCast(args.model_path));
            args.model_path = duplicated;
            args.owns_model_path = true;
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
            const value_arg = args_iter.next() orelse return ParseError.MissingSystemPrompt;
            const duplicated = try allocator.dupe(u8, value_arg);
            if (args.owns_system_prompt) allocator.free(@constCast(args.system_prompt));
            args.system_prompt = duplicated;
            args.owns_system_prompt = true;
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temp")) {
            const temp_str = args_iter.next() orelse return ParseError.MissingTemperature;
            args.temperature = try fmt.parseFloat(f32, temp_str);
        } else {
            print("Unknown argument: {s}\n", .{arg});
            args.show_help = true;
            break;
        }
    }

    return args;
}

pub fn freeArgs(allocator: std.mem.Allocator, args: *CliArgs) void {
    if (args.owns_model_path) {
        allocator.free(@constCast(args.model_path));
        args.owns_model_path = false;
    }
    if (args.owns_system_prompt) {
        allocator.free(@constCast(args.system_prompt));
        args.owns_system_prompt = false;
    }
}

pub fn printHelp(exe_name: []const u8) void {
    print(
        "Usage: {s} [options]\n" ++
            "\n" ++
            "A simple CLI chat interface for Qwen3 models using llama.cpp.\n" ++
            "\n" ++
            "Options:\n" ++
            "  -m, --model <path>    Path to the GGUF model file (required)\n" ++
            "  -s, --system <prompt> System prompt to use (default: \"{s}\")\n" ++
            "  -t, --temp <value>    Sampling temperature (default: {d:.1})\n" ++
            "  -h, --help            Show this help message\n\n",
        .{ exe_name, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE },
    );
}
