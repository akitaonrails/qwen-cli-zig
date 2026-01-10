// config.zig - Application configuration and CLI argument parsing

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Application configuration with sensible defaults.
pub const Config = struct {
    model_path: []const u8 = defaults.model_path,
    system_prompt: []const u8 = defaults.system_prompt,
    temperature: f32 = defaults.temperature,
    context_size: u32 = defaults.context_size,
    batch_size: u32 = defaults.batch_size,
    max_history_pairs: usize = defaults.max_history_pairs,
    max_new_tokens: u32 = defaults.max_new_tokens,

    // Ownership tracking for allocated strings
    _model_path_owned: ?[]u8 = null,
    _system_prompt_owned: ?[]u8 = null,
    _allocator: ?Allocator = null,

    pub const defaults = struct {
        pub const model_path = "models/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf";
        pub const system_prompt = "You are a helpful assistant.";
        pub const temperature: f32 = 0.7;
        pub const context_size: u32 = 4096;
        pub const batch_size: u32 = 64;
        pub const max_history_pairs: usize = 3;
        pub const max_new_tokens: u32 = 512;
        pub const max_prompt_bytes: usize = 8192;
        pub const token_buffer_size: usize = 1024;
        pub const response_buffer_size: usize = 4096;
        pub const piece_buffer_size: usize = 256;
    };

    pub fn deinit(self: *Config) void {
        if (self._allocator) |alloc| {
            if (self._model_path_owned) |p| alloc.free(p);
            if (self._system_prompt_owned) |p| alloc.free(p);
        }
        self.* = undefined;
    }
};

pub const ParseError = error{
    MissingModelPath,
    MissingSystemPrompt,
    MissingTemperature,
    InvalidTemperature,
    OutOfMemory,
};

pub const ParseResult = union(enum) {
    config: Config,
    help: void,
    @"error": ParseError,
};

/// Parse command line arguments into a Config.
pub fn parseArgs(allocator: Allocator) ParseResult {
    var args_iter = std.process.argsWithAllocator(allocator) catch |err| {
        return .{ .@"error" = switch (err) {
            error.OutOfMemory => ParseError.OutOfMemory,
        } };
    };
    defer args_iter.deinit();

    _ = args_iter.next(); // Skip executable name

    var config = Config{};
    config._allocator = allocator;

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            config.deinit();
            return .help;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            const value = args_iter.next() orelse {
                config.deinit();
                return .{ .@"error" = ParseError.MissingModelPath };
            };
            if (config._model_path_owned) |p| allocator.free(p);
            config._model_path_owned = allocator.dupe(u8, value) catch {
                config.deinit();
                return .{ .@"error" = ParseError.OutOfMemory };
            };
            config.model_path = config._model_path_owned.?;
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
            const value = args_iter.next() orelse {
                config.deinit();
                return .{ .@"error" = ParseError.MissingSystemPrompt };
            };
            if (config._system_prompt_owned) |p| allocator.free(p);
            config._system_prompt_owned = allocator.dupe(u8, value) catch {
                config.deinit();
                return .{ .@"error" = ParseError.OutOfMemory };
            };
            config.system_prompt = config._system_prompt_owned.?;
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temp")) {
            const value = args_iter.next() orelse {
                config.deinit();
                return .{ .@"error" = ParseError.MissingTemperature };
            };
            config.temperature = std.fmt.parseFloat(f32, value) catch {
                config.deinit();
                return .{ .@"error" = ParseError.InvalidTemperature };
            };
        }
        // Unknown arguments are silently ignored for forward compatibility
    }

    return .{ .config = config };
}

/// Print usage help.
pub fn printHelp(writer: anytype, exe_name: []const u8) !void {
    try writer.print(
        \\Usage: {s} [options]
        \\
        \\A CLI chat interface for Qwen3 models using llama.cpp.
        \\
        \\Options:
        \\  -m, --model <path>    Path to GGUF model file
        \\                        (default: {s})
        \\  -s, --system <prompt> System prompt
        \\                        (default: "{s}")
        \\  -t, --temp <value>    Sampling temperature (default: {d:.1})
        \\  -h, --help            Show this help message
        \\
        \\Examples:
        \\  {s} --model models/qwen.gguf
        \\  {s} --model models/qwen.gguf --temp 0.5
        \\  {s} -m models/qwen.gguf -s "You are a coding expert."
        \\
    , .{
        exe_name,
        Config.defaults.model_path,
        Config.defaults.system_prompt,
        Config.defaults.temperature,
        exe_name,
        exe_name,
        exe_name,
    });
}
