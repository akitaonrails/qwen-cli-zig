const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;

const llama = @import("llama_cpp.zig");
const config = @import("config.zig");
const chat = @import("chat.zig");
const utils = @import("utils.zig");

// --- Error Types ---
const AppError = std.fs.File.WriteError || error{
    MissingModelPath,
    MissingSystemPrompt,
    MissingTemperature,
    ModelLoadFailed,
    ContextCreateFailed,
    TokenizeFailed,
    DecodeFailed,
    OutOfMemory,
};

// --- Application State ---
// Global allocator is generally discouraged for this exact reason, but for CLI it's often fine.
// Better: move it to main.
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// --- Llama.cpp Logging Callback ---
fn logCallback(level: llama.ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    const level_int = @intFromEnum(level);
    switch (level) {
        .ERROR => std.debug.print("LLAMA: [ERROR] {s}", .{text}),
        .WARN => std.debug.print("LLAMA: [WARN] {s}", .{text}),
        .INFO => std.debug.print("LLAMA: [INFO] {s}", .{text}),
        .DEBUG => std.debug.print("LLAMA: [DEBUG] {s}", .{text}),
        .CONT => std.debug.print("{s}", .{text}),
        else => std.debug.print("LLAMA: [LEVEL={d}] {s}", .{ level_int, text }),
    }
}

pub fn main() !void {
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            print("Warning: Memory leak detected!\n", .{});
        }
    }

    const args = try config.parseArgs(allocator);
    defer {
        if (args.model_path.len > 0 and !std.mem.eql(u8, args.model_path, config.DEFAULT_MODEL_PATH)) allocator.free(args.model_path);
        if (args.system_prompt.len > 0 and !std.mem.eql(u8, args.system_prompt, config.DEFAULT_SYSTEM_PROMPT)) allocator.free(args.system_prompt);
    }

    if (args.show_help) {
        const exe_path = try std.fs.selfExePathAlloc(allocator);
        defer allocator.free(exe_path);
        const exe_name = std.fs.path.basename(exe_path);
        config.printHelp(exe_name);
        return;
    }

    // --- Llama.cpp Initialization ---
    print("Initializing llama.cpp backend...\n", .{});
    llama.llama_backend_init();
    llama.llama_log_set(&logCallback, null);
    print("llama.cpp log handler set.\n", .{});

    // --- Load Model ---
    var mparams = llama.llama_model_default_params();
    mparams.use_mmap = false;

    print("Loading model: {s}...\n", .{args.model_path});
    const c_model_path = try utils.toCString(allocator, args.model_path);
    defer allocator.free(c_model_path);

    const model = llama.llama_model_load_from_file(c_model_path, mparams);
    if (model == null) {
        print("Error: Failed to load model from '{s}'\n", .{args.model_path});
        llama.llama_backend_free();
        return error.ModelLoadFailed;
    }
    // Correct defer syntax: call the function
    defer llama.llama_model_free(model);

    // --- Create Context ---
    var cparams = llama.llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;

    print("Creating context (n_ctx = {d}, n_batch = {d})...\n", .{cparams.n_ctx, cparams.n_batch});
    const ctx = llama.llama_init_from_model(model, cparams); 
    if (ctx == null) {
        print("Error: Failed to create context\n", .{});
        llama.llama_backend_free();
        return error.ContextCreateFailed;
    }
    defer llama.llama_free(ctx);

    const n_ctx = llama.llama_n_ctx(ctx);
    print("Context created (n_ctx = {d})\n", .{n_ctx});
    print("Model loaded successfully.\n", .{});
    print("System Prompt: {s}\n", .{args.system_prompt});
    print("Temperature: {d:.2}\n", .{args.temperature});
    print("Enter your message (Ctrl+D or /quit to exit):\n", .{});

    // Get vocab for tokenization and add_bos check
    const vocab = llama.llama_model_get_vocab(model);

    try chat.runChatLoop(allocator, model, ctx, args, n_ctx, vocab);

    print("Cleaning up llama.cpp backend...\n", .{});
    llama.llama_backend_free();
}
