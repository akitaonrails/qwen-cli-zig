const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const io = std.io;
const fs = std.fs;
const process = std.process;
const fmt = std.fmt;
const ArrayList = std.ArrayList;
const math = std.math;

const llama = @import("llama_cpp.zig");

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

// --- Default Configuration ---
const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in software development tasks.";
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_MODEL_PATH = "";

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

// --- Command Line Arguments ---
const CliArgs = struct {
    model_path: []const u8 = DEFAULT_MODEL_PATH,
    system_prompt: []const u8 = DEFAULT_SYSTEM_PROMPT,
    temperature: f32 = DEFAULT_TEMPERATURE,
    show_help: bool = false,
};

fn parseArgs() !CliArgs {
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

fn printHelp(exe_name: []const u8) void {
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

// --- Chat History ---
const Message = struct {
    role: []const u8,
    content: []const u8,
};

fn toCString(slice: []const u8) ![:0]const u8 {
    const c_str = try allocator.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}

pub fn main() !void {
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            print("Warning: Memory leak detected!\n", .{});
        }
    }

    const args = try parseArgs();
    defer {
        if (args.model_path.len > 0 and !std.mem.eql(u8, args.model_path, DEFAULT_MODEL_PATH)) allocator.free(args.model_path);
        if (args.system_prompt.len > 0 and !std.mem.eql(u8, args.system_prompt, DEFAULT_SYSTEM_PROMPT)) allocator.free(args.system_prompt);
    }

    if (args.show_help) {
        const exe_path = try std.fs.selfExePathAlloc(allocator);
        defer allocator.free(exe_path);
        const exe_name = std.fs.path.basename(exe_path);
        printHelp(exe_name);
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
    const c_model_path = try toCString(args.model_path);
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

    var history = std.ArrayListUnmanaged(Message){};
    defer {
        for (history.items) |msg| {
            allocator.free(msg.content);
        }
        history.deinit(allocator);
    }

    const stdin_file = std.fs.File.stdin();
    const stdin = stdin_file.deprecatedReader();
    const stdout_file = std.fs.File.stdout();
    const stdout = stdout_file.deprecatedWriter();
    var input_buffer = std.ArrayListUnmanaged(u8){};
    defer input_buffer.deinit(allocator);

    // --- Chat Loop ---
    while (true) {
        try stdout.print("> ", .{});
        // stdout is unbuffered `DeprecatedWriter` (GenericWriter around File).
        // It flushes on each write (syscall). No explicit flush needed or available on GenericWriter context?
        // GenericWriter context is File. File has sync?
        // But print() calls write() which calls write() on context (File). File.write calls sycall.
        // So it is unbuffered.

        input_buffer.clearRetainingCapacity();
        stdin.streamUntilDelimiter(input_buffer.writer(allocator), '\n', null) catch |err| {
            if (err == error.EndOfStream) {
                print("\nExiting...\n", .{});
                break;
            } else {
                print("Error reading input: {}\n", .{err});
                return err;
            }
        };

        const user_input = std.mem.trim(u8, input_buffer.items, " \t\r\n");

        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "/quit")) {
            print("Exiting...\n", .{});
            break;
        }

        // --- Prepare Prompt ---
        var current_chat = std.ArrayListUnmanaged(llama.llama_chat_message){};
        defer current_chat.deinit(allocator);

        const c_system_role = try toCString("system");
        defer allocator.free(c_system_role);
        const c_system_prompt = try toCString(args.system_prompt);
        defer allocator.free(c_system_prompt);
        try current_chat.append(allocator, .{ .role = c_system_role, .content = c_system_prompt });

        var temp_slices = std.ArrayListUnmanaged([:0]const u8){};
        defer {
            for (temp_slices.items) |slice_to_free| {
                allocator.free(slice_to_free);
            }
            temp_slices.deinit(allocator);
        }
        for (history.items) |msg| {
            const c_role = try toCString(msg.role);
            try temp_slices.append(allocator, c_role);
            const c_content = try toCString(msg.content);
            try temp_slices.append(allocator, c_content);
            try current_chat.append(allocator, .{ .role = c_role, .content = c_content });
        }

        const c_user_role = try toCString("user");
        defer allocator.free(c_user_role);
        const c_user_input = try toCString(user_input);
        defer allocator.free(c_user_input);
        try current_chat.append(allocator, .{ .role = c_user_role, .content = c_user_input });

        var formatted_prompt_buf = std.ArrayListUnmanaged(u8){};
        defer formatted_prompt_buf.deinit(allocator);
        const initial_buf_size: i32 = 2048;
        try formatted_prompt_buf.resize(allocator, @intCast(initial_buf_size));

        // Get template from model
        const tmpl = llama.llama_model_chat_template(model, null);

        var required_len = llama.llama_chat_apply_template(
            tmpl,
            current_chat.items.ptr,
            current_chat.items.len,
            true,
            formatted_prompt_buf.items.ptr,
            initial_buf_size,
        );

        if (required_len < 0) {
            print("Error: llama_chat_apply_template failed ({d})\n", .{required_len});
            continue;
        }

        if (@as(u32, @intCast(required_len)) > initial_buf_size) {
            print("Resizing chat template buffer to {d} bytes\n", .{required_len});
            try formatted_prompt_buf.resize(allocator, @intCast(required_len));
            required_len = llama.llama_chat_apply_template(
                tmpl, current_chat.items.ptr, current_chat.items.len, true,
                formatted_prompt_buf.items.ptr, required_len
            );
            if (required_len < 0) {
                 print("Error: llama_chat_apply_template failed after resize ({d})\n", .{required_len});
                 continue;
            }
        }
        const full_prompt_bytes = formatted_prompt_buf.items[0..@intCast(required_len)];

        // --- Llama.cpp Inference ---
        print("Assistant: ", .{});

        // --- Tokenize ---
        const tokens = try allocator.alloc(llama.llama_token, n_ctx);
        defer allocator.free(tokens);

        const vocab = llama.llama_model_get_vocab(model);
        const add_bos = llama.llama_vocab_get_add_bos(vocab);
        
        const n_tokens = llama.llama_tokenize(
            vocab, 
            full_prompt_bytes.ptr, 
            @intCast(full_prompt_bytes.len), 
            tokens.ptr, 
            @intCast(n_ctx), 
            add_bos, 
            false
        );

        if (n_tokens < 0) {
            print("Error: Failed to tokenize prompt (token buffer too small?)\n", .{});
            continue;
        }

        // --- Evaluate Prompt ---
        // print("Decoding prompt ({d} tokens)...\n", .{n_tokens});
        const prompt_batch = llama.llama_batch_get_one(tokens.ptr, n_tokens); // n_tokens is i32, correct
        const decode_prompt_ret = llama.llama_decode(ctx, prompt_batch);
        if (decode_prompt_ret != 0) {
             print("Error: llama_decode failed during prompt processing (ret={d})\n", .{decode_prompt_ret});
             continue;
        }

        // --- Generation Loop ---
        var generated_response = std.ArrayListUnmanaged(u8){};
        defer generated_response.deinit(allocator);
        const writer = generated_response.writer(allocator);

        // cur_pos not strictly needed if we just track token count for limits
        var generated_token_count: u32 = 0;
        const max_new_tokens = n_ctx - @as(u32, @intCast(n_tokens));

        // --- Init Sampler Chain ---
        const chain_params = llama.llama_sampler_chain_default_params();
        const sampler_chain = llama.llama_sampler_chain_init(chain_params);
        if (sampler_chain == null) {
            print("Error: Failed to init sampler chain\n", .{});
            continue;
        }
        defer llama.llama_sampler_free(sampler_chain);

        const greedy_sampler = llama.llama_sampler_init_greedy();
        llama.llama_sampler_chain_add(sampler_chain, greedy_sampler);

        while (generated_token_count < max_new_tokens) {
            const id = llama.llama_sampler_sample(sampler_chain, ctx, -1);
            llama.llama_sampler_accept(sampler_chain, id);

            if (llama.llama_vocab_is_eog(vocab, id)) {
                break;
            }

            var piece_len = llama.llama_token_to_piece(vocab, id, null, 0, 0, false);
            if (piece_len < 0) {
                 piece_len = -piece_len;
            } else if (piece_len == 0) {
                 continue;
            }
            const piece_buf = try allocator.alloc(u8, @intCast(piece_len));
            defer allocator.free(piece_buf);
            
            const actual_len = llama.llama_token_to_piece(vocab, id, piece_buf.ptr, @intCast(piece_buf.len), 0, false);
            if (actual_len < 0) {
                 break;
            }
            const final_buf = piece_buf[0..@intCast(actual_len)];

            try stdout.print("{s}", .{final_buf});
            try writer.writeAll(final_buf);

            // Manual batch
            var token_id_val = id;
            const batch = llama.llama_batch {
                .n_tokens = 1,
                .token = @ptrCast(&token_id_val), // Pointer to stack variable
                .embd = null,
                .pos = null,
                .n_seq_id = null,
                .seq_id = null,
                .logits = null,
            };
            // Note: In Zig 0.15, pointer casting might need to be explicit if types mismatch
            // but &token_id_val is *i32 (llama_token), and batch.token is ?[*]llama_token.
            // Implicit coercion from single pointer to many-pointer usually works.
            // If not, use @ptrCast.

            const decode_gen_ret = llama.llama_decode(ctx, batch);
            if (decode_gen_ret != 0) {
                 break;
            }
            generated_token_count += 1;
        }
        try stdout.print("\n", .{});

        // --- Update History ---
        const user_input_copy = try allocator.dupe(u8, user_input);
        errdefer allocator.free(user_input_copy);
        const assistant_response_copy = try generated_response.toOwnedSlice(allocator);
        errdefer allocator.free(assistant_response_copy);

        try history.append(allocator, .{ .role = "user", .content = user_input_copy });
        try history.append(allocator, .{ .role = "assistant", .content = assistant_response_copy });
    }

    print("Cleaning up llama.cpp backend...\n", .{});
    llama.llama_backend_free();
}
