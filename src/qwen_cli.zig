const std = @import("std");
const c = @cImport({
    @cInclude("llama.h");
});

const Allocator = std.mem.Allocator;
const print = std.debug.print;
const io = std.io;
const fs = std.fs;
const process = std.process;
const fmt = std.fmt;

const AppError = std.os.WriteError || error{
    MissingModelPath,
    MissingSystemPrompt,
    MissingTemperature,
    ModelLoadFailed,
    ContextCreateFailed,
    TokenizeFailed,
    DecodeFailed,
    OutOfMemory,
};

const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in software development tasks.";
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_MODEL_PATH = "";

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

fn logCallback(level: c_uint, text_ptr: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    const text_c: [*:0]const u8 = @ptrCast(text_ptr);
    const slice = std.mem.sliceTo(text_c, 0);
    const level_int: u32 = @intCast(level);
    switch (level) {
        @as(c_uint, c.GGML_LOG_LEVEL_ERROR) => std.debug.print("LLAMA: [ERROR] {s}", .{slice}),
        @as(c_uint, c.GGML_LOG_LEVEL_WARN) => std.debug.print("LLAMA: [WARN] {s}", .{slice}),
        @as(c_uint, c.GGML_LOG_LEVEL_INFO) => std.debug.print("LLAMA: [INFO] {s}", .{slice}),
        else => std.debug.print("LLAMA: [LEVEL={d}] {s}", .{ level_int, slice }),
    }
}

const CliArgs = struct {
    model_path: []const u8 = DEFAULT_MODEL_PATH,
    system_prompt: []const u8 = DEFAULT_SYSTEM_PROMPT,
    temperature: f32 = DEFAULT_TEMPERATURE,
    show_help: bool = false,
    owns_model_path: bool = false,
    owns_system_prompt: bool = false,
};

fn parseArgs() !CliArgs {
    var args_iter = try process.argsWithAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.next();

    var args = CliArgs{};
    errdefer freeArgs(&args);

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            const value_arg = args_iter.next() orelse return error.MissingModelPath;
            const duplicated = try allocator.dupe(u8, value_arg);
            if (args.owns_model_path) allocator.free(@constCast(args.model_path));
            args.model_path = duplicated;
            args.owns_model_path = true;
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
            const value_arg = args_iter.next() orelse return error.MissingSystemPrompt;
            const duplicated = try allocator.dupe(u8, value_arg);
            if (args.owns_system_prompt) allocator.free(@constCast(args.system_prompt));
            args.system_prompt = duplicated;
            args.owns_system_prompt = true;
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

fn freeArgs(args: *CliArgs) void {
    if (args.owns_model_path) {
        allocator.free(@constCast(args.model_path));
        args.owns_model_path = false;
    }
    if (args.owns_system_prompt) {
        allocator.free(@constCast(args.system_prompt));
        args.owns_system_prompt = false;
    }
}

fn printHelp(exe_name: []const u8) void {
    print(
        "Usage: {s} [options]\n" ++ "\n" ++ "A simple CLI chat interface for Qwen3 models using llama.cpp.\n" ++ "\n" ++ "Options:\n" ++ "  -m, --model <path>    Path to the GGUF model file (required)\n" ++ "  -s, --system <prompt> System prompt to use (default: \"{s}\")\n" ++ "  -t, --temp <value>    Sampling temperature (default: {d:.1})\n" ++ "  -h, --help            Show this help message\n\n",
        .{ exe_name, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE },
    );
}

const Message = struct {
    role: []const u8,
    content: []const u8,
};

fn toCString(slice: []const u8) ![:0]u8 {
    const c_str = try allocator.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}

pub fn main() !void {
    defer {
        if (gpa.deinit() == .leak) {
            print("Warning: Memory leak detected!\n", .{});
        }
    }

    var args = try parseArgs();
    defer freeArgs(&args);

    if (args.show_help) {
        const exe_path = try fs.selfExePathAlloc(allocator);
        defer allocator.free(exe_path);
        const exe_name = fs.path.basename(exe_path);
        printHelp(exe_name);
        return;
    }

    c.llama_backend_init();
    defer c.llama_backend_free();

    c.llama_log_set(logCallback, null);

    var mparams = c.llama_model_default_params();
    mparams.use_mmap = false;

    const c_model_path = try toCString(args.model_path);
    defer allocator.free(c_model_path);

    const model_ptr = c.llama_model_load_from_file(c_model_path.ptr, mparams);
    if (model_ptr == null) {
        print("Error: Failed to load model from '{s}'\n", .{args.model_path});
        return error.ModelLoadFailed;
    }
    const model = model_ptr.?;
    defer c.llama_model_free(model);

    var cparams = c.llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;

    const ctx_ptr = c.llama_init_from_model(model, cparams);
    if (ctx_ptr == null) {
        print("Error: Failed to create context\n", .{});
        return error.ContextCreateFailed;
    }
    const ctx = ctx_ptr.?;
    defer c.llama_free(ctx);

    const n_ctx = c.llama_n_ctx(ctx);
    print("Context created (n_ctx = {d})\n", .{n_ctx});
    print("System Prompt: {s}\n", .{args.system_prompt});
    print("Temperature: {d:.2}\n", .{args.temperature});
    print("Enter your message (Ctrl+D or /quit to exit):\n", .{});

    var history = std.ArrayList(Message).empty;
    defer {
        for (history.items) |msg| {
            allocator.free(msg.content);
        }
        history.deinit(allocator);
    }

    const stdin_file = fs.File.stdin();
    const stdout_file = fs.File.stdout();
    var input_buffer = std.ArrayList(u8).empty;
    defer input_buffer.deinit(allocator);

    const vocab = c.llama_model_get_vocab(model);
    const template_ptr = c.llama_model_chat_template(model, null);
    const fallback_template = [_:0]u8{ 'c', 'h', 'a', 't', 'm', 'l', 0 };
    const fallback_template_slice: [:0]const u8 = fallback_template[0..];

    while (true) {
        try stdout_file.writeAll("> ");

        input_buffer.clearRetainingCapacity();
        var received_eof = false;
        while (true) {
            var byte_buf: [1]u8 = undefined;
            const read_len = stdin_file.read(byte_buf[0..]) catch |err| {
                print("Error reading input: {}\n", .{err});
                return err;
            };

            if (read_len == 0) {
                received_eof = true;
                break;
            }

            const b = byte_buf[0];
            if (b == '\n') break;
            try input_buffer.append(allocator, b);
        }

        if (received_eof and input_buffer.items.len == 0) {
            print("\nExiting...\n", .{});
            break;
        }

        const user_input = std.mem.trim(u8, input_buffer.items, " \t\r\n");

        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "/quit")) {
            print("Exiting...\n", .{});
            break;
        }

        var current_chat = std.ArrayList(c.llama_chat_message).empty;
        defer current_chat.deinit(allocator);

        const c_system_role = try toCString("system");
        defer allocator.free(c_system_role);
        const c_system_prompt = try toCString(args.system_prompt);
        defer allocator.free(c_system_prompt);
        try current_chat.append(allocator, .{ .role = c_system_role.ptr, .content = c_system_prompt.ptr });

        var temp_slices = std.ArrayList([:0]u8).empty;
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
            try current_chat.append(allocator, .{ .role = c_role.ptr, .content = c_content.ptr });
        }

        const c_user_role = try toCString("user");
        defer allocator.free(c_user_role);
        const c_user_input = try toCString(user_input);
        defer allocator.free(c_user_input);
        try current_chat.append(allocator, .{ .role = c_user_role.ptr, .content = c_user_input.ptr });

        var formatted_prompt_buf = std.ArrayList(u8).empty;
        defer formatted_prompt_buf.deinit(allocator);
        const initial_buf_size: usize = 2048;
        try formatted_prompt_buf.resize(allocator, initial_buf_size);

        const fallback_ptr: [*c]const u8 = @ptrCast(fallback_template_slice.ptr);
        const tmpl_arg: [*c]const u8 = if (template_ptr) |tmpl| @ptrCast(tmpl) else fallback_ptr;
        const first_buffer: [*c]u8 = @ptrCast(formatted_prompt_buf.items.ptr);
        const first_buffer_len: c_int = @intCast(formatted_prompt_buf.items.len);
        var required_len = c.llama_chat_apply_template(tmpl_arg, current_chat.items.ptr, current_chat.items.len, true, first_buffer, first_buffer_len);

        if (required_len < 0) {
            print("Error: llama_chat_apply_template failed ({d})\n", .{required_len});
            continue;
        }

        const required_u32: u32 = @intCast(required_len);
        if (required_u32 > formatted_prompt_buf.items.len) {
            try formatted_prompt_buf.resize(allocator, @intCast(required_len));
            const resized_buffer: [*c]u8 = @ptrCast(formatted_prompt_buf.items.ptr);
            const resized_buffer_len: c_int = @intCast(formatted_prompt_buf.items.len);
            required_len = c.llama_chat_apply_template(tmpl_arg, current_chat.items.ptr, current_chat.items.len, true, resized_buffer, resized_buffer_len);
            if (required_len < 0) {
                print("Error: llama_chat_apply_template failed after resize ({d})\n", .{required_len});
                continue;
            }
        }

        const prompt_len: usize = @intCast(required_len);
        const full_prompt_bytes = formatted_prompt_buf.items[0..prompt_len];

        const tokens = try allocator.alloc(c.llama_token, @intCast(n_ctx));
        defer allocator.free(tokens);

        const add_bos = c.llama_vocab_get_add_bos(vocab);
        const prompt_ptr: [*c]const u8 = @ptrCast(full_prompt_bytes.ptr);
        const token_ptr: [*c]c.llama_token = @ptrCast(tokens.ptr);
        const prompt_len_c: c_int = @intCast(full_prompt_bytes.len);
        const token_capacity_c: c_int = @intCast(tokens.len);
        const n_tokens = c.llama_tokenize(vocab, prompt_ptr, prompt_len_c, token_ptr, token_capacity_c, add_bos, false);

        if (n_tokens < 0) {
            print("Error: Failed to tokenize prompt\n", .{});
            continue;
        }

        print("Decoding prompt ({d} tokens)...\n", .{n_tokens});
        const n_tokens_c: c_int = @intCast(n_tokens);
        const prompt_batch = c.llama_batch_get_one(tokens.ptr, n_tokens_c);
        const decode_prompt_ret = c.llama_decode(ctx, prompt_batch);
        print("llama_decode for prompt returned: {d}\n", .{decode_prompt_ret});
        if (decode_prompt_ret != 0) {
            print("Error: llama_decode failed during prompt processing (ret={d})\n", .{decode_prompt_ret});
            continue;
        }

        print("Assistant: ", .{});

        var generated_response = std.ArrayList(u8).empty;
        defer generated_response.deinit(allocator);
        var writer = generated_response.writer(allocator);

        const n_tokens_u32: u32 = @intCast(n_tokens);
        const max_new_tokens = if (n_ctx > n_tokens_u32) n_ctx - n_tokens_u32 else 0;
        var generated_token_count: u32 = 0;

        const chain_params = c.llama_sampler_chain_default_params();
        const sampler_chain_ptr = c.llama_sampler_chain_init(chain_params);
        if (sampler_chain_ptr == null) {
            print("Error: Failed to init sampler chain\n", .{});
            continue;
        }
        const sampler_chain = sampler_chain_ptr.?;
        defer c.llama_sampler_free(sampler_chain);

        const greedy_sampler_ptr = c.llama_sampler_init_greedy();
        if (greedy_sampler_ptr == null) {
            print("Error: Failed to init greedy sampler\n", .{});
            continue;
        }
        c.llama_sampler_chain_add(sampler_chain, greedy_sampler_ptr.?);

        while (generated_token_count < max_new_tokens) {
            const id = c.llama_sampler_sample(sampler_chain, ctx, -1);
            c.llama_sampler_accept(sampler_chain, id);

            if (c.llama_vocab_is_eog(vocab, id)) {
                break;
            }

            var piece_len = c.llama_token_to_piece(vocab, id, null, 0, 0, false);
            var allow_special = false;

            if (piece_len == 0) {
                piece_len = c.llama_token_to_piece(vocab, id, null, 0, 0, true);
                allow_special = true;
            }

            if (piece_len < 0) {
                piece_len = -piece_len;
            }

            if (piece_len <= 0) {
                print("Error: llama_token_to_piece failed (token_id={d})\n", .{id});
                break;
            }

            const piece_buf = try allocator.alloc(u8, @intCast(piece_len));
            defer allocator.free(piece_buf);
            const piece_ptr: [*c]u8 = @ptrCast(piece_buf.ptr);
            const piece_capacity_c: c_int = @intCast(piece_buf.len);
            const actual_len = c.llama_token_to_piece(vocab, id, piece_ptr, piece_capacity_c, 0, allow_special);
            if (actual_len <= 0) {
                print("Error: llama_token_to_piece failed to produce text (token_id={d}, len={d})\n", .{ id, actual_len });
                break;
            }

            const piece_slice = piece_buf[0..@intCast(actual_len)];
            try stdout_file.writeAll(piece_slice);
            try writer.writeAll(piece_slice);

            var token_id = id;
            const token_batch = c.llama_batch_get_one(&token_id, 1);
            const decode_gen_ret = c.llama_decode(ctx, token_batch);
            if (decode_gen_ret != 0) {
                print("Error: llama_decode failed during generation (ret={d})\n", .{decode_gen_ret});
                break;
            }
            generated_token_count += 1;
        }
        try stdout_file.writeAll("\n");

        const user_input_copy = try allocator.dupe(u8, user_input);
        errdefer allocator.free(user_input_copy);
        const assistant_response_copy = try generated_response.toOwnedSlice(allocator);
        errdefer allocator.free(assistant_response_copy);

        try history.append(allocator, .{ .role = "user", .content = user_input_copy });
        try history.append(allocator, .{ .role = "assistant", .content = assistant_response_copy });
    }
}
