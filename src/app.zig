const std = @import("std");
const fs = std.fs;

const llama = @import("llama.zig");
const cli_args = @import("cli_args.zig");

const ArrayList = std.array_list.Managed;

const Message = struct {
    role: []const u8,
    content: []const u8,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

fn toCString(allocator: std.mem.Allocator, slice: []const u8) ![:0]const u8 {
    const c_str = try allocator.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}

fn logCallback(level: llama.ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    const level_int = @intFromEnum(level);
    switch (level_int) {
        @intFromEnum(llama.ggml_log_level.ERROR) => std.debug.print("LLAMA: [ERROR] {s}", .{text}),
        @intFromEnum(llama.ggml_log_level.WARN) => std.debug.print("LLAMA: [WARN] {s}", .{text}),
        @intFromEnum(llama.ggml_log_level.INFO) => std.debug.print("LLAMA: [INFO] {s}", .{text}),
        @intFromEnum(llama.ggml_log_level.CONT) => std.debug.print("{s}", .{text}),
        else => std.debug.print("LLAMA: [LEVEL={d}] {s}", .{ level_int, text }),
    }
}

pub fn main() !void {
    const allocator = gpa.allocator();
    defer {
        if (gpa.deinit() == .leak) {
            std.debug.print("Warning: Memory leak detected!\n", .{});
        }
    }

    var args = try cli_args.parseArgs(allocator);
    defer args.deinit(allocator);

    if (args.show_help) {
        const exe_path = try fs.selfExePathAlloc(allocator);
        defer allocator.free(exe_path);
        const exe_name = fs.path.basename(exe_path);
        cli_args.printHelp(exe_name);
        return;
    }

    std.debug.print("Initializing llama.cpp backend...\n", .{});
    llama.llama_backend_init();
    llama.llama_log_set(&logCallback, null);
    std.debug.print("llama.cpp log handler set.\n", .{});
    defer {
        std.debug.print("Cleaning up llama.cpp backend...\n", .{});
        llama.llama_backend_free();
    }

    var mparams = llama.llama_model_default_params();
    mparams.use_mmap = false;

    std.debug.print("Loading model: {s}...\n", .{args.model_path});
    const c_model_path = try toCString(allocator, args.model_path);
    defer allocator.free(c_model_path);

    const model = llama.llama_model_load_from_file(c_model_path, mparams);
    if (model == null) {
        std.debug.print("Error: Failed to load model from '{s}'\n", .{args.model_path});
        return error.ModelLoadFailed;
    }
    defer llama.llama_model_free(model);

    var cparams = llama.llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;

    std.debug.print("Creating context (n_ctx = {d}, n_batch = {d})...\n", .{ cparams.n_ctx, cparams.n_batch });
    const ctx = llama.llama_init_from_model(model, cparams);
    if (ctx == null) {
        std.debug.print("Error: Failed to create context\n", .{});
        return error.ContextCreateFailed;
    }
    defer llama.llama_free(ctx);

    const n_ctx = llama.llama_n_ctx(ctx);
    std.debug.print("Context created (n_ctx = {d})\n", .{n_ctx});
    std.debug.print("Model loaded successfully.\n", .{});
    std.debug.print("System Prompt: {s}\n", .{args.system_prompt});
    std.debug.print("Temperature: {d:.2}\n", .{args.temperature});
    std.debug.print("Enter your message (Ctrl+D or /quit to exit):\n", .{});

    var history = ArrayList(Message).init(allocator);
    defer {
        for (history.items) |msg| {
            allocator.free(msg.content);
        }
        history.deinit();
    }

    const stdin = fs.File.stdin().deprecatedReader();
    const stdout = fs.File.stdout().deprecatedWriter();

    var input_buffer = ArrayList(u8).init(allocator);
    defer input_buffer.deinit();

    while (true) {
        try stdout.print("> ", .{});

        input_buffer.clearRetainingCapacity();
        stdin.streamUntilDelimiter(input_buffer.writer(), '\n', null) catch |err| {
            if (err == error.EndOfStream) {
                std.debug.print("\nExiting...\n", .{});
                break;
            } else {
                std.debug.print("Error reading input: {}\n", .{err});
                return err;
            }
        };

        const user_input = std.mem.trim(u8, input_buffer.items, " \t\r\n");

        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "/quit")) {
            std.debug.print("Exiting...\n", .{});
            break;
        }

        var current_chat = ArrayList(llama.llama_chat_message).init(allocator);
        defer current_chat.deinit();

        const c_system_role = try toCString(allocator, "system");
        defer allocator.free(c_system_role);
        const c_system_prompt = try toCString(allocator, args.system_prompt);
        defer allocator.free(c_system_prompt);
        try current_chat.append(.{ .role = c_system_role, .content = c_system_prompt });

        var temp_slices = ArrayList([:0]const u8).init(allocator);
        defer {
            for (temp_slices.items) |slice_to_free| allocator.free(slice_to_free);
            temp_slices.deinit();
        }

        for (history.items) |msg| {
            const c_role = try toCString(allocator, msg.role);
            try temp_slices.append(c_role);
            const c_content = try toCString(allocator, msg.content);
            try temp_slices.append(c_content);
            try current_chat.append(.{ .role = c_role, .content = c_content });
        }

        const c_user_role = try toCString(allocator, "user");
        defer allocator.free(c_user_role);
        const c_user_input = try toCString(allocator, user_input);
        defer allocator.free(c_user_input);
        try current_chat.append(.{ .role = c_user_role, .content = c_user_input });

        var formatted_prompt_buf = ArrayList(u8).init(allocator);
        defer formatted_prompt_buf.deinit();
        const initial_buf_size: i32 = 2048;
        try formatted_prompt_buf.resize(@intCast(initial_buf_size));

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
            std.debug.print("Error: llama_chat_apply_template failed ({d})\n", .{required_len});
            continue;
        }

        if (@as(u32, @intCast(required_len)) > initial_buf_size) {
            std.debug.print("Resizing chat template buffer to {d} bytes\n", .{required_len});
            try formatted_prompt_buf.resize(@intCast(required_len));
            required_len = llama.llama_chat_apply_template(
                tmpl,
                current_chat.items.ptr,
                current_chat.items.len,
                true,
                formatted_prompt_buf.items.ptr,
                required_len,
            );
            if (required_len < 0) {
                std.debug.print("Error: llama_chat_apply_template failed after resize ({d})\n", .{required_len});
                continue;
            }
        }

        const full_prompt_bytes = formatted_prompt_buf.items[0..@intCast(required_len)];

        std.debug.print("Assistant: ", .{});

        const tokens = try allocator.alloc(llama.llama_token, n_ctx);
        defer allocator.free(tokens);

        const vocab = llama.llama_model_get_vocab(model);
        if (vocab == null) {
            std.debug.print("Error: Failed to get vocab\n", .{});
            continue;
        }

        const add_bos = llama.llama_vocab_get_add_bos(vocab);
        const n_tokens = llama.llama_tokenize(vocab, full_prompt_bytes.ptr, @intCast(full_prompt_bytes.len), tokens.ptr, @intCast(n_ctx), add_bos, false);
        if (n_tokens < 0) {
            std.debug.print("Error: Failed to tokenize prompt (token buffer too small?)\n", .{});
            continue;
        }

        std.debug.print("Decoding prompt ({d} tokens)...\n", .{n_tokens});
        const prompt_batch = llama.llama_batch_get_one(&tokens[0], @intCast(n_tokens));
        const decode_prompt_ret = llama.llama_decode(ctx, prompt_batch);
        std.debug.print("llama_decode for prompt returned: {d}\n", .{decode_prompt_ret});
        if (decode_prompt_ret != 0) {
            std.debug.print("Error: llama_decode failed during prompt processing (ret={d})\n", .{decode_prompt_ret});
            continue;
        }

        var generated_response = ArrayList(u8).init(allocator);
        defer generated_response.deinit();
        const writer = generated_response.writer();

        const cur_pos: llama.llama_pos = @intCast(n_tokens);
        const max_new_tokens = n_ctx - @as(u32, @intCast(n_tokens));
        var generated_token_count: u32 = 0;

        const chain_params = llama.llama_sampler_chain_default_params();
        const sampler_chain = llama.llama_sampler_chain_init(chain_params);
        if (sampler_chain == null) {
            std.debug.print("Error: Failed to init sampler chain\n", .{});
            continue;
        }
        defer llama.llama_sampler_free(sampler_chain);

        const greedy_sampler = llama.llama_sampler_init_greedy();
        llama.llama_sampler_chain_add(sampler_chain, greedy_sampler);

        while (cur_pos < n_ctx and generated_token_count < max_new_tokens) {
            var id = llama.llama_sampler_sample(sampler_chain, ctx, -1);
            llama.llama_sampler_accept(sampler_chain, id);

            if (llama.llama_vocab_is_eog(vocab, id)) {
                break;
            }

            var stack_piece_buf: [4096]u8 = undefined;
            var piece_buf = stack_piece_buf[0..];

            var actual_len = llama.llama_token_to_piece(vocab, id, piece_buf.ptr, @intCast(piece_buf.len), 0, false);
            if (actual_len < 0) {
                const needed: usize = @intCast(-actual_len);
                const heap_piece_buf = try allocator.alloc(u8, needed);
                defer allocator.free(heap_piece_buf);

                actual_len = llama.llama_token_to_piece(vocab, id, heap_piece_buf.ptr, @intCast(heap_piece_buf.len), 0, false);
                if (actual_len < 0) {
                    std.debug.print("Error: llama_token_to_piece failed (token_id={d}, ret={d})\n", .{ id, actual_len });
                    break;
                }

                const piece = heap_piece_buf[0..@intCast(actual_len)];
                try stdout.print("{s}", .{piece});
                try writer.writeAll(piece);
            } else {
                const piece = piece_buf[0..@intCast(actual_len)];
                try stdout.print("{s}", .{piece});
                try writer.writeAll(piece);
            }

            const token_id_ptr: ?*llama.llama_token = &id;
            const batch = llama.llama_batch{
                .n_tokens = 1,
                .token = token_id_ptr,
                .embd = null,
                .pos = null,
                .n_seq_id = null,
                .seq_id = null,
                .logits = null,
            };

            const decode_gen_ret = llama.llama_decode(ctx, batch);
            if (decode_gen_ret != 0) {
                std.debug.print("Error: llama_decode failed during generation (ret={d})\n", .{decode_gen_ret});
                break;
            }
            generated_token_count += 1;
        }

        try stdout.print("\n", .{});

        const user_input_copy = try allocator.dupe(u8, user_input);
        errdefer allocator.free(user_input_copy);

        const assistant_response_copy = try generated_response.toOwnedSlice();
        errdefer allocator.free(assistant_response_copy);

        try history.append(.{ .role = "user", .content = user_input_copy });
        try history.append(.{ .role = "assistant", .content = assistant_response_copy });
    }

}
