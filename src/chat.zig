const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const io = std.io;

const llama = @import("llama_cpp.zig");
const config = @import("config.zig");
const utils = @import("utils.zig");

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub fn runChatLoop(
    allocator: Allocator,
    model: ?*llama.llama_model,
    ctx: ?*llama.llama_context,
    args: config.CliArgs,
    n_ctx: u32,
    vocab: ?*llama.llama_vocab,
) !void {
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

    while (true) {
        try stdout.print("> ", .{});
        // stdout is unbuffered with deprecatedWriter

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

        const c_system_role = try utils.toCString(allocator, "system");
        defer allocator.free(c_system_role);
        const c_system_prompt = try utils.toCString(allocator, args.system_prompt);
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
            const c_role = try utils.toCString(allocator, msg.role);
            try temp_slices.append(allocator, c_role);
            const c_content = try utils.toCString(allocator, msg.content);
            try temp_slices.append(allocator, c_content);
            try current_chat.append(allocator, .{ .role = c_role, .content = c_content });
        }

        const c_user_role = try utils.toCString(allocator, "user");
        defer allocator.free(c_user_role);
        const c_user_input = try utils.toCString(allocator, user_input);
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
        const prompt_batch = llama.llama_batch_get_one(tokens.ptr, n_tokens);
        const decode_prompt_ret = llama.llama_decode(ctx, prompt_batch);
        if (decode_prompt_ret != 0) {
             print("Error: llama_decode failed during prompt processing (ret={d})\n", .{decode_prompt_ret});
             continue;
        }

        // --- Generation Loop ---
        var generated_response = std.ArrayListUnmanaged(u8){};
        defer generated_response.deinit(allocator);
        const writer = generated_response.writer(allocator);

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
}
