const std = @import("std");
const config = @import("config.zig");
const llama = @import("llama.zig");

const fs = std.fs;
const print = std.debug.print;
const Allocator = std.mem.Allocator;

const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub fn run(allocator: Allocator, args: config.CliArgs) !void {
    llama.initBackend();
    defer llama.deinitBackend();
    llama.setLogCallback();

    var runtime = try llama.Runtime.init(allocator, args.model_path);
    defer runtime.deinit();

    const n_ctx = runtime.contextSize();
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

    const system_role_c = try llama.toCString(allocator, "system");
    defer allocator.free(system_role_c);

    chat_loop: while (true) {
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

        var current_chat = std.ArrayList(llama.ChatMessage).empty;
        defer current_chat.deinit(allocator);

        var temp_slices = std.ArrayList([:0]u8).empty;
        defer {
            for (temp_slices.items) |slice_to_free| {
                allocator.free(slice_to_free);
            }
            temp_slices.deinit(allocator);
        }

        const c_system_prompt = try llama.toCString(allocator, args.system_prompt);
        defer allocator.free(c_system_prompt);
        try current_chat.append(allocator, .{ .role = system_role_c.ptr, .content = c_system_prompt.ptr });

        for (history.items) |msg| {
            const c_role = try llama.toCString(allocator, msg.role);
            try temp_slices.append(allocator, c_role);
            const c_content = try llama.toCString(allocator, msg.content);
            try temp_slices.append(allocator, c_content);
            try current_chat.append(allocator, .{ .role = c_role.ptr, .content = c_content.ptr });
        }

        const c_user_role = try llama.toCString(allocator, "user");
        defer allocator.free(c_user_role);
        const c_user_input = try llama.toCString(allocator, user_input);
        defer allocator.free(c_user_input);
        try current_chat.append(allocator, .{ .role = c_user_role.ptr, .content = c_user_input.ptr });

        const formatted_prompt = runtime.applyChatTemplate(allocator, current_chat.items) catch |err| switch (err) {
            llama.Error.ChatTemplateFailed => {
                continue :chat_loop;
            },
            else => return err,
        };
        defer allocator.free(formatted_prompt);

        var token_buffer = runtime.tokenize(allocator, formatted_prompt) catch |err| switch (err) {
            llama.Error.TokenizeFailed => {
                print("Error: Failed to tokenize prompt\n", .{});
                continue :chat_loop;
            },
            else => return err,
        };
        defer token_buffer.deinit(allocator);

        const n_tokens = token_buffer.len();
        print("Decoding prompt ({d} tokens)...\n", .{n_tokens});
        const decode_prompt_ret = runtime.decodePrompt(&token_buffer);
        print("llama_decode for prompt returned: {d}\n", .{decode_prompt_ret});
        if (decode_prompt_ret != 0) {
            print("Error: llama_decode failed during prompt processing (ret={d})\n", .{decode_prompt_ret});
            continue;
        }

        print("Assistant: ", .{});

        var generated_response = std.ArrayList(u8).empty;
        defer generated_response.deinit(allocator);

        const max_new_tokens = runtime.remainingTokens(n_tokens);
        var generated_token_count: usize = 0;

        var sampler_chain = llama.SamplerChain.init() catch |err| switch (err) {
            llama.Error.SamplerChainInitFailed => {
                print("Error: Failed to init sampler chain\n", .{});
                continue :chat_loop;
            },
            llama.Error.SamplerInitFailed => {
                print("Error: Failed to init greedy sampler\n", .{});
                continue :chat_loop;
            },
            else => return err,
        };
        defer sampler_chain.deinit();

        generation: while (generated_token_count < max_new_tokens) {
            const id = runtime.sampleToken(&sampler_chain);
            sampler_chain.accept(id);

            if (runtime.isEndOfGeneration(id)) {
                break;
            }

            const piece_slice = runtime.tokenToPiece(allocator, id) catch |err| switch (err) {
                llama.Error.TokenPieceFailed => {
                    print("Error: llama_token_to_piece failed (token_id={d})\n", .{id});
                    break :generation;
                },
                else => return err,
            };
            defer allocator.free(piece_slice);

            try stdout_file.writeAll(piece_slice);
            try generated_response.appendSlice(allocator, piece_slice);

            var token_id = id;
            const decode_gen_ret = runtime.decodeToken(&token_id);
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
