const std = @import("std");
const llama = @import("llama.zig");
const cli = @import("cli.zig");
const chat = @import("chat.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        if (gpa.deinit() == .leak) {
            std.debug.print("Warning: Memory leak detected!\n", .{});
        }
    }
    const allocator = gpa.allocator();

    // Parse command line arguments
    var args = cli.parseArgs(allocator) catch |err| {
        std.debug.print("Error parsing arguments: {}\n", .{err});
        return;
    };
    defer cli.freeArgs(allocator, &args);

    if (args.show_help) {
        const exe_path = std.fs.selfExePathAlloc(allocator) catch {
            cli.printHelp("qwen_cli");
            return;
        };
        defer allocator.free(exe_path);
        cli.printHelp(std.fs.path.basename(exe_path));
        return;
    }

    if (args.model_path.len == 0) {
        std.debug.print("Error: Model path is required. Use -m or --model to specify.\n", .{});
        return;
    }

    // Initialize llama.cpp backend
    std.debug.print("Initializing llama.cpp backend...\n", .{});
    llama.llama_backend_init();
    defer {
        std.debug.print("Cleaning up llama.cpp backend...\n", .{});
        llama.llama_backend_free();
    }

    // Set logging callback
    llama.llama_log_set(&llama.logCallback, null);
    std.debug.print("llama.cpp log handler set.\n", .{});

    // Load model
    var mparams = llama.llama_model_default_params();
    mparams.use_mmap = false;

    std.debug.print("Loading model: {s}...\n", .{args.model_path});
    const c_model_path = chat.toCString(allocator, args.model_path) catch {
        std.debug.print("Error: Failed to allocate model path string\n", .{});
        return;
    };
    defer allocator.free(c_model_path);

    const model = llama.llama_model_load_from_file(c_model_path, mparams);
    if (model == null) {
        std.debug.print("Error: Failed to load model from '{s}'\n", .{args.model_path});
        return;
    }
    defer llama.llama_model_free(model);

    // Create context
    var cparams = llama.llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;

    std.debug.print("Creating context (n_ctx = {d}, n_batch = {d})...\n", .{ cparams.n_ctx, cparams.n_batch });
    const ctx = llama.llama_init_from_model(model, cparams);
    if (ctx == null) {
        std.debug.print("Error: Failed to create context\n", .{});
        return;
    }
    defer llama.llama_free(ctx);

    const n_ctx = llama.llama_n_ctx(ctx);
    std.debug.print("Context created (n_ctx = {d})\n", .{n_ctx});
    std.debug.print("Model loaded successfully.\n", .{});
    std.debug.print("System Prompt: {s}\n", .{args.system_prompt});
    std.debug.print("Temperature: {d:.2}\n", .{args.temperature});
    std.debug.print("Enter your message (Ctrl+D or /quit to exit):\n", .{});

    // Initialize chat history
    var history = chat.ChatHistory.init(allocator);
    defer history.deinit();

    // Get vocab for tokenization
    const vocab = llama.llama_model_get_vocab(model);

    // IO setup - Zig 0.15 style (using deprecated API for simpler line reading)
    const stdin_file = std.fs.File.stdin();
    const stdout_file = std.fs.File.stdout();
    const stdin_reader = stdin_file.deprecatedReader();
    var input_buf: [4096]u8 = undefined;

    // Chat loop
    while (true) {
        stdout_file.writeAll("> ") catch {};

        // Read line from stdin
        const line = stdin_reader.readUntilDelimiterOrEof(&input_buf, '\n') catch |err| {
            std.debug.print("Error reading input: {}\n", .{err});
            return;
        } orelse {
            std.debug.print("\nExiting...\n", .{});
            break;
        };

        const user_input = std.mem.trim(u8, line, " \t\r\n");

        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "/quit")) {
            std.debug.print("Exiting...\n", .{});
            break;
        }

        // Build chat messages for template
        var temp_strings = chat.TempCStrings.init(allocator);
        defer temp_strings.deinit();

        var chat_messages = chat.buildChatMessages(
            allocator,
            args.system_prompt,
            &history,
            user_input,
            &temp_strings,
        ) catch {
            std.debug.print("Error: Failed to build chat messages\n", .{});
            continue;
        };
        defer chat_messages.deinit(allocator);

        // Apply template
        const full_prompt = chat.applyTemplate(allocator, model, chat_messages.items) catch {
            std.debug.print("Error: Failed to apply chat template\n", .{});
            continue;
        };
        defer allocator.free(full_prompt);

        // Tokenize
        const tokens = allocator.alloc(llama.llama_token, n_ctx) catch {
            std.debug.print("Error: Failed to allocate token buffer\n", .{});
            continue;
        };
        defer allocator.free(tokens);

        const add_bos = llama.llama_vocab_get_add_bos(vocab);
        const n_tokens = llama.llama_tokenize(
            vocab,
            full_prompt.ptr,
            @intCast(full_prompt.len),
            tokens.ptr,
            @intCast(n_ctx),
            add_bos,
            false,
        );

        if (n_tokens < 0) {
            std.debug.print("Error: Failed to tokenize prompt\n", .{});
            continue;
        }

        // Decode prompt
        std.debug.print("Assistant: ", .{});
        std.debug.print("Decoding prompt ({d} tokens)...\n", .{n_tokens});

        const prompt_batch = llama.llama_batch_get_one(tokens.ptr, n_tokens);
        const decode_ret = llama.llama_decode(ctx, prompt_batch);
        if (decode_ret != 0) {
            std.debug.print("Error: llama_decode failed (ret={d})\n", .{decode_ret});
            continue;
        }
        std.debug.print("Prompt decoded successfully.\n", .{});

        // Initialize sampler chain
        const chain_params = llama.llama_sampler_chain_default_params();
        const sampler_chain = llama.llama_sampler_chain_init(chain_params);
        if (sampler_chain == null) {
            std.debug.print("Error: Failed to init sampler chain\n", .{});
            continue;
        }
        defer llama.llama_sampler_free(sampler_chain);

        // Add greedy sampler
        const greedy_sampler = llama.llama_sampler_init_greedy();
        llama.llama_sampler_chain_add(sampler_chain, greedy_sampler);

        // Generation loop
        var generated_response = std.ArrayListUnmanaged(u8){};
        defer generated_response.deinit(allocator);

        const max_new_tokens = n_ctx - @as(u32, @intCast(n_tokens));
        var generated_count: u32 = 0;

        // Buffer for token-to-piece conversion (reused across iterations)
        var piece_buf: [256]u8 = undefined;

        while (generated_count < max_new_tokens) {
            // Sample next token
            const id = llama.llama_sampler_sample(sampler_chain, ctx, -1);
            llama.llama_sampler_accept(sampler_chain, id);

            // Check for end of generation
            if (llama.llama_vocab_is_eog(vocab, id)) {
                break;
            }

            // Convert token to text using a pre-allocated buffer
            // llama_token_to_piece returns positive length on success,
            // negative value means buffer too small (abs value = required size)
            const piece_len = llama.llama_token_to_piece(vocab, id, &piece_buf, @intCast(piece_buf.len), 0, false);

            if (piece_len < 0) {
                // Buffer too small - this shouldn't happen with 256 bytes but handle it
                std.debug.print("Warning: token {d} requires larger buffer ({d} bytes)\n", .{ id, -piece_len });
                continue;
            }

            if (piece_len == 0) {
                // Empty piece (might be a special token), skip
                continue;
            }

            const piece_slice = piece_buf[0..@intCast(piece_len)];

            // Output and store
            stdout_file.writeAll(piece_slice) catch {};
            generated_response.appendSlice(allocator, piece_slice) catch break;

            // Decode next token using llama_batch_get_one for simplicity
            var single_token = [_]llama.llama_token{id};
            const batch = llama.llama_batch_get_one(&single_token, 1);

            const gen_decode_ret = llama.llama_decode(ctx, batch);
            if (gen_decode_ret != 0) {
                std.debug.print("Error: llama_decode failed during generation\n", .{});
                break;
            }
            generated_count += 1;
        }

        stdout_file.writeAll("\n") catch {};

        // Update history
        history.append("user", user_input) catch {
            std.debug.print("Warning: Failed to save user message to history\n", .{});
        };

        const response_slice = generated_response.items;
        history.append("assistant", response_slice) catch {
            std.debug.print("Warning: Failed to save assistant response to history\n", .{});
        };
    }
}
