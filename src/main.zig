// main.zig - Qwen CLI application entry point
//
// A CLI chat interface for Qwen3 language models using llama.cpp.

const std = @import("std");
const llama = @import("llama.zig");
const chat = @import("chat.zig");
const config = @import("config.zig");

const Config = config.Config;
const History = chat.History;
const Format = chat.Format;

// ============================================================================
// Logging
// ============================================================================

fn llamaLogCallback(level: llama.LogLevel, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    const level_str = switch (@intFromEnum(level)) {
        @intFromEnum(llama.LogLevel.@"error") => "[ERROR]",
        @intFromEnum(llama.LogLevel.warn) => "[WARN]",
        @intFromEnum(llama.LogLevel.info) => "[INFO]",
        else => "[DEBUG]",
    };
    std.debug.print("llama {s} {s}", .{ level_str, text });
}

// ============================================================================
// Model Validation
// ============================================================================

fn validateModelFile(path: []const u8) !void {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        return switch (err) {
            error.FileNotFound => {
                std.debug.print("Error: Model file not found: '{s}'\n", .{path});
                return error.FileNotFound;
            },
            error.AccessDenied => {
                std.debug.print("Error: Access denied: '{s}'\n", .{path});
                return error.AccessDenied;
            },
            else => {
                std.debug.print("Error: Cannot access '{s}': {}\n", .{ path, err });
                return err;
            },
        };
    };
    defer file.close();

    // Check file size
    const stat = try file.stat();
    if (stat.size < 8) {
        std.debug.print("Error: File too small to be valid GGUF: {d} bytes\n", .{stat.size});
        return error.InvalidFileSize;
    }

    // Check GGUF magic bytes
    var magic: [4]u8 = undefined;
    _ = try file.readAll(&magic);
    if (!std.mem.eql(u8, &magic, "GGUF")) {
        std.debug.print("Error: Invalid GGUF magic: {any}\n", .{magic});
        return error.InvalidMagic;
    }
}

// ============================================================================
// Inference Engine
// ============================================================================

const InferenceEngine = struct {
    model: *llama.Model,
    ctx: *llama.Context,
    vocab: *llama.Vocab,
    cfg: Config,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, cfg: Config) !Self {
        // Initialize backend
        llama.init();
        llama.setLogCallback(&llamaLogCallback, null);

        // Validate model file
        try validateModelFile(cfg.model_path);

        // Load model
        std.debug.print("Loading model: {s}\n", .{cfg.model_path});
        const model_path_z = try allocator.dupeZ(u8, cfg.model_path);
        defer allocator.free(model_path_z);

        const model = try llama.loadModel(model_path_z, llama.defaultModelParams());
        errdefer llama.freeModel(model);

        // Create context
        var ctx_params = llama.defaultContextParams();
        ctx_params.n_ctx = cfg.context_size;
        ctx_params.n_batch = cfg.batch_size;
        ctx_params.n_ubatch = cfg.batch_size;

        const ctx = try llama.createContext(model, ctx_params);
        errdefer llama.freeContext(ctx);

        const vocab = try llama.getVocab(model);

        std.debug.print("Model loaded (context size: {d})\n", .{llama.getContextSize(ctx)});

        return .{
            .model = model,
            .ctx = ctx,
            .vocab = vocab,
            .cfg = cfg,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        llama.freeContext(self.ctx);
        llama.freeModel(self.model);
        llama.deinit();
    }

    /// Generate a response for the given prompt.
    pub fn generate(self: *Self, prompt: []const u8, writer: anytype) ![]const u8 {
        // Tokenize
        var tokens: [Config.defaults.token_buffer_size]llama.Token = undefined;
        const add_bos = llama.shouldAddBos(self.vocab);
        const n_tokens = try llama.tokenize(self.vocab, prompt, &tokens, add_bos);

        if (n_tokens == 0) {
            return error.EmptyPrompt;
        }

        // Clear KV cache for fresh context
        llama.clearKvCache(self.ctx);

        // Process prompt in batches
        var pos: usize = 0;
        while (pos < n_tokens) {
            const chunk_size = @min(self.cfg.batch_size, n_tokens - pos);
            const chunk = tokens[pos..][0..chunk_size];
            const batch = llama.batchFromTokens(chunk);
            try llama.decode(self.ctx, batch);
            pos += chunk_size;
        }

        // Initialize sampler
        var sampler = try llama.SamplerChain.init();
        defer sampler.deinit();

        if (self.cfg.temperature > 0) {
            try sampler.addTemperature(self.cfg.temperature);
        }
        try sampler.addGreedy();

        // Generate tokens
        var response_buf: [Config.defaults.response_buffer_size]u8 = undefined;
        var response_len: usize = 0;
        var cur_pos: llama.Pos = @intCast(n_tokens);
        const max_tokens = @min(self.cfg.max_new_tokens, self.cfg.context_size - @as(u32, @intCast(n_tokens)));

        var generated: u32 = 0;
        while (generated < max_tokens) : (generated += 1) {
            const token_id = sampler.sample(self.ctx);

            if (token_id < 0 or llama.isEndOfGeneration(self.vocab, token_id)) {
                break;
            }

            sampler.accept(token_id);

            // Convert token to text
            var piece_buf: [Config.defaults.piece_buffer_size]u8 = undefined;
            const piece = llama.tokenToPiece(self.vocab, token_id, &piece_buf) catch continue;

            if (piece.len == 0) continue;

            // Output immediately
            writer.print("{s}", .{piece}) catch {};

            // Store in response buffer
            if (response_len + piece.len < response_buf.len) {
                @memcpy(response_buf[response_len..][0..piece.len], piece);
                response_len += piece.len;
            }

            // Decode next token
            const batch = llama.Batch{
                .n_tokens = 1,
                .token = @constCast(&token_id),
                .embd = null,
                .pos = &cur_pos,
                .n_seq_id = null,
                .seq_id = null,
                .logits = null,
            };
            llama.decode(self.ctx, batch) catch break;
            cur_pos += 1;
        }

        // Return a copy of the response
        if (response_len > 0) {
            return try self.allocator.dupe(u8, response_buf[0..response_len]);
        }
        return "";
    }
};

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse arguments
    const parse_result = config.parseArgs(allocator);
    switch (parse_result) {
        .help => {
            const exe_path = try std.fs.selfExePathAlloc(allocator);
            defer allocator.free(exe_path);
            const exe_name = std.fs.path.basename(exe_path);
            try config.printHelp(std.fs.File.stdout().deprecatedWriter(), exe_name);
            return;
        },
        .@"error" => |err| {
            std.debug.print("Argument error: {}\n", .{err});
            std.debug.print("Use --help for usage information.\n", .{});
            return;
        },
        .config => {},
    }
    var cfg = parse_result.config;
    defer cfg.deinit();

    // Initialize inference engine
    var engine = InferenceEngine.init(allocator, cfg) catch |err| {
        std.debug.print("Failed to initialize: {}\n", .{err});
        return;
    };
    defer engine.deinit();

    // Print startup info
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("System: {s}\n", .{cfg.system_prompt});
    try stdout.print("Temperature: {d:.2}\n", .{cfg.temperature});
    try stdout.print("Enter message (Ctrl+D or /quit to exit)\n\n", .{});

    // Initialize chat history
    var history = History.init(allocator, cfg.max_history_pairs);
    defer history.deinit();

    // Chat loop - use fixed buffer for input to avoid ArrayList complexity
    const stdin = std.fs.File.stdin().deprecatedReader();
    var input_buf: [4096]u8 = undefined;

    while (true) {
        try stdout.print("> ", .{});

        const line = stdin.readUntilDelimiter(&input_buf, '\n') catch |err| {
            if (err == error.EndOfStream) {
                try stdout.print("\nGoodbye!\n", .{});
                break;
            }
            std.debug.print("Input error: {}\n", .{err});
            continue;
        };

        const user_input = std.mem.trim(u8, line, " \t\r\n");
        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "/quit")) {
            try stdout.print("Goodbye!\n", .{});
            break;
        }

        // Build prompt
        const prompt = Format.buildPrompt(allocator, cfg.system_prompt, &history, user_input) catch |err| {
            std.debug.print("Failed to build prompt: {}\n", .{err});
            continue;
        };
        defer allocator.free(prompt);

        if (prompt.len > Config.defaults.max_prompt_bytes) {
            std.debug.print("Prompt too long ({d} bytes). Try shorter messages.\n", .{prompt.len});
            continue;
        }

        // Generate response
        try stdout.print("Assistant: ", .{});
        const response = engine.generate(prompt, stdout) catch |err| {
            try stdout.print("\n", .{});
            std.debug.print("Generation error: {}\n", .{err});
            continue;
        };
        defer if (response.len > 0) allocator.free(response);
        try stdout.print("\n", .{});

        // Update history
        if (response.len > 0) {
            history.add(.user, user_input) catch continue;
            history.add(.assistant, response) catch {
                history.removeLast();
                continue;
            };
        }
    }
}
