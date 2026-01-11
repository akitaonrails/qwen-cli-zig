const std = @import("std");

pub const c = @cImport({
    @cInclude("llama.h");
});

const print = std.debug.print;
const Allocator = std.mem.Allocator;

fn logCallback(level: c_uint, text_ptr: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    const text_c: [*:0]const u8 = @ptrCast(text_ptr);
    const slice = std.mem.sliceTo(text_c, 0);
    const level_int: u32 = @intCast(level);
    switch (level) {
        @as(c_uint, c.GGML_LOG_LEVEL_ERROR) => print("LLAMA: [ERROR] {s}", .{slice}),
        @as(c_uint, c.GGML_LOG_LEVEL_WARN) => print("LLAMA: [WARN] {s}", .{slice}),
        @as(c_uint, c.GGML_LOG_LEVEL_INFO) => print("LLAMA: [INFO] {s}", .{slice}),
        else => print("LLAMA: [LEVEL={d}] {s}", .{ level_int, slice }),
    }
}

pub fn initBackend() void {
    c.llama_backend_init();
}

pub fn deinitBackend() void {
    c.llama_backend_free();
}

pub fn setLogCallback() void {
    c.llama_log_set(logCallback, null);
}

pub const Error = error{
    ModelLoadFailed,
    ContextCreateFailed,
    TokenizeFailed,
    DecodeFailed,
    ChatTemplateFailed,
    SamplerChainInitFailed,
    SamplerInitFailed,
    TokenPieceFailed,
};

const fallback_template = [_:0]u8{ 'c', 'h', 'a', 't', 'm', 'l', 0 };
const fallback_template_slice: [:0]const u8 = fallback_template[0..];

pub const ChatMessage = c.llama_chat_message;

pub fn toCString(allocator: Allocator, slice: []const u8) ![:0]u8 {
    const c_str = try allocator.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}

pub const Runtime = struct {
    model: *c.llama_model,
    ctx: *c.llama_context,
    vocab: *const c.llama_vocab,
    chat_template: [*:0]const u8,
    n_ctx: usize,

    pub fn init(allocator: Allocator, model_path: []const u8) !Runtime {
        var mparams = c.llama_model_default_params();
        mparams.use_mmap = false;

        const c_model_path = try toCString(allocator, model_path);
        defer allocator.free(c_model_path);

        const model_ptr = c.llama_model_load_from_file(c_model_path.ptr, mparams);
        if (model_ptr == null) {
            return Error.ModelLoadFailed;
        }
        const model = model_ptr.?;

        var cparams = c.llama_context_default_params();
        cparams.n_ctx = 4096;
        cparams.n_batch = 512;

        const ctx_ptr = c.llama_init_from_model(model, cparams);
        if (ctx_ptr == null) {
            c.llama_model_free(model);
            return Error.ContextCreateFailed;
        }
        const ctx = ctx_ptr.?;

        const vocab_ptr = c.llama_model_get_vocab(model);
        if (vocab_ptr == null) {
            c.llama_free(ctx);
            c.llama_model_free(model);
            return Error.ModelLoadFailed;
        }
        const vocab = vocab_ptr.?;
        const template_ptr = c.llama_model_chat_template(model, null);
        const chat_template = if (template_ptr) |tmpl| tmpl else fallback_template_slice.ptr;
        const n_ctx: usize = @intCast(c.llama_n_ctx(ctx));

        return Runtime{
            .model = model,
            .ctx = ctx,
            .vocab = vocab,
            .chat_template = chat_template,
            .n_ctx = n_ctx,
        };
    }

    pub fn deinit(self: *Runtime) void {
        c.llama_free(self.ctx);
        c.llama_model_free(self.model);
    }

    pub fn contextSize(self: *const Runtime) usize {
        return self.n_ctx;
    }

    pub fn applyChatTemplate(self: *const Runtime, allocator: Allocator, messages: []const ChatMessage) ![]u8 {
        var formatted_prompt_buf = std.ArrayList(u8).empty;
        errdefer formatted_prompt_buf.deinit(allocator);

        const initial_buf_size: usize = 2048;
        try formatted_prompt_buf.resize(allocator, initial_buf_size);

        const tmpl_arg: [*c]const u8 = @ptrCast(self.chat_template);
        const first_buffer: [*c]u8 = @ptrCast(formatted_prompt_buf.items.ptr);
        const first_buffer_len: c_int = @intCast(formatted_prompt_buf.items.len);
        var required_len = c.llama_chat_apply_template(tmpl_arg, messages.ptr, messages.len, true, first_buffer, first_buffer_len);

        if (required_len < 0) {
            print("Error: llama_chat_apply_template failed ({d})\n", .{required_len});
            return Error.ChatTemplateFailed;
        }

        const required_u32: u32 = @intCast(required_len);
        if (required_u32 > formatted_prompt_buf.items.len) {
            try formatted_prompt_buf.resize(allocator, @intCast(required_len));
            const resized_buffer: [*c]u8 = @ptrCast(formatted_prompt_buf.items.ptr);
            const resized_buffer_len: c_int = @intCast(formatted_prompt_buf.items.len);
            required_len = c.llama_chat_apply_template(tmpl_arg, messages.ptr, messages.len, true, resized_buffer, resized_buffer_len);
            if (required_len < 0) {
                print("Error: llama_chat_apply_template failed after resize ({d})\n", .{required_len});
                return Error.ChatTemplateFailed;
            }
        }

        const prompt_len: usize = @intCast(required_len);
        try formatted_prompt_buf.resize(allocator, prompt_len);
        return formatted_prompt_buf.toOwnedSlice(allocator);
    }

    pub const TokenBuffer = struct {
        buffer: []c.llama_token,
        count: usize,

        pub fn slice(self: *const TokenBuffer) []c.llama_token {
            return self.buffer[0..self.count];
        }

        pub fn len(self: *const TokenBuffer) usize {
            return self.count;
        }

        pub fn deinit(self: *TokenBuffer, allocator: Allocator) void {
            allocator.free(self.buffer);
        }
    };

    pub fn tokenize(self: *const Runtime, allocator: Allocator, prompt: []const u8) !TokenBuffer {
        const tokens = try allocator.alloc(c.llama_token, self.n_ctx);
        errdefer allocator.free(tokens);

        const add_bos = c.llama_vocab_get_add_bos(self.vocab);
        const prompt_ptr: [*c]const u8 = @ptrCast(prompt.ptr);
        const token_ptr: [*c]c.llama_token = @ptrCast(tokens.ptr);
        const prompt_len_c: c_int = @intCast(prompt.len);
        const token_capacity_c: c_int = @intCast(tokens.len);
        const n_tokens = c.llama_tokenize(self.vocab, prompt_ptr, prompt_len_c, token_ptr, token_capacity_c, add_bos, false);

        if (n_tokens < 0) {
            return Error.TokenizeFailed;
        }

        return TokenBuffer{ .buffer = tokens, .count = @intCast(n_tokens) };
    }

    pub fn decodePrompt(self: *Runtime, tokens: *const TokenBuffer) c_int {
        const slice = tokens.slice();
        const prompt_batch = c.llama_batch_get_one(@constCast(slice.ptr), @intCast(slice.len));
        return c.llama_decode(self.ctx, prompt_batch);
    }

    pub fn decodeToken(self: *Runtime, token: *c.llama_token) c_int {
        const token_batch = c.llama_batch_get_one(token, 1);
        return c.llama_decode(self.ctx, token_batch);
    }

    pub fn remainingTokens(self: *const Runtime, consumed: usize) usize {
        return if (self.n_ctx > consumed) self.n_ctx - consumed else 0;
    }

    pub fn isEndOfGeneration(self: *const Runtime, token: c.llama_token) bool {
        return c.llama_vocab_is_eog(self.vocab, token);
    }

    pub fn tokenToPiece(self: *Runtime, allocator: Allocator, token: c.llama_token) ![]u8 {
        var piece_len = c.llama_token_to_piece(self.vocab, token, null, 0, 0, false);
        var allow_special = false;

        if (piece_len == 0) {
            piece_len = c.llama_token_to_piece(self.vocab, token, null, 0, 0, true);
            allow_special = true;
        }

        if (piece_len < 0) {
            piece_len = -piece_len;
        }

        if (piece_len <= 0) {
            return Error.TokenPieceFailed;
        }

        const piece_buf = try allocator.alloc(u8, @intCast(piece_len));
        errdefer allocator.free(piece_buf);
        const piece_ptr: [*c]u8 = @ptrCast(piece_buf.ptr);
        const piece_capacity_c: c_int = @intCast(piece_buf.len);
        const actual_len = c.llama_token_to_piece(self.vocab, token, piece_ptr, piece_capacity_c, 0, allow_special);
        if (actual_len <= 0) {
            return Error.TokenPieceFailed;
        }

        return piece_buf[0..@intCast(actual_len)];
    }

    pub fn sampleToken(self: *Runtime, chain: *SamplerChain) c.llama_token {
        return c.llama_sampler_sample(chain.ptr, self.ctx, -1);
    }
};

pub const SamplerChain = struct {
    ptr: *c.llama_sampler,

    pub fn init() !SamplerChain {
        const chain_params = c.llama_sampler_chain_default_params();
        const chain_ptr = c.llama_sampler_chain_init(chain_params);
        if (chain_ptr == null) {
            return Error.SamplerChainInitFailed;
        }
        const sampler_chain = chain_ptr.?;

        const greedy_sampler_ptr = c.llama_sampler_init_greedy();
        if (greedy_sampler_ptr == null) {
            c.llama_sampler_free(sampler_chain);
            return Error.SamplerInitFailed;
        }

        c.llama_sampler_chain_add(sampler_chain, greedy_sampler_ptr.?);
        return SamplerChain{ .ptr = sampler_chain };
    }

    pub fn deinit(self: *SamplerChain) void {
        c.llama_sampler_free(self.ptr);
    }

    pub fn accept(self: *SamplerChain, token: c.llama_token) void {
        c.llama_sampler_accept(self.ptr, token);
    }
};
