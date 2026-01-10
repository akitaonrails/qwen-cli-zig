// llama.zig - Llama.cpp bindings and wrapper types
//
// This module provides Zig bindings for llama.cpp. Uses manual extern declarations
// instead of @cImport due to Zig C translation issues with llama.cpp headers.

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Opaque Types (pointers to C structs)
// ============================================================================

pub const Model = opaque {};
pub const Context = opaque {};
pub const Sampler = opaque {};
pub const Vocab = opaque {};
pub const Memory = opaque {};

// ============================================================================
// Basic Types
// ============================================================================

pub const Token = c_int;
pub const Pos = c_int;
pub const SeqId = c_int;

// ============================================================================
// Enums
// ============================================================================

pub const SplitMode = enum(c_int) {
    none = 0,
    layer = 1,
    row = 2,
};

pub const RopeScalingType = enum(c_int) {
    unspecified = -1,
    none = 0,
    linear = 1,
    yarn = 2,
    longrope = 3,
};

pub const PoolingType = enum(c_int) {
    unspecified = -1,
    none = 0,
    mean = 1,
    cls = 2,
    last = 3,
    rank = 4,
};

pub const AttentionType = enum(c_int) {
    unspecified = -1,
    causal = 0,
    non_causal = 1,
};

pub const FlashAttnType = enum(c_int) {
    auto = -1,
    disabled = 0,
    enabled = 1,
};

pub const LogLevel = enum(c_int) {
    @"error" = 2,
    warn = 3,
    info = 4,
    debug = 5,
};

// ============================================================================
// Structs
// ============================================================================

pub const Batch = extern struct {
    n_tokens: i32,
    token: ?*Token,
    embd: ?*f32,
    pos: ?*Pos,
    n_seq_id: ?*i32,
    seq_id: ?*?*SeqId,
    logits: ?*i8,
};

pub const ChatMessage = extern struct {
    role: [*c]const u8,
    content: [*c]const u8,
};

pub const SamplerChainParams = extern struct {
    no_perf: bool,
};

// Opaque placeholder types for complex fields
const GgmlBackendDev = opaque {};
const ModelTensorBuftOverride = opaque {};
const ModelKvOverride = opaque {};

const ProgressCallbackFn = *const fn (progress: f32, user_data: ?*anyopaque) callconv(.c) bool;
const ProgressCallback = ?ProgressCallbackFn;

pub const ModelParams = extern struct {
    devices: ?*GgmlBackendDev,
    tensor_buft_overrides: ?*const ModelTensorBuftOverride,
    n_gpu_layers: i32,
    split_mode: SplitMode,
    main_gpu: i32,
    tensor_split: ?*const f32,
    progress_callback: ProgressCallback,
    progress_callback_user_data: ?*anyopaque,
    kv_overrides: ?*const ModelKvOverride,
    vocab_only: bool,
    use_mmap: bool,
    use_mlock: bool,
    check_tensors: bool,
};

const EvalCallback = ?*const fn () callconv(.c) void;
const AbortCallback = ?*const fn (user_data: ?*anyopaque) callconv(.c) bool;

pub const ContextParams = extern struct {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    rope_scaling_type: RopeScalingType,
    pooling_type: PoolingType,
    attention_type: AttentionType,
    flash_attn_type: FlashAttnType,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: EvalCallback,
    cb_eval_user_data: ?*anyopaque,
    type_k: c_int,
    type_v: c_int,
    abort_callback: AbortCallback,
    abort_callback_data: ?*anyopaque,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: ?*anyopaque,
    n_samplers: usize,
};

// ============================================================================
// Extern Function Declarations
// ============================================================================

const LogCallbackFn = *const fn (level: LogLevel, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void;

extern "c" fn llama_backend_init() void;
extern "c" fn llama_backend_free() void;
extern "c" fn llama_log_set(log_callback: ?LogCallbackFn, user_data: ?*anyopaque) void;
extern "c" fn llama_model_default_params() ModelParams;
extern "c" fn llama_context_default_params() ContextParams;
extern "c" fn llama_sampler_chain_default_params() SamplerChainParams;
extern "c" fn llama_load_model_from_file(path_model: [*c]const u8, params: ModelParams) ?*Model;
extern "c" fn llama_free_model(model: ?*Model) void;
extern "c" fn llama_new_context_with_model(model: ?*Model, params: ContextParams) ?*Context;
extern "c" fn llama_free(ctx: ?*Context) void;
extern "c" fn llama_n_ctx(ctx: ?*Context) u32;
extern "c" fn llama_vocab_get_add_bos(vocab: ?*Vocab) bool;
extern "c" fn llama_tokenize(vocab: ?*Vocab, text: [*c]const u8, text_len: i32, tokens: [*c]Token, n_max_tokens: i32, add_special: bool, parse_special: bool) i32;
extern "c" fn llama_decode(ctx: ?*Context, batch: Batch) i32;
extern "c" fn llama_batch_get_one(tokens: ?*Token, n_tokens: i32) Batch;
extern "c" fn llama_sampler_chain_init(params: SamplerChainParams) ?*Sampler;
extern "c" fn llama_sampler_chain_add(chain: ?*Sampler, sampler: ?*Sampler) void;
extern "c" fn llama_sampler_init_temp(temp: f32) ?*Sampler;
extern "c" fn llama_sampler_init_greedy() ?*Sampler;
extern "c" fn llama_sampler_free(sampler: ?*Sampler) void;
extern "c" fn llama_sampler_sample(sampler: ?*Sampler, ctx: ?*Context, idx: i32) Token;
extern "c" fn llama_sampler_accept(sampler: ?*Sampler, token: Token) void;
extern "c" fn llama_vocab_is_eog(vocab: ?*Vocab, token: Token) bool;
extern "c" fn llama_token_to_piece(vocab: ?*Vocab, token: Token, buf: [*c]u8, length: i32, lstrip: i32, special: bool) i32;
extern "c" fn llama_get_model(ctx: ?*Context) ?*Model;
extern "c" fn llama_model_get_vocab(model: ?*Model) ?*Vocab;
extern "c" fn llama_get_memory(ctx: ?*Context) ?*Memory;
extern "c" fn llama_memory_clear(mem: ?*Memory, data: bool) void;

// ============================================================================
// High-Level Wrapper API
// ============================================================================

pub const Error = error{
    BackendInitFailed,
    ModelLoadFailed,
    ContextCreateFailed,
    VocabNotFound,
    SamplerInitFailed,
    TokenizeFailed,
    DecodeFailed,
    TokenConversionFailed,
};

/// Initialize the llama.cpp backend. Must be called before any other functions.
pub fn init() void {
    llama_backend_init();
}

/// Free the llama.cpp backend. Should be called at shutdown.
pub fn deinit() void {
    llama_backend_free();
}

/// Set a custom log callback for llama.cpp messages.
pub fn setLogCallback(callback: ?LogCallbackFn, user_data: ?*anyopaque) void {
    llama_log_set(callback, user_data);
}

/// Get default model parameters.
pub fn defaultModelParams() ModelParams {
    return llama_model_default_params();
}

/// Get default context parameters.
pub fn defaultContextParams() ContextParams {
    return llama_context_default_params();
}

/// Load a model from a GGUF file.
pub fn loadModel(path: [*c]const u8, params: ModelParams) Error!*Model {
    return llama_load_model_from_file(path, params) orelse error.ModelLoadFailed;
}

/// Free a loaded model.
pub fn freeModel(model: *Model) void {
    llama_free_model(model);
}

/// Create a context for inference.
pub fn createContext(model: *Model, params: ContextParams) Error!*Context {
    return llama_new_context_with_model(model, params) orelse error.ContextCreateFailed;
}

/// Free a context.
pub fn freeContext(ctx: *Context) void {
    llama_free(ctx);
}

/// Get the context size.
pub fn getContextSize(ctx: *Context) u32 {
    return llama_n_ctx(ctx);
}

/// Get the vocabulary from a model.
pub fn getVocab(model: *Model) Error!*Vocab {
    return llama_model_get_vocab(model) orelse error.VocabNotFound;
}

/// Check if BOS token should be added during tokenization.
pub fn shouldAddBos(vocab: *Vocab) bool {
    return llama_vocab_get_add_bos(vocab);
}

/// Tokenize text into tokens.
pub fn tokenize(vocab: *Vocab, text: []const u8, tokens: []Token, add_special: bool) Error!usize {
    const result = llama_tokenize(
        vocab,
        text.ptr,
        @intCast(text.len),
        tokens.ptr,
        @intCast(tokens.len),
        add_special,
        false,
    );
    if (result < 0) return error.TokenizeFailed;
    return @intCast(result);
}

/// Decode a batch of tokens.
pub fn decode(ctx: *Context, batch: Batch) Error!void {
    const result = llama_decode(ctx, batch);
    if (result != 0) return error.DecodeFailed;
}

/// Create a batch from a slice of tokens.
pub fn batchFromTokens(tokens: []Token) Batch {
    return llama_batch_get_one(@ptrCast(tokens.ptr), @intCast(tokens.len));
}

/// Clear the memory/KV cache.
pub fn clearKvCache(ctx: *Context) void {
    const mem = llama_get_memory(ctx);
    if (mem) |m| {
        llama_memory_clear(m, true);
    }
}

/// Check if a token is end-of-generation.
pub fn isEndOfGeneration(vocab: *Vocab, token: Token) bool {
    return llama_vocab_is_eog(vocab, token);
}

/// Convert a token to its string representation.
pub fn tokenToPiece(vocab: *Vocab, token: Token, buf: []u8) Error![]const u8 {
    const result = llama_token_to_piece(vocab, token, buf.ptr, @intCast(buf.len), 0, false);
    if (result < 0) return error.TokenConversionFailed;
    return buf[0..@intCast(result)];
}

// ============================================================================
// Sampler Wrapper
// ============================================================================

pub const SamplerChain = struct {
    ptr: *Sampler,

    pub fn init() Error!SamplerChain {
        const params = llama_sampler_chain_default_params();
        const ptr = llama_sampler_chain_init(params) orelse return error.SamplerInitFailed;
        return .{ .ptr = ptr };
    }

    pub fn deinit(self: SamplerChain) void {
        llama_sampler_free(self.ptr);
    }

    pub fn addGreedy(self: SamplerChain) Error!void {
        const sampler = llama_sampler_init_greedy() orelse return error.SamplerInitFailed;
        llama_sampler_chain_add(self.ptr, sampler);
    }

    pub fn addTemperature(self: SamplerChain, temp: f32) Error!void {
        const sampler = llama_sampler_init_temp(temp) orelse return error.SamplerInitFailed;
        llama_sampler_chain_add(self.ptr, sampler);
    }

    pub fn sample(self: SamplerChain, ctx: *Context) Token {
        return llama_sampler_sample(self.ptr, ctx, -1);
    }

    pub fn accept(self: SamplerChain, token: Token) void {
        llama_sampler_accept(self.ptr, token);
    }
};
