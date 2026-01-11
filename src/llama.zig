pub const llama_vocab = opaque {};
pub const llama_model = opaque {};
pub const llama_context = opaque {};
pub const llama_sampler = opaque {};

pub const llama_pos = i32;
pub const llama_token = i32;
pub const llama_seq_id = i32;

pub const llama_split_mode = enum(i32) {
    NONE = 0,
    LAYER = 1,
    ROW = 2,
};

pub const llama_rope_scaling_type = enum(i32) {
    UNSPECIFIED = -1,
    NONE = 0,
    LINEAR = 1,
    YARN = 2,
    LONGROPE = 3,
};

pub const llama_pooling_type = enum(i32) {
    UNSPECIFIED = -1,
    NONE = 0,
    MEAN = 1,
    CLS = 2,
    LAST = 3,
    RANK = 4,
};

pub const llama_attention_type = enum(i32) {
    UNSPECIFIED = -1,
    CAUSAL = 0,
    NON_CAUSAL = 1,
};

pub const llama_flash_attn_type = enum(i32) {
    AUTO = -1,
    DISABLED = 0,
    ENABLED = 1,
};

pub const ggml_log_level = enum(i32) {
    NONE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CONT = 5,
};

pub const ggml_log_callback = ?*const fn (level: ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void;

const ggml_tensor = opaque {};
pub const ggml_backend_sched_eval_callback = ?*const fn (t: ?*ggml_tensor, ask: bool, user_data: ?*anyopaque) callconv(.c) bool;
pub const ggml_abort_callback = ?*const fn (data: ?*anyopaque) callconv(.c) bool;

const ggml_backend_device = opaque {};
const ggml_backend_buffer_type = opaque {};

const ggml_backend_dev_t = *ggml_backend_device;
const ggml_backend_buffer_type_t = *ggml_backend_buffer_type;

const llama_model_tensor_buft_override = extern struct {
    pattern: [*c]const u8,
    buft: ggml_backend_buffer_type_t,
};

const llama_progress_callback_sig = fn (progress: f32, user_data: ?*anyopaque) callconv(.c) bool;
const llama_progress_callback = ?*const llama_progress_callback_sig;

const llama_model_kv_override = opaque {};

pub const llama_model_params = extern struct {
    devices: ?[*]ggml_backend_dev_t,
    tensor_buft_overrides: ?[*]const llama_model_tensor_buft_override,

    n_gpu_layers: i32,
    split_mode: llama_split_mode,

    main_gpu: i32,

    tensor_split: ?[*]const f32,

    progress_callback: llama_progress_callback,
    progress_callback_user_data: ?*anyopaque,

    kv_overrides: ?*const llama_model_kv_override,

    vocab_only: bool,
    use_mmap: bool,
    use_direct_io: bool,
    use_mlock: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
};

pub const llama_sampler_seq_config = extern struct {
    seq_id: llama_seq_id,
    sampler: ?*llama_sampler,
};

pub const llama_context_params = extern struct {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,

    rope_scaling_type: llama_rope_scaling_type,
    pooling_type: llama_pooling_type,
    attention_type: llama_attention_type,
    flash_attn_type: llama_flash_attn_type,

    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,

    cb_eval: ggml_backend_sched_eval_callback,
    cb_eval_user_data: ?*anyopaque,

    type_k: i32,
    type_v: i32,

    abort_callback: ggml_abort_callback,
    abort_callback_data: ?*anyopaque,

    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,

    samplers: ?*llama_sampler_seq_config,
    n_samplers: usize,
};

pub const llama_batch = extern struct {
    n_tokens: i32,

    token: ?*llama_token,
    embd: ?*f32,
    pos: ?*llama_pos,
    n_seq_id: ?*i32,
    seq_id: ?*?*llama_seq_id,
    logits: ?*i8,
};

pub const llama_chat_message = extern struct {
    role: [*c]const u8,
    content: [*c]const u8,
};

pub const llama_sampler_chain_params = extern struct {
    no_perf: bool,
};

pub extern "c" fn llama_backend_init() void;
pub extern "c" fn llama_backend_free() void;

pub extern "c" fn llama_log_set(log_callback: ggml_log_callback, user_data: ?*anyopaque) void;

pub extern "c" fn llama_model_default_params() llama_model_params;
pub extern "c" fn llama_context_default_params() llama_context_params;
pub extern "c" fn llama_sampler_chain_default_params() llama_sampler_chain_params;

pub extern "c" fn llama_model_load_from_file(path_model: [*c]const u8, params: llama_model_params) ?*llama_model;
pub extern "c" fn llama_model_free(model: ?*llama_model) void;

pub extern "c" fn llama_init_from_model(model: ?*llama_model, params: llama_context_params) ?*llama_context;
pub extern "c" fn llama_free(ctx: ?*llama_context) void;

pub extern "c" fn llama_n_ctx(ctx: ?*const llama_context) u32;

pub extern "c" fn llama_model_get_vocab(model: ?*const llama_model) ?*const llama_vocab;

pub extern "c" fn llama_vocab_get_add_bos(vocab: ?*const llama_vocab) bool;
pub extern "c" fn llama_vocab_is_eog(vocab: ?*const llama_vocab, token: llama_token) bool;

pub extern "c" fn llama_tokenize(
    vocab: ?*const llama_vocab,
    text: [*c]const u8,
    text_len: i32,
    tokens: [*c]llama_token,
    n_tokens_max: i32,
    add_special: bool,
    parse_special: bool,
) i32;

pub extern "c" fn llama_token_to_piece(
    vocab: ?*const llama_vocab,
    token: llama_token,
    buf: [*c]u8,
    length: i32,
    lstrip: i32,
    special: bool,
) i32;

pub extern "c" fn llama_decode(ctx: ?*llama_context, batch: llama_batch) i32;
pub extern "c" fn llama_batch_get_one(tokens: ?*llama_token, n_tokens: i32) llama_batch;

pub extern "c" fn llama_sampler_chain_init(params: llama_sampler_chain_params) ?*llama_sampler;
pub extern "c" fn llama_sampler_chain_add(chain: ?*llama_sampler, sampler: ?*llama_sampler) void;

pub extern "c" fn llama_sampler_init_temp(temp: f32) ?*llama_sampler;
pub extern "c" fn llama_sampler_init_greedy() ?*llama_sampler;

pub extern "c" fn llama_sampler_free(sampler: ?*llama_sampler) void;

pub extern "c" fn llama_sampler_sample(sampler: ?*llama_sampler, ctx: ?*llama_context, idx: i32) llama_token;
pub extern "c" fn llama_sampler_accept(sampler: ?*llama_sampler, token: llama_token) void;

pub extern "c" fn llama_get_model(ctx: ?*const llama_context) ?*const llama_model;

pub extern "c" fn llama_model_chat_template(model: ?*const llama_model, name: [*c]const u8) [*c]const u8;

pub extern "c" fn llama_chat_apply_template(
    tmpl: [*c]const u8,
    chat: [*c]const llama_chat_message,
    n_msg: usize,
    add_ass: bool,
    buf: [*c]u8,
    length: i32,
) i32;
