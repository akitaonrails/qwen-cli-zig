const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const io = std.io;
const fs = std.fs; // Need fs for path.basename and selfExePathAlloc
const process = std.process;
const fmt = std.fmt;
const ArrayList = std.ArrayList; // Explicit import for clarity
const math = std.math; // For maxInt

// --- Error Types ---
const AppError = @import("std").os.WriteError || // Errors from stdout.print
    error{ // Custom errors
    MissingModelPath,
    MissingSystemPrompt,
    MissingTemperature,
    ModelLoadFailed,
    ContextCreateFailed,
    TokenizeFailed,
    DecodeFailed,
    OutOfMemory, // From allocator
};

// --- Llama.cpp Extern Declarations ---
// Manually declare types and functions needed from llama.h
// This avoids @cImport issues but requires maintenance if llama.h changes.

// Opaque types (pointers)
const llama_model = *opaque{};
const llama_context = *opaque{};
const llama_sampler = *opaque{}; // Renamed from llama_sampling_context
const llama_vocab = *opaque{}; // Added for vocab functions

// Basic types
const llama_token = c_int;
const llama_pos = c_int;
const llama_seq_id = c_int;

// Enums needed for structs
const llama_split_mode = enum(c_int) { // Removed 'extern'
    NONE = 0,
    LAYER = 1,
    ROW = 2,
};

const llama_rope_scaling_type = enum(c_int) {
    UNSPECIFIED = -1,
    NONE = 0,
    LINEAR = 1,
    YARN = 2,
    LONGROPE = 3,
    // MAX_VALUE = 3, // Removed - Redundant alias for LONGROPE, causes Zig error
};

const llama_pooling_type = enum(c_int) {
    UNSPECIFIED = -1,
    NONE = 0,
    MEAN = 1,
    CLS = 2,
    LAST = 3,
    RANK = 4,
};

const llama_attention_type = enum(c_int) {
    UNSPECIFIED = -1,
    CAUSAL = 0,
    NON_CAUSAL = 1,
};

// Placeholder types for callbacks
const ggml_backend_sched_eval_callback = ?*const fn() callconv(.c) void; // Placeholder signature
const ggml_abort_callback = ?*const fn(user_data: ?*anyopaque) callconv(.c) bool; // Placeholder signature

// Structs (simplified - add fields as needed)
const llama_batch = extern struct {
    n_tokens: i32, // Use i32 for clarity, matches c_int on most platforms
    token: ?*llama_token, // Use optional single-item pointers
    embd: ?*f32,
    pos: ?*llama_pos,
    n_seq_id: ?*i32,
    seq_id: ?*?*llama_seq_id, // Pointer to pointer
    logits: ?*i8, // Use i8 for int8_t
    // Removed fields not present in llama.h definition:
    // all_pos_0: llama_pos,
    // all_pos_1: llama_pos,
    // all_seq_id: llama_seq_id,
};

const llama_model_params = extern struct {
    n_gpu_layers: c_int,
    main_gpu: c_int,
    tensor_split: ?[*]const f32,
    // progress_callback: ?fn (?*anyopaque, f32) callconv(.C) void, // Optional function pointers not directly compatible in extern struct
    // progress_callback_user_data: ?*anyopaque,
    // kv_overrides: ?[*]const llama_model_kv_override, // Need nested struct if used - Commented out for now
    vocab_only: bool,
    use_mmap: bool,
    use_mlock: bool,
    check_tensors: bool,
};

// Placeholder/Opaque types for complex fields in llama_model_params
const ggml_backend_dev_t = opaque {}; // Opaque struct
const llama_model_tensor_buft_override = opaque {}; // Opaque struct
// Define the function signature type first
const llama_progress_callback_sig = fn (progress: f32, user_data: ?*anyopaque) callconv(.c) bool;
// Now define the callback type as an optional pointer to that signature for extern struct compatibility
const llama_progress_callback = ?*const llama_progress_callback_sig;
const llama_model_kv_override = opaque {}; // Opaque struct

// Update llama_model_params to match llama.h
const llama_model_params_updated = extern struct {
    devices: ?*ggml_backend_dev_t, // Changed [*] to *
    tensor_buft_overrides: ?*const llama_model_tensor_buft_override, // Changed [*] to *
    n_gpu_layers: c_int,
    split_mode: llama_split_mode, // Added
    main_gpu: c_int,
    tensor_split: ?*const f32, // Changed [*] to *
    progress_callback: llama_progress_callback, // Added
    progress_callback_user_data: ?*anyopaque, // Added
    kv_overrides: ?*const llama_model_kv_override, // Changed [*] to *
    vocab_only: bool,
    use_mmap: bool,
    use_mlock: bool,
    check_tensors: bool,
};


const llama_context_params = extern struct {
    // NOTE: Fields ordered according to llama.h v1.0
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32, // Added
    n_seq_max: u32, // Added
    n_threads: i32, // Changed to i32
    n_threads_batch: i32, // Changed to i32

    rope_scaling_type: llama_rope_scaling_type, // Use enum
    pooling_type: llama_pooling_type, // Added
    attention_type: llama_attention_type, // Added

    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,

    cb_eval: ggml_backend_sched_eval_callback, // Use placeholder type
    cb_eval_user_data: ?*anyopaque,

    type_k: c_int, // enum ggml_type - Assuming c_int is compatible for now
    type_v: c_int, // enum ggml_type - Assuming c_int is compatible for now

    // Booleans grouped at the end in llama.h v1.0
    logits_all: bool,
    embeddings: bool,
    offload_kqv: bool,
    flash_attn: bool,
    no_perf: bool, // Added

    // Abort callback added at the end
    abort_callback: ggml_abort_callback, // Use placeholder type
    abort_callback_data: ?*anyopaque, // Added
};

// Added for chat template function
const llama_chat_message = extern struct {
    role: [*c]const u8,
    content: [*c]const u8,
};

// Replaced llama_sampling_params with llama_sampler_chain_params
const llama_sampler_chain_params = extern struct {
    no_perf: bool, // whether to measure performance timings
};

// Original llama_sampling_params fields are now mostly handled by individual samplers
// We still need the definition for the default function return type temporarily
const llama_sampling_params = extern struct {
    n_prev: c_int,
    n_probs: c_int,
    top_k: c_int,
    top_p: f32,
    min_p: f32,
    tfs_z: f32,
    typical_p: f32,
    temp: f32,
    dynatemp_range: f32,
    dynatemp_exponent: f32,
    penalty_last_n: c_int,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
    mirostat: c_int,
    mirostat_tau: f32,
    mirostat_eta: f32,
    penalize_nl: bool,
    samplers_sequence: [*c]const u8,
    grammar: [*c]const u8,
    cfg_scale: f32,
    cfg_smooth_factor: f32,
    cfg_negative_prompt: [*c]const u8,
    cfg_negative_prompt_tokens: [*c]llama_token,
    n_cfg_negative_prompt_tokens: c_int,
    penalty_prompt_tokens: [*c]llama_token,
    use_penalty_prompt_tokens: bool,
    n_penalty_prompt_tokens: c_int,
};

// Define the C function signature type
const ggml_log_callback_sig = fn (level: ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void;
// Define the type alias for the extern function parameter (optional pointer to the signature)
const ggml_log_callback_extern_t = ?*const ggml_log_callback_sig;

// Functions (Only those currently used)
extern "c" fn llama_backend_init() void;
extern "c" fn llama_backend_free() void;
extern "c" fn llama_log_set(log_callback: ggml_log_callback_extern_t, user_data: ?*anyopaque) void; // Use the optional pointer type alias
extern "c" fn llama_model_default_params() llama_model_params_updated; // Use updated struct
extern "c" fn llama_context_default_params() llama_context_params;
// extern "c" fn llama_sampling_default_params() llama_sampling_params; // Removed
extern "c" fn llama_sampler_chain_default_params() llama_sampler_chain_params; // Added
extern "c" fn llama_load_model_from_file(path_model: [*c]const u8, params: llama_model_params_updated) ?*llama_model; // Use updated struct
extern "c" fn llama_free_model(model: ?*llama_model) void;
extern "c" fn llama_new_context_with_model(model: ?*llama_model, params: llama_context_params) ?*llama_context;
extern "c" fn llama_free(ctx: ?*llama_context) void;
extern "c" fn llama_n_ctx(ctx: ?*llama_context) u32;
// extern "c" fn llama_should_add_bos_token(model: ?*llama_model) bool; // Removed - Symbol not found in library
extern "c" fn llama_vocab_get_add_bos(vocab: ?*llama_vocab) bool; // Added
extern "c" fn llama_tokenize(vocab: ?*llama_vocab, text: [*c]const u8, text_len: i32, tokens: [*c]llama_token, n_max_tokens: i32, add_special: bool, parse_special: bool) i32; // Use i32
extern "c" fn llama_decode(ctx: ?*llama_context, batch: llama_batch) i32; // Use i32
extern "c" fn llama_batch_get_one(tokens: ?*llama_token, n_tokens: i32) llama_batch; // Use ?*llama_token and i32
// extern "c" fn llama_sampling_init(params: llama_sampling_params) ?*llama_sampling_context; // Removed
extern "c" fn llama_sampler_chain_init(params: llama_sampler_chain_params) ?*llama_sampler; // Added
extern "c" fn llama_sampler_chain_add(chain: ?*llama_sampler, sampler: ?*llama_sampler) void; // Added
extern "c" fn llama_sampler_init_temp(temp: f32) ?*llama_sampler; // Added
extern "c" fn llama_sampler_init_greedy() ?*llama_sampler; // Added
// extern "c" fn llama_sampling_free(ctx: ?*llama_sampling_context) void; // Removed
extern "c" fn llama_sampler_free(sampler: ?*llama_sampler) void; // Added
// extern "c" fn llama_sampling_sample(ctx_sampling: ?*llama_sampling_context, ctx_main: ?*llama_context, ctx_cfg: ?*llama_context, idx: c_int) llama_token; // Removed
extern "c" fn llama_sampler_sample(sampler: ?*llama_sampler, ctx: ?*llama_context, idx: i32) llama_token; // Use i32
// extern "c" fn llama_sampling_accept(ctx_sampling: ?*llama_sampling_context, ctx_main: ?*llama_context, id: llama_token, apply_grammar: bool) void; // Removed
extern "c" fn llama_sampler_accept(sampler: ?*llama_sampler, token: llama_token) void; // Added
// extern "c" fn llama_token_is_eos(model: ?*llama_model, token: llama_token) bool; // Removed
extern "c" fn llama_vocab_is_eog(vocab: ?*llama_vocab, token: llama_token) bool; // Added
extern "c" fn llama_token_to_piece(vocab: ?*llama_vocab, token: llama_token, buf: [*c]u8, length: i32, lstrip: i32, special: bool) i32; // Use i32
extern "c" fn llama_get_model(ctx: ?*llama_context) ?*llama_model; // Added
extern "c" fn llama_model_get_vocab(model: ?*llama_model) ?*llama_vocab; // Added
extern "c" fn llama_chat_apply_template(
    model: ?*llama_model, // Model pointer (optional if tmpl provided)
    tmpl: [*c]const u8, // Custom template string (optional if model provided)
    chat: [*c]const llama_chat_message, // Array of messages
    n_msg: usize, // Number of messages
    add_ass: bool, // Add assistant start token
    buf: [*c]u8, // Output buffer
    length: i32, // Buffer length
) i32; // Returns required length or negative on error

// Define nested struct at top level if needed later
// const llama_model_kv_override = extern struct { ... }; // Now defined above as opaque


// --- Default Configuration ---
const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in software development tasks.";
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_MODEL_PATH = ""; // IMPORTANT: Provide path via --model argument

// --- Application State ---
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// --- Llama.cpp Logging Callback ---
// Define the log level enum based on ggml.h (adjust if different in your version)
const ggml_log_level = enum(c_int) {
    ERROR = 2,
    WARN = 3,
    INFO = 4,
    DEBUG = 5,
};

// Define the callback function
fn logCallback(level: ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data; // Unused for this simple callback
    // Switch on the underlying integer value to handle potential undefined values from C
    const level_int = @intFromEnum(level);
    switch (level_int) {
        @intFromEnum(ggml_log_level.ERROR) => std.debug.print("LLAMA: [ERROR] {s}", .{text}),
        @intFromEnum(ggml_log_level.WARN) => std.debug.print("LLAMA: [WARN] {s}", .{text}),
        @intFromEnum(ggml_log_level.INFO) => std.debug.print("LLAMA: [INFO] {s}", .{text}),
        // Don't print DEBUG by default to reduce noise, unless needed
        // @intFromEnum(ggml_log_level.DEBUG) => std.debug.print("LLAMA: [DEBUG] {s}", .{text}),
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

    var args = CliArgs{}; // We modify fields of args, so keep var here for now.

    // Use the standard next() method. The loop variable 'arg' is the unwrapped value.
    while (args_iter.next()) |arg| {
        // 'arg' is already []const u8 here, no need for orelse break

        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            // Get the next argument, which is the value. It's optional.
            // value_arg is already unwrapped []const u8 due to orelse
            const value_arg = args_iter.next() orelse return error.MissingModelPath;
            // Duplicate the string since the slice from next() is temporary
            args.model_path = try allocator.dupe(u8, value_arg);
            // Note: Potential leak of default model_path

        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
            // value_arg is already unwrapped []const u8 due to orelse
            const value_arg = args_iter.next() orelse return error.MissingSystemPrompt;
            // Duplicate the string
            args.system_prompt = try allocator.dupe(u8, value_arg);
            // Note: Potential leak of default system_prompt

        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temp")) {
            // temp_str is already unwrapped []const u8 due to orelse
            const temp_str = args_iter.next() orelse return error.MissingTemperature;
            // parseFloat works directly on the temporary slice
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

// --- Chat History and Prompt Formatting ---
// Qwen3 format (simplified):
// <|im_start|>system
// {system_prompt}<|im_end|>
// <|im_start|>user
// {user_prompt_1}<|im_end|>
// <|im_start|>assistant
// {assistant_response_1}<|im_end|>
// <|im_start|>user
// {user_prompt_2}<|im_end|>
// <|im_start|>assistant
//
// Note: Qwen3 also has a <think>...</think> mechanism, which might require
// more complex handling depending on how llama.cpp exposes it.
// This basic implementation doesn't handle the thinking part explicitly.

const Message = struct {
    role: []const u8,
    content: []const u8,
};

// Removed custom formatPrompt function - will use llama_chat_apply_template instead

// Helper to convert Zig slice to null-terminated C string (allocates)
fn toCString(slice: []const u8) ![:0]const u8 {
    const c_str = try allocator.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}

// --- Main Application Logic ---
pub fn main() !void {
    // No longer need '_ = llama;' as we removed the @cImport

    // Use const as args itself is not reassigned, only its fields might be updated during parsing
    const args = try parseArgs();

    if (args.show_help) {
        // Get the absolute path to the executable, allocating memory for it.
        const exe_path = try std.fs.selfExePathAlloc(allocator);
        // Free the allocated path when done
        defer allocator.free(exe_path);
        const exe_name = std.fs.path.basename(exe_path);
        printHelp(exe_name);
        return;
    }

    // --- Llama.cpp Initialization ---
    print("Initializing llama.cpp backend...\n", .{});
    // llama_backend_init(false); // Use false to disable NUMA if not needed or causing issues
    llama_backend_init();
    // Set the log handler to our Zig function (pass its address)
    llama_log_set(&logCallback, null);
    print("llama.cpp log handler set.\n", .{});

    // --- Load Model ---
    var mparams = llama_model_default_params(); // Changed to var to modify use_mmap
    mparams.use_mmap = false; // Disable memory mapping
    // TODO: Expose more model params as CLI args (e.g., n_gpu_layers, vocab_only)
    // mparams.n_gpu_layers = 99; // Example: Offload all possible layers to GPU

    print("Loading model: {s}...\n", .{args.model_path});
    const c_model_path = try toCString(args.model_path);
    defer allocator.free(c_model_path);

    const model = llama_load_model_from_file(c_model_path, mparams);
    if (model == null) {
        print("Error: Failed to load model from '{s}'\n", .{args.model_path});
        llama_backend_free();
        return error.ModelLoadFailed;
    }
    defer llama_free_model(model);

    // --- Create Context ---
    var cparams = llama_context_default_params();
    // TODO: Expose more context params (n_batch, seed, etc.)
    cparams.n_ctx = 4096; // Explicitly set a smaller context size for testing
    cparams.n_batch = 512; // REVERTED to default batch size
    // cparams.seed = 1234; // Removed - seed is not in the struct definition matching llama.h

    print("Creating context (n_ctx = {d}, n_batch = {d})...\n", .{cparams.n_ctx, cparams.n_batch}); // Log n_ctx and n_batch
    const ctx = llama_new_context_with_model(model, cparams);
    if (ctx == null) {
        print("Error: Failed to create context\n", .{});
        // Model is freed by defer, just free backend
        llama_backend_free();
        return error.ContextCreateFailed;
    }
    defer llama_free(ctx);

    const n_ctx = llama_n_ctx(ctx); // Get the actual context size
    print("Context created (n_ctx = {d})\n", .{n_ctx}); // Log the context size
    print("Model loaded successfully.\n", .{});
    print("System Prompt: {s}\n", .{args.system_prompt});
    print("Temperature: {d:.2}\n", .{args.temperature});
    print("Enter your message (Ctrl+D or /quit to exit):\n", .{});

    var history = try std.ArrayList(Message).initCapacity(allocator, 0);
    defer {
        for (history.items) |msg| {
            allocator.free(msg.content);
        }
        history.deinit(allocator);
    }

    var stdin_file = std.fs.File.stdin();
    defer stdin_file.close();

    var input_buffer = try std.ArrayList(u8).initCapacity(allocator, 128);
    defer input_buffer.deinit(allocator);

    // --- Chat Loop ---
    while (true) {
        print("> ", .{});

        const nread = try stdin_file.readAll(input_buffer.items[0..128]);
        if (nread == 0) {
            print("\nExiting...\n", .{});
            break;
        }

        const user_input = std.mem.trim(u8, input_buffer.items, " \t\r\n");
        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "/quit")) {
            print("Exiting...\n", .{});
            break;
        }

        // --- Prepare Prompt using llama_chat_apply_template ---
        var current_chat = try std.ArrayList(llama_chat_message).initCapacity(allocator, 0);
        defer current_chat.deinit(allocator);

        // Add system prompt
        const c_system_role = try toCString("system");
        defer allocator.free(c_system_role);
        const c_system_prompt = try toCString(args.system_prompt);
        defer allocator.free(c_system_prompt);
        try current_chat.append(allocator, .{ .role = c_system_role, .content = c_system_prompt });

        // Add history messages (need to convert Zig strings to C strings temporarily)
        var temp_slices = try std.ArrayList([:0]const u8).initCapacity(allocator, 0);
        defer {
            for (temp_slices.items) |slice_to_free| {
                allocator.free(slice_to_free);
            }
            temp_slices.deinit(allocator);
        }
        for (history.items) |msg| {
            const c_role = try toCString(msg.role);
            try temp_slices.append(allocator, c_role); // Track slice for freeing
            const c_content = try toCString(msg.content);
            try temp_slices.append(allocator, c_content); // Track slice for freeing
            try current_chat.append(allocator, .{ .role = c_role, .content = c_content });
        }

        // Add current user input
        const c_user_role = try toCString("user");
        defer allocator.free(c_user_role);
        const c_user_input = try toCString(user_input);
        defer allocator.free(c_user_input);
        try current_chat.append(allocator, .{ .role = c_user_role, .content = c_user_input });

        // Apply the template
        var formatted_prompt_buf = try std.ArrayList(u8).initCapacity(allocator, 2048);
        defer formatted_prompt_buf.deinit(allocator);
        const initial_buf_size: i32 = 2048; // Start with a reasonable buffer size
        try formatted_prompt_buf.resize(allocator, @intCast(initial_buf_size));

        var required_len = llama_chat_apply_template(
            model, // Use model's default template
            null, // No custom template
            current_chat.items.ptr,
            current_chat.items.len,
            true, // Add assistant start token
            formatted_prompt_buf.items.ptr,
            @intCast(formatted_prompt_buf.items.len),
        );

        if (required_len < 0) {
            print("Error: llama_chat_apply_template failed ({d})\n", .{required_len});
            continue;
        }

        if (@as(u32, @intCast(required_len)) > initial_buf_size) {
            // Buffer was too small, resize and try again
            print("Resizing chat template buffer to {d} bytes\n", .{required_len});
            try formatted_prompt_buf.resize(allocator, @intCast(required_len));
            required_len = llama_chat_apply_template(
                model, null, current_chat.items.ptr, current_chat.items.len, true,
                formatted_prompt_buf.items.ptr, required_len
            );
            if (required_len < 0) {
                 print("Error: llama_chat_apply_template failed after resize ({d})\n", .{required_len});
                 continue;
            }
        }
        // Now formatted_prompt_buf.items[0..@intCast(required_len)] contains the correct prompt
        const full_prompt_bytes = formatted_prompt_buf.items[0..@intCast(required_len)]; // Target usize inferred


        // --- Llama.cpp Inference ---
        print("Assistant: ", .{});
        // Removed: try stdout_file.sync(); // This caused crashes on stdout

        // --- Tokenize Prompt ---
        // Add 1 for BOS, 1 for potential extra token during processing
        // Use n_ctx declared earlier after context creation
        const tokens = try allocator.alloc(llama_token, n_ctx);
        defer allocator.free(tokens);

        // Get vocab for tokenization and add_bos check
        const vocab = llama_model_get_vocab(model); // Use this vocab variable consistently
        const add_bos = llama_vocab_get_add_bos(vocab);
        // Pass vocab, not model, to llama_tokenize
        // Target type i32 is inferred from llama_tokenize signature
        const n_tokens = llama_tokenize(vocab, full_prompt_bytes.ptr, @intCast(full_prompt_bytes.len), tokens.ptr, @intCast(n_ctx), add_bos, false);

        if (n_tokens < 0) {
            print("Error: Failed to tokenize prompt (token buffer too small?)\n", .{});
            continue; // Or handle error more gracefully
        }

        // --- Evaluate Prompt ---
        print("Decoding prompt ({d} tokens)...\n", .{n_tokens});
        // Add print right before the call
        print("Calling llama_decode for prompt...\n", .{});
        const prompt_batch = llama_batch_get_one(&tokens[0], @intCast(n_tokens));
        const decode_prompt_ret = llama_decode(ctx, prompt_batch);
        // Add print right after the call, including return value
        print("llama_decode for prompt returned: {d}\n", .{decode_prompt_ret});
        if (decode_prompt_ret != 0) {
             print("Error: llama_decode failed during prompt processing (ret={d})\n", .{decode_prompt_ret});
             continue;
        }
        print("Prompt decoded successfully.\n", .{});

        // --- Generation Loop ---
        var generated_response = try std.ArrayList(u8).initCapacity(allocator, 0);
        defer generated_response.deinit(allocator);

        const cur_pos: llama_pos = @intCast(n_tokens); // Changed var to const
        const max_new_tokens = n_ctx - @as(u32, @intCast(n_tokens)); // Cast n_tokens to u32 for subtraction
        var generated_token_count: u32 = 0;

        // --- Init Sampler Chain ---
        const chain_params = llama_sampler_chain_default_params();
        const sampler_chain = llama_sampler_chain_init(chain_params);
        if (sampler_chain == null) {
            print("Error: Failed to init sampler chain\n", .{});
            continue;
        }
        defer llama_sampler_free(sampler_chain);

        // Add samplers to the chain (order matters)
        // TODO: Add other samplers like top_k, top_p, penalties etc. based on CLI args
        // const temp_sampler = llama_sampler_init_temp(args.temperature);
        // llama_sampler_chain_add(sampler_chain, temp_sampler); // Temporarily disable temp sampler

        const greedy_sampler = llama_sampler_init_greedy(); // Use greedy as the final selector
        llama_sampler_chain_add(sampler_chain, greedy_sampler);
        // Note: The chain takes ownership, so we don't free temp_sampler or greedy_sampler directly

        // vocab variable already obtained before tokenize call, use it here too
        // const vocab = llama_model_get_vocab(model); // Already have 'vocab'

        while (cur_pos < n_ctx and generated_token_count < max_new_tokens) {
            // Sample the next token using the chain
            var id = llama_sampler_sample(sampler_chain, ctx, -1); // Use -1 for the last logit

            // Accept the token using the chain
            llama_sampler_accept(sampler_chain, id);

            // Check for End-Of-Sequence using vocab
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }

            // Convert token to piece and print/store
            // DEBUG: Print the sampled token ID
            // print(" [Debug] Sampled token ID: {d} ", .{id});

            // Get length first (pass 0 for lstrip, false for special)
            const piece_len = llama_token_to_piece(vocab, id, null, 0, 0, false);
            if (piece_len < 0) {
                 print("Error: llama_token_to_piece failed to get length (token_id={d})\n", .{id}); // Add token ID to error
                 break;
            }
            // Need to handle potential negative length if buffer is too small, though unlikely here
            const piece_buf = try allocator.alloc(u8, @intCast(piece_len));
            defer allocator.free(piece_buf);
            // Get actual piece (pass 0 for lstrip, false for special)
            // Target type i32 is inferred from llama_token_to_piece signature
            const actual_len = llama_token_to_piece(vocab, id, piece_buf.ptr, @intCast(piece_buf.len), 0, false);
             if (actual_len != piece_len) {
                 // This could happen if the buffer wasn't large enough, though alloc should prevent this.
                 print("Error: llama_token_to_piece length mismatch or error ({d} vs {d})\n", .{actual_len, piece_len});
                 break;
             }

            print("{s}", .{piece_buf});
            // No need to sync stdout after every piece
            try generated_response.appendSlice(allocator, piece_buf);

            // Prepare for next token - Manual batch construction (Attempt 2)
            // Ensure all fields match llama.h and null indicates default behavior
            const token_id_ptr: ?*llama_token = &id; // Changed var to const
            const batch = llama_batch{
                .n_tokens = 1,
                .token = token_id_ptr, // Point to the sampled token ID
                .embd = null,          // Not providing embeddings
                .pos = null,           // Explicitly null for automatic position tracking by llama_decode
                .n_seq_id = null,      // Use default sequence ID (0)
                .seq_id = null,        // Use default sequence ID (0)
                .logits = null,        // Request default logits (usually the last token)
            };

            // Decode the single token batch
            // print("Decoding token {d}...\n", .{id}); // Optional: Very verbose
            const decode_gen_ret = llama_decode(ctx, batch);
            if (decode_gen_ret != 0) {
                 print("Error: llama_decode failed during generation (ret={d})\n", .{decode_gen_ret});
                 break;
            }
            // cur_pos += 1; // Let llama_decode manage position tracking for single tokens when pos=null
            generated_token_count += 1;
        }
        try generated_response.appendSlice(allocator, "\n");

        // --- Update History ---
        const user_input_copy = try allocator.dupe(u8, user_input);
        errdefer allocator.free(user_input_copy);
        // Use the actual generated response
        const assistant_response_copy = try generated_response.toOwnedSlice(allocator);
        errdefer allocator.free(assistant_response_copy);

        try history.append(allocator, .{ .role = "user", .content = user_input_copy });
        try history.append(allocator, .{ .role = "assistant", .content = assistant_response_copy });

        // Sampling context is freed by defer

        // --- Llama.cpp Cleanup ---
        // Note: Context and Model are freed by their defers earlier in main()
        // No specific cleanup needed within the loop itself for this basic implementation
    }

    // --- Llama.cpp Cleanup ---
    // Context and Model are freed by their defers earlier in main()
    print("Cleaning up llama.cpp backend...\n", .{});
    llama_backend_free();

    // Final allocator check
    if (gpa.deinit() == .leak) {
        print("Warning: Memory leak detected!\n", .{});
    }
}
