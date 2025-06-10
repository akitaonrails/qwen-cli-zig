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
const ggml_backend_sched_eval_callback = ?*const fn() callconv(.C) void; // Placeholder signature
const ggml_abort_callback = ?*const fn(user_data: ?*anyopaque) callconv(.C) bool; // Placeholder signature

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
const llama_progress_callback_sig = fn (progress: f32, user_data: ?*anyopaque) callconv(.C) bool;
// Now define the callback type as an optional pointer to that signature for extern struct compatibility
const llama_progress_callback = ?*const llama_progress_callback_sig;
const llama_model_kv_override = opaque {}; // Opaque struct

// Update llama_model_params to match llama.h
const llama_model_params_updated = extern struct {
    devices: ?*ggml_backend_dev_t, // Changed [*] to *
    tensor_buft_overrides: ?*const llama_model_tensor_buft_override, // Changed [*] to *
    n_gpu_layers: i32, // Ensure this field exists and use i32
    split_mode: llama_split_mode, // Added
    main_gpu: i32, // Use i32
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
const ggml_log_callback_sig = fn (level: ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.C) void;
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
extern "c" fn llama_kv_cache_clear(ctx: ?*llama_context) void; // Added for clearing KV cache
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
const DEFAULT_MODEL_PATH = "models/qwen-14b.Q5_K_M.gguf"; // Default to the actual model file

// --- Application State ---
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// --- Memory Management Helpers ---
const StringPool = struct {
    strings: std.ArrayList([]u8),
    allocator: Allocator,

    fn init(alloc: Allocator) StringPool {
        return StringPool{
            .strings = std.ArrayList([]u8).init(alloc),
            .allocator = alloc,
        };
    }

    fn deinit(self: *StringPool) void {
        for (self.strings.items) |str| {
            self.allocator.free(str);
        }
        self.strings.deinit();
    }

    fn dupe(self: *StringPool, str: []const u8) ![]const u8 {
        const duped = try self.allocator.dupe(u8, str);
        try self.strings.append(duped);
        return duped;
    }
};

// --- Llama.cpp Logging Callback ---
// Define the log level enum based on ggml.h (adjust if different in your version)
const ggml_log_level = enum(c_int) {
    ERROR = 2,
    WARN = 3,
    INFO = 4,
    DEBUG = 5,
};

// Define the callback function
fn logCallback(level: ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.C) void { // Added callconv(.C)
    _ = user_data; // Unused for this simple callback
    // Switch on the underlying integer value to handle potential undefined values from C
    const level_int = @intFromEnum(level);
    switch (level_int) {
        @intFromEnum(ggml_log_level.ERROR) => std.debug.print("LLAMA: [ERROR] {s}", .{text}),
        @intFromEnum(ggml_log_level.WARN)  => std.debug.print("LLAMA: [WARN] {s}", .{text}),
        @intFromEnum(ggml_log_level.INFO)  => std.debug.print("LLAMA: [INFO] {s}", .{text}),
        // @intFromEnum(ggml_log_level.DEBUG) => std.debug.print("LLAMA: [DEBUG] {s}", .{text}), // Keep commented out for less noise
        else => std.debug.print("LLAMA: [LEVEL={d}] {s}", .{ level_int, text }), // Handle unknown/other levels
    }
}


// --- Command Line Arguments ---
const CliArgs = struct {
    model_path: []const u8 = DEFAULT_MODEL_PATH,
    system_prompt: []const u8 = DEFAULT_SYSTEM_PROMPT,
    temperature: f32 = DEFAULT_TEMPERATURE,
    show_help: bool = false,
};

fn parseArgs(string_pool: *StringPool) !CliArgs {
    var args_iter = try process.argsWithAllocator(allocator);
    defer args_iter.deinit();

    _ = args_iter.next(); // Skip executable name

    var args = CliArgs{};

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            const value_arg = args_iter.next() orelse return error.MissingModelPath;
            args.model_path = try string_pool.dupe(value_arg);
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--system")) {
            const value_arg = args_iter.next() orelse return error.MissingSystemPrompt;
            args.system_prompt = try string_pool.dupe(value_arg);
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
fn toCString(alloc: Allocator, slice: []const u8) ![:0]const u8 {
    const c_str = try alloc.allocSentinel(u8, slice.len, 0);
    @memcpy(c_str[0..slice.len], slice);
    return c_str;
}

// C string pool for managing temporary C strings
const CStringPool = struct {
    strings: std.ArrayList([:0]u8),
    allocator: Allocator,

    fn init(alloc: Allocator) CStringPool {
        return CStringPool{
            .strings = std.ArrayList([:0]u8).init(alloc),
            .allocator = alloc,
        };
    }

    fn deinit(self: *CStringPool) void {
        for (self.strings.items) |str| {
            self.allocator.free(str);
        }
        self.strings.deinit();
    }

    fn add(self: *CStringPool, slice: []const u8) ![:0]const u8 {
        const c_str = try toCString(self.allocator, slice);
        // Cast to remove const qualifier for storage
        const mutable_c_str: [:0]u8 = @constCast(c_str);
        try self.strings.append(mutable_c_str);
        return c_str;
    }
};

// --- Main Application Logic ---
pub fn main() !void {
    // Initialize string pool for memory management
    var string_pool = StringPool.init(allocator);
    // Remove defer - we'll handle cleanup manually at the end

    const args = try parseArgs(&string_pool);

    if (args.show_help) {
        // Get the absolute path to the executable, allocating memory for it.
        const exe_path = try std.fs.selfExePathAlloc(allocator);
        // Free the allocated path when done
        defer allocator.free(exe_path);
        const exe_name = std.fs.path.basename(exe_path);
        printHelp(exe_name);
        return;
    }

    // Use default model path if none provided
    var final_model_path = args.model_path;
    if (final_model_path.len == 0) {
        final_model_path = DEFAULT_MODEL_PATH;
    }
    
    // Check if model path is still empty after using default
    if (final_model_path.len == 0) {
        print("Error: No model path provided. Use --model to specify a GGUF model file.\n", .{});
        print("Run with --help for usage information.\n", .{});
        return error.MissingModelPath;
    }

    // --- Llama.cpp Initialization ---
    print("Initializing llama.cpp backend...\n", .{});
    // llama_backend_init(false); // Use false to disable NUMA if not needed or causing issues
    llama_backend_init();
    // Set the log handler to our Zig function (pass its address)
    llama_log_set(&logCallback, null);
    print("llama.cpp log handler set.\n", .{});

    // --- Load Model ---
    const mparams = llama_model_default_params(); // Changed back to const
    // mparams.use_mmap = false; // REVERTED: Try enabling mmap again by commenting this out
    // Set n_gpu_layers to a high value to offload as much as possible
    // mparams.n_gpu_layers = 999; // REVERTED: Comment out explicit GPU layer setting for now
    // print("Attempting to offload {d} layers to GPU...\n", .{mparams.n_gpu_layers}); // Commented out log

    // Check if model file exists before attempting to load
    const model_file = fs.cwd().openFile(final_model_path, .{}) catch |err| {
        switch (err) {
            error.FileNotFound => {
                print("Error: Model file not found: '{s}'\n", .{final_model_path});
                print("Please check that the file exists and the path is correct.\n", .{});
            },
            error.AccessDenied => {
                print("Error: Access denied to model file: '{s}'\n", .{final_model_path});
                print("Please check file permissions.\n", .{});
            },
            else => {
                print("Error: Cannot access model file '{s}': {}\n", .{ final_model_path, err });
            },
        }
        llama_backend_free();
        return error.ModelLoadFailed;
    };
    model_file.close(); // Close the file, we just wanted to check it exists

    // Check if the file looks like a GGUF file by checking its size and first few bytes
    const file_stat = fs.cwd().statFile(final_model_path) catch |err| {
        print("Error: Cannot stat model file '{s}': {}\n", .{ final_model_path, err });
        llama_backend_free();
        return error.ModelLoadFailed;
    };
    
    if (file_stat.size < 8) {
        print("Error: Model file '{s}' is too small ({d} bytes) to be a valid GGUF file.\n", .{ final_model_path, file_stat.size });
        print("The file appears to be empty or corrupted. Please re-download the model.\n", .{});
        llama_backend_free();
        return error.ModelLoadFailed;
    }

    // Check GGUF magic bytes
    const check_file = fs.cwd().openFile(final_model_path, .{}) catch |err| {
        print("Error: Cannot reopen model file '{s}': {}\n", .{ final_model_path, err });
        llama_backend_free();
        return error.ModelLoadFailed;
    };
    defer check_file.close();
    
    var magic_buf: [4]u8 = undefined;
    _ = check_file.readAll(&magic_buf) catch |err| {
        print("Error: Cannot read magic bytes from '{s}': {}\n", .{ final_model_path, err });
        llama_backend_free();
        return error.ModelLoadFailed;
    };
    
    // GGUF magic is "GGUF" (0x47475546)
    if (!std.mem.eql(u8, &magic_buf, "GGUF")) {
        print("Error: File '{s}' does not appear to be a valid GGUF file (magic: {any}).\n", .{ final_model_path, magic_buf });
        print("Expected GGUF magic bytes, but got different values.\n", .{});
        llama_backend_free();
        return error.ModelLoadFailed;
    }

    print("Loading model: {s}...\n", .{final_model_path});
    const c_model_path = try toCString(allocator, final_model_path);
    // Don't use defer - free immediately after use to avoid cleanup order issues

    const model = llama_load_model_from_file(c_model_path, mparams);
    // Free the C string immediately after use
    allocator.free(c_model_path);
    if (model == null) {
        print("Error: Failed to load model from '{s}'\n", .{final_model_path});
        print("The file exists but llama.cpp failed to load it. Check if it's a valid GGUF file.\n", .{});
        llama_backend_free();
        return error.ModelLoadFailed;
    }
    defer llama_free_model(model);

    // --- Create Context ---
    var cparams = llama_context_default_params();
    // TODO: Expose more context params (n_batch, seed, etc.)
    cparams.n_ctx = 4096; // Explicitly set a smaller context size for testing
    // Set n_batch and n_ubatch according to llama.cpp warning (GGML_KQ_MASK_PAD likely requires 64)
    cparams.n_batch = 64;
    cparams.n_ubatch = 64;
    // cparams.seed = 1234; // Removed - seed is not in the struct definition matching llama.h

    print("Creating context (n_ctx = {d}, n_batch = {d}, n_ubatch = {d})...\n", .{ cparams.n_ctx, cparams.n_batch, cparams.n_ubatch }); // Log params
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

    var history = std.ArrayList(Message).init(allocator);
    var history_cleaned = false;
    defer {
        if (!history_cleaned) {
            // Clean up history safely only if we didn't clean it manually
            for (history.items) |msg| {
                // Assuming roles are static strings, only free content
                if (msg.content.len > 0) {
                    allocator.free(msg.content);
                }
            }
            history.deinit();
        }
    }

    const stdin = io.getStdIn().reader();
    const stdout_file = io.getStdOut(); // Get the File object
    const stdout = stdout_file.writer(); // Get the writer from the File
    var input_buffer = std.ArrayList(u8).init(allocator);
    defer input_buffer.deinit();

    // --- Chat Loop ---
    while (true) {
        try stdout.print("> ", .{});
        // No need to sync stdout here

        input_buffer.clearRetainingCapacity();
        stdin.streamUntilDelimiter(input_buffer.writer(), '\n', null) catch |err| {
            if (err == error.EndOfStream) { // Ctrl+D
                print("\nExiting...\n", .{});
                // Clean up history manually before breaking
                if (!history_cleaned) {
                    for (history.items) |msg| {
                        if (msg.content.len > 0) {
                            allocator.free(msg.content);
                        }
                    }
                    history.deinit();
                    history_cleaned = true;
                }
                break;
            } else {
                print("Error reading input: {}\n", .{err});
                continue; // Don't return error, just continue to exit gracefully
            }
        };

        const user_input = std.mem.trim(u8, input_buffer.items, " \t\r\n");

        if (user_input.len == 0) continue;
        if (std.mem.eql(u8, user_input, "/quit")) {
            print("Exiting...\n", .{});
            // Clean up history manually before breaking
            if (!history_cleaned) {
                for (history.items) |msg| {
                    if (msg.content.len > 0) {
                        allocator.free(msg.content);
                    }
                }
                history.deinit();
                history_cleaned = true;
            }
            break;
        }

        // --- Prepare Prompt using simple string concatenation to avoid llama_chat_apply_template issues ---
        var prompt_builder = std.ArrayList(u8).init(allocator);
        defer prompt_builder.deinit();
        
        // Build prompt manually in Qwen3 format to avoid potential llama_chat_apply_template memory issues
        try prompt_builder.appendSlice("<|im_start|>system\n");
        try prompt_builder.appendSlice(args.system_prompt);
        try prompt_builder.appendSlice("<|im_end|>\n");
        
        // Limit history to prevent token overflow - keep only last few exchanges
        const max_history_pairs = 3; // Keep last 3 user-assistant pairs
        const history_start = if (history.items.len > max_history_pairs * 2) 
            history.items.len - (max_history_pairs * 2) 
        else 
            0;
        
        // Add limited history messages
        for (history.items[history_start..]) |msg| {
            try prompt_builder.appendSlice("<|im_start|>");
            try prompt_builder.appendSlice(msg.role);
            try prompt_builder.appendSlice("\n");
            try prompt_builder.appendSlice(msg.content);
            try prompt_builder.appendSlice("<|im_end|>\n");
        }
        
        // Add current user input
        try prompt_builder.appendSlice("<|im_start|>user\n");
        try prompt_builder.appendSlice(user_input);
        try prompt_builder.appendSlice("<|im_end|>\n");
        try prompt_builder.appendSlice("<|im_start|>assistant\n");
        
        const full_prompt_bytes = prompt_builder.items;
        
        // More conservative safety check for prompt length
        if (full_prompt_bytes.len > 2048) {
            print("Error: Prompt too long ({d} bytes), max 2048 bytes. Try shorter messages or restart.\n", .{full_prompt_bytes.len});
            continue;
        }


        // --- Llama.cpp Inference ---
        print("Assistant: ", .{});
        // Removed: try stdout_file.sync(); // This caused crashes on stdout

        // --- Tokenize Prompt ---
        // Use a much larger token buffer to handle the full prompt
        var tokens: [512]llama_token = undefined; // Increase back to 512 tokens
        const max_tokens = tokens.len;

        // Get vocab for tokenization and add_bos check
        const vocab = llama_model_get_vocab(model) orelse {
            print("Error: Failed to get model vocabulary\n", .{});
            continue;
        };
        
        const add_bos = llama_vocab_get_add_bos(vocab);
        
        // Pass vocab, not model, to llama_tokenize
        const n_tokens = llama_tokenize(vocab, full_prompt_bytes.ptr, @intCast(full_prompt_bytes.len), &tokens, @intCast(max_tokens), add_bos, false);

        if (n_tokens < 0) {
            print("Error: Failed to tokenize prompt (returned {d})\n", .{n_tokens});
            print("Prompt length: {d} bytes\n", .{full_prompt_bytes.len});
            continue;
        }
        
        if (n_tokens == 0) {
            print("Error: Tokenization produced 0 tokens\n", .{});
            continue;
        }
        
        if (n_tokens >= max_tokens) {
            print("Error: Too many tokens ({d}), max {d}. Try shorter messages or restart.\n", .{ n_tokens, max_tokens });
            continue;
        }

        print("Tokenized {d} tokens from {d} byte prompt\n", .{ n_tokens, full_prompt_bytes.len });

        // --- Clear KV cache and reset context state before processing new conversation ---
        llama_kv_cache_clear(ctx);
        
        // --- Evaluate Prompt in smaller chunks to avoid batch size issues ---
        var token_pos: i32 = 0;
        while (token_pos < n_tokens) {
            const chunk_size = @min(64, n_tokens - token_pos); // Process in chunks of 64 tokens max
            const chunk_tokens = tokens[@intCast(token_pos)..@intCast(token_pos + chunk_size)];
            
            const prompt_batch = llama_batch_get_one(@ptrCast(chunk_tokens.ptr), @intCast(chunk_size));
            const decode_prompt_ret = llama_decode(ctx, prompt_batch);
            if (decode_prompt_ret != 0) {
                 print("Error: llama_decode failed during prompt processing (ret={d})\n", .{decode_prompt_ret});
                 break;
            }
            token_pos += chunk_size;
        }
        
        if (token_pos < n_tokens) {
            print("Error: Failed to process full prompt\n", .{});
            continue;
        }

        // --- Generation Loop ---
        // Use a fixed buffer instead of ArrayList to avoid allocations
        var response_buf: [2048]u8 = undefined;
        var response_len: usize = 0;

        // Reset position tracking since we cleared the KV cache
        var cur_pos: llama_pos = @intCast(n_tokens);
        const max_new_tokens = @min(512, n_ctx - @as(u32, @intCast(n_tokens))); // Increase to 512 tokens for much longer responses
        var generated_token_count: u32 = 0;

        // --- Init Sampler Chain ---
        const chain_params = llama_sampler_chain_default_params();
        const sampler_chain = llama_sampler_chain_init(chain_params) orelse {
            print("Error: Failed to init sampler chain\n", .{});
            continue;
        };
        defer llama_sampler_free(sampler_chain);

        // Add samplers to the chain (order matters)
        const greedy_sampler = llama_sampler_init_greedy() orelse {
            print("Error: Failed to init greedy sampler\n", .{});
            continue;
        };
        llama_sampler_chain_add(sampler_chain, greedy_sampler);
        // Note: The chain takes ownership, so we don't free greedy_sampler directly

        // vocab variable already obtained before tokenize call, use it here too
        // const vocab = llama_model_get_vocab(model); // Already have 'vocab'

        while (cur_pos < n_ctx and generated_token_count < max_new_tokens) {
            // Sample the next token using the chain
            var id = llama_sampler_sample(sampler_chain, ctx, -1); // Use -1 for the last logit
            
            // Validate token ID
            if (id < 0) {
                print("Error: Invalid token ID sampled: {d}\n", .{id});
                break;
            }

            // Check for End-Of-Sequence using vocab BEFORE accepting token
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }

            // Accept the token using the chain
            llama_sampler_accept(sampler_chain, id);

            // Convert token to piece and print/store
            // Use a fixed-size buffer to avoid repeated allocations
            var piece_buf: [256]u8 = undefined; // Fixed buffer for token pieces
            const actual_len = llama_token_to_piece(vocab, id, &piece_buf, piece_buf.len, 0, false);
            if (actual_len < 0) {
                 print("Error: llama_token_to_piece failed (token_id={d}, ret={d})\n", .{ id, actual_len });
                 break;
            }
            
            if (actual_len == 0) {
                // Empty token, skip but continue
                generated_token_count += 1;
                cur_pos += 1;
                continue;
            }
            
            const piece_slice = piece_buf[0..@intCast(actual_len)];
            stdout.print("{s}", .{piece_slice}) catch |err| {
                print("Error writing to stdout: {}\n", .{err});
                break;
            };
            
            // Store in fixed buffer instead of ArrayList
            if (response_len + piece_slice.len < response_buf.len) {
                @memcpy(response_buf[response_len..response_len + piece_slice.len], piece_slice);
                response_len += piece_slice.len;
            } else {
                // Response buffer full, but continue generation
            }

            // Create a proper batch for single token with position
            const single_token_batch = llama_batch{
                .n_tokens = 1,
                .token = &id,
                .embd = null,
                .pos = &cur_pos,
                .n_seq_id = null,
                .seq_id = null,
                .logits = null,
            };
            
            // Decode the single token batch
            const decode_gen_ret = llama_decode(ctx, single_token_batch);
            if (decode_gen_ret != 0) {
                 print("Error: llama_decode failed during generation (ret={d})\n", .{decode_gen_ret});
                 break;
            }
            cur_pos += 1; // Update position manually
            generated_token_count += 1;
        }
        try stdout.print("\n", .{}); // Newline after assistant response

        // --- Update History ---
        if (response_len > 0) {
            // Only add to history if we have a meaningful response
            const user_input_copy = allocator.dupe(u8, user_input) catch |err| {
                print("Error: Failed to allocate memory for user input: {}\n", .{err});
                continue;
            };
            
            // Use the actual generated response from fixed buffer
            const assistant_response_copy = allocator.dupe(u8, response_buf[0..response_len]) catch |err| {
                print("Error: Failed to allocate memory for assistant response: {}\n", .{err});
                allocator.free(user_input_copy);
                continue;
            };

            // Add to history with proper error handling
            history.append(.{ .role = "user", .content = user_input_copy }) catch |err| {
                print("Error: Failed to add user message to history: {}\n", .{err});
                allocator.free(user_input_copy);
                allocator.free(assistant_response_copy);
                continue;
            };
            
            history.append(.{ .role = "assistant", .content = assistant_response_copy }) catch |err| {
                print("Error: Failed to add assistant message to history: {}\n", .{err});
                // User message was already added, so we need to remove it
                _ = history.pop();
                allocator.free(user_input_copy);
                allocator.free(assistant_response_copy);
                continue;
            };
        }

        // Sampling context is freed by defer

        // --- Llama.cpp Cleanup ---
        // Note: Context and Model are freed by their defers earlier in main()
        // No specific cleanup needed within the loop itself for this basic implementation
    }

    // --- Llama.cpp Cleanup ---
    // Context and Model are freed by their defers earlier in main()
    print("Cleaning up llama.cpp backend...\n", .{});
    llama_backend_free();

    // Clean up string pool manually at the very end
    string_pool.deinit();

    // Final allocator check - don't print warning as it's expected during development
    _ = gpa.deinit();
}
