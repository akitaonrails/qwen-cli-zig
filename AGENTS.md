# AGENTS.md

Instructions for AI agents working in this repository.

## Project Overview

A CLI chat interface for Qwen3 language models, written in Zig. Uses `llama.cpp` as a shared library for inference with CUDA/GPU acceleration support.

## Quick Reference

| Action | Command |
|--------|---------|
| Build llama.cpp dependency | `./build.sh` |
| Build Zig application | `zig build` |
| Run application | `./zig-out/bin/qwen_cli --model <path-to-gguf>` |
| Run unit tests | `zig build test` |
| Download models | `./download_model.sh` |

## Prerequisites

- **Zig**: 0.15+ (managed via `mise.toml`)
- **Build tools**: `cmake`, `git`, `gcc` or `clang`
- **GPU (optional)**: CUDA Toolkit for NVIDIA GPU acceleration
- **Model files**: GGUF format models in `models/` directory

## Directory Structure

```
.
├── build.sh              # Clones and builds llama.cpp as shared library
├── build.zig             # Zig build configuration
├── build.zig.zon         # Zig package manifest
├── download_model.sh     # Downloads pre-converted GGUF models
├── mise.toml             # Zig version management
├── src/
│   ├── main.zig          # Application entry point and inference engine
│   ├── llama.zig         # Llama.cpp bindings (extern declarations + wrapper API)
│   ├── chat.zig          # Chat formatting and history management
│   └── config.zig        # CLI argument parsing and configuration
├── vendor/               # (gitignored) llama.cpp source and build
├── models/               # (gitignored) GGUF model files
└── zig-out/              # (gitignored) Build output
```

## Code Architecture

### Module Responsibilities

- **main.zig**: Entry point, `InferenceEngine` struct, model validation, chat loop
- **llama.zig**: Low-level C bindings + high-level wrapper API (`SamplerChain`, helper functions)
- **chat.zig**: `History` struct, `Message` type, Qwen3 `Format` prompt builder
- **config.zig**: `Config` struct with defaults, `parseArgs()`, `printHelp()`

### Key Patterns

**Zig 0.15 ArrayList**: Uses unmanaged ArrayList (allocator passed to each operation):
```zig
var list: std.ArrayList(T) = .{};
try list.append(allocator, item);
list.deinit(allocator);
```

**Manual C Bindings**: Uses `extern "c"` declarations instead of `@cImport`:
```zig
const Model = opaque {};
extern "c" fn llama_load_model_from_file(path: [*c]const u8, params: ModelParams) ?*Model;
```

**Error Handling**: Functions return explicit error unions, caller handles or propagates.

### Configuration Constants

All tunable values are in `config.zig`:
```zig
pub const defaults = struct {
    pub const model_path = "models/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf";
    pub const context_size: u32 = 4096;
    pub const batch_size: u32 = 64;
    pub const max_history_pairs: usize = 3;
    pub const max_new_tokens: u32 = 512;
    // ... more constants
};
```

## Build Process

### Initial Setup

```bash
# 1. Build llama.cpp shared library (required once)
chmod +x build.sh download_model.sh
./build.sh

# 2. Download model files (~9-30GB each)
./download_model.sh

# 3. Build Zig application
zig build
```

### GPU Configuration

Default build enables CUDA. Edit `CMAKE_FLAGS` in `build.sh`:

```bash
# CPU only:
CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON"

# AMD ROCm:
CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON -DLLAMA_ROCM=ON"
```

After changing flags: `rm -rf vendor/llama.cpp/build && ./build.sh`

## Testing

```bash
# Run unit tests (chat.zig tests)
zig build test

# Manual testing (requires model)
./zig-out/bin/qwen_cli --model models/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf
```

## Gotchas

### Llama.cpp API Version

The code declares llama.cpp types manually. If updating llama.cpp:
1. Check `vendor/llama.cpp/include/llama.h` for changes
2. Update corresponding declarations in `src/llama.zig`
3. Key areas: struct definitions, function signatures, enum values

### Shared Library Path

RPATH is set to `vendor/llama.cpp/build/bin`. Moving the executable requires `libllama.so` in library search path.

### CUDA Dependencies

When built with CUDA, links against `libcuda`, `libcudart`, `libcublas`. Ensure CUDA runtime is available.

### Zig Version

Requires Zig 0.15+. The ArrayList API changed significantly from earlier versions (now unmanaged by default).
