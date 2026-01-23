# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CLI tool written in Zig to interact with Qwen language models via llama.cpp. Requires **Zig 0.15.x** (tested with 0.15.2). The core inference loop is partially implemented but may crash after GPU memory load.

## Build Commands

```bash
# First-time setup: build llama.cpp dependency (requires git, cmake, C compiler)
./build.sh

# Build the Zig application
zig build

# Run the CLI (requires a GGUF model file)
./zig-out/bin/qwen_cli -m models/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf

# Download a pre-quantized model (~9GB)
./download_model.sh
```

## Architecture

### Modular Structure
The code is organized into four files in `src/`:

1. **`llama.zig`** - C FFI declarations for llama.cpp API. Uses opaque types and explicit struct definitions because `@cImport` has issues with the llama.h header. Uses the new non-deprecated llama.cpp API (`llama_model_load_from_file`, `llama_init_from_model`, etc.).

2. **`cli.zig`** - Command-line argument parsing. `CliArgs` struct with `-m/--model`, `-s/--system`, `-t/--temp`, `-h/--help` flags.

3. **`chat.zig`** - Chat history management and template application. Handles building llama_chat_message arrays and applying the Qwen3 chat template via `llama_chat_apply_template`.

4. **`main.zig`** - Application entry point. Initialize backend → load model → create context → chat loop (read input → apply template → tokenize → decode → sample tokens → print response).

### Key Dependencies
- **llama.cpp**: Vendored in `vendor/llama.cpp/`, built as shared library with CUDA support by default
- **Models**: GGUF format files in `models/` (not tracked in git)

### Chat Template Format
Uses Qwen3 format: `<|im_start|>role\ncontent<|im_end|>` via `llama_chat_apply_template()`.

## Known Issues

- Inference loop incomplete - program crashes after GPU memory load
- Type casting complexity between Zig and C FFI
- Sampling currently uses greedy only (temperature sampler commented out)

## Build Configuration

Edit `CMAKE_FLAGS` in `build.sh` for different backends:
- CPU-only: Remove `-DLLAMA_CUDA=ON`
- AMD ROCm: Add `-DLLAMA_ROCM=ON`
