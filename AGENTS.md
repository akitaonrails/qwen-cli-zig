# Agent Guide for Qwen CLI Zig

This repository contains a Zig-based CLI for Qwen models, binding directly to `llama.cpp`.

## ⚡ Quick Start

1.  **Install dependencies** (git, cmake, C compiler):
    ```bash
    # Arch
    sudo pacman -S git cmake base-devel
    # Ubuntu
    sudo apt install git cmake build-essential
    ```

2.  **Build dependencies (llama.cpp) & Project**:
    ```bash
    ./build.sh   # Clones and builds llama.cpp vendor lib
    zig build    # Builds the Zig CLI wrapper
    ```

3.  **Get a Model**:
    ```bash
    ./download_model.sh  # Downloads Qwen3-14B-GGUF to models/
    ```

4.  **Run**:
    ```bash
    ./zig-out/bin/qwen_cli --model models/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf
    ```

## 🏗 Project Structure

-   `src/main.zig`: **Main Entry Point**. Contains the CLI logic, argument parsing, and inference loop.
-   `src/llama_cpp.zig`: **FFI Bindings**. Contains manually defined C structs, enums, and function declarations for `llama.cpp`.
-   `vendor/llama.cpp/`: Maintained by `build.sh`. **Do not edit manually**.
-   `build.zig`: Zig build logic (0.15.x compatible). Links against `libllama.so`.
-   `build.sh`: Helper to prepare the C++ dependency.

## 🧩 Code Architecture & Patterns

### 1. FFI Bindings (Critical)
**We do NOT use `@cImport`**.
Instead, `llama.cpp` C structs and functions are **manually defined** in `src/llama_cpp.zig`.

*   **Gotcha**: The FFI uses `callconv(.c)` (lowercase) in Zig 0.15+, not `.C`.
*   **Struct Alignment**: Access to `llama.cpp` types is centralized in `src/llama_cpp.zig`. Ensure structs match `llama.h` and `ggml.h` layout.

### 2. Memory Management
*   Uses `std.heap.GeneralPurposeAllocator`.
*   **ArrayList**: Uses `std.ArrayListUnmanaged` with explicit allocator passing, as `std.ArrayList(T).init` is removed in Zig 0.15.
*   **Streams**: `stdin` and `stdout` are accessed via `std.fs.File.stdin()`/`stdout()` and wrapped in buffered readers/writers because `File.reader()` now requires a buffer.

### 3. Inference Loop
The core loop in `src/main.zig` handles:
1.  **Tokenization**: `llama_tokenize`
2.  **Prompt Processing**: `llama_decode` (batch processing)
3.  **Generation Loop**: `llama_decode` (token by token) + `llama_sampler_sample`
4.  **Chat Templates**: managed via `llama_chat_apply_template` (Qwen3 specific formatting).

## 🛠 Common Tasks

### Updating llama.cpp
1.  Update `LLAMA_CPP_REPO` or checkout specific tag in `build.sh`.
2.  Run `./build.sh` to rebuild the C lib.
3.  Check `vendor/llama.cpp/include/llama.h` for API changes.
4.  Update extern definitions in `src/qwen_cli.zig`.

### Adding CLI Arguments
1.  Update `CliArgs` struct.
2.  Add parsing logic in `parseArgs`.
3.  Pass new config to `llama_model_params` or `llama_context_params`.

### Debugging
*   **LLM Logs**: `llama_log_set` is wired to a Zig callback. Logs print with `LLAMA: [LEVEL] ...`.
*   **Segfaults**: Almost always due to FFI struct mismatch or double-free. Check `extern struct` definitions against C header.

## ⚠️ Known Gotchas
*   **Hardware Acceleration**: CUDA is enabled by default in `build.sh` (`-DLLAMA_CUDA=ON`). Use `zig build` (which links against it). If CUDA is missing, `build.sh` might fail or fallback; check CMake output.
*   **Model Paths**: Paths must be valid GGUF files.
