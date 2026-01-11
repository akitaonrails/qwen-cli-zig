# Agent Guide

## Overview

This repository contains a Zig command-line client that drives Qwen models through the `llama.cpp` C API. The binary is now built with Zig 0.15+, relies on `@cImport` to consume the upstream headers, and performs the chat loop and sampling logic in `src/qwen_cli.zig`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/qwen_cli.zig` | Main executable entry point. Handles CLI parsing, C imports for `llama.cpp`, prompt construction, sampling loop, and stdout streaming. |
| `build.zig` | Zig build script wiring in the llama shared library, ggml headers, CUDA linkage, and run step. |
| `build.sh` | Helper script that clones and builds `llama.cpp` as a shared library under `vendor/`. |
| `download_model.sh` | Downloads pre-quantized GGUF weights for Qwen 3 models. |
| `mise.toml` | Pins Zig tooling through `mise` (`zig = "latest"`). |
| `README.md` | High-level setup, dependency, and usage instructions. |

The repository assumes `vendor/llama.cpp` exists (populated by `build.sh`) and `models/` contains converted GGUF artefacts (populated manually or via `download_model.sh`). Neither directory is tracked.

## Tooling & Environment

- **Zig version**: Target Zig 0.15.2 or newer. The build script already uses the `root_module` workflow and links ggml headers, so no manual patches are needed for current toolchains. Install via `mise install` if you rely on `mise`.
- **llama.cpp**: `build.sh` performs a shallow clone of `https://github.com/ggerganov/llama.cpp.git` and configures CMake with `-DBUILD_SHARED_LIBS=ON -DLLAMA_CUDA=ON -DLLAMA_FLASH_ATTN=ON`. Adjust that script if you need CPU-only or ROCm builds.
- **External tools**: `build.sh` requires `git`, `cmake`, and a working C/C++ compiler. Linking expects CUDA libraries by default (`cuda`, `cudart`, `cublas`).

## Build & Run Commands

1. Prepare the llama dependency (creates `vendor/llama.cpp` and builds `libllama.so`):
   ```bash
   ./build.sh
   ```
2. Build the Zig executable:
   ```bash
   zig build
   ```
3. Run the executable (extra arguments after `--` are forwarded to the program):
   ```bash
   zig build run -- --model path/to/model.gguf
   ```

## Model Assets

- `./download_model.sh` fetches GGUF weights from `ggml-org/Qwen3-14B-GGUF`, storing them under `models/Qwen3-14B-GGUF/`.
- To convert other checkpoints, follow `README.md` and run `llama.cpp`'s `convert.py`, then point the CLI to the resulting `.gguf` file.

## Source Code Patterns

- `src/qwen_cli.zig` imports `llama.h` via `@cImport`. Keep Zig bindings in sync with upstream by rebuilding the dependency (headers live in `vendor/llama.cpp/include` and `vendor/llama.cpp/ggml/include`).
- The build now compiles against Zig 0.15 APIs: `std.ArrayList` requires explicit allocators for `append`, `writer`, and `toOwnedSlice`; `std.fs.File` I/O uses the newer buffer-based `reader`/`writeAll` interfaces. Follow existing patterns when interacting with the standard library.
- Input handling manually reads from `stdin` byte-by-byte (terminating on `\n` or EOF) because the legacy `streamUntilDelimiter` helpers were removed. Reuse that logic when extending the REPL.
- Prompt formatting uses `llama_chat_apply_template`, resizing buffers as needed, and tokenization/generation rely on `llama_sampler_chain_init` with greedy sampling only. Extend the sampler chain to expose other strategies.
- Memory is managed via `std.heap.GeneralPurposeAllocator`. Every allocation tied to history or response buffers is explicitly freed; follow this convention to avoid leaks.

## Testing Status

- No Zig tests are defined. Validation is manual by running the CLI against a model. Add unit or integration tests under `zig build test` when you start filling in the TODO sections.

## Gotchas & Caveats

- `build.zig` adds include paths for both `vendor/llama.cpp/include` and `vendor/llama.cpp/ggml/include`. Keep those directories available when upgrading llama.cpp.
- CUDA linkage is unconditional. Remove or guard the `linkSystemLibrary` calls if you need CPU-only builds.
- `build.sh` aborts on missing prerequisites (`set -e`). Install the required toolchain before invoking it.
- Argument parsing duplicates user-provided strings using the global allocator. That’s acceptable for process lifetime but be mindful if you refactor to reuse allocators or add long-lived caches.
- Chat history stores owned copies of user/assistant turns. Remember to free inserted slices when you mutate the history vector.

## Additional References

- `README.md` covers setup, GPU notes, and manual conversion instructions.
- `llama.cpp/examples` (e.g., `examples/main/main.cpp`) remain the best reference for fleshing out inference details inside `qwen_cli.zig`.
