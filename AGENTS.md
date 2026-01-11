# AGENTS.md

## Repository overview

- **Project type:** Zig command-line application.
- **Purpose:** CLI chat interface for Qwen models via the `llama.cpp` C API (see `README.md`).
- **Main entrypoint:** `src/qwen_cli.zig`.
- **Build system:** Zig `build.zig`.
- **External dependency:** `llama.cpp` is expected to be cloned and built under `vendor/llama.cpp` via `build.sh`.

## Quick start (what to run)

### 1) Build `llama.cpp` dependency

This repo includes a helper script that clones and builds `llama.cpp` as a **shared library**:

```bash
./build.sh
```

Notes observed in `build.sh`:

- Clones `https://github.com/ggerganov/llama.cpp.git` into `vendor/llama.cpp` if missing.
- Configures CMake with:
  - `-DBUILD_SHARED_LIBS=ON`
  - `-DLLAMA_CUDA=ON`
  - `-DLLAMA_FLASH_ATTN=ON`
- Builds in `vendor/llama.cpp/build`.
- The shared library location the Zig build expects is `vendor/llama.cpp/build/bin`.

If you change `CMAKE_FLAGS` in `build.sh`, the `README.md` suggests cleaning the llama build directories before rebuilding.

### 2) Build the Zig app

```bash
zig build
```

`zig build` builds the project.

Expected output binary:

- `zig-out/bin/qwen_cli`

### 3) Run

From `build.zig`, there is a `run` step:

```bash
zig build run -- --model <path-to-gguf>
```

Or run the installed binary directly:

```bash
./zig-out/bin/qwen_cli --model <path-to-gguf>
```

### 4) Download a model (GGUF)

This repo also includes a downloader for pre-converted GGUF models:

```bash
./download_model.sh
```

Observed behavior:

- Downloads from `ggml-org/Qwen3-14B-GGUF` on Hugging Face.
- Writes into `models/Qwen3-14B-GGUF/`.
- Uses `wget` if present, otherwise `curl`.

## Tooling / environment

- `mise.toml` pins Zig to `latest` under mise:

  - `mise.toml:1-2`

If you’re using mise, install tools via your normal mise workflow; this repo does not include additional mise tasks.

## Project layout

```
.
├── build.zig
├── build.sh
├── download_model.sh
├── mise.toml
├── src/
│   └── qwen_cli.zig
└── README.md
```

### Ignored artifacts

From `.gitignore`:

- `models/` (downloaded models)
- `vendor/` (vendored `llama.cpp` clone and build output)
- `zig-out/`, `.zig-cache/`

## Build/linking expectations (important)

`build.zig` is hardcoded to the layout produced by `build.sh`:

- Headers: `vendor/llama.cpp/include`
- Shared library directory: `vendor/llama.cpp/build/bin`

On non-Windows targets, `build.zig`:

- Adds RPATH for `vendor/llama.cpp/build/bin`.
- Links system libs: `c`, `m`, `dl`, `pthread`.
- Links CUDA libs: `cuda`, `cudart`, `cublas`.
- Adds the llama/ggml shared objects from `vendor/llama.cpp/build/bin` to ensure the resulting binary can locate transitive `llama.cpp` deps at runtime.

If `llama.cpp` is built without CUDA (or on machines without CUDA), `build.zig` will likely need adjustments to the linked CUDA libraries.

## Source code patterns and conventions

### `src/qwen_cli.zig`

Key observed patterns:

- Uses a global `GeneralPurposeAllocator` (`gpa`) and `allocator` for allocations.
- Command line parsing via `std.process.argsWithAllocator`.
- CLI flags:
  - `--model` / `-m` (required in practice; default is empty string)
  - `--system` / `-s`
  - `--temp` / `-t`
  - `--help` / `-h`
- Uses **manual `extern "c"` declarations** for `llama.cpp` types and functions instead of `@cImport`.
  - This avoids Zig/C import issues but creates a maintenance burden: the declarations must stay compatible with the `llama.cpp` version in `vendor/`.
- Prompt formatting uses `llama_chat_apply_template`.
- Tokenization uses `llama_tokenize` via a vocab pointer acquired from `llama_model_get_vocab`.
- Sampling uses a sampler chain API (`llama_sampler_chain_*`), with a greedy sampler added.

### Logging

- Installs a llama/ggml log callback via `llama_log_set(&logCallback, null)`.
- The callback prints to `std.debug.print` with a `LLAMA:` prefix.

## Testing

No test targets or test directories were found in the current tree.

## Common gotchas (from code + scripts)

- `build.zig` assumes `vendor/llama.cpp` exists and shared objects are in `vendor/llama.cpp/build/bin`.
- `download_model.sh` downloads large files (GBs) and writes into `models/`, which is gitignored.
- The `llama.cpp` C API changes over time; keep `src/llama.zig` in sync with the `vendor/llama.cpp/include/llama.h` version.
