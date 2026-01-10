# Qwen CLI Zig

A CLI chat interface for Qwen3 language models, written in Zig using `llama.cpp`.

## Features

- Interactive chat with Qwen3 models
- CUDA/GPU acceleration support
- Configurable system prompt and temperature
- Conversation history with automatic context management

## Prerequisites

- **Zig**: 0.15+ (use [mise](https://mise.jdx.dev/) for version management)
- **Build tools**: `cmake`, `git`, `gcc` or `clang`
- **GPU (optional)**: CUDA Toolkit for NVIDIA GPUs

## Quick Start

```bash
# 1. Build llama.cpp dependency
chmod +x build.sh download_model.sh
./build.sh

# 2. Download a model (~9GB for Q4_K_M)
./download_model.sh

# 3. Build the application
zig build

# 4. Run
./zig-out/bin/qwen_cli --model models/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf
```

## Usage

```bash
./zig-out/bin/qwen_cli [options]

Options:
  -m, --model <path>    Path to GGUF model file (required)
  -s, --system <prompt> System prompt (default: "You are a helpful assistant.")
  -t, --temp <value>    Sampling temperature (default: 0.7)
  -h, --help            Show help message

Examples:
  ./zig-out/bin/qwen_cli --model models/qwen.gguf
  ./zig-out/bin/qwen_cli --model models/qwen.gguf --temp 0.5
  ./zig-out/bin/qwen_cli -m models/qwen.gguf -s "You are a coding expert."
```

Type messages and press Enter. Use `/quit` or Ctrl+D to exit.

## GPU Configuration

The default build enables NVIDIA CUDA. To change:

```bash
# Edit CMAKE_FLAGS in build.sh:

# CPU only:
CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON"

# AMD ROCm:
CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON -DLLAMA_ROCM=ON"

# Then rebuild:
rm -rf vendor/llama.cpp/build
./build.sh
zig build
```

## Project Structure

```
src/
├── main.zig    # Entry point and inference engine
├── llama.zig   # Llama.cpp bindings
├── chat.zig    # Chat formatting and history
└── config.zig  # CLI argument parsing
```

## Development

```bash
# Run unit tests
zig build test

# Build with optimizations
zig build -Doptimize=ReleaseFast
```

## License

MIT
