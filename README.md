# Qwen CLI Zig

This is an attempt at making Gemini 2.5 Pro Exp write usable Zig code. The veredict: do not try unless you have a robust RAG or LoRa solution for the most current Zig documentation or source code. 

It will try to write Zig but it doesn´t know what changed in the most current version. If you try to import C header files, it has a very hard time with type casting, pointer casting and makes frequent type and syntax errors. It is a true knighmare.

After spending upwards of USD 50 from OpenRouter.ai, I gave up. This program do load the Qwen3 model in the GPU memory using CUDA, but it's unable to retrieve responses and crashes. And even after dozens of attemps, Gemini was unable to add print/debug probes to figure out the problem.


# Introduction

A simple command-line interface written in Zig to interact with Qwen language models (like Qwen3-14B) using the `llama.cpp` library.

**Note:** This is currently a skeleton application. The core logic for interacting with the `llama.cpp` C API still needs to be implemented in `src/qwen_cli.zig`.

## Prerequisites

1.  **Zig Compiler:** Ensure you have a recent version of the Zig compiler installed. Check the [Ziglang Download Page](https://ziglang.org/download/).
2.  **C/C++ Toolchain:** You need `make`, `cmake`, and a C/C++ compiler (`gcc` or `clang`).
3.  **Git:** Required for cloning `llama.cpp`.
4.  **(Optional) GPU SDKs:** If you want GPU acceleration:
    *   **NVIDIA:** CUDA Toolkit (check Arch Wiki/Manjaro repos for `cuda`, `nvidia-utils`).
    *   **AMD:** ROCm SDK (check Arch Wiki/Manjaro repos for `rocm-hip-sdk`, `rocm-opencl-sdk`).

## Setup

### 1. Install Build Dependencies (Arch Linux / Manjaro Example)

```bash
# Install essential build tools
sudo pacman -Syu --needed base-devel cmake git

# Optional: Install GPU SDKs if needed (check package names)
# sudo pacman -Syu --needed cuda nvidia-utils # For NVIDIA
# sudo pacman -Syu --needed rocm-hip-sdk rocm-opencl-sdk # For AMD
```

### 2. Prepare `llama.cpp` Dependency using `build.sh`

This project includes a helper script `build.sh` to automate the cloning and building of the required `llama.cpp` library.

First, make the script executable:
```bash
chmod +x build.sh
```

Then, run the script:
```bash
./build.sh
```

This script will:
1. Check for required tools (`git`, `cmake`, `cc`).
2. Clone the `llama.cpp` repository into a `vendor/llama.cpp` subdirectory if it doesn't exist.
3. Configure and build `llama.cpp` as a **shared library** inside `vendor/llama.cpp/build`. **By default, it builds with NVIDIA CUDA and Flash Attention enabled.** Ensure you have the CUDA Toolkit installed (see Prerequisites). The shared library (`libllama.so`) will be placed in `vendor/llama.cpp/build/bin`.
4. Print the `zig build` command needed to build the main application.

**Customizing the `llama.cpp` Build (Optional):**

If you want to disable GPU support or add other specific `llama.cpp` features (like ROCm for AMD GPUs), you can edit the `CMAKE_FLAGS` variable near the top of the `build.sh` script *before* running it.

*   To disable CUDA and build for CPU only:
    ```bash
    # build.sh
    # ...
    CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON" # Remove CUDA flags
    # ...
    ```
*   Example for AMD ROCm (ensure ROCm SDK is installed):
    ```bash
    # build.sh
    # ...
    CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON -DLLAMA_ROCM=ON" # Add ROCm flag
    # ...
    ```
Remember to clean the build directory (`rm -rf vendor/llama.cpp/build vendor/llama.cpp/bin`) before running `./build.sh` if you change the `CMAKE_FLAGS`.

```bash
# build.sh
# ...
# Example: CMAKE_FLAGS="-DLLAMA_CUDA=ON -DLLAMA_FLASH_ATTN=ON"
CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON -DLLAMA_CUDA=ON" # Add your desired flags here
# ...
```

### 3. Download Model Files & Convert to GGUF

This application requires the language model to be in the **GGUF format** to be loaded by `llama.cpp`.

The provided `download_model.sh` script helps download the *original* model files (configuration, tokenizer, and Safetensors weights) from the official Hugging Face repository (`Qwen/Qwen3-14B`).

1.  **Make the script executable:**
    ```bash
    chmod +x download_model.sh
    ```
2.  **Run the script:**
    ```bash
    ./download_model.sh
    ```
    This will download the necessary files into the `models/Qwen3-14B-hf` directory.

3.  **Convert the downloaded files to GGUF format:**
    *   **IMPORTANT:** The downloaded files are **NOT** directly usable. You must convert them using the `convert.py` script from the `llama.cpp` project.
    *   Make sure you have Python installed along with the necessary libraries (usually listed in `llama.cpp`'s `requirements.txt`, e.g., `pip install -r requirements.txt` within the `llama.cpp` directory).
    *   Navigate to your separate `llama.cpp` clone directory.
    *   Run the conversion script, pointing it to the downloaded model directory and specifying an output file path and type. Example:
        ```bash
        # Assuming your qwen-cli-zig project is one level up from llama.cpp
        python convert.py ../qwen-cli-zig/models/Qwen3-14B-hf/ \
          --outfile ../qwen-cli-zig/models/qwen3-14b-converted.gguf \
          --outtype q5_k_m
        ```
        *(Adjust paths and `--outtype` (quantization level) as needed. Common types include `f16`, `q4_k_m`, `q5_k_m`, `q8_0`)*. Refer to `llama.cpp` documentation for details on `convert.py`.

### 4. Build `qwen_cli`

After running `./build.sh` successfully to prepare the `llama.cpp` dependency, simply run the Zig build command:

```bash
zig build
```

The `build.zig` file is configured to automatically find the `llama.cpp` headers and library within the `vendor` directory created by `build.sh`.

If the build is successful, the executable will be at `./zig-out/bin/qwen_cli`.

## Running

Execute the compiled program, pointing it to the **GGUF file you created** during the conversion step:

```bash
# Example using the converted file name from the instructions above:
./zig-out/bin/qwen_cli --model models/qwen3-14b-converted.gguf

# Or provide the full path otherwise:
./zig-out/bin/qwen_cli --model /path/to/your/qwen3-14b-converted.gguf
```

You can also specify other options:

```bash
./zig-out/bin/qwen_cli \
  --model /path/to/your/qwen3-14b.Q5_K_M.gguf \
  --system "You are a concise assistant." \
  --temp 0.5
```

Type your messages and press Enter. Use Ctrl+D or type `/quit` to exit.

## Development

The main logic for interacting with `llama.cpp` needs to be implemented in the `// TODO:` sections within `src/qwen_cli.zig`. This involves:
*   Using the `@cImport`ed `llama` functions.
*   Initializing the backend and model context.
*   Tokenizing the formatted prompt.
*   Setting up sampling parameters.
*   Running the inference loop (`llama_decode`/`llama_eval`, `llama_sampling_sample`, `llama_token_to_piece`).
*   Managing context and history.
*   Cleaning up resources (`llama_free`, `llama_free_model`, `llama_backend_free`).
Refer to the `llama.cpp` examples (like `examples/main/main.cpp`) for guidance on using the C API.
