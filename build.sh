#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
LLAMA_CPP_DIR="vendor/llama.cpp"
LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_DIR}/build"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"

# Add CMake flags here for custom builds (e.g., GPU support)
# Build as a shared library
CMAKE_FLAGS="-DBUILD_SHARED_LIBS=ON -DLLAMA_CUDA=ON -DLLAMA_FLASH_ATTN=ON" # Enable CUDA and Flash Attention

# --- Helper Functions ---
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --- Dependency Checks ---
echo "Checking dependencies..."
if ! command_exists git; then
    echo "Error: git is not installed. Please install it (e.g., 'sudo pacman -S git')."
    exit 1
fi
if ! command_exists cmake; then
    echo "Error: cmake is not installed. Please install it (e.g., 'sudo pacman -S cmake')."
    exit 1
fi
if ! command_exists cc; then
    echo "Error: C compiler (gcc or clang) not found. Please install base-devel (e.g., 'sudo pacman -S base-devel')."
    exit 1
fi
echo "Dependencies found."

# --- Clone llama.cpp ---
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp repository into ${LLAMA_CPP_DIR}..."
    git clone --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
else
    echo "llama.cpp directory already exists at ${LLAMA_CPP_DIR}. Skipping clone."
    # Optional: Add logic here to pull latest changes if desired
    # (cd "$LLAMA_CPP_DIR" && git pull)
fi

# --- Build llama.cpp ---
echo "Configuring and building llama.cpp..."
mkdir -p "$LLAMA_CPP_BUILD_DIR"
cd "$LLAMA_CPP_BUILD_DIR"

# Run CMake configuration
echo "Running CMake with flags: ${CMAKE_FLAGS}"
cmake .. ${CMAKE_FLAGS}

# Run the build
# Use $(nproc) to get number of processors for parallel build, fallback to 1 if nproc not found
NPROC=$(nproc 2>/dev/null || echo 1)
echo "Building with ${NPROC} jobs..."
cmake --build . --config Release -- -j${NPROC}

cd ../../ # Go back to the project root directory

echo ""
echo "--------------------------------------------------"
echo "llama.cpp shared build complete!"
echo "Header directory: ${LLAMA_CPP_DIR}"
echo "Library directory: ${LLAMA_CPP_BUILD_DIR}/bin" # Shared lib is in build/bin
echo ""
echo "You can now build the Zig application using:"
echo ""
echo "zig build"
echo "--------------------------------------------------"

exit 0
