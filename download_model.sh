#!/bin/bash
set -e

# --- Configuration ---
# Downloads pre-converted Qwen3-14B GGUF models from the ggml-org repository.
REPO_ID="ggml-org/Qwen3-14B-GGUF"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

# Files to download from the repository (GGUF format)
# Choose which ones you want by commenting/uncommenting
FILES=(
    "Qwen3-14B-Q4_K_M.gguf" # ~9 GB, Good balance
    "Qwen3-14B-Q8_0.gguf"   # ~15.7 GB, Higher quality quantization
    "Qwen3-14B-f16.gguf"    # ~29.5 GB, Full FP16 precision (large)
    # Add other quantizations if available/needed
)

# Destination directory for GGUF files
MODEL_DIR="models/Qwen3-14B-GGUF"

# --- Helper Functions ---
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --- Dependency Check ---
DOWNLOADER=""
if command_exists wget; then
    DOWNLOADER="wget"
elif command_exists curl; then
    DOWNLOADER="curl"
else
    echo "Error: Neither wget nor curl is installed. Please install one."
    exit 1
fi
echo "Using ${DOWNLOADER} to download."

# --- Download ---
echo "Creating directory ${MODEL_DIR}..."
mkdir -p "$MODEL_DIR"

for FILE in "${FILES[@]}"; do
    DEST_PATH="${MODEL_DIR}/${FILE}"
    MODEL_URL="${BASE_URL}/${FILE}?download=true" # GGUF files are usually LFS

    if [ -f "$DEST_PATH" ]; then
        echo "File already exists at ${DEST_PATH}. Skipping download."
    else
        echo "Downloading ${FILE} to ${DEST_PATH}..."
        if [ "$DOWNLOADER" = "wget" ]; then
            wget --show-progress -O "$DEST_PATH" "$MODEL_URL"
        elif [ "$DOWNLOADER" = "curl" ]; then
            # Use -C - to resume downloads if possible
            curl -L -C - -o "$DEST_PATH" "$MODEL_URL"
        fi

        if [ $? -ne 0 ]; then
            echo "Error: Download failed for ${FILE}."
            # Optional: remove partial file
            # rm -f "$DEST_PATH"
            # Decide if you want to exit on first failure or try downloading others
            # exit 1
        else
             echo "Downloaded ${FILE} successfully."
        fi
    fi
done


echo ""
echo "--------------------------------------------------"
echo "GGUF model files downloaded to: ${MODEL_DIR}"
echo ""
echo "These files are ready to be used directly with the application."
echo "Example run command (choose one of the downloaded files):"
echo ""
echo "./zig-out/bin/qwen_cli --model ${MODEL_DIR}/Qwen3-14B-Q4_K_M.gguf"
# echo "./zig-out/bin/qwen_cli --model ${MODEL_DIR}/Qwen3-14B-Q8_0.gguf"
# echo "./zig-out/bin/qwen_cli --model ${MODEL_DIR}/Qwen3-14B-f16.gguf"
echo "--------------------------------------------------"

exit 0
