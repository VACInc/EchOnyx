#!/bin/bash
set -e

# Model Download Script
# Downloads required GGUF models for video summarization

echo "========================================"
echo "  Model Download Script"
echo "========================================"

MODEL_DIR="${MODEL_CACHE_DIR:-./data/models}"
mkdir -p "$MODEL_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for huggingface-cli
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install huggingface_hub
fi

echo -e "\n${YELLOW}Downloading models to: $MODEL_DIR${NC}\n"

# Whisper models are downloaded automatically by faster-whisper
echo -e "${GREEN}[1/4] Whisper models${NC}"
echo "  Whisper models will be downloaded automatically on first use."
echo ""

# Download Qwen3 GGUF models
echo -e "${GREEN}[2/4] Qwen3-30B-A3B (Summarization)${NC}"
echo "  Downloading Q4_K_M quantization (~15GB)..."

# Note: Update these URLs when GGUF versions become available
# For now, we'll use placeholder logic
if [ ! -f "$MODEL_DIR/qwen3-30b-a3b-q4_k_m.gguf" ]; then
    echo -e "${YELLOW}  Model not found locally.${NC}"
    echo "  Please download manually from Hugging Face:"
    echo "  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF"
    echo ""
    echo "  Place the Q4_K_M.gguf file in: $MODEL_DIR/qwen3-30b-a3b-q4_k_m.gguf"
else
    echo "  Already downloaded."
fi
echo ""

# Download Qwen3-Omni for vision
echo -e "${GREEN}[3/4] Qwen3-Omni-30B-A3B (Vision)${NC}"
echo "  Downloading Q4_K_M quantization (~15GB)..."

if [ ! -f "$MODEL_DIR/qwen3-omni-30b-a3b-q4_k_m.gguf" ]; then
    echo -e "${YELLOW}  Model not found locally.${NC}"
    echo "  Please download manually from Hugging Face:"
    echo "  https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-GGUF"
    echo ""
    echo "  Place the Q4_K_M.gguf file in: $MODEL_DIR/qwen3-omni-30b-a3b-q4_k_m.gguf"
else
    echo "  Already downloaded."
fi
echo ""

# Embedding model
echo -e "${GREEN}[4/4] Nomic Embed Text (Embeddings)${NC}"
echo "  Will be downloaded automatically by sentence-transformers on first use."
echo ""

# pyannote models require HF token
echo "========================================"
echo -e "${YELLOW}Note about pyannote models:${NC}"
echo "========================================"
echo ""
echo "Speaker diarization models require a Hugging Face token."
echo "1. Get your token at: https://huggingface.co/settings/tokens"
echo "2. Accept the model agreements:"
echo "   - https://huggingface.co/pyannote/speaker-diarization-community-1"
echo "   - https://huggingface.co/pyannote/segmentation-3.0"
echo "3. Set HF_TOKEN in your .env file"
echo ""
echo "Models will be downloaded automatically on first use."
echo ""

echo "========================================"
echo -e "${GREEN}  Download script complete${NC}"
echo "========================================"
