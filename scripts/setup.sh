#!/bin/bash
set -e

# EchOnyx Setup Script
# Detects hardware and configures the environment

echo "========================================"
echo "  EchOnyx Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_requirements() {
    echo -e "\n${YELLOW}Checking requirements...${NC}"

    local missing=()
    command -v python3 >/dev/null 2>&1 || missing+=("python3")

    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        command -v ffmpeg >/dev/null 2>&1 || missing+=("ffmpeg")
    else
        command -v docker >/dev/null 2>&1 || missing+=("docker")
        command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1 || missing+=("docker-compose")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        echo -e "${RED}Missing required tools: ${missing[*]}${NC}"
        echo "Please install them and run this script again."
        exit 1
    fi

    echo -e "${GREEN}All requirements satisfied${NC}"
}

# Detect hardware
detect_hardware() {
    echo -e "\n${YELLOW}Detecting hardware...${NC}"

    HARDWARE_PROFILE=""
    GPU_BACKEND=""
    COMPOSE_FILE="docker-compose.yml"
    START_MODE="docker"

    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        echo -e "${GREEN}Detected Apple Silicon${NC}"
        HARDWARE_PROFILE="apple_silicon"
        GPU_BACKEND="metal"
        START_MODE="host"
        COMPOSE_FILE=""
    fi

    # Check for NVIDIA GPUs
    if [ -z "$GPU_BACKEND" ] && command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/1024}')

        echo -e "${GREEN}Detected ${GPU_COUNT} NVIDIA GPU(s): ${GPU_NAMES}${NC}"
        echo "Total VRAM: ${TOTAL_VRAM}GB"

        GPU_BACKEND="cuda"
        COMPOSE_FILE="docker-compose.yml -f docker-compose.nvidia.yml"

        if [ "$GPU_COUNT" -gt 1 ]; then
            HARDWARE_PROFILE="multi_gpu"
        elif echo "$GPU_NAMES" | grep -qi "5090"; then
            HARDWARE_PROFILE="rtx_5090"
        else
            HARDWARE_PROFILE="multi_gpu"
        fi
    fi

    # Check for AMD GPUs / Strix Halo
    if [ -z "$GPU_BACKEND" ]; then
        if [ -d "/dev/dri" ]; then
            # Check for ROCm
            if command -v rocm-smi >/dev/null 2>&1; then
                echo -e "${GREEN}Detected AMD GPU with ROCm${NC}"
                GPU_BACKEND="rocm"
                COMPOSE_FILE="docker-compose.yml -f docker-compose.amd.yml"
                HARDWARE_PROFILE="multi_gpu"
            else
                # Check for high RAM (Strix Halo indicator)
                TOTAL_RAM=$(grep MemTotal /proc/meminfo | awk '{print $2/1024/1024}')
                if (( $(echo "$TOTAL_RAM >= 96" | bc -l) )); then
                    echo -e "${GREEN}Detected AMD Strix Halo (high unified memory: ${TOTAL_RAM}GB)${NC}"
                    GPU_BACKEND="rocm"
                    COMPOSE_FILE="docker-compose.yml -f docker-compose.amd.yml"
                    HARDWARE_PROFILE="strix_halo"
                fi
            fi
        fi
    fi

    # Fallback to CPU
    if [ -z "$GPU_BACKEND" ]; then
        echo -e "${YELLOW}No GPU detected, using CPU mode${NC}"
        GPU_BACKEND="cpu"
        HARDWARE_PROFILE="cpu_only"
    fi

    echo ""
    echo "Hardware Profile: $HARDWARE_PROFILE"
    echo "GPU Backend: $GPU_BACKEND"
    if [ "$START_MODE" = "docker" ]; then
        echo "Compose File: $COMPOSE_FILE"
    else
        echo "Start Mode: host"
    fi
}

# Create .env file
create_env() {
    echo -e "\n${YELLOW}Creating .env file...${NC}"

    if [ -f ".env" ]; then
        echo -e "${YELLOW}.env file already exists. Backing up to .env.backup${NC}"
        cp .env .env.backup
    fi

    cp .env.example .env

    # Update hardware settings
    python3 - <<PY
from pathlib import Path
path = Path(".env")
payload = path.read_text(encoding="utf-8")
updates = {
    "HARDWARE_PROFILE": "$HARDWARE_PROFILE",
    "GPU_BACKEND": "$GPU_BACKEND",
    "MODEL_LOADING": "sequential" if "$HARDWARE_PROFILE" in {"strix_halo", "cpu_only", "apple_silicon"} else "parallel",
}
if "$HARDWARE_PROFILE" == "apple_silicon":
    updates.update({
        "WHISPER_MODEL": "small",
        "EMBEDDING_MODEL": "nomic-ai/nomic-embed-text-v1.5",
        "VISION_MODEL": "Qwen2.5-VL-3B-Instruct.Q4_K_M.gguf",
        "VISION_MMPROJ": "Qwen2.5-VL-3B-Instruct.mmproj-fp16.gguf",
        "VISION_CHAT_FORMAT": "qwen2.5-vl",
        "SUMMARIZATION_MODEL": "Qwen2.5-3B-Instruct.Q4_K_M.gguf",
        "GPU_MEMORY_FRACTION": "0.65",
    })
for key, value in updates.items():
    lines = payload.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    payload = "\\n".join(lines) + "\\n"
path.write_text(payload, encoding="utf-8")
PY

    echo -e "${GREEN}.env file created${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT: You need to set your Hugging Face token in .env${NC}"
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo "Then accept the model agreements at:"
    echo "  - https://huggingface.co/pyannote/speaker-diarization-community-1"
    echo "  - https://huggingface.co/pyannote/segmentation-3.0"
}

# Create data directories
create_directories() {
    echo -e "\n${YELLOW}Creating data directories...${NC}"

    mkdir -p data/uploads data/models data/chroma

    echo -e "${GREEN}Directories created${NC}"
}

# Download models
download_models() {
    echo -e "\n${YELLOW}Would you like to download models now? (y/n)${NC}"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        ./scripts/download-models.sh
    else
        echo "You can download models later with: ./scripts/download-models.sh"
    fi
}

# Print startup instructions
print_instructions() {
    echo ""
    echo "========================================"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo "========================================"
    echo ""
    echo "To start the application:"
    echo ""
    if [ "$START_MODE" = "docker" ]; then
        echo "  docker compose $COMPOSE_FILE up -d"
    else
        echo "  Metal on Apple Silicon runs on the host, not Docker."
        echo "  Follow backend/README.md for the host startup commands."
        echo "  Use Celery --pool=solo for the host worker on Apple Silicon."
    fi
    echo ""
    echo "Then open http://localhost:3000 in your browser."
    echo ""
    echo "To view logs:"
    if [ "$START_MODE" = "docker" ]; then
        echo "  docker compose logs -f"
    else
        echo "  tail -f /tmp/echonyx-backend.log /tmp/echonyx-worker.log"
    fi
    echo ""
    echo "To stop:"
    if [ "$START_MODE" = "docker" ]; then
        echo "  docker compose down"
    else
        echo "  pkill -f 'uvicorn app.main:app' && pkill -f 'celery -A app.workers.celery_app worker'"
    fi
    echo ""
}

# Main
main() {
    cd "$(dirname "$0")/.."

    check_requirements
    detect_hardware
    create_env
    create_directories
    # download_models  # Optional, can be slow
    print_instructions
}

main "$@"
