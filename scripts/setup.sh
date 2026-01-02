#!/bin/bash
set -e

# Video Summarizer Setup Script
# Detects hardware and configures the environment

echo "========================================"
echo "  Video Summarizer Setup"
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

    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v docker-compose >/dev/null 2>&1 || command -v "docker compose" >/dev/null 2>&1 || missing+=("docker-compose")

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

    # Check for NVIDIA GPUs
    if command -v nvidia-smi >/dev/null 2>&1; then
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
                    GPU_BACKEND="vulkan"
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
    echo "Compose File: $COMPOSE_FILE"
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
    sed -i "s/^HARDWARE_PROFILE=.*/HARDWARE_PROFILE=$HARDWARE_PROFILE/" .env
    sed -i "s/^GPU_BACKEND=.*/GPU_BACKEND=$GPU_BACKEND/" .env

    # Set model loading strategy
    if [ "$HARDWARE_PROFILE" = "strix_halo" ] || [ "$HARDWARE_PROFILE" = "cpu_only" ]; then
        sed -i "s/^MODEL_LOADING=.*/MODEL_LOADING=sequential/" .env
    else
        sed -i "s/^MODEL_LOADING=.*/MODEL_LOADING=parallel/" .env
    fi

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
    echo "  docker compose $COMPOSE_FILE up -d"
    echo ""
    echo "Then open http://localhost:3000 in your browser."
    echo ""
    echo "To view logs:"
    echo "  docker compose logs -f"
    echo ""
    echo "To stop:"
    echo "  docker compose down"
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
