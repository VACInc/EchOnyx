#!/bin/bash

# Hardware Detection Script
# Outputs hardware information in JSON format

detect_nvidia() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi

    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        name=$(echo "$line" | cut -d',' -f1 | xargs)
        vram=$(echo "$line" | cut -d',' -f2 | xargs)
        vram_gb=$(echo "scale=2; $vram / 1024" | bc)
        echo "    {\"name\": \"$name\", \"vram_gb\": $vram_gb},"
    done
}

detect_amd() {
    if ! command -v rocm-smi >/dev/null 2>&1; then
        return
    fi

    rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU Memory" | while read line; do
        echo "    {\"name\": \"AMD GPU\", \"vram_gb\": 0},"
    done
}

detect_apple() {
    if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
        return
    fi

    chip_name=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    total_mem=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    total_mem_gb=$(echo "scale=2; $total_mem / 1024 / 1024 / 1024" | bc)
    echo "    {\"name\": \"$chip_name\", \"vram_gb\": $total_mem_gb},"
}

get_total_ram() {
    if [ "$(uname -s)" = "Darwin" ]; then
        sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.2f", $1/1024/1024/1024}'
    else
        grep MemTotal /proc/meminfo | awk '{printf "%.2f", $2/1024/1024}'
    fi
}

# Main output
echo "{"
echo "  \"nvidia_gpus\": ["
detect_nvidia | sed '$ s/,$//'
echo "  ],"
echo "  \"amd_gpus\": ["
detect_amd | sed '$ s/,$//'
echo "  ],"
echo "  \"apple_gpus\": ["
detect_apple | sed '$ s/,$//'
echo "  ],"
echo "  \"total_ram_gb\": $(get_total_ram),"

# Determine recommended profile
PROFILE="cpu_only"
BACKEND="cpu"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/1024}')

    BACKEND="cuda"
    if [ "$GPU_COUNT" -gt 1 ]; then
        PROFILE="multi_gpu"
    elif [ "$(echo "$TOTAL_VRAM >= 30" | bc)" -eq 1 ]; then
        PROFILE="rtx_5090"
    else
        PROFILE="multi_gpu"
    fi
elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    PROFILE="apple_silicon"
    BACKEND="metal"
elif [ -d "/dev/dri" ]; then
    TOTAL_RAM=$(get_total_ram)
    if [ "$(echo "$TOTAL_RAM >= 96" | bc)" -eq 1 ]; then
        PROFILE="strix_halo"
        BACKEND="rocm"
    elif command -v rocm-smi >/dev/null 2>&1; then
        PROFILE="multi_gpu"
        BACKEND="rocm"
    fi
fi

echo "  \"recommended_profile\": \"$PROFILE\","
echo "  \"recommended_backend\": \"$BACKEND\""
echo "}"
