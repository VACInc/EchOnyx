import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_detect_hardware(tmp_path, memtotal_kb: int) -> str:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    bc = fake_bin / "bc"
    bc.write_text(
        """#!/bin/sh
python3 -c '
import sys
expr = sys.stdin.read().strip()
left, right = expr.split(">=")
print(1 if float(left.strip()) >= float(right.strip()) else 0)
'
""",
        encoding="utf-8",
    )
    bc.chmod(0o755)
    dri_path = tmp_path / "dri"
    dri_path.mkdir()
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text(f"MemTotal:       {memtotal_kb} kB\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "ECHONYX_DRI_PATH": str(dri_path),
            "ECHONYX_MEMINFO_PATH": str(meminfo_path),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SETUP_SCRIPT": str(ROOT / "scripts" / "setup.sh"),
        }
    )
    harness = r"""
set -e
command() {
    if [ "$1" = "-v" ]; then
        case "$2" in
            nvidia-smi) return 1 ;;
            rocm-smi) return 0 ;;
        esac
    fi
    builtin command "$@"
}
source <(sed '/^# Main/,$d' "$SETUP_SCRIPT")
detect_hardware
printf 'PROFILE=%s\nBACKEND=%s\n' "$HARDWARE_PROFILE" "$GPU_BACKEND"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout


def test_detect_hardware_classifies_high_ram_rocm_as_strix_halo(tmp_path):
    output = _run_detect_hardware(tmp_path, memtotal_kb=128 * 1024 * 1024)

    assert "PROFILE=strix_halo" in output
    assert "BACKEND=rocm" in output


def test_detect_hardware_classifies_lower_ram_rocm_as_multi_gpu(tmp_path):
    output = _run_detect_hardware(tmp_path, memtotal_kb=64 * 1024 * 1024)

    assert "PROFILE=multi_gpu" in output
    assert "BACKEND=rocm" in output
