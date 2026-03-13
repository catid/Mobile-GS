#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12 is required but was not found in PATH." >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required but was not found in PATH." >&2
    exit 1
fi

uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(
    .venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available after installing PyTorch nightly.")

arch_list = sorted(
    {
        f"{major}.{minor}"
        for major, minor in (
            torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
        )
    }
)
print(";".join(arch_list))
PY
)}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

uv pip install --python .venv/bin/python --no-build-isolation -r requirements.txt

.venv/bin/python scripts/verify_env.py
