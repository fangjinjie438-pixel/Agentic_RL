#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "$(uname -s)" == Linux ]] || { echo 'This installer requires Linux.' >&2; exit 1; }
[[ -n "${CONDA_PREFIX:-}${VIRTUAL_ENV:-}" ]] || { echo 'Activate a dedicated Python 3.9 conda/venv environment first.' >&2; exit 1; }
python -c 'import sys; assert sys.version_info[:2] == (3,9), "Use Python 3.9"'
python scripts/bootstrap_search_r1.py
python -m pip install 'pip==24.3.1' 'setuptools==75.3.0' 'wheel==0.44.0'
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/linux-candidate.txt
if [[ -n "${FLASH_ATTN_WHEEL:-}" ]]; then
  [[ -f "$FLASH_ATTN_WHEEL" ]] || { echo 'FLASH_ATTN_WHEEL must identify an existing local wheel.' >&2; exit 1; }
  python -m pip install --no-deps "$FLASH_ATTN_WHEEL"
elif [[ "${BUILD_FLASH_ATTN:-0}" == 1 ]]; then
  [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] || { echo 'Set CUDA_HOME to a CUDA 12.1 toolkit with nvcc.' >&2; exit 1; }
  "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.1' || { echo 'This recipe requires a CUDA 12.1 toolkit for building FlashAttention.' >&2; exit 1; }
  export MAX_JOBS="${MAX_JOBS:-4}"
  python -m pip install 'flash-attn==2.7.0.post2' --no-build-isolation --no-deps
else
  echo 'FlashAttention is required by the actor. Set FLASH_ATTN_WHEEL to a compatible local wheel, or BUILD_FLASH_ATTN=1 with CUDA_HOME pointing to a CUDA 12.1 toolkit, then rerun.' >&2
  exit 1
fi
python -c 'import torch, flash_attn; assert torch.version.cuda == "12.1"; assert flash_attn.__version__ == "2.7.0.post2"; print(torch.__version__, flash_attn.__version__)'
python -m pip install --no-deps -e third_party/Search-R1
python -m pip install --no-deps -e .
python -m pip check
mkdir -p outputs/environment
python -m pip freeze > outputs/environment/pip-freeze.txt
python -c 'import platform, torch; print(platform.platform()); print(torch.__config__.show())' > outputs/environment/torch-build.txt
echo 'Installed candidate stack. Complete preflight and a GPU smoke test before treating it as validated.'
