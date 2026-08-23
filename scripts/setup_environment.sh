#!/usr/bin/env bash
# Create .venv and install MEMORA dependencies.
#   bash scripts/setup_environment.sh analysis   # analyze saved outputs (no GPU)
#   bash scripts/setup_environment.sh gpu        # + vLLM 0.19 paper stack
set -euo pipefail

MODE="${1:-analysis}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ "$MODE" != "analysis" ] && [ "$MODE" != "gpu" ]; then
  echo "Usage: bash scripts/setup_environment.sh {analysis|gpu}"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN to a Python 3.10-3.12 executable."
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))'; then
  PYTHON_VERSION="$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
  echo "ERROR: MEMORA supports Python 3.10-3.12; $PYTHON_BIN is Python $PYTHON_VERSION."
  echo "Choose a supported interpreter, for example:"
  echo "  PYTHON_BIN=python3.12 bash scripts/setup_environment.sh $MODE"
  exit 1
fi

echo "Using $PYTHON_BIN ($($PYTHON_BIN -c 'import platform; print(platform.python_version())'))"
"$PYTHON_BIN" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -U pip wheel setuptools

if [ "$MODE" = "analysis" ]; then
  pip install -e ".[analysis]" --no-build-isolation
else
  echo ""
  echo "Installing the full paper stack (vLLM 0.19.1)..."
  if ! pip install -e ".[analysis,gpu]" --no-build-isolation; then
    echo ""
    echo "WARN: GPU dependency installation failed."
    echo "      See docs/SETUP.md and https://docs.vllm.ai/en/v0.19.1/getting_started/installation.html"
    exit 1
  fi
  echo ""
  echo "GPU stack installed. Verify with:"
  echo "  python3 -c \"import vllm; print(vllm.__version__)\"  # expect 0.19.1"
fi

echo ""
echo "OK: venv ready ($MODE). Next:"
echo "  source .venv/bin/activate"
echo "  export MEMORA_DATA_ROOT=...  HF_HOME=...  HF_TOKEN=..."
echo "  source scripts/configure_paths.sh"
if [ "$MODE" = "analysis" ]; then
  echo "  bash scripts/data/verify.sh"
else
  echo "  memora-eam-qa --help"
fi
echo ""
echo "Full guide: docs/SETUP.md"
