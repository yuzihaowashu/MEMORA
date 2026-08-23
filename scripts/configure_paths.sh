# Source from the repository root before running data or result scripts:
#   source scripts/configure_paths.sh

if [ -z "${MEMORA_ROOT:-}" ]; then
    if [ -n "${BASH_VERSION:-}" ]; then
        _MEMORA_ENV_FILE="${BASH_SOURCE[0]}"
    elif [ -n "${ZSH_VERSION:-}" ]; then
        _MEMORA_ENV_FILE="${(%):-%N}"
    else
        _MEMORA_ENV_FILE="$0"
    fi
    export MEMORA_ROOT="$(cd "$(dirname "$_MEMORA_ENV_FILE")/.." && pwd)"
    unset _MEMORA_ENV_FILE
fi

# Downloaded participant memory, saved outputs, and optional model cache.
# Set MEMORA_DATA_ROOT before sourcing this file when possible.
export MEMORA_DATA_ROOT="${MEMORA_DATA_ROOT:-$MEMORA_ROOT/data}"

export MEMORA_BENCH_DIR="${MEMORA_BENCH_DIR:-$MEMORA_ROOT/src/memora_bench}"
export MEMORA_PLANNING_BENCH_DIR="${MEMORA_PLANNING_BENCH_DIR:-$MEMORA_BENCH_DIR/planning}"
export EAM_QA_BENCH_DIR="${EAM_QA_BENCH_DIR:-$MEMORA_BENCH_DIR/eam_qa/questions}"
if [ -z "${PLANNING_MEMORY_ROOT:-}" ]; then
    export PLANNING_MEMORY_ROOT="$MEMORA_DATA_ROOT/participant_memory/memora_paper"
fi
