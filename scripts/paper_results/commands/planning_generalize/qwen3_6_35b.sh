#!/usr/bin/env bash
# Reproduce one paper-reported number: planning_generalize_qwen35b_memora_full
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck source=../../../configure_paths.sh
source "$REPO_ROOT/scripts/configure_paths.sh"
exec python3 "$REPO_ROOT/scripts/paper_results/reproduce_results.py" planning_generalize_qwen35b_memora_full --refresh --strict "$@"
