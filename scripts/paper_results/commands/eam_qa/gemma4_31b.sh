#!/usr/bin/env bash
# Reproduce one paper-reported number: eam_qa_memora_gemma31b
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck source=../../../configure_paths.sh
source "$REPO_ROOT/scripts/configure_paths.sh"
exec python3 "$REPO_ROOT/scripts/paper_results/reproduce_results.py" eam_qa_memora_gemma31b --refresh --strict "$@"
