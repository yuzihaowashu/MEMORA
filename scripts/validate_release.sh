#!/usr/bin/env bash
# Validate repository hygiene before a public release.
#   Default: no progress/design md|txt under src/ (allow-list only).
#   --release: also fail if release-excluded or ignored local files exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IGNORE="$ROOT/.releaseignore"
RELEASE_MODE=0
[ "${1:-}" = "--release" ] && RELEASE_MODE=1
fail=0

# Build the scan set from Git rather than a hand-maintained directory list, so
# newly tracked public text cannot silently bypass the terminology checks.
tracked_text=()
while IFS= read -r -d '' rel; do
  [ "$rel" = "scripts/validate_release.sh" ] && continue
  [ -f "$ROOT/$rel" ] || continue
  tracked_text+=("$ROOT/$rel")
done < <(
  git -C "$ROOT" ls-files -z --cached --others --exclude-standard -- \
    '*.py' '*.sh' '*.md' '*.html' '*.css' '*.js' '*.toml' '*.yml' '*.yaml' \
    '*.cff' '*.txt' '*.vtt' '*.svg'
)

# Public code and documentation use paper terminology, not experiment-tracking
# aliases from development. JSON/CSV benchmark content is excluded because it
# may contain ordinary words such as "layered" as part of an answer choice.
internal_terms='\bM[1-7]_[A-Za-z0-9_]+|v3c2|\bv6\b|de[-_ ]?novo(_v2)?|c_plus|expertskill|offline_phase_c|knowledge[_ -]?bases?|\bKBs?\b|socratic|graph2d|typed-layered|layered_raw|LayeredMemory|offline_patterns|viewpoint_log|memora_typed|typed_memory_raw|pre_editing|judged_|participant_semantic_evidence|skill_memory|semantic_profile|routine_profile|participant[_ -]?profiles?|routine[_ -]?records|routine_skills|generated[_ -]?preferences|derived_enrichment|pre[_ -]?curated|object[_ -]?detect|pre[_ -]?detect|detections-dir|skip-detections|--memory-path|video_understanding|\bego4d\b|install_hf_data|quick_check|pipeline/perception|pipeline[./]generalization|pipeline[./]configs|memory_editor[./]flat_memory|memory_agent[./]task_environment|memory_agent/environment|evaluation/planning/environment|analyze_eam_qa_results|OmniVideoProcessor|VLLMOmniVideoProcessor|APIVideoProcessor|MEMORA_FOUR_CHOICE_MODE|all_18pid|with_unanswerable|with_unans|eam_qa_panel|PLAN [A-Z]:'
while IFS= read -r match; do
  [ -z "$match" ] && continue
  echo "[FORBIDDEN internal terminology] $match"
  fail=1
done < <(
  rg -n -i "$internal_terms" "${tracked_text[@]}" 2>/dev/null || true
)
while IFS= read -r match; do
  [ -z "$match" ] && continue
  echo "[FORBIDDEN internal QA label] $match"
  fail=1
done < <(
  rg -n '\bM[1-7]\b' "${tracked_text[@]}" 2>/dev/null || true
)

if [ "$RELEASE_MODE" -eq 1 ]; then
  if [ ! -f "$IGNORE" ]; then
    echo "[FORBIDDEN for release archive] missing .releaseignore"
    fail=1
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="${line// /}"
    [ -z "$line" ] && continue
    if [[ "$line" == *"*"* || "$line" == *"?"* || "$line" == *"["* ]]; then
      while IFS= read -r match; do
        [ -z "$match" ] && continue
        rel="${match#$ROOT/}"
        echo "[FORBIDDEN for release archive] $rel"
        fail=1
      done < <(compgen -G "$ROOT/$line" || true)
    else
      path="$ROOT/$line"
      if [ -e "$path" ]; then
        echo "[FORBIDDEN for release archive] $line"
        fail=1
      fi
    fi
  done < "$IGNORE"

  while read -r status path; do
    [ "$status" = "!!" ] || continue
    case "$path" in
      .secret|.secrets|.memora_secrets|.env|\
      .venv/*|.venv_api/*|\
      data/*|local/*|participant_memory/*|outputs/*|hf_models/*|containers/*|\
      *__pycache__/|*__pycache__/*|*.pyc|*.sif|*.DS_Store)
        echo "[IGNORED local file present] $path"
        fail=1
        ;;
    esac
  done < <(cd "$ROOT" && git status --short --ignored --untracked-files=normal)
fi

# Scan for common work-in-progress note filenames anywhere under src/.
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  case "$base" in
    progress.txt|plan.txt|HANDOFF.md|CHANGES.md|existing_property.md)
      echo "[FORBIDDEN] ${f#$ROOT/}"
      fail=1
      ;;
  esac
done < <(find "$ROOT/src" -type f \( -name '*.md' -o -name '*.txt' \) -print0 2>/dev/null)

# Allowed md/txt under src/ (whitelist)
allowed_md_txt=(
  "src/memora/README.md"
  "src/memora/pipeline/README.md"
  "src/memora/evaluation/README.md"
  "src/memora/memory_agent/README.md"
  "src/memora/memory_agent/tools/README.md"
  "src/memora_bench/README.md"
  "src/memora_bench/eam_qa/README.md"
  "src/memora_bench/planning/suites/README.md"
)

while IFS= read -r -d '' f; do
  rel="${f#$ROOT/}"
  case "$rel" in
    *.egg-info/*) continue ;;
  esac
  ok=0
  for a in "${allowed_md_txt[@]}"; do
    [ "$rel" = "$a" ] && ok=1 && break
  done
  if [ "$ok" -eq 0 ]; then
    echo "[UNLISTED md/txt in src] $rel  (add to whitelist in validate_release.sh or remove)"
    fail=1
  fi
done < <(find "$ROOT/src" -type f \( -name '*.md' -o -name '*.txt' \) -print0 2>/dev/null)

# Forbidden planning JSON outside the canonical suite layout.
pb="$ROOT/src/memora_bench/planning"
stale_pb="$ROOT/src/memora/planning_benchmark"
for stale in \
  "$ROOT/src/memora/benchmarks" \
  "$ROOT/src/memora/memora_bench" \
  "$ROOT/src/memora/memora_planning" \
  "$ROOT/src/memora_bench/memora_planning"; do
  if [ -d "$stale" ]; then
    rel="${stale#$ROOT/}"
    echo "[FORBIDDEN stale benchmark dir] $rel  (use memora_bench/)"
    fail=1
  fi
done
if [ -d "$stale_pb" ]; then
  echo "[FORBIDDEN noncanonical dir] src/memora/planning_benchmark  (use src/memora_bench/planning/)"
  fail=1
fi
if [ -d "$pb" ]; then
  while IFS= read -r -d '' f; do
    rel="${f#$ROOT/}"
    echo "[FORBIDDEN planning file] $rel"
    fail=1
  done < <(find "$pb" -maxdepth 1 -type f \( \
    -name '*_pre_llm_query.json' -o \
    -name 'planning_tasks_raw_*.json' -o \
    -name 'planning_tasks_draft_*.json' -o \
    -name 'planning_tasks_scratch_*.json' -o \
    -name '*_backup.json' -o \
    -name 'planning_tasks.json' -o \
    -name 'planning_tasks_rewrite.json' -o \
    -name 'gt_review_*.json' -o \
    -name 'checkpoint_queries.json' \
    \) -print0 2>/dev/null)
  while IFS= read -r -d '' f; do
    rel="${f#$ROOT/}"
    echo "[FORBIDDEN flat planning name] $rel  (use suites/replay|generalize/)"
    fail=1
  done < <(find "$pb" -maxdepth 1 -type f -name 'planning_tasks_*.json' -print0 2>/dev/null)
fi

if [ -d "$pb" ]; then
  for d in "$pb"/*; do
    [ -d "$d" ] || continue
    case "$(basename "$d")" in
      data|suites) ;;
      *)
        rel="${d#$ROOT/}"
        echo "[FORBIDDEN benchmark work dir] $rel"
        fail=1
        ;;
    esac
  done
  while IFS= read -r -d '' f; do
    rel="${f#$ROOT/}"
    echo "[FORBIDDEN source annotation copy] $rel"
    fail=1
  done < <(find "$pb/data" -maxdepth 1 -type f ! -name 'participant_video_ids.jsonl' -print0 2>/dev/null)
fi

if [ "$fail" -eq 0 ]; then
  if [ "$RELEASE_MODE" -eq 1 ]; then
    echo "OK: strict release check passed ($ROOT)"
  else
    echo "OK: public source and documentation hygiene passed ($ROOT)"
  fi
else
  echo "FAILED: see messages above"
  exit 1
fi
