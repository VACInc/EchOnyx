#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:8000"
PRIMARY_FIXTURE=""
SECONDARY_FIXTURE=""
SEARCH_QUERY=""
ASK_QUESTION=""
ASK_EXPECTS=""
GPU_IDLE_COMMAND=""
READ_ONLY=0
RUN_BATCH=0
TIMEOUT_SECONDS=900
POLL_INTERVAL_SECONDS=5
TITLE_PREFIX="acceptance-$(date +%Y%m%d-%H%M%S)"
BATCH_FIXTURES=()

usage() {
  cat <<'EOF'
Usage: scripts/acceptance.sh [options]

Options:
  --base-url URL              API base URL. Default: http://127.0.0.1:8000
  --primary-fixture PATH      Primary video fixture for single-upload checks.
  --secondary-fixture PATH    Secondary fixture for similarity checks.
  --batch-fixture PATH        Fixture to include in batch upload. Repeatable.
  --search-query TEXT         Query for /api/search against the primary video.
  --ask-question TEXT         Question for /api/search/ask against the primary video.
  --ask-expects TEXT          Substring expected in the ask answer.
  --gpu-idle-command CMD      Optional shell command to print GPU idle state.
  --timeout SECONDS           Overall wait timeout for video/batch completion.
  --poll-interval SECONDS     Poll interval while waiting for jobs.
  --title-prefix TEXT         Title prefix for uploaded videos/batches.
  --run-batch                 Run batch upload acceptance.
  --read-only                 Health/settings/models checks only. No uploads.
  --help                      Show this help.
EOF
}

log() {
  printf '[acceptance] %s\n' "$*"
}

fail() {
  printf '[acceptance] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

urlencode() {
  jq -rn --arg value "$1" '$value|@uri'
}

mime_type_for() {
  case "${1##*.}" in
    mp4|MP4) echo "video/mp4" ;;
    mov|MOV) echo "video/quicktime" ;;
    webm|WEBM) echo "video/webm" ;;
    mkv|MKV) echo "video/x-matroska" ;;
    avi|AVI) echo "video/x-msvideo" ;;
    *) echo "application/octet-stream" ;;
  esac
}

api_get() {
  curl -fsS "${BASE_URL}$1"
}

api_post_json() {
  local endpoint="$1"
  local payload="$2"
  curl -fsS -X POST \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "${BASE_URL}${endpoint}"
}

api_put_json() {
  local endpoint="$1"
  local payload="$2"
  curl -fsS -X PUT \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "${BASE_URL}${endpoint}"
}

api_delete() {
  curl -fsS -X DELETE "${BASE_URL}$1"
}

upload_video() {
  local fixture="$1"
  local title="$2"
  local mime
  mime="$(mime_type_for "$fixture")"
  curl -fsS \
    -F "file=@${fixture};type=${mime}" \
    -F "title=${title}" \
    "${BASE_URL}/api/videos/upload"
}

upload_batch() {
  local name="$1"
  shift
  local args=()
  local fixture
  for fixture in "$@"; do
    args+=(-F "files=@${fixture};type=$(mime_type_for "$fixture")")
  done
  args+=(-F "name=${name}")
  curl -fsS "${args[@]}" "${BASE_URL}/api/batch"
}

assert_runtime_endpoints() {
  log "runtime settings"
  api_get "/api/settings" | jq -e '.runtime_planner != null and .models != null' >/dev/null \
    || fail "/api/settings missing runtime planner or models"
  log "hardware settings"
  api_get "/api/settings/hardware" | jq -e '.active_profile != null and .active_backend != null and .runtime_plan != null' >/dev/null \
    || fail "/api/settings/hardware missing expected fields"
}

poll_video() {
  local video_id="$1"
  local started_at
  started_at="$(date +%s)"

  while true; do
    local video_json job_json status current_step progress error_message now
    video_json="$(api_get "/api/videos/${video_id}")"
    job_json="$(api_get "/api/jobs?video_id=${video_id}&page_size=1")"
    status="$(printf '%s' "$video_json" | jq -r '.status')"
    current_step="$(printf '%s' "$job_json" | jq -r '.jobs[0].current_step // "pending"')"
    progress="$(printf '%s' "$job_json" | jq -r '.jobs[0].progress // 0')"
    error_message="$(printf '%s' "$job_json" | jq -r '.jobs[0].error_message // empty')"

    log "video ${video_id} status=${status} step=${current_step} progress=${progress}"

    if [[ "$status" == "completed" ]]; then
      return 0
    fi
    if [[ "$status" == "failed" ]]; then
      fail "video ${video_id} failed at ${current_step}: ${error_message}"
    fi

    now="$(date +%s)"
    if (( now - started_at > TIMEOUT_SECONDS )); then
      fail "video ${video_id} timed out after ${TIMEOUT_SECONDS}s"
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
}

poll_batch() {
  local batch_id="$1"
  local started_at
  started_at="$(date +%s)"

  while true; do
    local batch_json status progress now
    batch_json="$(api_get "/api/batch/${batch_id}")"
    status="$(printf '%s' "$batch_json" | jq -r '.status')"
    progress="$(printf '%s' "$batch_json" | jq -r '.progress')"

    log "batch ${batch_id} status=${status} progress=${progress}"

    if [[ "$status" == "completed" ]]; then
      return 0
    fi
    if [[ "$status" == "failed" || "$status" == "cancelled" ]]; then
      fail "batch ${batch_id} ended with status=${status}"
    fi

    now="$(date +%s)"
    if (( now - started_at > TIMEOUT_SECONDS )); then
      fail "batch ${batch_id} timed out after ${TIMEOUT_SECONDS}s"
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
}

assert_jobs_list() {
  local video_id="$1"
  api_get "/api/jobs?video_id=${video_id}&page_size=5" | jq -e '.total >= 1 and (.jobs | length) >= 1' >/dev/null \
    || fail "jobs listing missing rows for ${video_id}"
  log "jobs listing ok for ${video_id}"
}

assert_summary() {
  local video_id="$1"
  local summary_json
  summary_json="$(api_get "/api/summaries/${video_id}")"
  printf '%s' "$summary_json" | jq -e '.summary != null and (.summary.executive_summary | length) > 0' >/dev/null \
    || fail "summary missing for video ${video_id}"
  log "summary ok for ${video_id}"
}

update_video_tags() {
  local video_id="$1"
  local payload="$2"
  api_put_json "/api/videos/${video_id}/tags" "$payload"
}

create_action_item() {
  local payload="$1"
  api_post_json "/api/action-items" "$payload"
}

update_action_item_api() {
  local action_item_id="$1"
  local payload="$2"
  api_put_json "/api/action-items/${action_item_id}" "$payload"
}

delete_action_item_api() {
  local action_item_id="$1"
  api_delete "/api/action-items/${action_item_id}"
}

assert_action_items() {
  local video_id="$1"
  local tag="acceptance"
  local item_id open_json completed_json deleted_json

  update_video_tags "$video_id" "$(jq -cn --arg tag "$tag" '{tags:[$tag]}')" >/dev/null

  item_id="$(create_action_item "$(jq -cn --arg video_id "$video_id" '{video_id:$video_id, text:"Follow up on acceptance flow", source:"manual"}')" | jq -r '.id')"
  [[ -n "$item_id" && "$item_id" != "null" ]] || fail "action item create failed for ${video_id}"

  open_json="$(api_get "/api/action-items?video_id=${video_id}&status=open&tags=${tag}")"
  printf '%s' "$open_json" | jq -e --arg item_id "$item_id" '.total >= 1 and (.items[] | select(.id == $item_id and .completed == false))' >/dev/null \
    || fail "open action item missing for ${video_id}"

  update_action_item_api "$item_id" '{"completed":true}' >/dev/null
  completed_json="$(api_get "/api/action-items?video_id=${video_id}&status=completed&tags=${tag}")"
  printf '%s' "$completed_json" | jq -e --arg item_id "$item_id" '.total >= 1 and (.items[] | select(.id == $item_id and .completed == true))' >/dev/null \
    || fail "completed action item missing for ${video_id}"

  delete_action_item_api "$item_id" >/dev/null
  deleted_json="$(api_get "/api/action-items?video_id=${video_id}&status=all&tags=${tag}")"
  printf '%s' "$deleted_json" | jq -e --arg item_id "$item_id" 'all(.items[]?; .id != $item_id)' >/dev/null \
    || fail "deleted action item still present for ${video_id}"
  log "action items ok for ${video_id}"
}

assert_search() {
  local query="$1"
  local video_id="$2"
  local encoded_query encoded_video search_json
  encoded_query="$(urlencode "$query")"
  encoded_video="$(urlencode "$video_id")"
  search_json="$(api_get "/api/search?q=${encoded_query}&video_id=${encoded_video}")"
  printf '%s' "$search_json" | jq -e '.total > 0 and (.results | length) > 0' >/dev/null \
    || fail "search returned no results for query=${query}"
  log "search ok for query=${query}"
}

assert_ask() {
  local question="$1"
  local video_id="$2"
  local expects="$3"
  local payload answer ask_json
  payload="$(jq -cn --arg question "$question" --arg video_id "$video_id" '{question:$question, video_ids:[$video_id]}')"
  ask_json="$(api_post_json "/api/search/ask" "$payload")"
  answer="$(printf '%s' "$ask_json" | jq -r '.answer')"
  [[ -n "$answer" && "$answer" != "null" ]] || fail "ask returned empty answer"
  if [[ -n "$expects" ]]; then
    printf '%s' "$answer" | grep -Fqi "$expects" || fail "ask answer did not contain expected text: $expects"
  fi
  log "ask ok"
}

assert_similar() {
  local source_video_id="$1"
  local expected_video_id="${2:-}"
  local similar_json results_count
  similar_json="$(api_get "/api/search/similar/${source_video_id}?limit=5")"
  results_count="$(printf '%s' "$similar_json" | jq -r '.results | length')"
  [[ "$results_count" -gt 0 ]] || fail "similar returned no results for ${source_video_id}"
  if [[ -n "$expected_video_id" ]]; then
    printf '%s' "$similar_json" | jq -e --arg video_id "$expected_video_id" '.results[] | select(.video_id == $video_id)' >/dev/null \
      || fail "similar did not include expected video ${expected_video_id}"
  fi
  log "similar ok for ${source_video_id}"
}

run_gpu_idle_probe() {
  [[ -n "$GPU_IDLE_COMMAND" ]] || return 0
  log "gpu idle probe"
  bash -lc "$GPU_IDLE_COMMAND"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --primary-fixture)
      PRIMARY_FIXTURE="$2"
      shift 2
      ;;
    --secondary-fixture)
      SECONDARY_FIXTURE="$2"
      shift 2
      ;;
    --batch-fixture)
      BATCH_FIXTURES+=("$2")
      shift 2
      ;;
    --search-query)
      SEARCH_QUERY="$2"
      shift 2
      ;;
    --ask-question)
      ASK_QUESTION="$2"
      shift 2
      ;;
    --ask-expects)
      ASK_EXPECTS="$2"
      shift 2
      ;;
    --gpu-idle-command)
      GPU_IDLE_COMMAND="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --title-prefix)
      TITLE_PREFIX="$2"
      shift 2
      ;;
    --run-batch)
      RUN_BATCH=1
      shift
      ;;
    --read-only)
      READ_ONLY=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

require_cmd curl
require_cmd jq

log "health check ${BASE_URL}"
api_get "/health" | jq .
assert_runtime_endpoints
log "model status"
api_get "/api/settings/models/status" | jq .
run_gpu_idle_probe

if (( READ_ONLY )); then
  log "read-only acceptance complete"
  exit 0
fi

[[ -n "$PRIMARY_FIXTURE" ]] || fail "--primary-fixture is required unless --read-only is used"
[[ -f "$PRIMARY_FIXTURE" ]] || fail "Primary fixture not found: $PRIMARY_FIXTURE"

primary_upload="$(upload_video "$PRIMARY_FIXTURE" "${TITLE_PREFIX}-primary")"
primary_video_id="$(printf '%s' "$primary_upload" | jq -r '.id')"
log "uploaded primary video ${primary_video_id}"
poll_video "$primary_video_id"
assert_jobs_list "$primary_video_id"
assert_summary "$primary_video_id"
assert_action_items "$primary_video_id"

if [[ -n "$SEARCH_QUERY" ]]; then
  api_post_json "/api/search/warm" '{"mode":"search"}' >/dev/null || true
  assert_search "$SEARCH_QUERY" "$primary_video_id"
fi

if [[ -n "$ASK_QUESTION" ]]; then
  api_post_json "/api/search/warm" '{"mode":"ask"}' >/dev/null || true
  assert_ask "$ASK_QUESTION" "$primary_video_id" "$ASK_EXPECTS"
fi

secondary_video_id=""
if [[ -n "$SECONDARY_FIXTURE" ]]; then
  [[ -f "$SECONDARY_FIXTURE" ]] || fail "Secondary fixture not found: $SECONDARY_FIXTURE"
  secondary_upload="$(upload_video "$SECONDARY_FIXTURE" "${TITLE_PREFIX}-secondary")"
  secondary_video_id="$(printf '%s' "$secondary_upload" | jq -r '.id')"
  log "uploaded secondary video ${secondary_video_id}"
  poll_video "$secondary_video_id"
  assert_summary "$secondary_video_id"
  assert_similar "$primary_video_id"
fi

if (( RUN_BATCH )); then
  if [[ ${#BATCH_FIXTURES[@]} -eq 0 ]]; then
    BATCH_FIXTURES=("$PRIMARY_FIXTURE")
    [[ -n "$SECONDARY_FIXTURE" ]] && BATCH_FIXTURES+=("$SECONDARY_FIXTURE")
  fi
  [[ ${#BATCH_FIXTURES[@]} -gt 0 ]] || fail "No batch fixtures provided"
  local_missing=0
  for fixture in "${BATCH_FIXTURES[@]}"; do
    if [[ ! -f "$fixture" ]]; then
      log "missing batch fixture: $fixture"
      local_missing=1
    fi
  done
  (( local_missing == 0 )) || fail "One or more batch fixtures are missing"

  batch_upload="$(upload_batch "${TITLE_PREFIX}-batch" "${BATCH_FIXTURES[@]}")"
  batch_id="$(printf '%s' "$batch_upload" | jq -r '.id')"
  log "uploaded batch ${batch_id}"
  poll_batch "$batch_id"
  api_get "/api/batch?page_size=5" | jq -e '.total >= 1 and (.batches | length) >= 1' >/dev/null \
    || fail "batch listing missing rows"
  log "batch listing ok"
fi

run_gpu_idle_probe
log "acceptance complete"
