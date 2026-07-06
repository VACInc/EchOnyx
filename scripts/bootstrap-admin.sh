#!/usr/bin/env bash
set -euo pipefail

MODE="compose"
BASE_URL="http://127.0.0.1:8000"
PASSWORD="${ECHONYX_PASSWORD:-}"

BOOTSTRAP_PY='
import json
import os
import sys
import urllib.error
import urllib.request


def summarize_detail(body):
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:500] if body else "no response body"

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            location = ".".join(str(part) for part in item.get("loc", []))
            message = str(item.get("msg", "validation error"))
            messages.append(f"{location}: {message}" if location else message)
        return "; ".join(messages) if messages else "validation error"
    if detail is not None:
        return json.dumps(detail, separators=(",", ":"))
    return body[:500] if body else "no response body"


password = sys.stdin.read()
if password.endswith("\n"):
    password = password[:-1]
if not password:
    print("[bootstrap-admin] ERROR: password is empty", file=sys.stderr)
    sys.exit(1)

base_url = os.environ.get("ECHONYX_BOOTSTRAP_URL", "http://127.0.0.1:8000").rstrip("/")
request = urllib.request.Request(
    f"{base_url}/api/auth/setup",
    data=json.dumps({"password": password}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.status
        body = response.read().decode("utf-8", "replace")
except urllib.error.HTTPError as exc:
    status = exc.code
    body = exc.read().decode("utf-8", "replace")
except urllib.error.URLError as exc:
    print(f"[bootstrap-admin] ERROR: could not reach {base_url}/api/auth/setup: {exc}", file=sys.stderr)
    sys.exit(1)

detail = summarize_detail(body)
if status in {200, 201}:
    print("[bootstrap-admin] Admin account created.")
    sys.exit(0)
if status == 409 and "already" in detail.lower():
    print("[bootstrap-admin] Authentication is already configured; no changes made.")
    sys.exit(0)

print(f"[bootstrap-admin] ERROR: setup failed with HTTP {status}: {detail}", file=sys.stderr)
sys.exit(1)
'

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap-admin.sh [options]

Options:
  --compose       POST /api/auth/setup from inside the Compose backend container. Default.
  --url URL       POST /api/auth/setup directly from this host against URL.
  --help          Show this help.

Password input:
  Set ECHONYX_PASSWORD, or run interactively and enter the password twice.
EOF
}

log() {
  printf '[bootstrap-admin] %s\n' "$*"
}

fail() {
  printf '[bootstrap-admin] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    printf 'python3\n'
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    printf 'python\n'
    return 0
  fi
  fail "Missing required command: python3 or python"
}

prompt_password() {
  if [[ -n "$PASSWORD" ]]; then
    return 0
  fi

  local first
  local second
  read -r -s -p "Admin password: " first
  printf '\n'
  read -r -s -p "Confirm admin password: " second
  printf '\n'

  [[ -n "$first" ]] || fail "password is empty"
  [[ "$first" == "$second" ]] || fail "passwords did not match"
  PASSWORD="$first"
}

run_compose_bootstrap() {
  require_cmd docker
  docker compose version >/dev/null 2>&1 || fail "Docker Compose plugin is required"
  log "Posting setup request from inside the backend container"
  printf '%s' "$PASSWORD" | env -u ECHONYX_PASSWORD docker compose exec -T backend python -c "$BOOTSTRAP_PY"
}

run_url_bootstrap() {
  local python_bin
  python_bin="$(find_python)"
  log "Posting setup request to ${BASE_URL}/api/auth/setup"
  printf '%s' "$PASSWORD" | env -u ECHONYX_PASSWORD ECHONYX_BOOTSTRAP_URL="$BASE_URL" "$python_bin" -c "$BOOTSTRAP_PY"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose)
      MODE="compose"
      shift
      ;;
    --url)
      [[ $# -ge 2 ]] || fail "--url requires a URL"
      MODE="url"
      BASE_URL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

prompt_password

case "$MODE" in
  compose)
    run_compose_bootstrap
    ;;
  url)
    run_url_bootstrap
    ;;
  *)
    fail "unknown mode: $MODE"
    ;;
esac
