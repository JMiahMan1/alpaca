#!/usr/bin/env bash
# restart-when-idle.sh — restart a compose service without killing a benchmark.
#
# Polls the web dashboard until no benchmark run is active, then restarts the
# given service so a pulled update goes live. Safe to launch via nohup and
# forget; it only acts once the box is idle.
#
# Usage:
#   ./scripts/restart-when-idle.sh [service] [max_wait_seconds] [poll_seconds]
#
# Defaults: service=alpaca-web, max_wait=86400 (24h), poll=60s.
# Env: STATUS_URL (default http://localhost:5000/api/status).
# Logs progress to .tmp/restart-when-idle.log.
set -euo pipefail

SERVICE="${1:-alpaca-web}"
MAX_WAIT_S="${2:-86400}"
POLL_S="${3:-60}"
STATUS_URL="${STATUS_URL:-http://localhost:5000/api/status}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO_DIR/.tmp/restart-when-idle.log"
mkdir -p "$(dirname "$LOG")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

is_idle() {
    local body status
    body="$(curl -s -m 10 "$STATUS_URL" 2>/dev/null)" || return 1
    status="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' <<<"$body" 2>/dev/null)" || return 1
    [ "$status" = "idle" ]
}

log "watching for idle (service=$SERVICE, max_wait=${MAX_WAIT_S}s, poll=${POLL_S}s)"
deadline=$(( $(date +%s) + MAX_WAIT_S ))
while true; do
    if is_idle; then
        # Re-check once to avoid racing a run that just started.
        sleep 5
        if is_idle; then
            log "box idle — restarting $SERVICE"
            (cd "$REPO_DIR" && sudo docker compose restart "$SERVICE")
            log "restart issued for $SERVICE"
            exit 0
        fi
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        log "gave up after ${MAX_WAIT_S}s: a run is still active; restart NOT performed"
        exit 1
    fi
    sleep "$POLL_S"
done
