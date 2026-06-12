#!/usr/bin/env bash
# Ingest a local file into the RAG pipeline.
#
# Usage:
#   ./scripts/ingest.sh <file> [server_url] [source_id] [title]
#
# Examples:
#   ./scripts/ingest.sh report.pdf
#   ./scripts/ingest.sh notes.md http://localhost:8000
#   ./scripts/ingest.sh paper.pdf http://localhost:8000 my-paper "Attention Is All You Need"
#
# Environment variables (override defaults):
#   API_KEY     — X-API-Key header value  (default: value from .env, then "changeme")
#   SERVER_URL  — base URL of the server  (default: http://localhost:8000)

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────

die() { echo "error: $*" >&2; exit 1; }

usage() {
  echo "Usage: $0 <file> [server_url] [source_id] [title]"
  echo ""
  echo "Supported file types: .pdf  .txt  .text  .rst  .md  .markdown"
  echo ""
  echo "Options (also settable via environment):"
  echo "  API_KEY     X-API-Key header  (default: read from .env or 'changeme')"
  echo "  SERVER_URL  Server base URL   (default: http://localhost:8000)"
  exit 1
}

# ── load .env if present (for API_KEY) ────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
fi

# ── arguments ─────────────────────────────────────────────────────────────────

[[ $# -lt 1 ]] && usage

FILE="$1"
SERVER_URL="${2:-${SERVER_URL:-http://localhost:8000}}"
SOURCE_ID="${3:-}"
TITLE="${4:-}"
API_KEY="${API_KEY:-changeme}"

# ── validate ──────────────────────────────────────────────────────────────────

[[ -f "$FILE" ]] || die "file not found: $FILE"

command -v curl >/dev/null 2>&1 || die "curl is required but not installed"

EXT="${FILE##*.}"
EXT="${EXT,,}"   # lowercase
case "$EXT" in
  pdf|txt|text|rst|md|markdown) ;;
  *) die "unsupported file type '.${EXT}'. Supported: pdf txt text rst md markdown" ;;
esac

# ── default source_id to filename ─────────────────────────────────────────────

if [[ -z "$SOURCE_ID" ]]; then
  SOURCE_ID="$(basename "$FILE")"
fi

# ── build curl args ───────────────────────────────────────────────────────────

CURL_ARGS=(
  --silent
  --show-error
  --fail-with-body
  -X POST
  "${SERVER_URL}/v1/ingest/file"
  -H "X-API-Key: ${API_KEY}"
  -F "file=@${FILE}"
  -F "source_id=${SOURCE_ID}"
)

if [[ -n "$TITLE" ]]; then
  CURL_ARGS+=(-F "title=${TITLE}")
fi

# ── run ───────────────────────────────────────────────────────────────────────

echo "Ingesting: $FILE"
echo "  server:    $SERVER_URL"
echo "  source_id: $SOURCE_ID"
[[ -n "$TITLE" ]] && echo "  title:     $TITLE"
echo ""

RESPONSE=$(curl "${CURL_ARGS[@]}") || {
  echo ""
  die "request failed. Is the server running at $SERVER_URL?"
}

echo "$RESPONSE" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f\"  status:      {d.get('status', '?')}\")
    print(f\"  source_id:   {d.get('source_id', '?')}\")
    print(f\"  chunk_count: {d.get('chunk_count', '?')}\")
except Exception:
    print(sys.stdin.read())
" 2>/dev/null || echo "$RESPONSE"
