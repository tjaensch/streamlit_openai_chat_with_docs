#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# sanitize-report.sh — Redact tokens and auth headers from CSI reports
# ────────────────────────────────────────────────────────────────────────────
# Usage: sanitize-report.sh <input-file> [output-file]
# If output-file is omitted, writes to stdout.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

INPUT="${1:?Usage: sanitize-report.sh <input-file> [output-file]}"
OUTPUT="${2:-/dev/stdout}"

if [[ ! -f "$INPUT" ]]; then
  echo "Error: Input file '$INPUT' does not exist." >&2
  exit 1
fi

sed -E \
  -e 's/(^|[^[:alnum:]_])(gh[porsu]_[A-Za-z0-9_]{20,})([^[:alnum:]_]|$)/\1[REDACTED_TOKEN]\3/g' \
  -e 's/(^|[^[:alnum:]_])(github_pat_[A-Za-z0-9_]{20,})([^[:alnum:]_]|$)/\1[REDACTED_TOKEN]\3/g' \
  -e 's/(^|[^[:alnum:]_])(sk-[A-Za-z0-9_-]{20,})([^[:alnum:]_]|$)/\1[REDACTED_TOKEN]\3/g' \
  -e 's/(authorization:)[[:space:]]*(Bearer|token|Basic)[[:space:]]+[^[:space:]]+/\1 \2 [REDACTED]/gI' \
  -e 's/(x-access-token:)[[:space:]]*[^[:space:]]+/\1 [REDACTED]/gI' \
  -e 's/(OPENAI_API_KEY|COPILOT_TOKEN|CSI_PAT|GH_TOKEN|GITHUB_TOKEN)=[^[:space:]"'"'"']+/\1=[REDACTED]/g' \
  "$INPUT" > "$OUTPUT"
