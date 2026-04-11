#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# install-copilot-cli.sh — Install GitHub Copilot CLI with fallback strategy
# ────────────────────────────────────────────────────────────────────────────
# Tries the latest release first, then falls back through known-good versions.
# Each version is attempted up to COPILOT_RETRIES+1 times total (default: 3).
#
# Usage:
#   .github/scripts/install-copilot-cli.sh [--pin <tag>]
#
# Arguments:
#   --pin <tag>  Pin to a specific release tag (e.g., v1.0.24). Skips latest
#                and fallback versions — only the pinned tag is attempted.
#
# Environment variables:
#   GH_TOKEN              Optional but recommended — GitHub token for authenticated
#                         downloads. Unauthenticated requests may hit API rate limits.
#   COPILOT_FALLBACK_VERSIONS   Comma-separated fallback tags to try after latest.
#   COPILOT_FALLBACK_COUNT      How many recent releases to auto-discover (default: 3).
#   COPILOT_RETRIES       Retries per version attempt (default: 2).
#   GITHUB_OUTPUT         If set, writes cli_version and cli_version_output.
#   GITHUB_PATH           If set, appends install dir to $PATH for later steps.
#
# Outputs (GITHUB_OUTPUT when running in Actions):
#   cli_version          Parsed version tag  (e.g., v1.0.24)
#   cli_version_output   Raw `copilot --version` line
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Helpers ────────────────────────────────────────────────────────────────
validate_non_negative_integer() {
  local var_name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "::error::${var_name} must be a non-negative integer, got '${value}'"
    exit 1
  fi
  printf '%s\n' "$value"
}

# ── Defaults ──────────────────────────────────────────────────────────────
FALLBACK_VERSIONS="${COPILOT_FALLBACK_VERSIONS:-}"
FALLBACK_COUNT="$(validate_non_negative_integer "COPILOT_FALLBACK_COUNT" "${COPILOT_FALLBACK_COUNT:-3}")"
MAX_RETRIES="$(validate_non_negative_integer "COPILOT_RETRIES" "${COPILOT_RETRIES:-2}")"
INSTALL_DIR="$HOME/.local/bin"
PATTERN="copilot-linux-x64.tar.gz"
REPO="github/copilot-cli"

PIN_VERSION=""

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pin)
      shift
      PIN_VERSION="${1:-}"
      if [[ -z "$PIN_VERSION" ]]; then
        echo "::error::--pin requires a version tag argument (e.g., v1.0.24)"
        exit 1
      fi
      if [[ ! "$PIN_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([._-][0-9A-Za-z]+)*$ ]]; then
        echo "::error::Invalid pinned version '$PIN_VERSION'. Expected tag format like v1.0.24"
        exit 1
      fi
      shift
      ;;
    *)
      echo "::error::Unknown argument '$1'. Usage: install-copilot-cli.sh [--pin <tag>]"
      exit 1
      ;;
  esac
done

# ── Build ordered version list ────────────────────────────────────────────
declare -a VERSIONS_TO_TRY=()

if [[ -n "$PIN_VERSION" ]]; then
  VERSIONS_TO_TRY=("$PIN_VERSION")
  echo "ℹ Pinned to Copilot CLI $PIN_VERSION (no fallback)."
else
  VERSIONS_TO_TRY=("latest")

  if [[ -n "$FALLBACK_VERSIONS" ]]; then
    IFS=',' read -ra FALLBACK_ARRAY <<< "$FALLBACK_VERSIONS"
    for v in "${FALLBACK_ARRAY[@]}"; do
      v="$(printf '%s' "$v" | xargs)"
      [[ -n "$v" ]] && VERSIONS_TO_TRY+=("$v")
    done
    echo "ℹ Will try: latest → ${FALLBACK_VERSIONS}"
  else
    DISCOVER_LIMIT=$((FALLBACK_COUNT + 1))
    echo "ℹ Auto-discovering last ${FALLBACK_COUNT} stable fallback releases from ${REPO}..."
    DISCOVERED="$(gh release list --repo "$REPO" \
                    --exclude-drafts --exclude-pre-releases \
                    --limit "$DISCOVER_LIMIT" \
                    --json tagName --jq '.[].tagName' 2>/dev/null || true)"
    if [[ -n "$DISCOVERED" ]]; then
      declare -A SEEN_TAGS=()
      SKIP_FIRST=1
      while IFS= read -r tag; do
        [[ -z "$tag" ]] && continue
        if [[ "$SKIP_FIRST" -eq 1 ]]; then
          SKIP_FIRST=0
          continue
        fi
        if [[ -z "${SEEN_TAGS[$tag]:-}" ]]; then
          SEEN_TAGS["$tag"]=1
          VERSIONS_TO_TRY+=("$tag")
        fi
      done <<< "$DISCOVERED"
      if [[ ${#VERSIONS_TO_TRY[@]} -eq 1 ]]; then
        echo "ℹ Will try: latest (no additional fallbacks discovered)"
      else
        echo "ℹ Will try: latest → ${VERSIONS_TO_TRY[*]:1}"
      fi
    else
      VERSIONS_TO_TRY+=("v1.0.24")
      echo "::warning::Could not list releases from ${REPO}; using hardcoded fallback v1.0.24"
    fi
  fi
fi

# ── Helper: attempt a download + install for one version ──────────────────
try_install() {
  local tag="$1"
  local download_dir
  download_dir="$(mktemp -d)"
  local stderr_log="$download_dir/stderr.log"

  local release_args=()
  if [[ "$tag" != "latest" ]]; then
    release_args=("$tag")
  fi

  if ! gh release download "${release_args[@]}" \
       --repo "$REPO" \
       --pattern "$PATTERN" \
       --dir "$download_dir" 2>"$stderr_log"; then
    local snippet
    snippet="$(head -c 200 "$stderr_log" 2>/dev/null || true)"
    [[ -n "$snippet" ]] && echo "  stderr: $snippet"
    rm -rf "$download_dir"
    return 1
  fi

  local tarball="$download_dir/$PATTERN"
  if [[ ! -s "$tarball" ]]; then
    echo "  Downloaded tarball is missing or empty"
    rm -rf "$download_dir"
    return 1
  fi

  # Integrity verification — SHA256SUMS if available
  local checksum_verified=false
  if gh release download "${release_args[@]}" \
       --repo "$REPO" \
       --pattern "SHA256SUMS.txt" \
       --dir "$download_dir" 2>/dev/null; then
    if [[ -s "$download_dir/SHA256SUMS.txt" ]]; then
      if ! (
        cd "$download_dir" && \
        awk -v expected="$PATTERN" '
          {
            filename = $0
            sub(/^[^[:space:]]+[[:space:]]+\*?/, "", filename)
            if (filename == expected) { print; count++ }
          }
          END { exit(count == 1 ? 0 : 1) }
        ' SHA256SUMS.txt | sha256sum -c - >/dev/null 2>&1
      ); then
        echo "  ⚠ SHA256 verification failed for $tag"
        rm -rf "$download_dir"
        return 1
      fi
      checksum_verified=true
      echo "  ✓ SHA256 checksum verified"
    fi
  fi

  if [[ "$checksum_verified" != "true" ]]; then
    if [[ -n "${PIN_VERSION:-}" ]]; then
      echo "  ⚠ SHA256SUMS.txt unavailable for pinned version $tag — failing closed"
      rm -rf "$download_dir"
      return 1
    else
      echo "  ⚠ SHA256SUMS.txt unavailable for $tag — proceeding without checksum verification"
    fi
  fi

  local extract_dir
  extract_dir="$(mktemp -d)"

  # Safety: reject archives containing absolute paths or '..' components
  local bad_entries
  bad_entries="$(tar -tzf "$tarball" 2>/dev/null | grep -E '(^/|\.\./)' || true)"
  if [[ -n "$bad_entries" ]]; then
    echo "  ⚠ Archive contains unsafe paths — rejecting: $bad_entries"
    rm -rf "$download_dir" "$extract_dir"
    return 1
  fi

  if ! tar -xzf "$tarball" -C "$extract_dir" --no-anchored 'copilot' --no-wildcards 2>"$stderr_log"; then
    if ! tar -xzf "$tarball" -C "$extract_dir" 2>"$stderr_log"; then
      local snippet
      snippet="$(head -c 200 "$stderr_log" 2>/dev/null || true)"
      [[ -n "$snippet" ]] && echo "  tar stderr: $snippet"
      rm -rf "$download_dir" "$extract_dir"
      return 1
    fi
  fi
  rm -rf "$download_dir"

  if [[ ! -f "$extract_dir/copilot" ]]; then
    rm -rf "$extract_dir"
    return 1
  fi

  if ! chmod +x "$extract_dir/copilot"; then
    rm -rf "$extract_dir"
    return 1
  fi

  mkdir -p "$INSTALL_DIR"
  mv -f "$extract_dir/copilot" "$INSTALL_DIR/copilot"
  rm -rf "$extract_dir"

  if ! "$INSTALL_DIR/copilot" --version >/dev/null 2>&1; then
    rm -f "$INSTALL_DIR/copilot"
    return 1
  fi

  return 0
}

# ── Main loop: try each version with retries ──────────────────────────────
INSTALLED=false

for version in "${VERSIONS_TO_TRY[@]}"; do
  label="$version"
  attempt=0

  while (( attempt <= MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "→ Attempting Copilot CLI install: $label (try $attempt/$((MAX_RETRIES + 1)))"

    if try_install "$version"; then
      INSTALLED=true
      echo "✓ Successfully installed Copilot CLI ($label)"
      break 2
    fi

    if (( attempt <= MAX_RETRIES )); then
      sleep_secs=$((attempt * 5))
      echo "  ✗ Attempt $attempt failed for $label — retrying in ${sleep_secs}s..."
      sleep "$sleep_secs"
    else
      echo "  ✗ All $((MAX_RETRIES + 1)) attempts exhausted for $label."
    fi
  done
done

if [[ "$INSTALLED" != "true" ]]; then
  echo "::error::Failed to install Copilot CLI. Tried: ${VERSIONS_TO_TRY[*]}. Check network connectivity, GH_TOKEN permissions, and rate limits."
  exit 1
fi

# ── Capture and validate version ──────────────────────────────────────────
COPILOT_VERSION_OUTPUT="$("$INSTALL_DIR/copilot" --version | head -n 1)"
echo "$COPILOT_VERSION_OUTPUT"

COPILOT_VERSION_PARSED="$(printf '%s' "$COPILOT_VERSION_OUTPUT" \
  | sed -nE 's/.*CLI[[:space:]]+v?([0-9]+\.[0-9]+\.[0-9]+([._-][0-9A-Za-z]+)*).*/v\1/p')"

if [[ -z "$COPILOT_VERSION_PARSED" ]]; then
  COPILOT_VERSION_PARSED="$(printf '%s' "$COPILOT_VERSION_OUTPUT" \
    | sed -nE 's/.*[^0-9]?v?([0-9]+\.[0-9]+\.[0-9]+).*/v\1/p' | head -1)"
fi

if [[ -z "$COPILOT_VERSION_PARSED" ]]; then
  echo "::warning::Unable to parse Copilot CLI version from: $COPILOT_VERSION_OUTPUT"
  COPILOT_VERSION_PARSED="unknown"
fi

if [[ -n "$PIN_VERSION" ]]; then
  if [[ "$COPILOT_VERSION_PARSED" == "unknown" ]]; then
    echo "::error::Unable to verify installed Copilot CLI version for pinned version '$PIN_VERSION'. Raw output: $COPILOT_VERSION_OUTPUT"
    exit 1
  fi
  if [[ "$COPILOT_VERSION_PARSED" != "$PIN_VERSION" ]]; then
    echo "::error::Installed Copilot CLI version '$COPILOT_VERSION_PARSED' does not match pinned version '$PIN_VERSION'."
    exit 1
  fi
fi

# ── Expose to PATH and GITHUB_OUTPUT ──────────────────────────────────────
if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "$INSTALL_DIR" >> "$GITHUB_PATH"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "cli_version=${COPILOT_VERSION_PARSED}" >> "$GITHUB_OUTPUT"
  echo "cli_version_output=${COPILOT_VERSION_OUTPUT}" >> "$GITHUB_OUTPUT"
fi

echo "✓ Copilot CLI $COPILOT_VERSION_PARSED ready at $INSTALL_DIR/copilot"
