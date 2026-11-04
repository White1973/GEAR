#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${GEAR_SFT_CONFIG:-${SCRIPT_DIR}/configs/qwen2_vl_3b_full_sft.yaml}"

command -v llamafactory-cli >/dev/null 2>&1 || {
  echo "llamafactory-cli was not found. Install or activate LLaMA-Factory first." >&2
  exit 1
}

exec llamafactory-cli train "${CONFIG_PATH}" "$@"
