#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/check_pretrain.sh \
    --pt <PT_ID> \
    [--gen-script <gen_yamls_*.py>] \
    [--device <cuda:0|cpu>] \
    [--log-dir <path>] \
    [--debug]

Examples:
  bash scripts/check_pretrain.sh --pt 1 --gen-script gen_yamls_r01.py
  bash scripts/check_pretrain.sh --pt 1 --gen-script gen_yamls_r01.py --device cuda:0 --debug
EOF
}

PT_ID=""
GEN_SCRIPT="gen_yamls_r01.py"
DEVICE="cuda:0"
DEBUG=0
LOG_DIR="./logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pt)
      PT_ID="${2:?--pt requires value}"
      shift 2
      ;;
    --gen-script)
      GEN_SCRIPT="${2:?--gen-script requires value}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?--device requires value}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:?--log-dir requires value}"
      shift 2
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PT_ID" ]]; then
  echo "[ERR] --pt is required."
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RES_DIR="${PROJECT_DIR}/resources"
GEN_SCRIPT_PATH="${RES_DIR}/${GEN_SCRIPT}"

if [[ ! -f "$GEN_SCRIPT_PATH" ]]; then
  echo "[ERR] generator script not found: ${GEN_SCRIPT_PATH}"
  exit 1
fi

PT_BLOCK=$((PT_ID / 100))
PT_FILE="pt$(printf '%04d' "${PT_ID}").yaml"
PT_DIR="$(printf 'pt%02ds' "${PT_BLOCK}")"
PT_PATH="${RES_DIR}/pretrain_yamls/${PT_DIR}/${PT_FILE}"
LOG_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${PROJECT_DIR}/${LOG_DIR}/pretrain_${PT_ID}_${LOG_STAMP}.log"

mkdir -p "${PROJECT_DIR}/${LOG_DIR}"

echo "[1/3] generate pretrain yamls -> ${GEN_SCRIPT_PATH}"
cd "$RES_DIR"
python "$GEN_SCRIPT_PATH"

if [[ ! -f "$PT_PATH" ]]; then
  echo "[ERR] YAML not found: ${PT_PATH}"
  echo "[ERR] Please check --pt and generator script output."
  exit 1
fi
echo "[OK] YAML exists: ${PT_PATH}"

echo "[2/3] run pretraining (pt=${PT_ID}, device=${DEVICE}, debug=${DEBUG})"
cd "$PROJECT_DIR"
if [[ "$DEBUG" == "1" ]]; then
  CMD=(python pretrain.py --pt "${PT_ID}" --device "${DEVICE}" --debug)
else
  CMD=(python pretrain.py --pt "${PT_ID}" --device "${DEVICE}")
fi

"${CMD[@]}" | tee "$LOG_FILE"

echo "[3/3] verify log tail"
echo "--- log tail: ${LOG_FILE} ---"
tail -n 60 "$LOG_FILE"
echo "[DONE] log saved: ${LOG_FILE}"
